#include "h3img_vae.h"
#include "h3_weights.h"
#include "h3img_gpu_ext.h"

#include <Accelerate/Accelerate.h>
#include <vecLib/vForce.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The converted checkpoint (vae_diffusers/) stores every tensor in F32 with
 * diffusers key names. The encoder is a causal Conv3D stack, but an image is
 * a single frame and the causal padding prepends only zero frames, so every
 * k3 kernel reads just its last temporal tap: the whole encoder is a 2D
 * convolution stack in NCHW fp32. */

enum {
    IMG_LATENT_DIV = 16, /* spatial compression ratio = patch size */
    LATENT_CHANNELS = 24,
    QUANT_CHANNELS = 48,
    H3IMG_MAX_TILES = 34, /* released decode tiling: the official size table
                           * tops out at 6272 px/side (21:9), which needs 33
                           * tiles; starts/overlaps stack arrays size with
                           * this constant */
    VIT_DIM = 1024,
    VIT_HEADS = 16,
    VIT_HEAD_DIM = 64,
    VIT_ROTARY = 32,
    VIT_LAYERS = 24,
    VIT_REGISTERS = 4,
    VIT_FFN = 4096,
    GROUPS = 32
};

static const int block_channels[6] = {128, 256, 256, 512, 512, 1024};
static const int block_downsample[6] = {1, 1, 1, 1, 0, 0};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float *read_f32(h3_weight_store *store, const char *name,
                       size_t elements, char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != elements) {
        fail(error, error_size, "VAE tensor %s is absent or malformed", name);
        return NULL;
    }
    float *values = malloc(elements * sizeof(float));
    if (!values) {
        fail(error, error_size, "out of memory reading VAE tensor %s", name);
        return NULL;
    }
    char detail[256];
    if (!h3_st_read_data(header, tensor, values, elements * sizeof(float),
                         detail, sizeof(detail))) {
        fail(error, error_size, "cannot read VAE tensor %s: %s", name, detail);
        free(values);
        return NULL;
    }
    return values;
}

/* Slice the live temporal tap (kt index 2) out of a [out][in][3][3][3]
 * causal Conv3D weight: an image's only frame is preceded by two zero
 * frames, so the taps at kt 0/1 multiply zeros and vanish. */
static float *tap_slice(const float *weight5, int out_channels, int channels,
                        char *error, size_t error_size) {
    float *tap = malloc((size_t)out_channels * channels * 9 * sizeof(float));
    if (!tap) {
        fail(error, error_size, "out of memory slicing VAE conv tap");
        return NULL;
    }
    for (int oc = 0; oc < out_channels; oc++)
        for (int ic = 0; ic < channels; ic++)
            for (int k = 0; k < 9; k++)
                tap[((size_t)oc * channels + ic) * 9 + k] =
                    weight5[((size_t)oc * channels + ic) * 27 + 2 * 9 + k];
    return tap;
}

static void dump_stage(const char *path, const float *data, size_t count);

/* ---------------------------------------------------------------- conv ---- */

/* One 2D convolution with 1-px reflect padding on all sides and stride 1 or
 * 2. With stride 2 the reflect pad is applied first (matching the reference,
 * which pads bottom/right by 1 and then runs a valid conv), which for a
 * 1-px pad is identical to reflecting on the padded extent. Scalar reference
 * path (H3IMG_VAE_SCALAR=1). */
static void conv2d_k3_scalar(const float *in, int channels, int height,
                             int width, const float *weight,
                             const float *bias, int out_channels, int stride,
                             float *out, int out_height, int out_width,
                             int padding) {
    for (int oc = 0; oc < out_channels; oc++) {
        const float *plane = weight + (size_t)oc * channels * 9;
        for (int oy = 0; oy < out_height; oy++) {
            for (int ox = 0; ox < out_width; ox++) {
                float accumulator = bias ? bias[oc] : 0.0f;
                /* Input coordinates covered by the 3x3 window centered on the
                 * reflect-padded input: input pixel (iy, ix) maps to output
                 * (iy / stride) with the pad occupying the border. */
                int base_y = oy * stride - padding;
                int base_x = ox * stride - padding;
                for (int ic = 0; ic < channels; ic++) {
                    const float *input = in + (size_t)ic * height * width;
                    const float *kernel = plane + (size_t)ic * 9;
                    float sum = 0.0f;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = base_y + ky;
                        if (padding) {
                            if (iy < 0) iy = -iy;
                            else if (iy >= height) iy = 2 * (height - 1) - iy;
                            if (iy < 0 || iy >= height) continue;
                        } else if (iy < 0 || iy >= height) {
                            continue;
                        }
                        const float *line = input + (size_t)iy * width;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = base_x + kx;
                            float sample;
                            if (padding) {
                                if (ix < 0) ix = -ix;
                                else if (ix >= width)
                                    ix = 2 * (width - 1) - ix;
                                if (ix < 0 || ix >= width) continue;
                            } else if (ix < 0 || ix >= width) {
                                continue;
                            }
                            sample = line[ix];
                            sum += sample * kernel[ky * 3 + kx];
                        }
                    }
                    accumulator += sum;
                }
                out[(size_t)oc * out_height * out_width +
                    (size_t)oy * out_width + ox] = accumulator;
            }
        }
    }
}

/* GEMM engine. The weight arrives as [out][in][9] (one temporal tap of the
 * checkpoint's [out][in][3][3][3]). The input is repacked once into a
 * channels-last padded ring: head guard rows, the padded grid, tail guard
 * rows. Within that layout the input IS the col-major (K=channels) x
 * (M=grid) matrix of the convolution, so every 3x3 tap becomes one
 * Accumulate-handling cblas_sgemm over the full padded grid; the interior is
 * gathered (with the bias) into the planar output afterwards. Output rows are
 * processed in blocks to bound the scratch matrix. Accumulation runs through
 * the AMX-backed sgemm instead of the scalar per-pixel loop — different
 * rounding order, same math (H3IMG_VAE_SCALAR=1 restores the reference
 * kernel). */
#define CONV_TILE_ROWS 64

static void conv2d_k3_gemm(const float *in, int channels, int height,
                           int width, const float *weight, const float *bias,
                           int out_channels, int stride, float *out,
                           int out_height, int out_width, int padding) {
    int pad = padding ? 1 : 0;
    int h2 = height + 2 * pad, w2 = width + 2 * pad;
    size_t grid = (size_t)h2 * w2;
    /* Guard rows on both sides: the deepest tap view reads head-2 plus two
     * full grid rows past the end, so the head must cover 2*w2+2 samples. */
    size_t head = (size_t)2 * w2 + 3;
    size_t span = head + grid + head;         /* head + grid + tail */
    size_t plane_in = (size_t)height * width;

    /* Channels-last padded grid (zeros double as the out-of-range samples the
     * scalar kernel skips after reflection). */
    float *cl = calloc(span * (size_t)channels, sizeof(float));
    float *repacked = malloc((size_t)9 * channels * out_channels *
                             sizeof(float));
    if (!cl || !repacked) {
        free(cl);
        free(repacked);
        /* Fail soft: the callers own their buffers; falling back keeps the
         * pipeline correct (slow) rather than aborting a render. */
        conv2d_k3_scalar(in, channels, height, width, weight, bias,
                         out_channels, stride, out, out_height, out_width,
                         padding);
        return;
    }
    for (int iy = -pad; iy < height + pad; iy++) {
        int sy = iy;
        if (pad) {
            if (sy < 0) sy = -sy;
            else if (sy >= height) sy = 2 * (height - 1) - sy;
        }
        for (int ix = -pad; ix < width + pad; ix++) {
            int sx = ix;
            if (pad) {
                if (sx < 0) sx = -sx;
                else if (sx >= width) sx = 2 * (width - 1) - sx;
            }
            float *dst = cl + (head + (size_t)(iy + pad) * w2 + (ix + pad)) *
                channels;
            const float *src = in + (size_t)sy * width + sx;
            for (int c = 0; c < channels; c++)
                dst[c] = src[(size_t)c * plane_in];
        }
    }
    for (int oc = 0; oc < out_channels; oc++) {
        const float *plane = weight + (size_t)oc * channels * 9;
        for (int ic = 0; ic < channels; ic++)
            for (int tap = 0; tap < 9; tap++)
                repacked[(size_t)tap * channels * out_channels +
                         (size_t)ic + (size_t)oc * channels] =
                    plane[(size_t)ic * 9 + tap];
    }

    for (int oy0 = 0; oy0 < out_height; oy0 += CONV_TILE_ROWS) {
        int rows = out_height - oy0 < CONV_TILE_ROWS ?
            out_height - oy0 : CONV_TILE_ROWS;
        /* Padded-grid rows spanned by this tile of output rows. */
        size_t m_lo = (size_t)stride * oy0 * w2;
        size_t m_hi = (size_t)stride * (oy0 + rows - 1) * w2 +
            (size_t)2 * w2 + w2;
        if (m_hi > grid) m_hi = grid;
        size_t tile_rows = m_hi - m_lo;
        float *temp = malloc(tile_rows * (size_t)out_channels *
                             sizeof(float));
        if (!temp) {
            free(cl);
            free(repacked);
            conv2d_k3_scalar(in, channels, height, width, weight, bias,
                             out_channels, stride, out, out_height, out_width,
                             padding);
            return;
        }
        for (int tap = 0; tap < 9; tap++) {
            int ky = tap / 3, kx = tap % 3;
            const float *a = cl + (head + m_lo + (size_t)ky * w2 + kx) *
                channels;
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        (int)tile_rows, out_channels, channels, 1.0f, a,
                        channels, repacked + (size_t)tap * channels *
                        out_channels, channels,
                        tap == 0 ? 0.0f : 1.0f, temp, (int)tile_rows);
        }
        for (int oc = 0; oc < out_channels; oc++) {
            float bias_value = bias ? bias[oc] : 0.0f;
            const float *column = temp + (size_t)oc * tile_rows;
            float *target = out + (size_t)oc * out_height * out_width +
                (size_t)oy0 * out_width;
            for (int oy = 0; oy < rows; oy++) {
                /* The tile's C rows are the padded grid from m_lo; output row
                 * (oy0+oy) gathers at grid row stride*(oy0+oy). */
                const float *line = column +
                    (size_t)stride * oy * w2;
                float *dst = target + (size_t)oy * out_width;
                for (int ox = 0; ox < out_width; ox++)
                    dst[ox] = line[(size_t)stride * ox] + bias_value;
            }
        }
        free(temp);
    }
    free(cl);
    free(repacked);
}

static void conv2d_k3(const float *in, int channels, int height, int width,
                      const float *weight /* [out][in][3][3] */,
                      const float *bias, int out_channels, int stride,
                      float *out, int out_height, int out_width,
                      int padding /* 1 = reflect border, 0 = valid */) {
    if (getenv("H3IMG_VAE_SCALAR")) {
        conv2d_k3_scalar(in, channels, height, width, weight, bias,
                         out_channels, stride, out, out_height, out_width,
                         padding);
        return;
    }
    conv2d_k3_gemm(in, channels, height, width, weight, bias, out_channels,
                   stride, out, out_height, out_width, padding);
}

/* 1x1 convolution over NCHW planes: out[oc][index] =
 * bias + sum_ic weight[oc][ic] * in[ic][index]. The planar layout makes this
 * a single zero-copy sgemm (H3IMG_VAE_SCALAR=1 restores the scalar loop). */
static void conv1x1_scalar(const float *in, int channels, int count,
                           const float *weight /* [out][in] */,
                           const float *bias, int out_channels, float *out) {
    for (int oc = 0; oc < out_channels; oc++) {
        const float *row = weight + (size_t)oc * channels;
        for (int index = 0; index < count; index++) {
            double sum = bias ? bias[oc] : 0.0;
            for (int ic = 0; ic < channels; ic++)
                sum += (double)in[(size_t)ic * count + index] * row[ic];
            out[(size_t)oc * count + index] = (float)sum;
        }
    }
}

static void conv1x1(const float *in, int channels, int count,
                    const float *weight /* [out][in] */, const float *bias,
                    int out_channels, float *out) {
    if (getenv("H3IMG_VAE_SCALAR")) {
        conv1x1_scalar(in, channels, count, weight, bias, out_channels, out);
        return;
    }
    /* The planar input is col-major (M=count x K=channels) with lda=count;
     * the [out][in] weight is col-major (K x N) with ldb=channels; the
     * planar output is col-major (M x N) with ldc=count. */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, count,
                out_channels, channels, 1.0f, in, count, weight, channels,
                0.0f, out, count);
    if (bias)
        for (int oc = 0; oc < out_channels; oc++) {
            float *plane = out + (size_t)oc * count;
            for (int index = 0; index < count; index++)
                plane[index] += bias[oc];
        }
}

static void group_norm_silu(float *data, int channels, int count,
                            const float *weight, const float *bias) {
    int per_group = channels / GROUPS;
    int scalar = getenv("H3IMG_VAE_SCALAR") != NULL;
    float *scratch = NULL;
    if (!scalar) {
        scratch = malloc((size_t)count * sizeof(float));
        if (!scratch) scalar = 1;
    }
    for (int group = 0; group < GROUPS; group++) {
        int first = group * per_group;
        double sum = 0.0, squares = 0.0;
        for (int c = first; c < first + per_group; c++) {
            const float *plane = data + (size_t)c * count;
            for (int index = 0; index < count; index++) {
                double value = plane[index];
                sum += value;
                squares += value * value;
            }
        }
        double elements = (double)per_group * count;
        double mean = sum / elements;
        double variance = squares / elements - mean * mean;
        double inverse = 1.0 / sqrt(variance + 1e-6);
        for (int c = first; c < first + per_group; c++) {
            float gamma = weight ? weight[c] : 1.0f;
            float beta = bias ? bias[c] : 0.0f;
            float *plane = data + (size_t)c * count;
            if (scalar) {
                for (int index = 0; index < count; index++) {
                    double normalized =
                        ((double)plane[index] - mean) * inverse;
                    double activated = normalized * gamma + beta;
                    plane[index] =
                        (float)(activated / (1.0 + exp(-activated)));
                }
                continue;
            }
            /* SiLU via vvsExpf over the whole plane; the mean/inverse math
             * keeps the reference's double precision. */
            for (int index = 0; index < count; index++) {
                double normalized = ((double)plane[index] - mean) * inverse;
                double activated = normalized * gamma + beta;
                plane[index] = (float)activated;
                scratch[index] = (float)-activated;
            }
            int vector_count = count;
            vvexpf(scratch, scratch, &vector_count);
            for (int index = 0; index < count; index++)
                plane[index] =
                    plane[index] / (1.0f + scratch[index]);
        }
    }
    free(scratch);
}

/* ------------------------------------------------------------ encoder ---- */

static int resnet_block(h3_weight_store *store, char *name_prefix,
                        const float *input, int in_channels, int out_channels,
                        int height, int width, float *output,
                        char *error, size_t error_size) {
    int ok = 1;
    float *normed = malloc((size_t)in_channels * height * width * sizeof(float));
    float *hidden = malloc((size_t)out_channels * height * width * sizeof(float));
    float *shortcut = malloc((size_t)out_channels * height * width * sizeof(float));
    float *w1 = NULL, *b1 = NULL, *w2 = NULL, *b2 = NULL, *n1 = NULL, *b1n = NULL,
          *n2 = NULL, *b2n = NULL, *ws = NULL, *bs = NULL;
    if (!normed || !hidden || !shortcut) {
        fail(error, error_size, "out of memory in VAE resnet");
        ok = 0;
        goto cleanup;
    }
    char name[160];
    snprintf(name, sizeof(name), "%s.norm1.weight", name_prefix);
    n1 = read_f32(store, name, in_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm1.bias", name_prefix);
    b1n = read_f32(store, name, in_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1.weight", name_prefix);
    w1 = read_f32(store, name, (size_t)out_channels * in_channels * 27, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1.bias", name_prefix);
    b1 = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2.weight", name_prefix);
    n2 = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2.bias", name_prefix);
    b2n = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.conv2.weight", name_prefix);
    w2 = read_f32(store, name, (size_t)out_channels * out_channels * 27,
                  error, error_size);
    snprintf(name, sizeof(name), "%s.conv2.bias", name_prefix);
    b2 = read_f32(store, name, out_channels, error, error_size);
    if (!n1 || !b1n || !w1 || !b1 || !n2 || !b2n || !w2 || !b2) {
        ok = 0;
        goto cleanup;
    }
    float *w1t = tap_slice(w1, out_channels, in_channels, error, error_size);
    float *w2t = tap_slice(w2, out_channels, out_channels, error, error_size);
    if (!w1t || !w2t) {
        ok = 0;
        free(w1t);
        free(w2t);
        goto cleanup;
    }
    static int resnet_counter = 0;
    int instance = resnet_counter++;
    memcpy(normed, input, (size_t)in_channels * height * width * sizeof(float));
    group_norm_silu(normed, in_channels, height * width, n1, b1n);
    conv2d_k3(normed, in_channels, height, width, w1t, b1, out_channels, 1,
              hidden, height, width, 1);
    if (getenv("H3IMG_VAE_DUMP") && instance == 2) {
        dump_stage("/tmp/h3img_vae_r4_normed.f32", normed,
                   (size_t)in_channels * height * width);
        dump_stage("/tmp/h3img_vae_r4_conv1.f32", hidden,
                   (size_t)out_channels * height * width);
        dump_stage("/tmp/h3img_vae_r4_input.f32", input,
                   (size_t)in_channels * height * width);
    }
    group_norm_silu(hidden, out_channels, height * width, n2, b2n);
    conv2d_k3(hidden, out_channels, height, width, w2t, b2, out_channels, 1,
              output, height, width, 1);
    if (getenv("H3IMG_VAE_DUMP") && instance == 2)
        dump_stage("/tmp/h3img_vae_r4_conv2.f32", output,
                   (size_t)out_channels * height * width);
    free(w1t);
    free(w2t);
    if (in_channels != out_channels) {
        snprintf(name, sizeof(name), "%s.conv_shortcut.weight", name_prefix);
        ws = read_f32(store, name, (size_t)out_channels * in_channels,
                      error, error_size);
        snprintf(name, sizeof(name), "%s.conv_shortcut.bias", name_prefix);
        bs = read_f32(store, name, out_channels, error, error_size);
        if (!ws || !bs) {
            ok = 0;
            goto cleanup;
        }
        conv1x1(input, in_channels, height * width, ws, bs, out_channels,
                shortcut);
        if (getenv("H3IMG_VAE_DUMP") && instance == 2)
            dump_stage("/tmp/h3img_vae_r4_shortcut.f32", shortcut,
                       (size_t)out_channels * height * width);
    } else {
        memcpy(shortcut, input,
               (size_t)in_channels * height * width * sizeof(float));
    }
    for (size_t index = 0;
         index < (size_t)out_channels * height * width; index++)
        output[index] += shortcut[index];

cleanup:
    free(normed);
    free(hidden);
    free(shortcut);
    free(w1); free(b1); free(w2); free(b2);
    free(n1); free(b1n); free(n2); free(b2n);
    free(ws); free(bs);
    return ok;
}

static void dump_stage(const char *path, const float *data, size_t count) {
    FILE *file = fopen(path, "wb");
    if (file) {
        fwrite(data, sizeof(float), count, file);
        fclose(file);
    }
}

/* ------------------------------------------------------- GPU encoder ---- */

/* The qwenMetal P5 approach: every 3x3 convolution runs as nine
 * accumulate-mode TensorOps GEMMs over a reflect ring, channels-last
 * throughout. Stages mirror encode_core; the encoder runs once per
 * reference, so conv weights are uploaded and repacked per stage instead of
 * staying resident. Every stage submits before its scratch tensors are
 * freed, and a half-open command chain is drained on failure (the shared
 * GPU context must survive the fallback to the CPU path). */

static void gpu_drain(h3_gpu *gpu, int was_open) {
    if (!h3_gpu_submit(gpu)) {
        /* A failed submit leaves nothing pending to restore; the caller
         * falls back to the CPU path and frees the context. */
        return;
    }
    if (was_open) h3_gpu_begin(gpu);
}

static int gpu_conv3x3(h3_gpu *gpu, h3_gpu_tensor **output,
                       const h3_gpu_tensor *input, const float *weight9,
                       const float *bias, uint32_t in_channels,
                       uint32_t out_channels, uint32_t height, uint32_t width,
                       char *error, size_t error_size) {
    uint32_t co_pad = (out_channels + 63u) & ~63u;
    uint32_t wp = width + 2u;
    uint32_t m_span = (uint32_t)(((size_t)(height - 1) * wp + width + 63) /
                                 64 * 64);
    uint32_t pad_rows = m_span + 2u * wp + 2u;
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight9,
        (size_t)out_channels * in_channels * 9);
    h3_gpu_tensor *w9 = h3_gpu_tensor_new_f32(gpu,
        (size_t)9 * co_pad * in_channels);
    h3_gpu_tensor *b = bias ?
        h3_gpu_tensor_from_f32(gpu, bias, out_channels) : NULL;
    h3_gpu_tensor *arena = h3_gpu_tensor_new_f32(gpu,
        (size_t)pad_rows * in_channels);
    *output = h3_gpu_tensor_new_f32(gpu,
        (size_t)height * width * out_channels);
    if (!w || !w9 || (bias && !b) || !arena || !*output) {
        fail(error, error_size, "out of memory in the GPU VAE encoder");
        h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(w9);
        h3_gpu_tensor_free(b);
        h3_gpu_tensor_free(arena);
        h3_gpu_tensor_free(*output);
        *output = NULL;
        return 0;
    }
    int was_open = !h3_gpu_begin(gpu);
    int ok = h3img_gpu_vae_conv_repack_f32(gpu, w9, w, in_channels,
                                           out_channels, co_pad) &&
        h3img_gpu_vae_conv_mm_f32(gpu, *output, input, w9, b, arena, height,
                                  width, in_channels, out_channels);
    if (!ok) {
        fail(error, error_size, "GPU VAE convolution failed: %s",
             h3_gpu_error(gpu));
        gpu_drain(gpu, was_open);
    } else if (!h3_gpu_submit(gpu)) {
        fail(error, error_size, "GPU VAE convolution submit failed: %s",
             h3_gpu_error(gpu));
        ok = 0;
        if (was_open) h3_gpu_begin(gpu);
    } else if (was_open) {
        h3_gpu_begin(gpu);
    }
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(w9);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(arena);
    if (!ok) {
        h3_gpu_tensor_free(*output);
        *output = NULL;
    }
    return ok;
}

static int gpu_resnet_block(h3_weight_store *store, const char *name_prefix,
                            h3_gpu *gpu, h3_gpu_tensor *input,
                            h3_gpu_tensor **output, uint32_t in_channels,
                            uint32_t out_channels, uint32_t height,
                            uint32_t width, char *error, size_t error_size) {
    char name[160];
    snprintf(name, sizeof(name), "%s.norm1.weight", name_prefix);
    float *n1 = read_f32(store, name, in_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm1.bias", name_prefix);
    float *b1n = read_f32(store, name, in_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1.weight", name_prefix);
    float *w1 = read_f32(store, name,
                         (size_t)out_channels * in_channels * 27, error,
                         error_size);
    snprintf(name, sizeof(name), "%s.conv1.bias", name_prefix);
    float *b1 = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2.weight", name_prefix);
    float *n2 = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2.bias", name_prefix);
    float *b2n = read_f32(store, name, out_channels, error, error_size);
    snprintf(name, sizeof(name), "%s.conv2.weight", name_prefix);
    float *w2 = read_f32(store, name,
                         (size_t)out_channels * out_channels * 27, error,
                         error_size);
    snprintf(name, sizeof(name), "%s.conv2.bias", name_prefix);
    float *b2 = read_f32(store, name, out_channels, error, error_size);
    if (!n1 || !b1n || !w1 || !b1 || !n2 || !b2n || !w2 || !b2) {
        free(n1); free(b1n); free(w1); free(b1);
        free(n2); free(b2n); free(w2); free(b2);
        return 0;
    }
    float *w1t = tap_slice(w1, out_channels, in_channels, error, error_size);
    float *w2t = tap_slice(w2, out_channels, out_channels, error, error_size);
    if (!w1t || !w2t) {
        free(n1); free(b1n); free(w1); free(b1);
        free(n2); free(b2n); free(w2); free(b2);
        free(w1t); free(w2t);
        return 0;
    }
    size_t in_count = (size_t)in_channels * height * width;
    size_t out_count = (size_t)out_channels * height * width;
    h3_gpu_tensor *n1g = h3_gpu_tensor_from_f32(gpu, n1, in_channels);
    h3_gpu_tensor *b1ng = h3_gpu_tensor_from_f32(gpu, b1n, in_channels);
    h3_gpu_tensor *n2g = h3_gpu_tensor_from_f32(gpu, n2, out_channels);
    h3_gpu_tensor *b2ng = h3_gpu_tensor_from_f32(gpu, b2n, out_channels);
    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu, in_count);
    h3_gpu_tensor *normed2 = h3_gpu_tensor_new_f32(gpu, out_count);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(gpu, out_count);
    h3_gpu_tensor *shortcut = h3_gpu_tensor_new_f32(gpu, out_count);
    if (!n1g || !b1ng || !n2g || !b2ng || !normed || !normed2 || !hidden ||
        !shortcut) {
        fail(error, error_size, "out of memory in the GPU VAE encoder");
        free(n1); free(b1n); free(w1); free(b1);
        free(n2); free(b2n); free(w2); free(b2);
        free(w1t); free(w2t);
        h3_gpu_tensor_free(n1g); h3_gpu_tensor_free(b1ng);
        h3_gpu_tensor_free(n2g); h3_gpu_tensor_free(b2ng);
        h3_gpu_tensor_free(normed); h3_gpu_tensor_free(normed2);
        h3_gpu_tensor_free(hidden); h3_gpu_tensor_free(shortcut);
        return 0;
    }
    int ok = 1;
    int was_open = !h3_gpu_begin(gpu);
    ok = h3_gpu_vae_encoder_group_norm_silu_f32(gpu, normed, input, n1g,
                                                b1ng, 1, 1, height, width,
                                                in_channels, 32, 1e-6f) &&
        h3_gpu_submit(gpu);
    if (!ok) {
        fail(error, error_size, "GPU VAE resnet stage failed: %s",
             h3_gpu_error(gpu));
        gpu_drain(gpu, was_open);
    } else if (was_open) {
        h3_gpu_begin(gpu);
    }
    if (ok) ok = gpu_conv3x3(gpu, &hidden, normed, w1t, b1, in_channels,
                             out_channels, height, width, error, error_size);
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3_gpu_vae_encoder_group_norm_silu_f32(gpu, normed2, hidden, n2g,
                                                    b2ng, 1, 1, height, width,
                                                    out_channels, 32,
                                                    1e-6f) &&
            h3_gpu_submit(gpu);
        if (!ok) {
            fail(error, error_size, "GPU VAE resnet stage failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        } else if (was_open) {
            h3_gpu_begin(gpu);
        }
    }
    if (ok) ok = gpu_conv3x3(gpu, output, normed2, w2t, b2, out_channels,
                             out_channels, height, width, error, error_size);
    if (ok) {
        if (in_channels != out_channels) {
            snprintf(name, sizeof(name), "%s.conv_shortcut.weight",
                     name_prefix);
            float *ws = read_f32(store, name,
                                 (size_t)out_channels * in_channels, error,
                                 error_size);
            snprintf(name, sizeof(name), "%s.conv_shortcut.bias", name_prefix);
            float *bs = read_f32(store, name, out_channels, error,
                                 error_size);
            h3_gpu_tensor *wsg = ws ?
                h3_gpu_tensor_from_f32(gpu, ws,
                                       (size_t)out_channels * in_channels) :
                NULL;
            h3_gpu_tensor *bsg = bs ?
                h3_gpu_tensor_from_f32(gpu, bs, out_channels) : NULL;
            if (ws && bs && wsg && bsg) {
                was_open = !h3_gpu_begin(gpu);
                ok = h3_gpu_linear_f32(gpu, shortcut, input, wsg, bsg,
                                       (uint32_t)(height * width),
                                       in_channels, out_channels) &&
                    h3_gpu_submit(gpu);
                if (!ok) {
                    fail(error, error_size, "GPU VAE resnet stage failed: %s",
                         h3_gpu_error(gpu));
                    gpu_drain(gpu, was_open);
                } else if (was_open) {
                    h3_gpu_begin(gpu);
                }
            } else {
                fail(error, error_size, "cannot load the GPU VAE shortcut");
                ok = 0;
            }
            free(ws);
            free(bs);
            h3_gpu_tensor_free(wsg);
            h3_gpu_tensor_free(bsg);
        } else {
            /* out = 0 * input + 1 * input — the alias-safe elementwise copy. */
            was_open = !h3_gpu_begin(gpu);
            ok = h3_gpu_add_scaled_f32(gpu, shortcut, input, input, 0.0f,
                                       1.0f, (uint32_t)in_count) &&
                h3_gpu_submit(gpu);
            if (!ok) {
                fail(error, error_size, "GPU VAE resnet stage failed: %s",
                     h3_gpu_error(gpu));
                gpu_drain(gpu, was_open);
            } else if (was_open) {
                h3_gpu_begin(gpu);
            }
        }
    }
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3_gpu_add_scaled_f32(gpu, *output, *output, shortcut, 1.0f,
                                   1.0f, (uint32_t)out_count) &&
            h3_gpu_submit(gpu);
        if (!ok) {
            fail(error, error_size, "GPU VAE resnet stage failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        } else if (was_open) {
            h3_gpu_begin(gpu);
        }
    }
    free(n1); free(b1n); free(w1); free(b1);
    free(n2); free(b2n); free(w2); free(b2);
    free(w1t); free(w2t);
    h3_gpu_tensor_free(n1g);
    h3_gpu_tensor_free(b1ng);
    h3_gpu_tensor_free(n2g);
    h3_gpu_tensor_free(b2ng);
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(normed2);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(shortcut);
    return ok;
}

static int gpu_encode_core(h3_weight_store *store, const float *image,
                           int width, int height, float *mean, float *logvar,
                           char *error, size_t error_size) {
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, error_size);
    if (!gpu) return 0;
    int ok = 1;
    h3_gpu_tensor *current = NULL, *next = NULL;
    char name[160];
    int channels = 3, plane_w = width, plane_h = height;

    /* The reference image arrives planar; convert to channels-last on the
     * host (3 x H x W floats — negligible) and upload. */
    float *image_cl = malloc((size_t)3 * height * width * sizeof(float));
    if (!image_cl) {
        fail(error, error_size, "out of memory in the GPU VAE encoder");
        h3_gpu_free(gpu);
        return 0;
    }
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                image_cl[((size_t)y * width + x) * 3 + c] =
                    image[(size_t)c * height * width + y * width + x];
    current = h3_gpu_tensor_from_f32(gpu, image_cl,
                                     (size_t)3 * height * width);
    free(image_cl);
    if (!current) {
        fail(error, error_size, "cannot upload the VAE encoder input");
        h3_gpu_free(gpu);
        return 0;
    }

    /* conv_in (3 -> 128). */
    if (ok) {
        snprintf(name, sizeof(name), "encoder.conv_in.weight");
        float *w = read_f32(store, name,
                            (size_t)block_channels[0] * 3 * 27, error,
                            error_size);
        snprintf(name, sizeof(name), "encoder.conv_in.bias");
        float *b = read_f32(store, name, block_channels[0], error,
                            error_size);
        float *tap = w ? tap_slice(w, block_channels[0], 3, error,
                                   error_size) : NULL;
        /* A missing bias must fail the stage, not silently run bias-free
         * (downsample below would upload a NULL as a live bias buffer). */
        if (tap && b) {
            channels = block_channels[0];
            ok = gpu_conv3x3(gpu, &next, current, tap, b, 3, channels,
                             (uint32_t)height, (uint32_t)width, error,
                             error_size);
            h3_gpu_tensor_free(current);
            current = next;
            next = NULL;
        } else {
            ok = 0;
        }
        free(w);
        free(b);
        free(tap);
    }

    for (int level = 0; ok && level < 6; level++) {
        for (int layer = 0; ok && layer < 2; layer++) {
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.resnets.%d",
                     level, layer);
            int in_channels = channels;
            int out_channels = block_channels[level];
            ok = gpu_resnet_block(store, name, gpu, current, &next,
                                  (uint32_t)in_channels,
                                  (uint32_t)out_channels,
                                  (uint32_t)plane_h, (uint32_t)plane_w,
                                  error, error_size);
            if (ok) {
                h3_gpu_tensor_free(current);
                current = next;
                next = NULL;
                channels = out_channels;
            }
        }
        if (!ok || !block_downsample[level]) continue;
        /* Downsample: reflect ring on the bottom/right only, then a valid
         * stride-2 convolution. */
        uint32_t hp = (uint32_t)plane_h + 1, wp = (uint32_t)plane_w + 1;
        uint32_t out_h = (hp - 3) / 2 + 1, out_w = (wp - 3) / 2 + 1;
        snprintf(name, sizeof(name),
                 "encoder.down_blocks.%d.downsamplers.0.conv.weight", level);
        float *w = read_f32(store, name,
                            (size_t)channels * channels * 27, error,
                            error_size);
        snprintf(name, sizeof(name),
                 "encoder.down_blocks.%d.downsamplers.0.conv.bias", level);
        float *b = read_f32(store, name, channels, error, error_size);
        float *tap = w ? tap_slice(w, channels, channels, error,
                                   error_size) : NULL;
        h3_gpu_tensor *ring = NULL, *wg = NULL, *bg = NULL;
        if (tap && b) {
            ring = h3_gpu_tensor_new_f32(gpu,
                (size_t)hp * wp * channels);
            wg = h3_gpu_tensor_from_f32(gpu, tap,
                (size_t)channels * channels * 9);
            bg = h3_gpu_tensor_from_f32(gpu, b, (size_t)channels);
            next = h3_gpu_tensor_new_f32(gpu,
                (size_t)out_h * out_w * channels);
            if (ring && wg && bg && next) {
                int was_open = !h3_gpu_begin(gpu);
                ok = h3_gpu_vae_encoder_pad_f32(gpu, ring, current, 1, 1,
                                                (uint32_t)plane_h,
                                                (uint32_t)plane_w, channels,
                                                0, 0, 1, 0, 1) &&
                    h3img_gpu_vae_downsample_f32(gpu, next, ring, wg, bg,
                                                 out_h, out_w, hp, wp,
                                                 (uint32_t)channels,
                                                 (uint32_t)channels) &&
                    h3_gpu_submit(gpu);
                if (!ok) {
                    fail(error, error_size,
                         "GPU VAE downsample failed: %s", h3_gpu_error(gpu));
                    gpu_drain(gpu, was_open);
                } else if (was_open) {
                    h3_gpu_begin(gpu);
                }
            } else {
                fail(error, error_size,
                     "out of memory in the GPU VAE encoder");
                ok = 0;
            }
        } else {
            ok = 0;
        }
        h3_gpu_tensor_free(ring);
        h3_gpu_tensor_free(wg);
        h3_gpu_tensor_free(bg);
        free(w);
        free(b);
        free(tap);
        if (ok) {
            h3_gpu_tensor_free(current);
            current = next;
            next = NULL;
            plane_h = (int)out_h;
            plane_w = (int)out_w;
        } else {
            /* next was allocated before the failure that aborted the
             * dispatch chain; without this it leaks. */
            h3_gpu_tensor_free(next);
            next = NULL;
        }
    }

    /* norm_out + conv_out (48) + quant_conv, then split mean | logvar. */
    if (ok) {
        snprintf(name, sizeof(name), "encoder.norm_out.weight");
        float *gnw = read_f32(store, name, channels, error, error_size);
        snprintf(name, sizeof(name), "encoder.norm_out.bias");
        float *gnb = read_f32(store, name, channels, error, error_size);
        h3_gpu_tensor *gnwg = NULL, *gnbg = NULL, *normed = NULL;
        if (gnw && gnb) {
            gnwg = h3_gpu_tensor_from_f32(gpu, gnw, (size_t)channels);
            gnbg = h3_gpu_tensor_from_f32(gpu, gnb, (size_t)channels);
            normed = h3_gpu_tensor_new_f32(gpu,
                (size_t)channels * plane_h * plane_w);
        }
        if (gnwg && gnbg && normed) {
            int was_open = !h3_gpu_begin(gpu);
            ok = h3_gpu_vae_encoder_group_norm_silu_f32(gpu, normed, current,
                                                        gnwg, gnbg, 1, 1,
                                                        (uint32_t)plane_h,
                                                        (uint32_t)plane_w,
                                                        (uint32_t)channels,
                                                        32, 1e-6f) &&
                h3_gpu_submit(gpu);
            if (!ok) {
                fail(error, error_size, "GPU VAE norm_out failed: %s",
                     h3_gpu_error(gpu));
                gpu_drain(gpu, was_open);
            } else if (was_open) {
                h3_gpu_begin(gpu);
            }
            if (ok) {
                snprintf(name, sizeof(name), "encoder.conv_out.weight");
                float *w = read_f32(store, name,
                                    (size_t)QUANT_CHANNELS * channels * 27,
                                    error, error_size);
                snprintf(name, sizeof(name), "encoder.conv_out.bias");
                float *b = read_f32(store, name, QUANT_CHANNELS, error,
                                    error_size);
                float *tap = w ? tap_slice(w, QUANT_CHANNELS, channels,
                                           error, error_size) : NULL;
                if (tap) {
                    ok = gpu_conv3x3(gpu, &next, normed, tap, b,
                                     (uint32_t)channels, QUANT_CHANNELS,
                                     (uint32_t)plane_h, (uint32_t)plane_w,
                                     error, error_size);
                    h3_gpu_tensor_free(current);
                    current = next;
                    next = NULL;
                    channels = QUANT_CHANNELS;
                } else {
                    ok = 0;
                }
                free(w);
                free(b);
                free(tap);
            }
        } else if (ok) {
            fail(error, error_size, "cannot load the GPU VAE norm_out");
            ok = 0;
        }
        h3_gpu_tensor_free(gnwg);
        h3_gpu_tensor_free(gnbg);
        h3_gpu_tensor_free(normed);
        /* h3_gpu_tensor_from_f32 copies into device storage; the host weight
         * copies are no longer needed. Release them on every exit of this
         * block (success or the load-failure else branch). */
        free(gnw);
        free(gnb);
    }

    if (ok) {
        snprintf(name, sizeof(name), "quant_conv.weight");
        float *w = read_f32(store, name,
                            (size_t)QUANT_CHANNELS * QUANT_CHANNELS, error,
                            error_size);
        snprintf(name, sizeof(name), "quant_conv.bias");
        float *b = read_f32(store, name, QUANT_CHANNELS, error, error_size);
        h3_gpu_tensor *wg = NULL, *bg = NULL, *quant = NULL;
        int count = plane_h * plane_w;
        if (w && b) {
            wg = h3_gpu_tensor_from_f32(gpu, w,
                                        (size_t)QUANT_CHANNELS *
                                        QUANT_CHANNELS);
            bg = h3_gpu_tensor_from_f32(gpu, b, QUANT_CHANNELS);
            quant = h3_gpu_tensor_new_f32(gpu,
                (size_t)QUANT_CHANNELS * count);
        }
        if (wg && bg && quant) {
            int was_open = !h3_gpu_begin(gpu);
            ok = h3_gpu_linear_f32(gpu, quant, current, wg, bg,
                                   (uint32_t)count, QUANT_CHANNELS,
                                   QUANT_CHANNELS) &&
                h3_gpu_submit(gpu);
            if (!ok) {
                fail(error, error_size, "GPU VAE quant failed: %s",
                     h3_gpu_error(gpu));
                gpu_drain(gpu, was_open);
            } else if (was_open) {
                h3_gpu_begin(gpu);
            }
            if (ok) {
                float *host = malloc((size_t)QUANT_CHANNELS * count *
                                     sizeof(float));
                if (host &&
                    h3_gpu_tensor_read_f32(quant, host,
                                           (size_t)QUANT_CHANNELS * count)) {
                    /* [count, 48] channels-last -> planar mean | logvar. */
                    for (int c = 0; c < LATENT_CHANNELS; c++)
                        for (int index = 0; index < count; index++) {
                            mean[(size_t)c * count + index] =
                                host[(size_t)index * QUANT_CHANNELS + c];
                            if (logvar)
                                logvar[(size_t)c * count + index] =
                                    host[(size_t)index * QUANT_CHANNELS +
                                         LATENT_CHANNELS + c];
                        }
                } else {
                    fail(error, error_size,
                         "cannot read back the GPU VAE latent");
                    ok = 0;
                }
                free(host);
            }
        } else if (ok) {
            fail(error, error_size, "cannot load the GPU VAE quant_conv");
            ok = 0;
        }
        h3_gpu_tensor_free(wg);
        h3_gpu_tensor_free(bg);
        h3_gpu_tensor_free(quant);
        free(w);
        free(b);
    }

    h3_gpu_tensor_free(current);
    h3_gpu_free(gpu);
    return ok;
}

/* Shared encoder core: fills mean with the posterior's first 24 quant
 * channels and, when logvar is non-NULL, the remaining 24. The GPU path
 * (TensorOps GEMM convolutions) is the default; H3IMG_VAE_CPU=1 skips it and
 * H3IMG_VAE_SCALAR=1 additionally restores the scalar convolution kernels.
 * A GPU failure falls back to the CPU path transparently. */
static int encode_core(h3_weight_store *store, const float *image, int width,
                       int height, float *mean, float *logvar,
                       char *error, size_t error_size) {
    if (!getenv("H3IMG_VAE_SCALAR") && !getenv("H3IMG_VAE_CPU")) {
        /* Zeroed: a GPU failure that never wrote an error (e.g. a dispatch
         * failure before fail()) must not print uninitialized stack bytes. */
        char gpu_error[512] = {0};
        if (gpu_encode_core(store, image, width, height, mean, logvar,
                            gpu_error, sizeof(gpu_error)))
            return 1;
        fprintf(stderr, "h3img: GPU VAE encoder failed (%s); "
                "falling back to the CPU path\n",
                gpu_error[0] ? gpu_error : "unknown error");
        if (error && error_size) error[0] = '\0';
    }
    char detail[256];
    int ok = 1;
    float *current = NULL, *next = NULL;
    char name[160];
    int level = 0;
    int channels = 0, plane_w = width, plane_h = height;

    /* conv_in: k3, reflect pad 1, temporal tap 2 (the only live tap for an
     * image). */
    size_t conv_in_elements = (size_t)block_channels[0] * 3 * 27;
    snprintf(name, sizeof(name), "encoder.conv_in.weight");
    float *w = read_f32(store, name, conv_in_elements, error, error_size);
    snprintf(name, sizeof(name), "encoder.conv_in.bias");
    float *b = read_f32(store, name, block_channels[0], error, error_size);
    if (!w || !b) {
        ok = 0;
        goto cleanup;
    }
    /* Slice the last temporal tap: [out][in][3][3][3] -> [out][in][3][3]. */
    float *tap = tap_slice(w, block_channels[0], 3, error, error_size);
    if (!tap) {
        ok = 0;
        goto cleanup;
    }
    current = malloc((size_t)block_channels[0] * height * width * sizeof(float));
    if (!current) {
        fail(error, error_size, "out of memory in VAE encoder");
        ok = 0;
        goto cleanup;
    }
    conv2d_k3(image, 3, height, width, tap, b, block_channels[0], 1,
              current, height, width, 1);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_enc_conv_in.f32", current,
                   (size_t)block_channels[0] * height * width);
    free(w);
    free(b);
    free(tap);
    channels = block_channels[0];

    for (level = 0; ok && level < 6; level++) {
        char prefix[64];
        for (int layer = 0; ok && layer < 2; layer++) {
            snprintf(prefix, sizeof(prefix), "encoder.down_blocks.%d.resnets.%d",
                     level, layer);
            int in_channels = channels;
            int out_channels = block_channels[level];
            next = malloc((size_t)out_channels * plane_h * plane_w *
                          sizeof(float));
            if (!next) {
                fail(error, error_size, "out of memory in VAE encoder");
                ok = 0;
                break;
            }
            ok = resnet_block(store, prefix, current, in_channels,
                              out_channels, plane_h, plane_w, next, error,
                              error_size);
            if (ok && getenv("H3IMG_VAE_DUMP")) {
                char dump_path[96];
                snprintf(dump_path, sizeof(dump_path),
                         "/tmp/h3img_vae_b%d_resnet%d.f32", level, layer);
                dump_stage(dump_path, next,
                           (size_t)out_channels * plane_h * plane_w);
            }

            free(current);
            current = next;
            next = NULL;
            channels = out_channels;
        }
        if (!ok || !block_downsample[level]) continue;
        /* Downsample: reflect pad bottom/right by 1, then a valid k3 conv
         * with stride 2 (temporal tap 2). */
        int padded_h = plane_h + 1, padded_w = plane_w + 1;
        float *padded = malloc((size_t)channels * padded_h * padded_w *
                               sizeof(float));
        next = malloc((size_t)channels * ((plane_h + 1) / 2) *
                      ((plane_w + 1) / 2) * sizeof(float));
        if (!padded || !next) {
            fail(error, error_size, "out of memory in VAE encoder");
            ok = 0;
            free(padded);
            break;
        }
        for (int c = 0; c < channels; c++) {
            const float *plane = current + (size_t)c * plane_h * plane_w;
            float *target = padded + (size_t)c * padded_h * padded_w;
            for (int y = 0; y < padded_h; y++)
                for (int x = 0; x < padded_w; x++) {
                    /* Torch reflect excludes the edge: a 1-px pad repeats
                     * row/col H-2, not the edge row itself. */
                    int sy = y < plane_h ? y : 2 * (plane_h - 1) - y;
                    int sx = x < plane_w ? x : 2 * (plane_w - 1) - x;
                    if (sy < 0) sy = -sy;
                    if (sx < 0) sx = -sx;
                    target[y * padded_w + x] = plane[sy * plane_w + sx];
                }
        }
        snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsamplers.0.conv.weight", level);
        w = read_f32(store, name, (size_t)channels * channels * 27, error, error_size);
        snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsamplers.0.conv.bias", level);
        b = read_f32(store, name, channels, error, error_size);
        if (!w || !b) {
            ok = 0;
            free(padded);
            free(next);
            break;
        }
        float *tap2 = tap_slice(w, channels, channels, error, error_size);
        if (!tap2) {
            ok = 0;
            free(padded);
            free(next);
            break;
        }
        int out_h = (padded_h - 3) / 2 + 1;
        int out_w = (padded_w - 3) / 2 + 1;
        conv2d_k3(padded, channels, padded_h, padded_w, tap2, b, channels, 2,
                  next, out_h, out_w, 0);
        free(padded);
        free(current);
        free(w);
        free(b);
        free(tap2);
        current = next;
        next = NULL;
        plane_h = out_h;
        plane_w = out_w;

    }
    if (!ok) goto cleanup;

    /* norm_out + silu, conv_out (48 channels), quant_conv, mode = first 24. */
    snprintf(name, sizeof(name), "encoder.norm_out.weight");
    float *gnw = read_f32(store, name, channels, error, error_size);
    snprintf(name, sizeof(name), "encoder.norm_out.bias");
    float *gnb = read_f32(store, name, channels, error, error_size);
    if (!gnw || !gnb) {
        ok = 0;
        goto cleanup;
    }
    group_norm_silu(current, channels, plane_h * plane_w, gnw, gnb);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_enc_normout.f32", current,
                   (size_t)channels * plane_h * plane_w);
    snprintf(name, sizeof(name), "encoder.conv_out.weight");
    w = read_f32(store, name, (size_t)QUANT_CHANNELS * channels * 27, error, error_size);
    snprintf(name, sizeof(name), "encoder.conv_out.bias");
    b = read_f32(store, name, QUANT_CHANNELS, error, error_size);
    if (!w || !b) {
        ok = 0;
        goto cleanup;
    }
    float *tap3 = tap_slice(w, QUANT_CHANNELS, channels, error, error_size);
    next = malloc((size_t)QUANT_CHANNELS * plane_h * plane_w * sizeof(float));
    if (!tap3 || !next) {
        fail(error, error_size, "out of memory in VAE encoder");
        ok = 0;
        free(tap3);
        free(next);
        goto cleanup;
    }
    conv2d_k3(current, channels, plane_h, plane_w, tap3, b, QUANT_CHANNELS, 1,
              next, plane_h, plane_w, 1);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_enc_convout.f32", next,
                   (size_t)QUANT_CHANNELS * plane_h * plane_w);
    free(tap3);
    free(current);
    free(w);
    free(b);
    current = next;
    next = NULL;
    snprintf(name, sizeof(name), "quant_conv.weight");
    w = read_f32(store, name, (size_t)QUANT_CHANNELS * QUANT_CHANNELS, error, error_size);
    snprintf(name, sizeof(name), "quant_conv.bias");
    b = read_f32(store, name, QUANT_CHANNELS, error, error_size);
    if (!w || !b) {
        ok = 0;
        goto cleanup;
    }
    int count = plane_h * plane_w;
    float *quant = malloc((size_t)QUANT_CHANNELS * count * sizeof(float));
    if (!quant) {
        fail(error, error_size, "out of memory in VAE encoder");
        ok = 0;
        goto cleanup;
    }
    conv1x1(current, QUANT_CHANNELS, count, w, b, QUANT_CHANNELS, quant);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_enc_quant.f32", quant,
                   (size_t)QUANT_CHANNELS * count);
    /* The posterior is [mean(24) | logvar(24)]. */
    for (int c = 0; c < LATENT_CHANNELS; c++) {
        memcpy(mean + (size_t)c * count, quant + (size_t)c * count,
               count * sizeof(float));
        if (logvar)
            memcpy(logvar + (size_t)c * count,
                   quant + (size_t)(LATENT_CHANNELS + c) * count,
                   count * sizeof(float));
    }
    free(quant);

cleanup:
    free(current);
    free(next);
    (void)detail;
    (void)level;
    return ok;
}

int h3img_vae_encode(const char *weight_directory, const float *image,
                     int width, int height, float *latent,
                     char *error, size_t error_size) {
    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                  error_size);
    if (!store) return 0;
    int ok = encode_core(store, image, width, height, latent, NULL, error,
                         error_size);
    h3_weight_store_free(store);
    return ok;
}

int h3img_vae_encode_sampled(const char *weight_directory, const float *image,
                             int width, int height, h3_rng *rng, float *latent,
                             char *error, size_t error_size) {
    /* The encoder's stride-2 plane arithmetic assumes 16-aligned inputs;
     * anything else silently desynchronizes the latent element count. */
    if (width % 16 || height % 16) {
        fail(error, error_size,
             "VAE encoder needs multiples of 16 (got %dx%d)", width, height);
        return 0;
    }
    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                  error_size);
    if (!store) return 0;
    int count = (width / 16) * (height / 16);
    float *mean = malloc((size_t)LATENT_CHANNELS * count * sizeof(float));
    float *logvar = malloc((size_t)LATENT_CHANNELS * count * sizeof(float));
    int ok = mean && logvar;
    if (!ok) fail(error, error_size, "out of memory in VAE encoder");
    if (ok) ok = encode_core(store, image, width, height, mean, logvar,
                             error, error_size);
    if (ok) {
        /* The reference recipe: posterior sample rounded to float16. */
        for (int c = 0; c < LATENT_CHANNELS; c++)
            for (int index = 0; index < count; index++) {
                double mu = mean[(size_t)c * count + index];
                double variance = exp(logvar[(size_t)c * count + index] * 0.5);
                double sample = mu + variance * (double)h3_rng_normal(rng);
                union { float f; uint32_t u; } round;
                round.f = (float)sample;
                round.u = (round.u + 0x7FFFu + ((round.u >> 16) & 1u)) &
                          0xFFFF0000u;
                latent[(size_t)c * count + index] = round.f;
            }
    }
    free(mean);
    free(logvar);
    h3_weight_store_free(store);
    return ok;
}

/* ------------------------------------------------------------ decoder ---- */

static void rms_norm(float *data, int count, int dim, const float *weight,
                     float eps) {
    for (int index = 0; index < count; index++) {
        float *row = data + (size_t)index * dim;
        double sum = 0.0;
        for (int d = 0; d < dim; d++) {
            double value = row[d];
            sum += value * value;
        }
        double inverse = 1.0 / sqrt(sum / dim + eps);
        for (int d = 0; d < dim; d++)
            row[d] = (float)((double)row[d] * inverse * weight[d]);
    }
}

static void layer_norm(float *data, int count, int dim, const float *weight,
                       const float *bias, float eps) {
    for (int index = 0; index < count; index++) {
        float *row = data + (size_t)index * dim;
        double mean = 0.0;
        for (int d = 0; d < dim; d++) mean += row[d];
        mean /= dim;
        double squares = 0.0;
        for (int d = 0; d < dim; d++) {
            double centered = row[d] - mean;
            squares += centered * centered;
        }
        double inverse = 1.0 / sqrt(squares / dim + eps);
        for (int d = 0; d < dim; d++)
            row[d] = (float)(((double)row[d] - mean) * inverse * weight[d] +
                             bias[d]);
    }
}

static void linear(const float *input, int count, int in_dim,
                   const float *weight /* [out][in] */, const float *bias,
                   int out_dim, float *output) {
    if (getenv("H3IMG_VAE_SCALAR")) {
        for (int index = 0; index < count; index++) {
            const float *row = input + (size_t)index * in_dim;
            float *target = output + (size_t)index * out_dim;
            for (int oc = 0; oc < out_dim; oc++) {
                const float *weights = weight + (size_t)oc * in_dim;
                double sum = bias ? bias[oc] : 0.0;
                for (int ic = 0; ic < in_dim; ic++)
                    sum += (double)row[ic] * weights[ic];
                target[oc] = (float)sum;
            }
        }
        return;
    }
    /* Row-major (count x in_dim) input, [out][in] weight (an (N x K) matrix,
     * hence the transpose flag), and (count x out_dim) output. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, out_dim,
                in_dim, 1.0f, input, in_dim, weight, in_dim, 0.0f, output,
                out_dim);
    if (bias)
        for (int index = 0; index < count; index++) {
            float *row = output + (size_t)index * out_dim;
            for (int oc = 0; oc < out_dim; oc++)
                row[oc] += bias[oc];
        }
}

/* Returns 1 on success, 0 on failure. The caller must check the return value:
 * error is optional, so a void-returning block that only reports via error[]
 * silently succeeds (with a half-updated stream) when error == NULL. */
static int vit_block(h3_weight_store *store, int index, float *stream,
                     int tokens, const float *cos, const float *sin,
                     char *error, size_t error_size) {
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
    char name[160];
    int ok = 1;
    /* Load everything the block needs. */
    float *norm1 = NULL, *q_w = NULL, *q_b = NULL, *k_w = NULL, *k_b = NULL,
          *v_w = NULL, *v_b = NULL, *o_w = NULL, *o_b = NULL, *scale1 = NULL,
          *norm2 = NULL, *f1_w = NULL, *f1_b = NULL, *f2_w = NULL, *f2_b = NULL,
          *scale2 = NULL;
    snprintf(name, sizeof(name), "%snorm1.weight", prefix);
    norm1 = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_q.weight", prefix);
    q_w = read_f32(store, name, (size_t)VIT_DIM * VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_q.bias", prefix);
    q_b = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_k.weight", prefix);
    k_w = read_f32(store, name, (size_t)VIT_DIM * VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_k.bias", prefix);
    k_b = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_v.weight", prefix);
    v_w = read_f32(store, name, (size_t)VIT_DIM * VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_v.bias", prefix);
    v_b = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_out.0.weight", prefix);
    o_w = read_f32(store, name, (size_t)VIT_DIM * VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sattn.to_out.0.bias", prefix);
    o_b = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sscale1", prefix);
    scale1 = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%snorm2.weight", prefix);
    norm2 = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sff.net.0.proj.weight", prefix);
    f1_w = read_f32(store, name, (size_t)VIT_DIM * 2 * VIT_FFN, error, error_size);
    snprintf(name, sizeof(name), "%sff.net.0.proj.bias", prefix);
    f1_b = read_f32(store, name, 2 * VIT_FFN, error, error_size);
    snprintf(name, sizeof(name), "%sff.net.2.weight", prefix);
    f2_w = read_f32(store, name, (size_t)VIT_DIM * VIT_FFN, error, error_size);
    snprintf(name, sizeof(name), "%sff.net.2.bias", prefix);
    f2_b = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "%sscale2", prefix);
    scale2 = read_f32(store, name, VIT_DIM, error, error_size);
    if (!norm1 || !q_w || !q_b || !k_w || !k_b || !v_w || !v_b || !o_w ||
        !o_b || !scale1 || !norm2 || !f1_w || !f1_b || !f2_w || !f2_b ||
        !scale2) {
        ok = 0;
        goto cleanup;
    }
    float *normed = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    float *qkv = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    float *attention = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    float *attnout = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    float *ff = malloc((size_t)tokens * 2 * VIT_FFN * sizeof(float));
    float *act = malloc((size_t)tokens * VIT_FFN * sizeof(float));
    float *mlp = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    if (!normed || !qkv || !attention || !attnout || !ff || !act || !mlp) {
        fail(error, error_size, "out of memory in VAE decoder block");
        ok = 0;
        goto cleanup;
    }
    /* Attention half. */
    memcpy(normed, stream, (size_t)tokens * VIT_DIM * sizeof(float));
    rms_norm(normed, tokens, VIT_DIM, norm1, 1e-5f);
    linear(normed, tokens, VIT_DIM, q_w, q_b, VIT_DIM, qkv);
    if (getenv("H3IMG_VAE_DUMP") && index == 0) {
        dump_stage("/tmp/h3img_vae_d0_normed.f32", normed,
                   (size_t)tokens * VIT_DIM);
        dump_stage("/tmp/h3img_vae_d0_qraw.f32", qkv,
                   (size_t)tokens * VIT_DIM);
        dump_stage("/tmp/h3img_vae_d0_hidden.f32", stream,
                   (size_t)tokens * VIT_DIM);
    }
    float *key = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    float *value = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    if (!key || !value) {
        fail(error, error_size, "out of memory in VAE decoder block");
        ok = 0;
        free(key);
        free(value);
        goto cleanup;
    }
    linear(normed, tokens, VIT_DIM, k_w, k_b, VIT_DIM, key);
    linear(normed, tokens, VIT_DIM, v_w, v_b, VIT_DIM, value);
    /* Per-head QK RMS norm (no affine), rope, attention. */
    for (int token = 0; token < tokens && ok; token++) {
        for (int head = 0; head < VIT_HEADS && ok; head++) {
            float *q = qkv + (size_t)token * VIT_DIM + head * VIT_HEAD_DIM;
            float *kk = key + (size_t)token * VIT_DIM + head * VIT_HEAD_DIM;
            double qsum = 0.0, ksum = 0.0;
            for (int d = 0; d < VIT_HEAD_DIM; d++) {
                qsum += (double)q[d] * q[d];
                ksum += (double)kk[d] * kk[d];
            }
            double qinv = 1.0 / sqrt(qsum / (double)VIT_HEAD_DIM + 1e-5);
            double kinv = 1.0 / sqrt(ksum / (double)VIT_HEAD_DIM + 1e-5);
            for (int d = 0; d < VIT_HEAD_DIM; d++) {
                q[d] = (float)(q[d] * qinv);
                kk[d] = (float)(kk[d] * kinv);
            }
            /* Rotate-half on the first VIT_ROTARY dims: pairs (i, i+16). */
            const float *cosrow = cos + (size_t)token * VIT_ROTARY;
            const float *sinrow = sin + (size_t)token * VIT_ROTARY;
            float rotated[2][VIT_ROTARY];
            for (int d = 0; d < VIT_ROTARY / 2; d++) {
                rotated[0][d] = -q[VIT_ROTARY / 2 + d];
                rotated[0][VIT_ROTARY / 2 + d] = q[d];
                rotated[1][d] = -kk[VIT_ROTARY / 2 + d];
                rotated[1][VIT_ROTARY / 2 + d] = kk[d];
            }
            for (int d = 0; d < VIT_ROTARY; d++) {
                q[d] = q[d] * cosrow[d] + rotated[0][d] * sinrow[d];
                kk[d] = kk[d] * cosrow[d] + rotated[1][d] * sinrow[d];
            }
        }
    }
    if (ok) {
        /* Attention per head over the 21 tokens. */
        double scale = 1.0 / sqrt((double)VIT_HEAD_DIM);
        double *weights = malloc((size_t)tokens * sizeof(*weights));
        if (!weights) {
            fail(error, error_size, "out of memory for attention scores");
            ok = 0;
        }
        for (int head = 0; ok && head < VIT_HEADS; head++) {
            for (int row = 0; row < tokens; row++) {
                const float *qrow =
                    qkv + (size_t)row * VIT_DIM + head * VIT_HEAD_DIM;
                double maximum = -1e300;
                for (int col = 0; col < tokens; col++) {
                    const float *kcol =
                        key + (size_t)col * VIT_DIM + head * VIT_HEAD_DIM;
                    double score = 0.0;
                    for (int d = 0; d < VIT_HEAD_DIM; d++)
                        score += (double)qrow[d] * kcol[d];
                    score *= scale;
                    weights[col] = score;
                    if (score > maximum) maximum = score;
                }
                double denominator = 0.0;
                for (int col = 0; col < tokens; col++) {
                    weights[col] = exp(weights[col] - maximum);
                    denominator += weights[col];
                }
                float *outrow =
                    attention + (size_t)row * VIT_DIM + head * VIT_HEAD_DIM;
                for (int d = 0; d < VIT_HEAD_DIM; d++) outrow[d] = 0.0f;
                for (int col = 0; col < tokens; col++) {
                    const float *vcol =
                        value + (size_t)col * VIT_DIM + head * VIT_HEAD_DIM;
                    double weight = weights[col] / denominator;
                    for (int d = 0; d < VIT_HEAD_DIM; d++)
                        outrow[d] += (float)(weight * (double)vcol[d]);
                }
            }
        }
        free(weights);
        weights = NULL;
        linear(attention, tokens, VIT_DIM, o_w, o_b, VIT_DIM, attnout);
        if (getenv("H3IMG_VAE_DUMP") && index == 0) {
            dump_stage("/tmp/h3img_vae_d0_q.f32", qkv,
                       (size_t)tokens * VIT_DIM);
            dump_stage("/tmp/h3img_vae_d0_k.f32", key,
                       (size_t)tokens * VIT_DIM);
            dump_stage("/tmp/h3img_vae_d0_v.f32", value,
                       (size_t)tokens * VIT_DIM);
            dump_stage("/tmp/h3img_vae_d0_attn.f32", attention,
                       (size_t)tokens * VIT_DIM);
            dump_stage("/tmp/h3img_vae_d0_attnout.f32", attnout,
                       (size_t)tokens * VIT_DIM);
        }
        for (int token = 0; token < tokens; token++)
            for (int d = 0; d < VIT_DIM; d++)
                stream[(size_t)token * VIT_DIM + d] +=
                    attnout[(size_t)token * VIT_DIM + d] * scale1[d];
        if (getenv("H3IMG_VAE_DUMP") && index == 0)
            dump_stage("/tmp/h3img_vae_d0_postattn.f32", stream,
                       (size_t)tokens * VIT_DIM);
        /* MLP half. */
        memcpy(normed, stream, (size_t)tokens * VIT_DIM * sizeof(float));
        rms_norm(normed, tokens, VIT_DIM, norm2, 1e-5f);
        linear(normed, tokens, VIT_DIM, f1_w, f1_b, 2 * VIT_FFN, ff);
        for (int token = 0; token < tokens; token++)
            for (int d = 0; d < VIT_FFN; d++) {
                /* diffusers swiglu: silu(second half) * first half. */
                double first = ff[(size_t)token * 2 * VIT_FFN + d];
                double second = ff[(size_t)token * 2 * VIT_FFN + VIT_FFN + d];
                act[(size_t)token * VIT_FFN + d] =
                    (float)(second / (1.0 + exp(-second)) * first);
            }
        linear(act, tokens, VIT_FFN, f2_w, f2_b, VIT_DIM, mlp);
        if (getenv("H3IMG_VAE_DUMP") && index == 0)
            dump_stage("/tmp/h3img_vae_d0_mlp.f32", mlp,
                       (size_t)tokens * VIT_DIM);
        for (int token = 0; token < tokens; token++)
            for (int d = 0; d < VIT_DIM; d++)
                stream[(size_t)token * VIT_DIM + d] +=
                    mlp[(size_t)token * VIT_DIM + d] * scale2[d];
    }
    free(normed);
    free(qkv);
    free(attention);
    free(attnout);
    free(ff);
    free(act);
    free(mlp);
    free(key);
    free(value);
cleanup:
    free(norm1); free(q_w); free(q_b); free(k_w); free(k_b);
    free(v_w); free(v_b); free(o_w); free(o_b); free(scale1);
    free(norm2); free(f1_w); free(f1_b); free(f2_w); free(f2_b);
    free(scale2);
    return ok;
}

/* ------------------------------------------------------- GPU decoder ---- */

/* Per-block weight names (identical layout for every VIT_LAYERS block). */
static const struct { const char *suffix; size_t count; }
GPU_DECODER_LOADS[] = {
    {"norm1.weight", VIT_DIM}, {"attn.to_q.weight", (size_t)VIT_DIM * VIT_DIM},
    {"attn.to_q.bias", VIT_DIM}, {"attn.to_k.weight", (size_t)VIT_DIM * VIT_DIM},
    {"attn.to_k.bias", VIT_DIM}, {"attn.to_v.weight", (size_t)VIT_DIM * VIT_DIM},
    {"attn.to_v.bias", VIT_DIM}, {"attn.to_out.0.weight", (size_t)VIT_DIM * VIT_DIM},
    {"attn.to_out.0.bias", VIT_DIM}, {"scale1", VIT_DIM},
    {"norm2.weight", VIT_DIM}, {"ff.net.0.proj.weight", (size_t)VIT_DIM * 2 * VIT_FFN},
    {"ff.net.0.proj.bias", 2 * VIT_FFN}, {"ff.net.2.weight", (size_t)VIT_DIM * VIT_FFN},
    {"ff.net.2.bias", VIT_DIM}, {"scale2", VIT_DIM},
};
enum {
    W_N1, W_Q, W_QB, W_K, W_KB, W_V, W_VB, W_O, W_OB, W_S1,
    W_N2, W_F1, W_F1B, W_F2, W_F2B, W_S2, W_COUNT
};

/* All decoder weights uploaded once per decode and reused across tiles —
 * re-reading and re-uploading ~80 MB of block weights per tile (24 layers
 * x tiles) dominated the tiled decode time against the reference stack,
 * which keeps the whole ~2.2 GB VAE resident. */
typedef struct {
    h3_gpu_tensor *block[VIT_LAYERS][W_COUNT];
    /* Head weights: post_quant/proj_in/register tokens are consumed on the
     * host per tile; norm_out/proj_out run on the GPU. */
    float *pq_w, *pq_b, *pi_w, *pi_b, *registers;
    h3_gpu_tensor *lnw, *lnb, *proj_w, *proj_b;
} gpu_decoder_weights;

static int gpu_decoder_weights_build(h3_weight_store *store, h3_gpu *gpu,
                                     gpu_decoder_weights *weights,
                                     char *error, size_t error_size) {
    memset(weights, 0, sizeof(*weights));
    char name[160];
    for (int index = 0; index < VIT_LAYERS; index++) {
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.", index);
        size_t prefix_len = strlen(name);
        for (int w = 0; w < W_COUNT; w++) {
            snprintf(name + prefix_len, sizeof(name) - prefix_len, "%s",
                     GPU_DECODER_LOADS[w].suffix);
            float *host = read_f32(store, name, GPU_DECODER_LOADS[w].count,
                                   error, error_size);
            if (host)
                weights->block[index][w] = h3_gpu_tensor_from_f32(
                    gpu, host, GPU_DECODER_LOADS[w].count);
            free(host);
            if (!weights->block[index][w]) {
                fail(error, error_size, "cannot load %s", name);
                return 0;
            }
        }
    }
    const struct { const char *name; size_t count; float **slot; }
    host_loads[] = {
        {"post_quant_conv.weight", (size_t)LATENT_CHANNELS * LATENT_CHANNELS,
         &weights->pq_w},
        {"post_quant_conv.bias", LATENT_CHANNELS, &weights->pq_b},
        {"decoder.proj_in.weight", (size_t)VIT_DIM * LATENT_CHANNELS,
         &weights->pi_w},
        {"decoder.proj_in.bias", VIT_DIM, &weights->pi_b},
        {"decoder.register_tokens", (size_t)VIT_REGISTERS * VIT_DIM,
         &weights->registers},
    };
    for (size_t i = 0; i < sizeof(host_loads) / sizeof(host_loads[0]); i++) {
        *host_loads[i].slot = read_f32(store, host_loads[i].name,
                                       host_loads[i].count, error,
                                       error_size);
        if (!*host_loads[i].slot) return 0;
    }
    const struct { const char *name; size_t count;
                   h3_gpu_tensor **slot; } device_loads[] = {
        {"decoder.norm_out.weight", VIT_DIM, &weights->lnw},
        {"decoder.norm_out.bias", VIT_DIM, &weights->lnb},
        {"decoder.proj_out.weight", (size_t)VIT_DIM * 768, &weights->proj_w},
        {"decoder.proj_out.bias", 768, &weights->proj_b},
    };
    for (size_t i = 0;
         i < sizeof(device_loads) / sizeof(device_loads[0]); i++) {
        float *host = read_f32(store, device_loads[i].name,
                               device_loads[i].count, error, error_size);
        if (host)
            *device_loads[i].slot = h3_gpu_tensor_from_f32(
                gpu, host, device_loads[i].count);
        free(host);
        if (!*device_loads[i].slot) {
            fail(error, error_size, "cannot load %s", device_loads[i].name);
            return 0;
        }
    }
    return 1;
}

static void gpu_decoder_weights_free(gpu_decoder_weights *weights) {
    for (int index = 0; index < VIT_LAYERS; index++)
        for (int w = 0; w < W_COUNT; w++)
            h3_gpu_tensor_free(weights->block[index][w]);
    free(weights->pq_w);
    free(weights->pq_b);
    free(weights->pi_w);
    free(weights->pi_b);
    free(weights->registers);
    h3_gpu_tensor_free(weights->lnw);
    h3_gpu_tensor_free(weights->lnb);
    h3_gpu_tensor_free(weights->proj_w);
    h3_gpu_tensor_free(weights->proj_b);
    memset(weights, 0, sizeof(*weights));
}

/* One ViT block on the GPU over resident weights. Every stage submits
 * before its scratch tensors are freed. The rope tables are uploaded once per
 * tile by the caller and shared across all blocks. */
static int gpu_vit_block(const gpu_decoder_weights *weights, int index,
                         h3_gpu *gpu, h3_gpu_tensor *stream, uint32_t tokens,
                         const h3_gpu_tensor *cosg, const h3_gpu_tensor *sing,
                         char *error, size_t error_size) {
    h3_gpu_tensor *const *dev = weights->block[index];
    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu,
        (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *attention = h3_gpu_tensor_new_f32(gpu,
        (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *attnout = h3_gpu_tensor_new_f32(gpu,
        (size_t)tokens * VIT_DIM);
    h3_gpu_tensor *ff = h3_gpu_tensor_new_f32(gpu,
        (size_t)tokens * 2 * VIT_FFN);
    h3_gpu_tensor *act = h3_gpu_tensor_new_f32(gpu,
        (size_t)tokens * VIT_FFN);
    h3_gpu_tensor *mlp = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * VIT_DIM);
    if (!normed || !q || !k || !v || !attention ||
        !attnout || !ff || !act || !mlp) {
        fail(error, error_size, "out of memory in the GPU VAE decoder");
        h3_gpu_tensor_free(normed);
        h3_gpu_tensor_free(q);
        h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(attention);
        h3_gpu_tensor_free(attnout);
        h3_gpu_tensor_free(ff);
        h3_gpu_tensor_free(act);
        h3_gpu_tensor_free(mlp);
        return 0;
    }
    int ok = 1;
    /* Attention half. */
    int was_open = !h3_gpu_begin(gpu);
    ok = h3_gpu_rms_norm_f32(gpu, normed, stream, dev[W_N1], tokens, VIT_DIM,
                             1e-5f) &&
        h3_gpu_linear_f32(gpu, q, normed, dev[W_Q], dev[W_QB], tokens,
                          VIT_DIM, VIT_DIM) &&
        h3_gpu_linear_f32(gpu, k, normed, dev[W_K], dev[W_KB], tokens,
                          VIT_DIM, VIT_DIM) &&
        h3_gpu_linear_f32(gpu, v, normed, dev[W_V], dev[W_VB], tokens,
                          VIT_DIM, VIT_DIM) &&
        h3_gpu_continue(gpu);
    if (!ok) {
        fail(error, error_size, "GPU ViT qkv failed: %s", h3_gpu_error(gpu));
        gpu_drain(gpu, was_open);
    }
    /* continue() leaves a freshly opened command buffer, so the next stage's
     * !begin() naturally reports "already open" and keeps the chain. Scratch
     * tensors stay alive until the tile's terminal submit. */
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3img_gpu_vit_qkrms_rope_f32(gpu, q, k, cosg, sing, tokens,
                                          VIT_HEADS, VIT_HEAD_DIM,
                                          VIT_ROTARY) &&
            h3img_gpu_vit_attention_f32(gpu, attention, q, k, v, tokens,
                                        VIT_HEADS, VIT_HEAD_DIM) &&
            h3_gpu_continue(gpu);
        if (!ok) {
            fail(error, error_size, "GPU ViT attention failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        }
    }
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3_gpu_linear_f32(gpu, attnout, attention, dev[W_O], dev[W_OB],
                               tokens, VIT_DIM, VIT_DIM) &&
            h3_gpu_scale_add_f32(gpu, stream, stream, attnout, dev[W_S1],
                                 tokens, VIT_DIM) &&
            h3_gpu_continue(gpu);
        if (!ok) {
            fail(error, error_size, "GPU ViT attention-out failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        }
    }
    /* MLP half. */
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3_gpu_rms_norm_f32(gpu, normed, stream, dev[W_N2], tokens,
                                 VIT_DIM, 1e-5f) &&
            h3_gpu_linear_f32(gpu, ff, normed, dev[W_F1], dev[W_F1B], tokens,
                              VIT_DIM, 2 * VIT_FFN) &&
            h3_gpu_continue(gpu);
        if (!ok) {
            fail(error, error_size, "GPU ViT ff1 failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        }
    }
    if (ok) {
        was_open = !h3_gpu_begin(gpu);
        ok = h3img_gpu_swiglu_swap_f32(gpu, act, ff, tokens, VIT_FFN) &&
            h3_gpu_linear_f32(gpu, mlp, act, dev[W_F2], dev[W_F2B], tokens,
                              VIT_FFN, VIT_DIM) &&
            h3_gpu_scale_add_f32(gpu, stream, stream, mlp, dev[W_S2], tokens,
                                 VIT_DIM) &&
            h3_gpu_submit(gpu);
        if (!ok) {
            fail(error, error_size, "GPU ViT ff2 failed: %s",
                 h3_gpu_error(gpu));
            gpu_drain(gpu, was_open);
        } else if (was_open) {
            h3_gpu_begin(gpu);
        }
    }
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(attention);
    h3_gpu_tensor_free(attnout);
    h3_gpu_tensor_free(ff);
    h3_gpu_tensor_free(act);
    h3_gpu_tensor_free(mlp);
    return ok;
}

/* One released-pipeline decode tile: the ViT decoder runs on a single tile
 * of latents with its own local rope grid (the released decoder tiles
 * anything larger than 256x256 pixels, so a 261-token decode is the
 * trained shape — a whole-image decode changes the attention context and
 * produces a grid artifact). */
static int gpu_decode_tile(const gpu_decoder_weights *weights, h3_gpu *gpu,
                           const float *latent, int latent_width,
                           int latent_height, float *pixels,
                           char *error, size_t error_size) {
    int ok = 1;
    int tokens = latent_width * latent_height + VIT_REGISTERS + 1;
    int count = latent_width * latent_height;

    /* post_quant_conv + token build stay on the CPU (the NCHW latent, the
     * planar/CL juggling, and the register/zero suffix are all tiny). The
     * small head weights are cached in the resident weight set. */
    float *tokens_data = NULL, *projected = NULL;
    {
        if (weights->pq_w && weights->pq_b && weights->pi_w &&
            weights->pi_b && weights->registers) {
            tokens_data = malloc((size_t)tokens * VIT_DIM * sizeof(float));
            float *projected_latent =
                malloc((size_t)LATENT_CHANNELS * count * sizeof(float));
            float *flat = malloc((size_t)LATENT_CHANNELS * count *
                                 sizeof(float));
            if (tokens_data && projected_latent && flat) {
                conv1x1(latent, LATENT_CHANNELS, count, weights->pq_w,
                        weights->pq_b, LATENT_CHANNELS, projected_latent);
                for (int c = 0; c < LATENT_CHANNELS; c++)
                    for (int i = 0; i < count; i++)
                        flat[(size_t)i * LATENT_CHANNELS + c] =
                            projected_latent[(size_t)c * count + i];
                linear(flat, count, LATENT_CHANNELS, weights->pi_w,
                       weights->pi_b, VIT_DIM, tokens_data);
                for (int r = 0; r < VIT_REGISTERS; r++)
                    memcpy(tokens_data + (size_t)(count + r) * VIT_DIM,
                           weights->registers + (size_t)r * VIT_DIM,
                           VIT_DIM * sizeof(float));
                memset(tokens_data +
                           (size_t)(count + VIT_REGISTERS) * VIT_DIM, 0,
                       VIT_DIM * sizeof(float));
            } else {
                fail(error, error_size, "out of memory in the GPU VAE decoder");
                ok = 0;
            }
            free(projected_latent);
            free(flat);
        } else {
            ok = 0;
        }
    }

    /* The rope table: coordinates [-1, 1), angle = 2*pi*pos*inv_freq, freqs
     * 8 per axis, concatenated and doubled; the zero-position suffix covers
     * the registers and the output token. */
    float *cos_table = NULL, *sin_table = NULL;
    if (ok) {
        cos_table = malloc((size_t)tokens * VIT_ROTARY * sizeof(float));
        sin_table = malloc((size_t)tokens * VIT_ROTARY * sizeof(float));
        if (cos_table && sin_table) {
            for (int token = 0; token < tokens; token++) {
                double position[2] = {0.0, 0.0};
                if (token < count) {
                    int r = token / latent_width;
                    int c = token % latent_width;
                    position[0] = 2.0 * ((double)r + 0.5) /
                                  (double)latent_height - 1.0;
                    position[1] = 2.0 * ((double)c + 0.5) /
                                  (double)latent_width - 1.0;
                }
                for (int f = 0; f < VIT_ROTARY; f++) {
                    int base = f % (VIT_ROTARY / 2);
                    int axis = base / (VIT_ROTARY / 4);
                    int freq = base % (VIT_ROTARY / 4);
                    double inv = 1.0 / pow(100.0,
                        (2.0 * (double)freq) / (double)(VIT_ROTARY / 2));
                    double angle = 2.0 * M_PI * position[axis] * inv;
                    cos_table[(size_t)token * VIT_ROTARY + f] =
                        (float)cos(angle);
                    sin_table[(size_t)token * VIT_ROTARY + f] =
                        (float)sin(angle);
                }
            }
        } else {
            fail(error, error_size, "out of memory in the GPU VAE decoder");
            ok = 0;
        }
    }

    h3_gpu_tensor *stream = NULL;
    h3_gpu_tensor *cosg = NULL, *sing = NULL;
    if (ok) {
        stream = h3_gpu_tensor_from_f32(gpu, tokens_data,
                                        (size_t)tokens * VIT_DIM);
        if (!stream) {
            fail(error, error_size, "cannot upload the VAE decoder tokens");
            ok = 0;
        }
    }
    /* Upload the per-tile rope tables once and share them across all 24 ViT
     * blocks; the historical code re-uploaded the identical tables in every
     * block (24x the bandwidth and small-buffer churn per tile). */
    if (ok) {
        cosg = h3_gpu_tensor_from_f32(gpu, cos_table,
                                      (size_t)tokens * VIT_ROTARY);
        sing = h3_gpu_tensor_from_f32(gpu, sin_table,
                                      (size_t)tokens * VIT_ROTARY);
        if (!cosg || !sing) {
            fail(error, error_size, "cannot upload the VAE decoder rope tables");
            ok = 0;
        }
    }
    for (int index = 0; ok && index < VIT_LAYERS; index++)
        ok = gpu_vit_block(weights, index, gpu, stream, (uint32_t)tokens,
                           cosg, sing, error, error_size);

    h3_gpu_tensor_free(cosg);
    h3_gpu_tensor_free(sing);

    if (ok) {
        h3_gpu_tensor *normed = NULL, *projected_g = NULL;
        if (weights->lnw && weights->lnb && weights->proj_w &&
            weights->proj_b) {
            normed = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * VIT_DIM);
            projected_g = h3_gpu_tensor_new_f32(gpu, (size_t)tokens * 768);
        }
        if (normed && projected_g) {
            int was_open = !h3_gpu_begin(gpu);
            ok = h3_gpu_layer_norm_f32(gpu, normed, stream, weights->lnw,
                                       weights->lnb, (uint32_t)tokens,
                                       VIT_DIM, 1e-5f) &&
                h3_gpu_linear_f32(gpu, projected_g, normed, weights->proj_w,
                                  weights->proj_b, (uint32_t)tokens, VIT_DIM,
                                  768) &&
                h3_gpu_submit(gpu);
            if (!ok) {
                fail(error, error_size, "GPU VAE decoder out failed: %s",
                     h3_gpu_error(gpu));
                gpu_drain(gpu, was_open);
            } else if (was_open) {
                h3_gpu_begin(gpu);
            }
            if (ok) {
                projected = malloc((size_t)tokens * 768 * sizeof(float));
                if (projected &&
                    h3_gpu_tensor_read_f32(projected_g, projected,
                                           (size_t)tokens * 768)) {
                    int pixel_h = latent_height * IMG_LATENT_DIV;
                    int pixel_w = latent_width * IMG_LATENT_DIV;
                    for (int r = 0; r < latent_height; r++)
                        for (int c = 0; c < latent_width; c++) {
                            const float *token =
                                projected + (size_t)(r * latent_width + c) *
                                768;
                            for (int ch = 0; ch < 3; ch++)
                                for (int i = 0; i < IMG_LATENT_DIV; i++)
                                    for (int j = 0; j < IMG_LATENT_DIV; j++)
                                        pixels[(size_t)ch * pixel_h * pixel_w +
                                               (size_t)(r * 16 + i) * pixel_w +
                                               (c * 16 + j)] =
                                            token[ch * 256 + i * 16 + j];
                        }
                } else {
                    fail(error, error_size,
                         "cannot read back the GPU VAE pixels");
                    ok = 0;
                }
            }
        } else if (ok) {
            fail(error, error_size, "cannot load the GPU VAE decoder out");
            ok = 0;
        }
        h3_gpu_tensor_free(normed);
        h3_gpu_tensor_free(projected_g);
    }

    free(tokens_data);
    free(projected);
    free(cos_table);
    free(sin_table);
    h3_gpu_tensor_free(stream);
    return ok;
}

/* The released tiling: tile_sample_min 256x256 pixels with a >=64-pixel
 * overlap, slack distributed round-robin in 16-pixel (latent-aligned)
 * steps. Returns the pixel-space start of every tile, the uniform tile
 * length (the requested length itself when it fits in a single tile), and
 * the per-boundary overlaps. */
static int vae_split_tiles(int length, int tile_size, int min_overlap,
                           int *starts, int *overlaps, int *tile_count,
                           int *tile_length) {
    if (tile_size >= length) {
        starts[0] = 0;
        *tile_count = 1;
        *tile_length = length;
        return 1;
    }
    int count = (length + tile_size - 1) / tile_size;
    while (tile_size * count - min_overlap * (count - 1) - length < 0)
        count += 1;
    /* starts/overlaps at the caller are H3IMG_MAX_TILES-sized stack arrays;
     * an unbounded count (canvas sides beyond ~4672 px) used to write past
     * them. */
    if (count > H3IMG_MAX_TILES) return 0;
    int overlaps_total[H3IMG_MAX_TILES] = {0};
    int remaining = tile_size * count - min_overlap * (count - 1) - length;
    for (int i = 0; i < remaining / 16; i++)
        overlaps_total[i % (count - 1)] += 16;
    int start = 0;
    starts[0] = 0;
    for (int i = 0; i < count - 1; i++) {
        overlaps[i] = min_overlap + overlaps_total[i];
        start += tile_size - overlaps[i];
        starts[i + 1] = start;
    }
    *tile_count = count;
    *tile_length = tile_size;
    return 1;
}

/* Stitch the decoded tiles back together (the released _stitch_tiles: the
 * head overlap of every tile is cross-faded against the raw tile above it
 * (row direction) and the already-processed tile to its left (column
 * direction), then every non-last tile trims its own overlap tail). */
static void vae_blend_into(float *tile, const float *previous, int channels,
                           int height, int width, int blend, int dim) {
    if (blend > height && dim == 0) blend = height;
    if (blend > width && dim == 1) blend = width;
    for (int p = 0; p < blend; p++) {
        double weight_b = (double)p / blend;
        double weight_a = 1.0 - weight_b;
        for (int c = 0; c < channels; c++)
            for (int q = 0; q < (dim == 0 ? width : height); q++) {
                size_t a_index = dim == 0
                    ? (size_t)c * height * width +
                      (size_t)(height - blend + p) * width + q
                    : (size_t)c * height * width +
                      (size_t)q * width + (width - blend + p);
                size_t b_index = dim == 0
                    ? (size_t)c * height * width + (size_t)p * width + q
                    : (size_t)c * height * width +
                      (size_t)q * width + p;
                tile[b_index] = (float)(previous[a_index] * weight_a +
                                        tile[b_index] * weight_b);
            }
    }
}

static int vae_stitch_tiles(float **tiles, int rows, int columns,
                            const int *y_overlaps, const int *x_overlaps,
                            int tile_height, int tile_width,
                            float *pixels, int out_height, int out_width,
                            char *error, size_t error_size) {
    /* Working copies so the blends can read the raw tile above. */
    int count = rows * columns;
    float **processed = malloc((size_t)count * sizeof(*processed));
    if (!processed) {
        fail(error, error_size, "out of memory stitching VAE tiles");
        return 0;
    }
    int ok = 1;
    for (int index = 0; index < count; index++) {
        size_t elements = (size_t)3 * tile_height * tile_width;
        processed[index] = malloc(elements * sizeof(float));
        /* A silent skip used to leave that tile's region of the output as
         * uninitialized garbage with no failure signal. */
        if (processed[index]) {
            memcpy(processed[index], tiles[index], elements * sizeof(float));
        } else {
            fail(error, error_size, "out of memory stitching VAE tiles");
            ok = 0;
        }
    }
    for (int i = 0; i < rows && ok; i++)
        for (int j = 0; j < columns && ok; j++) {
            float *tile = processed[(size_t)i * columns + j];
            /* With any failed working copy the blend neighbors could read
             * NULL — bail out (ok is already 0). */
            if (!tile) continue;
            if (i > 0 && tiles[(size_t)(i - 1) * columns + j])
                vae_blend_into(tile, tiles[(size_t)(i - 1) * columns + j],
                               3, tile_height, tile_width, y_overlaps[i - 1],
                               0);
            if (j > 0 && processed[(size_t)i * columns + j - 1])
                vae_blend_into(tile, processed[(size_t)i * columns + j - 1],
                               3, tile_height, tile_width, x_overlaps[j - 1],
                               1);
            int height = i == rows - 1 ? tile_height
                                       : tile_height - y_overlaps[i];
            int width = j == columns - 1 ? tile_width
                                        : tile_width - x_overlaps[j];
            int y_base = 0;
            for (int above = 0; above < i; above++)
                y_base += above == rows - 1 ? tile_height
                                            : tile_height - y_overlaps[above];
            int x_base = 0;
            for (int left = 0; left < j; left++)
                x_base += left == columns - 1 ? tile_width
                                              : tile_width - x_overlaps[left];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        pixels[(size_t)c * out_height * out_width +
                               (size_t)(y_base + y) * out_width +
                               (x_base + x)] =
                            tile[(size_t)c * tile_height * tile_width +
                                 (size_t)y * tile_width + x];
        }
    for (int index = 0; index < count; index++) free(processed[index]);
    free(processed);
    return ok;
}

int h3img_vae_decode(const char *weight_directory, const float *latent,
                     int latent_width, int latent_height, float *pixels,
                     char *error, size_t error_size) {
    if (!getenv("H3IMG_VAE_SCALAR") && !getenv("H3IMG_VAE_CPU")) {
        h3_weight_store *gpu_store = h3_weight_store_open(weight_directory,
                                                          error, error_size);
        if (gpu_store) {
            h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, error_size);
            int ok = gpu != NULL;
            if (ok) {
                /* The released decoder tiles anything larger than 256x256
                 * pixels: every tile is a separate 261-token decode with its
                 * own local rope grid, and the overlaps are cross-faded. A
                 * whole-image decode changes the attention context and
                 * prints a fixed 16-pixel grid into the output. */
                int pixel_height = latent_height * 16;
                int pixel_width = latent_width * 16;
                int y_starts[H3IMG_MAX_TILES], x_starts[H3IMG_MAX_TILES];
                int y_overlaps[H3IMG_MAX_TILES], x_overlaps[H3IMG_MAX_TILES];
                int y_count = 0, x_count = 0, tile_lh_px = 256, tile_lw_px = 256;
                if (!vae_split_tiles(pixel_height, 256, 64, y_starts,
                                     y_overlaps, &y_count, &tile_lh_px) ||
                    !vae_split_tiles(pixel_width, 256, 64, x_starts,
                                     x_overlaps, &x_count, &tile_lw_px)) {
                    /* Hard error, not a CPU fallback: the scalar path would
                     * decode the whole oversized image in one attention. */
                    fail(error, error_size,
                         "VAE decode needs at most %d tiles per side "
                         "(canvas side %d x %d exceeds the tiling range)",
                         H3IMG_MAX_TILES, pixel_width, pixel_height);
                    if (gpu) h3_gpu_free(gpu);
                    h3_weight_store_free(gpu_store);
                    return 0;
                }
                int tile_lh = tile_lh_px / 16, tile_lw = tile_lw_px / 16;
                /* Resident decoder weights: uploaded once, reused by every
                 * tile (per-tile re-upload dominated the decode time). */
                gpu_decoder_weights weights;
                if (ok)
                    ok = gpu_decoder_weights_build(gpu_store, gpu, &weights,
                                                   error, error_size);
                float **tiles = NULL;
                if (ok) {
                    tiles = malloc((size_t)y_count * x_count *
                                   sizeof(*tiles));
                    ok = tiles != NULL;
                    if (tiles)
                        for (int index = 0; index < y_count * x_count; index++)
                            tiles[index] = NULL;
                }
                for (int i = 0; ok && i < y_count; i++)
                    for (int j = 0; ok && j < x_count; j++) {
                        int ly = y_starts[i] / 16, lx = x_starts[j] / 16;
                        float *tile_latent =
                            malloc((size_t)24 * tile_lh * tile_lw *
                                   sizeof(float));
                        ok = tile_latent != NULL;
                        for (int c = 0; ok && c < 24; c++)
                            for (int y = 0; y < tile_lh; y++)
                                for (int x = 0; x < tile_lw; x++)
                                    tile_latent[(size_t)c * tile_lh *
                                                tile_lw +
                                                (size_t)y * tile_lw + x] =
                                        latent[(size_t)c * latent_height *
                                               latent_width +
                                               (size_t)(ly + y) * latent_width +
                                               (lx + x)];
                        if (ok) {
                            tiles[(size_t)i * x_count + j] =
                                malloc((size_t)3 * tile_lh_px * tile_lw_px *
                                       sizeof(float));
                            ok = tiles[(size_t)i * x_count + j] != NULL &&
                                gpu_decode_tile(&weights, gpu, tile_latent,
                                                tile_lw, tile_lh,
                                                tiles[(size_t)i * x_count + j],
                                                error, error_size);
                        }
                        free(tile_latent);
                    }
                if (ok)
                    ok = vae_stitch_tiles(tiles, y_count, x_count, y_overlaps,
                                          x_overlaps, tile_lh_px, tile_lw_px,
                                          pixels, pixel_height, pixel_width,
                                          error, error_size);
                if (tiles) {
                    for (int index = 0; index < y_count * x_count; index++)
                        free(tiles[index]);
                    free(tiles);
                }
                /* Safe on a partial build: every slot is zeroed up front. */
                gpu_decoder_weights_free(&weights);
            }
            if (gpu) h3_gpu_free(gpu);
            h3_weight_store_free(gpu_store);
            if (ok) return 1;
            fprintf(stderr, "h3img: GPU VAE decoder failed (%s); "
                    "falling back to the CPU path\n",
                    (error && error[0]) ? error : "unknown error");
            if (error && error_size) error[0] = '\0';
        }
    }
    char detail[256];
    (void)detail;
    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                  error_size);
    if (!store) return 0;
    int ok = 1;
    char name[160];
    int tokens = latent_width * latent_height + VIT_REGISTERS + 1;
    int count = latent_width * latent_height;
    /* Host-side weight copies are owned by this function and released once at
     * cleanup. Declared up front (before any goto cleanup) so the unified
     * frees below never read an uninitialized pointer on early-exit paths. */
    float *piw = NULL, *pib = NULL, *registers = NULL;
    float *lnw = NULL, *lnb = NULL, *proj_w = NULL, *pob = NULL;

    /* post_quant_conv k1 (24 -> 24). */
    snprintf(name, sizeof(name), "post_quant_conv.weight");
    float *w = read_f32(store, name, (size_t)LATENT_CHANNELS * LATENT_CHANNELS,
                        error, error_size);
    snprintf(name, sizeof(name), "post_quant_conv.bias");
    float *b = read_f32(store, name, LATENT_CHANNELS, error, error_size);
    if (!w || !b) {
        ok = 0;
        goto cleanup;
    }
    float *tokens_data = malloc((size_t)tokens * VIT_DIM * sizeof(float));
    if (!tokens_data) {
        fail(error, error_size, "out of memory in VAE decoder");
        ok = 0;
        goto cleanup;
    }
    float *flat = malloc((size_t)LATENT_CHANNELS * count * sizeof(float));
    if (!flat) {
        fail(error, error_size, "out of memory in VAE decoder");
        ok = 0;
        free(tokens_data);
        goto cleanup;
    }
    /* post_quant_conv runs on the NCHW latent; the token build then reads
     * channel-last, so keep two views. */
    float *projected_latent =
        malloc((size_t)LATENT_CHANNELS * count * sizeof(float));
    if (!projected_latent) {
        fail(error, error_size, "out of memory in VAE decoder");
        ok = 0;
        free(flat);
        free(w);
        free(b);
        goto cleanup;
    }
    conv1x1(latent, LATENT_CHANNELS, count, w, b, LATENT_CHANNELS,
            projected_latent);
    for (int c = 0; c < LATENT_CHANNELS; c++)
        for (int index = 0; index < count; index++)
            flat[(size_t)index * LATENT_CHANNELS + c] =
                projected_latent[(size_t)c * count + index];
    if (getenv("H3IMG_VAE_DUMP")) {
        dump_stage("/tmp/h3img_vae_pq.f32", projected_latent,
                   (size_t)LATENT_CHANNELS * count);
        dump_stage("/tmp/h3img_vae_flat.f32", flat,
                   (size_t)LATENT_CHANNELS * count);
    }
    free(projected_latent);
    snprintf(name, sizeof(name), "decoder.proj_in.weight");
    piw = read_f32(store, name, (size_t)VIT_DIM * LATENT_CHANNELS, error, error_size);
    snprintf(name, sizeof(name), "decoder.proj_in.bias");
    pib = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "decoder.register_tokens");
    registers = read_f32(store, name, (size_t)VIT_REGISTERS * VIT_DIM, error, error_size);
    if (!piw || !pib || !registers) {
        ok = 0;
        free(flat);
        free(tokens_data);
        goto cleanup;
    }
    for (int index = 0; index < count; index++)
        for (int d = 0; d < VIT_DIM; d++) {
            double sum = pib[d];
            for (int c = 0; c < LATENT_CHANNELS; c++)
                sum += (double)flat[(size_t)index * LATENT_CHANNELS + c] *
                       piw[(size_t)d * LATENT_CHANNELS + c];
            tokens_data[(size_t)index * VIT_DIM + d] = (float)sum;
        }
    for (int r = 0; r < VIT_REGISTERS; r++)
        memcpy(tokens_data + (size_t)(count + r) * VIT_DIM,
               registers + (size_t)r * VIT_DIM, VIT_DIM * sizeof(float));
    memset(tokens_data + (size_t)(count + VIT_REGISTERS) * VIT_DIM, 0,
           VIT_DIM * sizeof(float));
    free(flat);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_dec_tokens.f32", tokens_data,
                   (size_t)tokens * VIT_DIM);
    free(w);
    free(b);

    /* The rope table is passed in by the caller through the fixture; for the
     * standalone path rebuild it here. Coordinates: [-1, 1) grids, angle =
     * 2*pi*pos*inv_freq, freqs 8 per axis, concatenated and doubled. */
    float *cos_table = malloc((size_t)tokens * VIT_ROTARY * sizeof(float));
    float *sin_table = malloc((size_t)tokens * VIT_ROTARY * sizeof(float));
    if (!cos_table || !sin_table) {
        fail(error, error_size, "out of memory in VAE decoder");
        ok = 0;
        free(cos_table);
        free(sin_table);
        free(tokens_data);
        goto cleanup;
    }
    {
        for (int r = 0; r < latent_height; r++)
            for (int c = 0; c < latent_width; c++) {
                /* Reference: angles per token = [h*invfreq[0..7],
                 * w*invfreq[0..7]], then tile(2) duplicates all 16. */
                (void)r;
                (void)c;
            }
        /* (The loop above documents the intent; the real table is built
         * below over all tokens including the zero-position suffix.) */
        for (int token = 0; token < tokens; token++) {
            double position[2] = {0.0, 0.0};
            if (token < count) {
                int r = token / latent_width;
                int c = token % latent_width;
                position[0] = 2.0 * ((double)r + 0.5) / (double)latent_height - 1.0;
                position[1] = 2.0 * ((double)c + 0.5) / (double)latent_width - 1.0;
            }
            for (int f = 0; f < VIT_ROTARY; f++) {
                /* tile(2): index f wraps to f % 16 of the concatenated
                 * [axis0 x 8 freqs | axis1 x 8 freqs] angle vector. */
                int base = f % (VIT_ROTARY / 2);
                int axis = base / (VIT_ROTARY / 4); /* 2 axes x 8 freqs */
                int freq = base % (VIT_ROTARY / 4);
                double inv = 1.0 / pow(100.0, (2.0 * (double)freq) / (double)(VIT_ROTARY / 2));
                double angle = 2.0 * M_PI * position[axis] * inv;
                cos_table[(size_t)token * VIT_ROTARY + f] = (float)cos(angle);
                sin_table[(size_t)token * VIT_ROTARY + f] = (float)sin(angle);
            }
        }
        if (getenv("H3IMG_VAE_DUMP")) {
            dump_stage("/tmp/h3img_vae_dec_cos.f32", cos_table,
                       (size_t)tokens * VIT_ROTARY);
            dump_stage("/tmp/h3img_vae_dec_sin.f32", sin_table,
                       (size_t)tokens * VIT_ROTARY);
        }
    }

    for (int index = 0; ok && index < VIT_LAYERS; index++) {
        char block_error[512] = {0};
        int block_ok = vit_block(store, index, tokens_data, tokens,
                                 cos_table, sin_table,
                                 block_error, sizeof(block_error));
        if (!block_ok) {
            /* Never pass the same buffer as both vsnprintf's destination and
             * a %s argument — that aliases source and destination (UB). */
            fail(error, error_size, "VAE decoder block %d: %s", index,
                 block_error);
            ok = 0;
            break;
        }
        if (ok && getenv("H3IMG_VAE_DUMP") && index < 2) {
            char dump_path[96];
            snprintf(dump_path, sizeof(dump_path),
                     "/tmp/h3img_vae_dec_b%d.f32", index);
            dump_stage(dump_path, tokens_data, (size_t)tokens * VIT_DIM);
        }
    }
    if (!ok) {
        free(cos_table);
        free(sin_table);
        free(tokens_data);
        goto cleanup;
    }

    snprintf(name, sizeof(name), "decoder.norm_out.weight");
    lnw = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "decoder.norm_out.bias");
    lnb = read_f32(store, name, VIT_DIM, error, error_size);
    snprintf(name, sizeof(name), "decoder.proj_out.weight");
    proj_w = read_f32(store, name, (size_t)VIT_DIM * 768, error, error_size);
    snprintf(name, sizeof(name), "decoder.proj_out.bias");
    pob = read_f32(store, name, 768, error, error_size);
    if (!lnw || !lnb || !proj_w || !pob) {
        ok = 0;
        free(cos_table);
        free(sin_table);
        free(tokens_data);
        goto cleanup;
    }
    layer_norm(tokens_data, tokens, VIT_DIM, lnw, lnb, 1e-5f);
    if (getenv("H3IMG_VAE_DUMP"))
        dump_stage("/tmp/h3img_vae_dec_normout.f32", tokens_data,
                   (size_t)tokens * VIT_DIM);
    float *projected = malloc((size_t)tokens * 768 * sizeof(float));
    if (!projected) {
        fail(error, error_size, "out of memory in VAE decoder");
        ok = 0;
        free(cos_table);
        free(sin_table);
        free(tokens_data);
        goto cleanup;
    }
    linear(tokens_data, tokens, VIT_DIM, proj_w, pob, 768, projected);
    /* Unpatchify: token[r*w+c][channel*256 + i*16 + j] -> pixel[c][r*16+i][c*16+j]. */
    int pixel_h = latent_height * IMG_LATENT_DIV;
    int pixel_w = latent_width * IMG_LATENT_DIV;
    for (int r = 0; r < latent_height; r++)
        for (int c = 0; c < latent_width; c++) {
            const float *token = projected + (size_t)(r * latent_width + c) * 768;
            for (int ch = 0; ch < 3; ch++)
                for (int i = 0; i < IMG_LATENT_DIV; i++)
                    for (int j = 0; j < IMG_LATENT_DIV; j++) {
                        float value = token[ch * 256 + i * 16 + j];
                        pixels[(size_t)ch * pixel_h * pixel_w +
                               (size_t)(r * 16 + i) * pixel_w +
                               (c * 16 + j)] = value;
                    }
        }
    free(projected);
    free(cos_table);
    free(sin_table);
    free(tokens_data);

cleanup:
    /* Host-side decoder weight copies (proj_in/registers/norm_out/proj_out).
     * The GPU path never reaches here, but the CPU fallback and all its
     * failure branches must not leak ~4 MB per decode. */
    free(piw); free(pib); free(registers);
    free(lnw); free(lnb); free(proj_w); free(pob);
    h3_weight_store_free(store);
    return ok;
}
