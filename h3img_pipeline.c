#include "h3img_pipeline.h"
#include "h3_dit.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3img_imgproc.h"
#include "h3img_png.h"
#include "h3img_vae.h"
#include "h3_multimodal.h"
#include "h3_vision_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Optional phase timing (H3IMG_TIMING=1). Emits cumulative wall seconds at
 * each pipeline phase boundary to stderr; default-off and allocation-free. */
static double h3img_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#define H3IMG_TIMING_BEGIN() \
    int h3img_timing = getenv("H3IMG_TIMING") != NULL; \
    double h3img_t0 = h3img_now_seconds(), h3img_tprev = h3img_t0
#define H3IMG_TIMING_MARK(label) do { \
    if (h3img_timing) { \
        double h3img_tnow = h3img_now_seconds(); \
        fprintf(stderr, "[timing] %-22s %7.3fs (+%7.3fs)\n", \
                (label), h3img_tnow - h3img_t0, h3img_tnow - h3img_tprev); \
        h3img_tprev = h3img_tnow; \
    } \
} while (0)

/* The released edit/t2i encoders truncate the prompt (and each edit label
 * block) at max_sequence_length = 4096 tokens. */
#define H3IMG_MAX_TOKENS 4096
#define H3IMG_CHANNELS 24
#define H3IMG_STEPS 6

/* The distilled sampler grid from pdd_config.json, consumed verbatim by the
 * official MiniMaxH3Scheduler (shift_provided_sigmas is unset, so provided
 * sigmas are used unshifted). Six model evaluations, terminal zero. */
static const float H3IMG_SIGMAS[H3IMG_STEPS + 1] = {
    1.0f,
    0.9675236940383911f,
    0.8875485062599182f,
    0.75f,
    0.5327739119529724f,
    0.23200887441635132f,
    0.0f
};

/* The VAE's per-channel latent statistics (vae config latents_mean/std).
 * The sampler runs entirely in the normalized space; the decode undoes the
 * normalization channel-wise. */
static const float H3IMG_LATENTS_MEAN[24] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
    -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
    -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
    -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f,
    0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
    -1.1206848621368408f, 0.3581933379173279f, -0.04225143790245056f,
    0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f
};
static const float H3IMG_LATENTS_STD[24] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f,
    1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
    0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
    0.772995293140411f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
    0.6936293244361877f, 2.961095094680786f, 2.7694199085235596f,
    3.0496184825897217f, 2.1088054180145264f, 3.276226282119751f,
    3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f
};

static int h3img_multiple_of_32(int value) {
    return value > 0 && value % 32 == 0;
}

/* Optional initial-latent injection: `x.latents` fp32 [24, h, w]. */
static int h3img_load_latents(const char *path, float *latent, size_t elements,
                              char *error, size_t error_size) {
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    const h3_st_tensor *tensor = h3_st_find(&header, "x.latents");
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != elements) {
        h3_st_free_header(&header);
        snprintf(error, error_size,
                 "latents file %s must hold fp32 x.latents [%zu] elements",
                 path, elements);
        return 0;
    }
    int ok = h3_st_read_data(&header, tensor, latent,
                             elements * sizeof(float), error, error_size);
    h3_st_free_header(&header);
    return ok;
}

/* PIL bound_short_edge: a short edge within [min, max] keeps its size,
 * otherwise the image scales so the short edge lands on the nearer bound;
 * both sides round to `multiple`. */
static void bound_short_edge(int width, int height, int min_edge, int max_edge,
                             int multiple, int *out_w, int *out_h) {
    int short_edge = width < height ? width : height;
    double scale = 1.0;
    if (short_edge < min_edge) scale = (double)min_edge / short_edge;
    else if (short_edge > max_edge) scale = (double)max_edge / short_edge;
    *out_w = (int)(width * scale / multiple + 0.5) * multiple;
    *out_h = (int)(height * scale / multiple + 0.5) * multiple;
    if (*out_w < multiple) *out_w = multiple;
    if (*out_h < multiple) *out_h = multiple;
}

static int load_reference_rgb(const char *path, uint8_t **rgb_out,
                              int *width, int *height, char *error,
                              size_t error_size) {
    uint32_t w = 0, h = 0;
    uint8_t *rgba = NULL;
    /* The qwen-derived PNG reader returns 0 on success. */
    if (h3img_png_read_rgba(path, &w, &h, &rgba, error, error_size) != 0)
        return 0;
    size_t pixels = (size_t)w * h;
    uint8_t *rgb = malloc(pixels * 3);
    if (!rgb) {
        free(rgba);
        snprintf(error, error_size, "out of memory for reference image");
        return 0;
    }
    /* The reference treats alpha as white-background composite (PIL
     * convert("RGB") on a transparent PNG flattens onto white). */
    h3img_imgproc_white_composite(rgba, (int)pixels, rgb);
    free(rgba);
    *rgb_out = rgb;
    *width = (int)w;
    *height = (int)h;
    return 1;
}

/* One reference image fully prepared: the vision tensors, the conditioning
 * latents (posterior sample, normalized, noised to t = 0.999 and patchified)
 * and the layout row count. */
typedef struct {
    h3_vision_output vision;
    float *condition_rows;   /* patchified, [rows][96] */
    size_t row_count;
    int latent_w, latent_h;
} h3img_reference_state;

static void free_reference_state(h3img_reference_state *state) {
    h3_vision_output_free(&state->vision);
    free(state->condition_rows);
    memset(state, 0, sizeof(*state));
}

int h3img_generate_ex(const char *model_dir, const char *prompt,
                             const char *const *reference_paths,
                             size_t reference_count, int width, int height,
                             uint64_t seed, const char *latents_path,
                             const char *output_png, int *total_steps,
                             h3img_progress progress, void *progress_opaque,
                             char *error, size_t error_size);

int h3img_generate(const char *model_dir, const char *prompt, int width,
                   int height, uint64_t seed, const char *latents_path,
                   const char *output_png, int *total_steps,
                   h3img_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    return h3img_generate_ex(model_dir, prompt, NULL, 0, width, height, seed,
                             latents_path, output_png, total_steps, progress,
                             progress_opaque, error, error_size);
}

/* Load one reference's official posterior latents and noising noise from the
 * A/B fixture (x.ref_latents_<i> / x.ref_noise_<i>, fp32 [24, h, w]).
 * Returns 0 when the fixture does not carry them (own-draws mode). */
static int edit_fixture_refs(const char *path, int index, float *latent,
                             float *noise, size_t elements) {
    h3_st_header header;
    char error_dummy[256];
    if (!h3_st_read_header(path, &header, error_dummy, sizeof(error_dummy)))
        return 0;
    char name[64];
    const h3_st_tensor *lt = NULL, *nt = NULL;
    snprintf(name, sizeof(name), "x.ref_latents_%d", index);
    lt = h3_st_find(&header, name);
    snprintf(name, sizeof(name), "x.ref_noise_%d", index);
    nt = h3_st_find(&header, name);
    int ok = lt && nt && lt->dtype == H3_DTYPE_F32 &&
             nt->dtype == H3_DTYPE_F32 &&
             h3_st_tensor_elements(lt) == elements &&
             h3_st_tensor_elements(nt) == elements;
    if (ok) {
        ok = h3_st_read_data(&header, lt, latent, elements * sizeof(float),
                             error_dummy, sizeof(error_dummy)) &&
             h3_st_read_data(&header, nt, noise, elements * sizeof(float),
                             error_dummy, sizeof(error_dummy));
    }
    h3_st_free_header(&header);
    return ok;
}

int h3img_generate_ex(const char *model_dir, const char *prompt,
                      const char *const *reference_paths,
                      size_t reference_count, int width, int height,
                      uint64_t seed, const char *latents_path,
                      const char *output_png, int *total_steps,
                      h3img_progress progress, void *progress_opaque,
                      char *error, size_t error_size) {
    H3IMG_TIMING_BEGIN();
    if (!model_dir || !prompt || !*prompt || !output_png) {
        snprintf(error, error_size, "h3img_generate: missing arguments");
        return 0;
    }
    if (!h3img_multiple_of_32(width) || !h3img_multiple_of_32(height)) {
        snprintf(error, error_size,
                 "image size must be a multiple of 32 (got %dx%d)", width,
                 height);
        return 0;
    }
    /* The VAE compresses 16x spatially; each DiT patch then covers a 2x2
     * latent block, i.e. 32 pixels. */
    /* The released pipeline generates on a canvas of at least
     * min_generation_area (default 2048x2048) at the requested aspect ratio
     * and bicubic-resizes back to the requested size after decoding — the
     * model is trained on 2048-class canvases and small direct sizes are
     * out of distribution (the official default output is 2048x2048). The
     * A/B harness (--latents) pins the exact fixture geometry instead.
     * H3IMG_MIN_AREA overrides the floor (the released pipeline accepts
     * 1024²..4096², where the floor value means "generate at the requested
     * size"). */
    int canvas_w = width, canvas_h = height;
    if (!latents_path) {
        const char *area_env = getenv("H3IMG_MIN_AREA");
        long min_area = (long)2048 * 2048;
        /* Upper clamp too: an extreme H3IMG_MIN_AREA used to overflow the
         * int canvas conversion (the documented range is 1024²..4096²). */
        if (area_env && *area_env) {
            char *end = NULL;
            min_area = strtol(area_env, &end, 10);
            if (!end || *end || min_area <= 0) min_area = (long)2048 * 2048;
        }
        if (min_area < (long)1024 * 1024) min_area = (long)1024 * 1024;
        if (min_area > (long)4096 * 4096) min_area = (long)4096 * 4096;
        double area = (double)width * (double)height;
        double scale = area < (double)min_area ?
            sqrt((double)min_area / area) : 1.0;
        canvas_w = (int)((double)width * scale / 32.0 + 0.5) * 32;
        canvas_h = (int)((double)height * scale / 32.0 + 0.5) * 32;
        if (canvas_w < 32) canvas_w = 32;
        if (canvas_h < 32) canvas_h = 32;
    }
    /* The official size table tops out at 6272 px per side (21:9, all
     * buckets ~16.8 MP); the VAE decode tiling supports up to 33 tiles per
     * side at that length. Anything beyond is rejected up front. */
    if (canvas_w > 6272 || canvas_h > 6272) {
        snprintf(error, error_size,
                 "generation canvas %dx%d exceeds the supported maximum "
                 "(6272 px per side, official size table)",
                 canvas_w, canvas_h);
        return 0;
    }
    int latent_w = canvas_w / 16;
    int latent_h = canvas_h / 16;
    if (latent_w < 4 || latent_h < 4 || latent_w % 2 || latent_h % 2) {
        snprintf(error, error_size, "latent grid %dx%d is not supported",
                 latent_w, latent_h);
        return 0;
    }
    if (total_steps) *total_steps = H3IMG_STEPS;
    /* M4 runs the reference-faithful kernels: the fused-MLP fast path has a
     * known Metal failure on this checkpoint (pinned by the M2 DiT parity,
     * which sets the same switches), and the final head stays F32 like the
     * checkpoint's pdd heads. */
    setenv("H3_DISABLE_FUSED_MLP", "1", 1);
    setenv("H3_DIT_F32_FINAL", "1", 1);
    if (progress) progress("tokenize", 0, 1, progress_opaque);

    /* 1. Verbatim-prompt tokenization (no template, no special tokens). */
    char *tokenizer_path = malloc(strlen(model_dir) + 64);
    if (!tokenizer_path) {
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    sprintf(tokenizer_path, "%s/tokenizer/tokenizer.json", model_dir);
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                error_size);
    free(tokenizer_path);
    if (!tokenizer) return 0;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    int ok = 1;
    if (!reference_count)
        ok = h3_tokenizer_encode(tokenizer, prompt, 0, &ids, &token_count,
                                 error, error_size);
    if (!ok) {
        h3_tokenizer_free(tokenizer);
        return 0;
    }
    if (ids && token_count > H3IMG_MAX_TOKENS) token_count = H3IMG_MAX_TOKENS;

    /* 2. The released first 50 Qwen3-VL language layers, pre-norm state. */
    if (progress) progress("text", 0, 50, progress_opaque);
    char *text_path = malloc(strlen(model_dir) + 32);
    char *dit_path = malloc(strlen(model_dir) + 32);
    char *vae_path = malloc(strlen(model_dir) + 16);
    if (!text_path || !dit_path || !vae_path) {
        free(text_path); free(dit_path); free(vae_path);
        h3_tokenizer_ids_free(ids);
        h3_tokenizer_free(tokenizer);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    sprintf(text_path, "%s/text_encoder", model_dir);
    sprintf(dit_path, "%s/transformer", model_dir);
    sprintf(vae_path, "%s/vae_diffusers", model_dir);
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3img_reference_state *references = NULL;
    h3_layout_ref *layout_refs = NULL;
    if (reference_count) {
        /* Edit: one vision block per reference ("Picture i: " label, no
         * angle brackets), then the verbatim prompt. */
        references = calloc(reference_count, sizeof(*references));
        layout_refs = calloc(reference_count, sizeof(*layout_refs));
        h3_vision_output *visions = calloc(reference_count,
                                           sizeof(*visions));
        if (!references || !layout_refs || !visions) {
            free(references); free(layout_refs); free(visions);
            free(dit_path); free(vae_path); free(text_path);
            h3_tokenizer_free(tokenizer);
            snprintf(error, error_size, "out of memory for references");
            return 0;
        }
        /* The engine's noise RNG also draws the posterior samples, matching
         * the reference's single-generator sequencing. */
        h3_rng rng;
        h3_rng_seed(&rng, seed);
        static const float pixel_mean[3] = {0.485f, 0.456f, 0.406f};
        static const float pixel_std[3] = {0.229f, 0.224f, 0.225f};
        for (size_t index = 0; index < reference_count; index++) {
            uint8_t *rgb = NULL;
            int sw = 0, sh = 0;
            if (!reference_paths || !reference_paths[index]) {
                snprintf(error, error_size,
                         "reference image %zu is missing", index);
                ok = 0;
                break;
            }
            if (getenv("H3IMG_EDIT_TRACE"))
                fprintf(stderr, "[edit: loading ref %zu]\n", index);
            if (!load_reference_rgb(reference_paths[index], &rgb, &sw, &sh,
                                    error, error_size)) {
                free(rgb);
                ok = 0;
                break;
            }
            /* Vision input: short edge pinned to 1024, both sides rounded
             * to 32. */
            int vw = 0, vh = 0;
            bound_short_edge(sw, sh, 1024, 1024, 32, &vw, &vh);
            uint8_t *vision_rgb = malloc((size_t)vw * vh * 3);
            float *vision_pixels = malloc((size_t)3 * vh * vw * sizeof(float));
            /* VAE input: short edge kept within [1024, 2048], rounded to 32
             * (a reference keeps its own size when in range). */
            int aw = 0, ah = 0;
            bound_short_edge(sw, sh, 1024, 2048, 32, &aw, &ah);
            uint8_t *vae_rgb = malloc((size_t)aw * ah * 3);
            float *vae_pixels = malloc((size_t)3 * ah * aw * sizeof(float));
            if (!vision_rgb || !vision_pixels || !vae_rgb || !vae_pixels) {
                free(rgb); free(vision_rgb); free(vision_pixels);
                free(vae_rgb); free(vae_pixels);
                snprintf(error, error_size,
                         "out of memory for reference %zu buffers", index);
                ok = 0;
                break;
            }
            if (!h3img_imgproc_resize_bicubic_u8(rgb, sw, sh, 3, vision_rgb,
                                                 vw, vh) ||
                !h3img_imgproc_resize_bicubic_u8(rgb, sw, sh, 3, vae_rgb,
                                                 aw, ah)) {
                free(rgb); free(vision_rgb); free(vision_pixels);
                free(vae_rgb); free(vae_pixels);
                snprintf(error, error_size, "reference resize failed");
                ok = 0;
                break;
            }
            free(rgb);
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < vh; y++)
                    for (int x = 0; x < vw; x++)
                        vision_pixels[(size_t)c * vh * vw + (size_t)y * vw +
                                      x] =
                            vision_rgb[((size_t)y * vw + x) * 3 + c] /
                            255.0f;
            /* ImageNet normalization for the VAE input. */
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < ah; y++)
                    for (int x = 0; x < aw; x++) {
                        float value =
                            vae_rgb[((size_t)y * aw + x) * 3 + c] / 255.0f;
                        vae_pixels[(size_t)c * ah * aw + (size_t)y * aw + x] =
                            (value - pixel_mean[c]) / pixel_std[c];
                    }
            free(vision_rgb);
            free(vae_rgb);
            if (progress) progress("vision", (int)index,
                                   (int)reference_count, progress_opaque);
            int vision_ok = h3_vision_encode_bf16(text_path, "h3_shaders.metal",
                                       vision_pixels, 1, vh, vw, NULL, NULL,
                                       &references[index].vision, error,
                                       error_size);
            if (!vision_ok) {
                free(vision_pixels);
                free(vae_pixels);
                ok = 0;
                break;
            }
            free(vision_pixels);
            if (getenv("H3IMG_DUMP_LATENTS")) {
                char dump_path[128];
                snprintf(dump_path, sizeof(dump_path),
                         "/tmp/h3img_vision_merged_%zu.bf16", index);
                FILE *dump = fopen(dump_path, "wb");
                if (dump) {
                    fwrite(references[index].vision.merged, sizeof(uint16_t),
                           references[index].vision.tokens * 5120, dump);
                    fclose(dump);
                }
            }
            visions[index] = references[index].vision;
            /* Conditioning latents: posterior sample, normalized, noised to
             * the 0.999 conditioning level (x_t = t*x0 + (1-t)*noise), then
             * patchified rows. */
            int alh = ah / 16, alw = aw / 16;
            float *latent = malloc((size_t)24 * alh * alw * sizeof(float));
            float *noise = malloc((size_t)24 * alh * alw * sizeof(float));
            if (!latent || !noise) {
                free(latent); free(noise);
                free(vae_pixels);
                snprintf(error, error_size,
                         "out of memory for reference %zu latents", index);
                ok = 0;
                break;
            }
            /* The edit A/B harness injects the reference's official posterior
             * latents and noising noise (x.ref_latents / x.ref_noise in the
             * latents fixture) so both stacks replay identical draws. */
            if (latents_path && edit_fixture_refs(latents_path, (int)index,
                                                  latent, noise,
                                                  (size_t)24 * alh * alw)) {
                free(vae_pixels);
            } else {
                /* A latents_path that lacks this ref's tensors (or carries a
                 * mismatched geometry) silently degraded to own-drawn noise
                 * and desynchronized the A/B; trace it loudly. */
                if (latents_path && getenv("H3IMG_EDIT_TRACE"))
                    fprintf(stderr,
                            "[edit: ref %zu missing/mismatched fixture "
                            "latents — own-draw fallback]\n", index);
                if (!h3img_vae_encode_sampled(vae_path, vae_pixels, aw, ah,
                                              &rng, latent, error,
                                              error_size)) {
                    free(latent); free(noise);
                    free(vae_pixels);
                    ok = 0;
                    break;
                }
                free(vae_pixels);
                for (int c = 0; c < 24; c++)
                    for (int i = 0; i < alh * alw; i++) {
                        size_t offset = (size_t)c * alh * alw + i;
                        latent[offset] =
                            (latent[offset] - H3IMG_LATENTS_MEAN[c]) /
                            H3IMG_LATENTS_STD[c];
                        noise[offset] = h3_rng_normal(&rng);
                    }
            }
            /* Patchify into rows of 96: (h//2, w//2) rows, (c, dh, dw) tap
             * order — same convention as h3_dit_patchify_video. */
            size_t rows = (size_t)(alh / 2) * (alw / 2);
            float *rows_buffer = malloc(rows * 96 * sizeof(float));
            if (!rows_buffer) {
                free(latent); free(noise);
                snprintf(error, error_size,
                         "out of memory for reference %zu condition rows",
                         index);
                ok = 0;
                break;
            }
            static const float aug = 0.999f;
            for (int h2 = 0; h2 < alh / 2; h2++)
                for (int w2 = 0; w2 < alw / 2; w2++) {
                    float *row = rows_buffer +
                        ((size_t)h2 * (alw / 2) + w2) * 96;
                    for (int c = 0; c < 24; c++)
                        for (int dh = 0; dh < 2; dh++)
                            for (int dw = 0; dw < 2; dw++) {
                                size_t offset =
                                    (size_t)c * alh * alw +
                                    (size_t)(h2 * 2 + dh) * alw +
                                    (w2 * 2 + dw);
                                row[(c * 2 + dh) * 2 + dw] =
                                    aug * latent[offset] +
                                    (1.0f - aug) * noise[offset];
                            }
                }
            free(latent);
            free(noise);
            references[index].condition_rows = rows_buffer;
            references[index].row_count = rows;
            references[index].latent_w = alw;
            references[index].latent_h = alh;
            layout_refs[index].kind = H3_LAYOUT_REF_IMAGE;
            layout_refs[index].latent_t = 1;
            layout_refs[index].latent_h = alh;
            layout_refs[index].latent_w = alw;
            layout_refs[index].audio_t = 0;
        }
        if (ok) {
            ok = h3_multimodal_encode_edit_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt, visions,
                reference_count, NULL, NULL, &text, error, error_size);
        }
        free(visions);
        h3_tokenizer_free(tokenizer);
        if (!ok) {
            for (size_t index = 0; index < reference_count; index++)
                free_reference_state(&references[index]);
            free(references);
            free(layout_refs);
            free(text_path);
            free(dit_path);
            free(vae_path);
            return 0;
        }
    } else {
        ok = h3_text_encode_bf16(text_path, "h3_shaders.metal", ids,
                                 token_count, NULL, NULL, &text, error,
                                 error_size);
        h3_tokenizer_free(tokenizer);
        if (!ok) {
            free(text_path);
            free(dit_path);
            free(vae_path);
            return 0;
        }
    }
    H3IMG_TIMING_MARK("text+vision encode");
    /* The text encoder directory path is no longer needed past this point
     * (it previously leaked on every path, including success). */
    free(text_path);
    text_path = NULL;
    h3_tokenizer_ids_free(ids);

    /* 3. Packed-sequence layout: text rows first, one latent frame, no
     * audio (validated against the reference in the M2 DiT parity). */
    /* frame_count 5 is the layout builder's structural minimum (the image
     * partition carries no audio, so it only feeds validation). */
    h3_layout_spec spec = {(int)text.tokens, 1, latent_h, latent_w, 0,
                           5, NULL, 0, layout_refs,
                           reference_count};
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    if (!h3_layout_build(&spec, &layout, error, error_size)) {
        h3_text_embedding_free(&text);
        if (reference_count) {
            for (size_t index = 0; index < reference_count; index++)
                free_reference_state(&references[index]);
            free(references);
            free(layout_refs);
        }
        free(dit_path);
        free(vae_path);
        return 0;
    }

    /* 4. The distilled sampler grid, consumed verbatim. */
    h3_sigma_schedule sigmas;
    if (!h3_schedule_build_explicit(H3IMG_SIGMAS, H3IMG_STEPS, &sigmas)) {
        h3_layout_free(&layout);
        h3_text_embedding_free(&text);
        if (reference_count) {
            for (size_t index = 0; index < reference_count; index++)
                free_reference_state(&references[index]);
            free(references);
            free(layout_refs);
        }
        free(dit_path);
        free(vae_path);
        snprintf(error, error_size, "cannot build the pdd sigma schedule");
        return 0;
    }

    if (progress) progress("dit", 0, 50, progress_opaque);
    size_t condition_elements = 0;
    float *condition_rows = NULL;
    if (reference_count) {
        size_t total = 0;
        for (size_t index = 0; index < reference_count; index++)
            total += references[index].row_count;
        condition_rows = malloc(total * 96 * sizeof(float));
        if (!condition_rows) {
            snprintf(error, error_size, "out of memory for condition rows");
            ok = 0;
        } else {
            size_t offset = 0;
            for (size_t index = 0; index < reference_count; index++) {
                memcpy(condition_rows + offset,
                       references[index].condition_rows,
                       references[index].row_count * 96 * sizeof(float));
                offset += references[index].row_count * 96;
            }
            condition_elements = total * 96;
        }
    }
    if (getenv("H3IMG_EDIT_TRACE")) {
        fprintf(stderr, "[layout] seq_len %zu\n", layout.seq_len);
        for (size_t s = 0; s < layout.segment_count; s++)
            fprintf(stderr, "[layout] seg %zu: kind %d rows %zu..%zu\n", s,
                    (int)layout.segments[s].kind, layout.segments[s].start,
                    layout.segments[s].stop);
        fprintf(stderr, "[layout] img_cond %zu img_target %zu\n",
                layout.img_cond_rows, layout.img_target_rows);
    }
    h3_dit *dit = NULL;
    if (ok) {
        dit = h3_dit_load_conditioned(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas, 50, 1, 0,
            0, 1.0f, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            condition_rows, condition_elements, NULL, 0,
            progress, progress_opaque, error, error_size);
    }
    if (reference_count) {
        for (size_t index = 0; index < reference_count; index++)
            free_reference_state(&references[index]);
        free(references);
        free(layout_refs);
        free(condition_rows);
    }
    if (getenv("H3IMG_DUMP_LATENTS")) {
        {
            FILE *dump = fopen("/tmp/h3img_layout.f32", "wb");
            if (dump) {
                fwrite(layout.positions, sizeof(h3_position),
                       layout.seq_len, dump);
                fclose(dump);
            }
        }
        FILE *dump = fopen("/tmp/h3img_text_embeds.f32", "wb");
        if (dump) {
            fwrite(text.values, sizeof(uint16_t),
                   (size_t)text.tokens * 5120, dump);
            fclose(dump);
        }
    }
    h3_layout_free(&layout);
    h3_text_embedding_free(&text);
    free(dit_path);
    H3IMG_TIMING_MARK("DiT load+AdaLN");
    if (!dit) {
        free(vae_path);
        return 0;
    }

    /* 5. Initial noise in the normalized latent space, or the injected
     * parity latents. */
    size_t latent_elements = (size_t)H3IMG_CHANNELS * latent_h * latent_w;
    float *latents = malloc(latent_elements * sizeof(float));
    float *velocity = malloc(latent_elements * sizeof(float));
    float *audio_dummy = malloc((size_t)32 * 2 * sizeof(float) * 8);
    if (!latents || !velocity || !audio_dummy) {
        free(latents); free(velocity); free(audio_dummy);
        h3_dit_free(dit);
        free(vae_path);
        snprintf(error, error_size, "out of memory for latents");
        return 0;
    }
    memset(audio_dummy, 0, (size_t)32 * 2 * sizeof(float) * 8);
    if (latents_path) {
        ok = h3img_load_latents(latents_path, latents, latent_elements, error,
                                error_size);
    } else {
        h3_rng rng;
        h3_rng_seed(&rng, seed);
        for (size_t index = 0; index < latent_elements; index++)
            latents[index] = h3_rng_normal(&rng);
        ok = 1;
    }
    if (!ok) {
        free(latents); free(velocity); free(audio_dummy);
        h3_dit_free(dit);
        free(vae_path);
        return 0;
    }

    /* 6. The six-step Euler/velocity loop, float32 throughout. */
    for (int step = 0; ok && step < H3IMG_STEPS; step++) {
        if (progress) progress("denoise", step, H3IMG_STEPS,
                               progress_opaque);
        ok = h3_dit_forward(dit, step, latents, audio_dummy, velocity,
                            audio_dummy, error, error_size);
        if (!ok) break;
        float delta = H3IMG_SIGMAS[step] - H3IMG_SIGMAS[step + 1];
        for (size_t index = 0; index < latent_elements; index++)
            latents[index] += delta * velocity[index];
    }
    h3_dit_free(dit);
    free(audio_dummy);
    free(velocity);
    H3IMG_TIMING_MARK("6-step denoise");
    if (!ok) {
        free(latents);
        free(vae_path);
        return 0;
    }

    if (getenv("H3IMG_DUMP_LATENTS")) {
        FILE *dump = fopen("/tmp/h3img_final_latents.f32", "wb");
        if (dump) {
            fwrite(latents, sizeof(float), latent_elements, dump);
            fclose(dump);
        }
    }
    /* 7. Denormalize channel-wise, then decode through the ViT VAE. */
    for (int c = 0; c < H3IMG_CHANNELS; c++)
        for (int index = 0; index < latent_h * latent_w; index++)
            latents[(size_t)c * latent_h * latent_w + index] =
                latents[(size_t)c * latent_h * latent_w + index] *
                    H3IMG_LATENTS_STD[c] + H3IMG_LATENTS_MEAN[c];
    if (progress) progress("vae", 0, 1, progress_opaque);
    size_t pixel_elements = (size_t)3 * canvas_h * canvas_w;
    float *pixels = malloc(pixel_elements * sizeof(float));
    if (!pixels) {
        free(latents);
        free(vae_path);
        snprintf(error, error_size, "out of memory for pixels");
        return 0;
    }
    ok = h3img_vae_decode(vae_path, latents, latent_w, latent_h, pixels,
                          error, error_size);
    free(latents);
    free(vae_path);
    H3IMG_TIMING_MARK("VAE decode");
    if (!ok) {
        free(pixels);
        return 0;
    }

    /* 8. PNG: quantize on the generation canvas (the released processor
     * resizes the 8-bit image), then bicubic-resize back to the requested
     * size when the canvas was scaled up. The VAE decodes into ImageNet-
     * normalized RGB — revert with pixel_mean/pixel_std first, exactly like
     * the released DecodeStep (a [-1,1] mapping here used to blow out every
     * highlight: normalized values above ~1 saturated at 255). The released
     * pipeline quantizes by truncation after the clamp. */
    if (progress) progress("png", 1, 1, progress_opaque);
    static const float pixel_mean[3] = {0.485f, 0.456f, 0.406f};
    static const float pixel_std[3] = {0.229f, 0.224f, 0.225f};
    uint8_t *canvas_rgb8 = malloc((size_t)3 * canvas_h * canvas_w);
    if (!canvas_rgb8) {
        free(pixels);
        snprintf(error, error_size, "out of memory for PNG");
        return 0;
    }
    for (int y = 0; y < canvas_h; y++)
        for (int x = 0; x < canvas_w; x++)
            for (int c = 0; c < 3; c++) {
                float value = pixels[(size_t)c * canvas_h * canvas_w +
                                     (size_t)y * canvas_w + x];
                float scaled = (value * pixel_std[c] + pixel_mean[c]) * 255.0f;
                if (scaled < 0.0f) scaled = 0.0f;
                if (scaled > 255.0f) scaled = 255.0f;
                canvas_rgb8[((size_t)y * canvas_w + x) * 3 + c] =
                    (uint8_t)(scaled + 0.5f);
            }
    free(pixels);
    uint8_t *rgb8 = canvas_rgb8;
    if (canvas_w != width || canvas_h != height) {
        rgb8 = malloc((size_t)3 * width * height);
        if (!rgb8) {
            free(canvas_rgb8);
            snprintf(error, error_size, "out of memory for PNG");
            return 0;
        }
        /* A failed resize leaves rgb8 uninitialized; encoding it would
         * write a garbage PNG and report success. */
        if (!h3img_imgproc_resize_bicubic_u8(canvas_rgb8, canvas_w, canvas_h,
                                             3, rgb8, width, height)) {
            free(rgb8);
            free(canvas_rgb8);
            snprintf(error, error_size, "cannot resize the canvas to %dx%d",
                     width, height);
            return 0;
        }
        free(canvas_rgb8);
    }
    uint8_t *rgba8 = malloc((size_t)width * height * 4);
    if (!rgba8) {
        free(rgb8);
        snprintf(error, error_size, "out of memory for PNG");
        return 0;
    }
    for (size_t index = 0; index < (size_t)width * height; index++) {
        rgba8[index * 4 + 0] = rgb8[index * 3 + 0];
        rgba8[index * 4 + 1] = rgb8[index * 3 + 1];
        rgba8[index * 4 + 2] = rgb8[index * 3 + 2];
        rgba8[index * 4 + 3] = 255;
    }
    free(rgb8);
    /* qwen-derived PNG writer returns 0 on success; this module's callers
     * expect the h3 true-on-success convention. */
    ok = h3img_png_write_rgba(output_png, (uint32_t)width, (uint32_t)height,
                              rgba8, error, error_size) == 0;
    free(rgba8);
    H3IMG_TIMING_MARK("PNG write+resize");
    return ok;
}
