#ifndef H3IMG_GPU_EXT_H
#define H3IMG_GPU_EXT_H

/* GPU wrappers backing the H3-Image VAE encoder (h3img_vae.c). The kernels
 * live in h3_shaders.metal under the h3img_vae_* names; see the shader tail
 * for the layout contracts (channels-last activations, reflect ring, 9-tap
 * GEMM). */

#include "h3_gpu.h"

/* One 3x3 convolution as nine accumulate-mode TensorOps GEMMs. `input` and
 * `output` are channels-last [height*width, C]; `w9` is the repacked
 * [9, out_pad64, in] weight; `pad_arena` receives the reflect ring (must
 * hold (m_span + 2*(width+2) + 2) * in_channels floats). Dispatches only —
 * the caller owns the begin/submit timing. */
int h3img_gpu_vae_conv_mm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *w9,
                              const h3_gpu_tensor *bias,
                              h3_gpu_tensor *pad_arena, uint32_t height,
                              uint32_t width, uint32_t in_channels,
                              uint32_t out_channels);

/* Repack a tap-sliced [out][in][9] conv weight into the [9, out_pad64, in]
 * GEMM layout on the device. Dispatches only. */
int h3img_gpu_vae_conv_repack_f32(h3_gpu *gpu, h3_gpu_tensor *w9,
                                  const h3_gpu_tensor *w,
                                  uint32_t in_channels, uint32_t out_channels,
                                  uint32_t out_pad);

/* Reflect-padded stride-2 3x3 convolution over an after-only reflect ring
 * [hp, wp, C] (hp = height+1, wp = width+1): output pixel (oy, ox) reads
 * the ring at (2*oy + ky, 2*ox + kx). Dispatches only. */
int h3img_gpu_vae_downsample_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias, uint32_t oh,
                                 uint32_t ow, uint32_t hp, uint32_t wp,
                                 uint32_t in_channels, uint32_t out_channels);

/* ---- H3-Image VAE decoder ViT (all dispatch-only) ---- */

/* Per-head QK RMS norm (no affine) + rotate-half RoPE on the first `rotary`
 * dims, in place on query and key ([tokens, heads*head_dim] rows). */
int h3img_gpu_vit_qkrms_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 const h3_gpu_tensor *cos_table,
                                 const h3_gpu_tensor *sin_table,
                                 uint32_t tokens, uint32_t heads,
                                 uint32_t head_dim, uint32_t rotary);

/* Single-kernel SDPA (one threadgroup per (head, row)); supports up to 4608
 * tokens (threadgroup score staging). */
int h3img_gpu_vit_attention_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *query,
                                const h3_gpu_tensor *key,
                                const h3_gpu_tensor *value, uint32_t tokens,
                                uint32_t heads, uint32_t head_dim);

/* The VAE swiglu gates the second half: silu(fused[width + d]) * fused[d]. */
int h3img_gpu_swiglu_swap_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *fused, uint32_t rows,
                              uint32_t width);

#endif
