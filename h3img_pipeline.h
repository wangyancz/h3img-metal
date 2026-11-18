#ifndef H3IMG_PIPELINE_H
#define H3IMG_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

/* The MiniMax-H3-Image text-to-image pipeline: verbatim-prompt Qwen3-VL text
 * encoding (first 50 layers, pre-norm state), the pdd-distilled six-step
 * Euler/velocity sampler over the packed DiT, per-channel latent
 * denormalization, and the ViT VAE decode. No CFG, no negative prompt — the
 * checkpoint is guidance-distilled. Latents live in the normalized space
 * (the per-channel mean/std statistics are only applied for the decode). */

typedef void (*h3img_progress)(const char *phase, int completed, int total,
                               void *opaque);

/* Generate one image.
 *   model_dir      the MiniMax-H3-Image checkpoint root
 *   prompt         the verbatim prompt (no template, no special tokens)
 *   width/height   requested pixel size (multiples of 32; clamped to the
 *                  trained canvas range like the reference)
 *   seed           seeds the engine's own noise RNG (the reference cannot be
 *                  matched RNG-for-RNG; pass latents_path for exact A/B)
 *   latents_path   optional safetensors with `x.latents` [24, H/16, W/16]
 *                  fp32 — when present it replaces the drawn noise (used by
 *                  the end-to-end parity harness to share noise with the
 *                  reference stack)
 *   output_png     destination PNG path (RGB)
 *   total_steps    receives the schedule length (6 for this checkpoint) */
/* Edit variant: reference_paths are PNG files used as conditioning images
 * ("Picture i: " vision blocks; the prompt may refer to them as <Picture i>). */
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
                   char *error, size_t error_size);

#endif
