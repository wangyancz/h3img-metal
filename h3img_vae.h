#ifndef H3IMG_VAE_H
#define H3IMG_VAE_H

#include "h3_safetensors.h"
#include "h3_host.h"

#include <stddef.h>
#include <stdint.h>

/* MiniMax-H3-Image VAE: a causal Conv3D encoder (an image degenerates to a
 * single frame, so every k3 kernel only ever reads its last temporal tap and
 * the whole encoder is a pure 2D convolution stack) and a ViT decoder (24
 * fp32 transformer blocks over one token per latent pixel plus 4 register
 * tokens and one zero token, unpatchified through a 16x16 output head).
 *
 * Latents are normalized per channel with latents_mean / latents_std rather
 * than a scaling factor; this module works on the raw (unnormalized) latent
 * space, callers apply the normalization.
 *
 * M3 milestone implementation: fp32 host reference (Accelerate GEMMs) for
 * parity; GPU kernels follow in the performance phase. */

typedef struct {
    /* Encoder spatial sizes: the image is square in the parity path but the
     * implementation handles rectangles everywhere. */
    int width;
    int height;
    /* Latent grid (width/16, height/16). */
    int latent_width;
    int latent_height;
} h3img_vae_info;

/* Encode one RGB image (fp32, channel-major [3][h][w]) into the 24-channel
 * posterior mode. latent receives 24 * latent_h * latent_w floats. */
int h3img_vae_encode(const char *weight_directory, const float *image,
                     int width, int height, float *latent,
                     char *error, size_t error_size);

/* Encode with a posterior sample instead of the mode: sample = mean +
 * exp(logvar / 2) * eps, eps drawn from the caller's RNG. The edit recipe
 * rounds the sample to float16 precision. */
int h3img_vae_encode_sampled(const char *weight_directory, const float *image,
                             int width, int height, h3_rng *rng, float *latent,
                             char *error, size_t error_size);

/* Decode 24-channel raw latents (the latent * std + mean normalization is the
 * caller's job) into RGB pixels [3][h*16][w*16]. */
int h3img_vae_decode(const char *weight_directory, const float *latent,
                     int latent_width, int latent_height, float *pixels,
                     char *error, size_t error_size);

#endif
