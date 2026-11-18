#include "h3img_vae.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MiniMax-H3-Image VAE parity: encode the fixture image and compare every
 * boundary the fixture captured, then decode and compare the pixels. The
 * reference is the official AutoencoderKLMiniMaxH3Image run in fp32. */

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_h3img_vae.c: %s\n", message);
    exit(1);
}

static void read_exact(const h3_st_header *fixture, const char *name,
                       void *output, size_t bytes) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) * h3_dtype_size(H3_DTYPE_F32) != bytes)
        die("fixture tensor is absent or malformed");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, output, bytes, error, sizeof(error)))
        die(error);
}

static double compare(const char *label, const float *got, const float *want,
                      size_t count, double bound) {
    double maximum = 0.0, scale = 0.0, squares = 0.0, reference = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(want[index]) > scale) scale = fabs(want[index]);
        squares += delta * delta;
        reference += (double)want[index] * want[index];
    }
    double rel_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double rel_l2 = sqrt(squares / (reference > 1e-24 ? reference : 1e-24));
    printf("%-14s rel-max %.6g abs %.6g rel-L2 %.6g\n", label, rel_max,
           maximum, rel_l2);
    if (rel_max >= bound || rel_l2 >= bound)
        die("VAE parity bound exceeded");
    return rel_l2;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/Volumes/Work/models/MiniMax-H3-Image";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3img_vae_t2i.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *image_tensor = h3_st_find(&fixture, "x.image");
    const h3_st_tensor *latent_tensor = h3_st_find(&fixture, "want.latent");
    const h3_st_tensor *pixels_tensor = h3_st_find(&fixture, "want.pixels");
    if (!image_tensor || image_tensor->ndim != 3 ||
        image_tensor->shape[0] != 3 || image_tensor->shape[1] != 64 ||
        image_tensor->shape[2] != 64 ||
        !latent_tensor || latent_tensor->ndim != 3 ||
        latent_tensor->shape[0] != 24 || latent_tensor->shape[1] != 4 ||
        latent_tensor->shape[2] != 4 ||
        !pixels_tensor || pixels_tensor->ndim != 3 ||
        pixels_tensor->shape[0] != 3)
        die("fixture geometry is malformed");

    enum { WIDTH = 64, HEIGHT = 64, LATENT_W = 4, LATENT_H = 4 };
    float *image = malloc(3 * HEIGHT * WIDTH * sizeof(float));
    float *latent_want = malloc(24 * LATENT_H * LATENT_W * sizeof(float));
    float *z_denorm = malloc(24 * LATENT_H * LATENT_W * sizeof(float));
    float *pixels_want = malloc(3 * HEIGHT * WIDTH * sizeof(float));
    float *latent_got = malloc(24 * LATENT_H * LATENT_W * sizeof(float));
    float *pixels_got = malloc(3 * HEIGHT * WIDTH * sizeof(float));
    float *stage = malloc(3 * HEIGHT * WIDTH * sizeof(float));
    if (!image || !latent_want || !pixels_want || !latent_got || !pixels_got ||
        !stage)
        die("out of memory loading fixture");
    read_exact(&fixture, "x.image", image,
               3 * HEIGHT * WIDTH * sizeof(float));
    read_exact(&fixture, "want.latent", latent_want,
               24 * LATENT_H * LATENT_W * sizeof(float));
    read_exact(&fixture, "want.z_denorm", z_denorm,
               24 * LATENT_H * LATENT_W * sizeof(float));
    read_exact(&fixture, "want.pixels", pixels_want,
               3 * HEIGHT * WIDTH * sizeof(float));

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/vae_diffusers", model_root);
    if (!h3img_vae_encode(weights, image, WIDTH, HEIGHT, latent_got, error,
                          sizeof(error)))
        die(error);
    compare("latent", latent_got, latent_want,
            24 * LATENT_H * LATENT_W, 5e-4);
    /* The decoder consumes denormalized latents: raw encode output must be
     * scaled by latents_std and shifted by latents_mean first (the fixture
     * captures that as want.z_denorm). */
    if (!h3img_vae_decode(weights, z_denorm, LATENT_W, LATENT_H, pixels_got,
                          error, sizeof(error)))
        die(error);
    compare("pixels", pixels_got, pixels_want, 3 * HEIGHT * WIDTH, 5e-4);

    h3_st_free_header(&fixture);
    free(image);
    free(latent_want);
    free(z_denorm);
    free(pixels_want);
    free(latent_got);
    free(pixels_got);
    free(stage);
    printf("ok: native VAE matches the official fp32 oracle\n");
    return 0;
}
