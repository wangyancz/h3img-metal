#include "h3img_pipeline.h"
#include "h3img_png.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

/* End-to-end edit acceptance against the official reference stack. The
 * fixture (tests/gen_edit_ab.py) holds the shared draws — initial noise,
 * reference posterior latents, noising noise — plus the official text
 * embeds and final latents produced by the released edit semantics
 * (reference rows held at t = 0.999, every other row stepping with the
 * step's timestep).
 *
 * Acceptance mirrors test_real_h3img_e2e.c: the C engine replays the exact
 * same conditioning and noise, and the final latents are compared against
 * the fixture's `x.final_latents`. The measured rel-L2 is 0.13 — the same
 * bf16 intrinsic band as the t2i e2e (0.20); wiring bugs measure >= 0.9
 * (the pre-fix state of this A/B measured 1.13 from a fixture-side timestep
 * semantics bug, and a genuine C-side divergence cannot hide below the
 * chaos amplification of the chain). The decoded images are additionally
 * compared against the official reference PNG with the t2i semantic gates
 * (brightness histogram correlation, texture-energy Laplacian variance),
 * which the same-seed edit passes comfortably (2.9/255 mean, 0.9999 corr).
 *
 * The final latents are obtained through the pipeline's H3IMG_DUMP_LATENTS
 * diagnostics (the dump file is unlinked first so a stale run cannot pass
 * this test). */

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_h3img_edit.c: %s\n", message);
    exit(1);
}

static void read_tensor(const h3_st_header *fixture, const char *name,
                        h3_dtype dtype, void *values, size_t bytes,
                        size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements)
        die("fixture tensor is absent or malformed");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, values, bytes,
                         error, sizeof(error))) die(error);
}

static double pixel_luma(const uint8_t *rgba, int width, int x, int y) {
    const uint8_t *pixel = rgba + ((size_t)y * width + x) * 4;
    return (pixel[0] + pixel[1] + pixel[2]) / 3.0;
}

/* Luminance Laplacian variance over the interior (texture energy). */
static double laplacian_variance(const uint8_t *rgba, int width, int height) {
    double squares = 0.0;
    double count = 0.0;
    for (int y = 1; y < height - 1; y++)
        for (int x = 1; x < width - 1; x++) {
            double center = pixel_luma(rgba, width, x, y);
            double neighbours =
                (pixel_luma(rgba, width, x, y - 1) +
                 pixel_luma(rgba, width, x, y + 1) +
                 pixel_luma(rgba, width, x - 1, y) +
                 pixel_luma(rgba, width, x + 1, y)) / 4.0;
            double laplacian = center - neighbours;
            squares += laplacian * laplacian;
            count += 1.0;
        }
    return squares / (count > 0 ? count : 1.0);
}

static double brightness_histogram_correlation(const uint8_t *a,
                                               const uint8_t *b,
                                               size_t pixels) {
    double ha[16] = {0}, hb[16] = {0};
    for (size_t index = 0; index < pixels; index++) {
        double la = (a[index * 4] + a[index * 4 + 1] + a[index * 4 + 2]) / 3.0;
        double lb = (b[index * 4] + b[index * 4 + 1] + b[index * 4 + 2]) / 3.0;
        ha[(int)(la / 16.0)] += 1.0;
        hb[(int)(lb / 16.0)] += 1.0;
    }
    double sum_a = 0, sum_b = 0;
    for (int bin = 0; bin < 16; bin++) {
        sum_a += ha[bin];
        sum_b += hb[bin];
    }
    double mean_a = 0, mean_b = 0;
    for (int bin = 0; bin < 16; bin++) {
        ha[bin] /= sum_a;
        hb[bin] /= sum_b;
        mean_a += ha[bin];
        mean_b += hb[bin];
    }
    mean_a /= 16;
    mean_b /= 16;
    double covariance = 0, var_a = 0, var_b = 0;
    for (int bin = 0; bin < 16; bin++) {
        double da = ha[bin] - mean_a;
        double db = hb[bin] - mean_b;
        covariance += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    return covariance / sqrt(var_a * var_b);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/Volumes/Work/models/MiniMax-H3-Image";
    const char *fixture_dir = argc > 2 ? argv[2] : "misc/fixtures";
    char error[512];
    error[0] = '?';
    char latents[512], reference[512], reference_out[512], output[256];
    snprintf(latents, sizeof(latents), "%s/h3img_edit_ab_fixtures.safetensors",
             fixture_dir);
    snprintf(reference, sizeof(reference), "%s/h3img_edit_ab_reference.png",
             fixture_dir);
    snprintf(reference_out, sizeof(reference_out),
             "%s/h3img_edit_ab_reference_out.png", fixture_dir);
    snprintf(output, sizeof(output), "/tmp/h3img_edit_output.png");
    setenv("H3IMG_DUMP_LATENTS", "1", 1);
    unlink("/tmp/h3img_final_latents.f32");

    const char *prompt =
        "Turn the apple into a green lime while keeping the composition";
    const char *references[1] = {reference};
    int steps = 0;
    if (!h3img_generate_ex(model_root, prompt, references, 1, 256, 256,
                           20260718, latents, output, &steps, NULL, NULL,
                           error, sizeof(error))) {
        fprintf(stderr, "generate failed: len=%zu bytes:", strlen(error));
        for (size_t i = 0; error[i] && i < 64; i++)
            fputc(isprint((unsigned char)error[i]) ? error[i] : '.', stderr);
        fputc('\n', stderr);
        die("h3img_generate_ex returned failure");
    }
    if (steps != 6) die("unexpected step count");

    /* Final latents vs the fixture (24 x 16 x 16 fp32). */
    h3_st_header fixture;
    if (!h3_st_read_header(latents, &fixture, error, sizeof(error)))
        die("cannot open the edit fixture");
    size_t latent_elements = 24u * 16u * 16u;
    float *want = malloc(latent_elements * sizeof(*want));
    float *got = malloc(latent_elements * sizeof(*got));
    if (!want || !got) die("out of memory for latents");
    read_tensor(&fixture, "x.final_latents", H3_DTYPE_F32, want,
                latent_elements * sizeof(*want), latent_elements);
    FILE *dump = fopen("/tmp/h3img_final_latents.f32", "rb");
    if (!dump) die("the pipeline did not dump the final latents");
    if (fread(got, sizeof(*got), latent_elements, dump) != latent_elements)
        die("final latents dump is truncated");
    fclose(dump);
    double total = 0.0, reference_norm = 0.0;
    for (size_t index = 0; index < latent_elements; index++) {
        double delta = (double)got[index] - want[index];
        total += delta * delta;
        reference_norm += (double)want[index] * want[index];
    }
    double rel = sqrt(total) / sqrt(reference_norm);
    free(want);
    free(got);
    h3_st_free_header(&fixture);
    printf("edit e2e: final latents rel-L2 %.4f\n", rel);
    if (rel >= 0.35)
        die("final latents gate exceeded (wiring bugs measure >= 0.9)");

    /* Decoded image vs the official reference PNG. */
    uint32_t cw = 0, ch = 0, rw = 0, rh = 0;
    uint8_t *c_rgba = NULL, *ref_rgba = NULL;
    if (h3img_png_read_rgba(output, &cw, &ch, &c_rgba, error, sizeof(error)))
        error[0] = 0;
    if (!c_rgba) die("cannot read the generated PNG");
    if (h3img_png_read_rgba(reference_out, &rw, &rh, &ref_rgba, error,
                            sizeof(error)))
        error[0] = 0;
    if (!ref_rgba) die("cannot read the reference PNG");
    if (cw != rw || ch != rh || cw != 256 || ch != 256)
        die("generated/reference size mismatch");

    size_t pixels = (size_t)cw * ch;
    double absolute = 0.0;
    for (size_t index = 0; index < pixels; index++) {
        for (int c = 0; c < 3; c++) {
            double delta = (double)c_rgba[index * 4 + c] -
                           ref_rgba[index * 4 + c];
            absolute += fabs(delta);
        }
    }
    absolute /= (double)pixels * 3.0;
    double correlation = brightness_histogram_correlation(c_rgba, ref_rgba,
                                                          pixels);
    double lap_c = laplacian_variance(c_rgba, (int)cw, (int)ch);
    double lap_r = laplacian_variance(ref_rgba, (int)rw, (int)rh);
    double lap_ratio = lap_c / (lap_r > 1e-9 ? lap_r : 1e-9);
    printf("edit image: mean abs diff %.2f/255, hist corr %.4f, "
           "LapVar %.0f vs %.0f (ratio %.3f)\n",
           absolute, correlation, lap_c, lap_r, lap_ratio);
    if (absolute >= 16.0) die("pixel difference gate exceeded");
    if (correlation < 0.95) die("composition gate exceeded");
    if (lap_ratio < 0.5 || lap_ratio > 2.0)
        die("texture-energy gate exceeded");

    free(c_rgba);
    free(ref_rgba);
    printf("ok: native edit matches the official stack semantically\n");
    return 0;
}
