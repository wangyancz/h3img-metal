#include "h3_safetensors.h"
#include "h3_text_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_h3img_text.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed % 10 == 0 || completed == total)
        fprintf(stderr, "native t2i Qwen: %d/%d layers\n", completed, total);
}

static void read_exact(const h3_st_header *fixture, const char *name,
                       h3_dtype dtype, void *output, size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements)
        die("fixture tensor is absent or malformed");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, output,
                         elements * h3_dtype_size(dtype),
                         error, sizeof(error))) die(error);
}

/* MiniMax-H3-Image text-to-image conditions on the Qwen3-VL hidden state
 * after decoder layer 50 of a verbatim prompt presentation (no chat template,
 * no special tokens, all-text modality). This test replays the fixture's
 * token ids through the native 50-layer tower and compares the conditioning
 * state (fixture x.layer_49, 0-based) against the official reference. */
int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/Volumes/Work/models/MiniMax-H3-Image";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3img_text_t2i.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *ids_tensor = h3_st_find(&fixture, "x.ids");
    const h3_st_tensor *want_tensor = h3_st_find(&fixture, "x.layer_49");
    if (!ids_tensor || ids_tensor->dtype != H3_DTYPE_I32 ||
        ids_tensor->ndim != 2 || ids_tensor->shape[0] != 1 ||
        !want_tensor || want_tensor->dtype != H3_DTYPE_BF16 ||
        want_tensor->ndim != 2 || want_tensor->shape[1] != 5120)
        die("fixture geometry is malformed");
    size_t tokens = (size_t)ids_tensor->shape[1];
    if (tokens != (size_t)want_tensor->shape[0])
        die("fixture ids and reference disagree on token count");

    uint32_t *ids = malloc(tokens * sizeof(*ids));
    uint16_t *want = malloc(tokens * 5120 * sizeof(*want));
    if (!ids || !want) die("out of memory loading fixture");
    read_exact(&fixture, "x.ids", H3_DTYPE_I32, ids, tokens);
    read_exact(&fixture, "x.layer_49", H3_DTYPE_BF16, want, tokens * 5120);

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/text_encoder", model_root);
    h3_text_embedding got;
    if (!h3_text_encode_bf16(weights, "h3_shaders.metal", ids, tokens,
                             progress, NULL, &got, error, sizeof(error)))
        die(error);
    if (got.tokens != tokens || got.width != 5120)
        die("native t2i presentation metadata mismatch");

    double maximum = 0.0, scale = 0.0, squares = 0.0, reference = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < tokens * 5120; index++) {
        double actual = bf16_to_f32(got.values[index]);
        double expected = bf16_to_f32(want[index]);
        if (!isfinite(actual) || !isfinite(expected)) {
            nonfinite++;
            continue;
        }
        double delta = actual - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        squares += delta * delta;
        reference += expected * expected;
    }
    double rel_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double rel_l2 = sqrt(squares / (reference > 1e-24 ? reference : 1e-24));
    printf("t2i text layer-49: max abs %.7g, rel-max %.7g, "
           "rel-L2 %.7g, nonfinite %zu\n",
           maximum, rel_max, rel_l2, nonfinite);
    /* Same convention as test_real_multimodal_text: 50-layer reduction-order
     * differences between the Metal tower and the PyTorch oracle live in a
     * representation-level band. */
    double bound = 0.15;
    if (nonfinite || rel_max >= bound || rel_l2 >= bound)
        die("t2i text parity bound exceeded");
    h3_text_embedding_free(&got);
    h3_st_free_header(&fixture);
    free(ids);
    free(want);
    printf("ok: native t2i text encoding matches the official layer-49 oracle\n");
    return 0;
}
