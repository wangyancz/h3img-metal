#include "h3img_pipeline.h"
#include "h3img_serve.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR --t2i -p PROMPT -o OUT.png --width W --height H [options]\n"
        "       %s serve -d MODEL_DIR       # JSONL worker\n"
        "       %s repl -d MODEL_DIR        # interactive session\n\n"
        "Native MiniMax-H3-Image inference for Apple Silicon (C11 / Metal).\n\n"
        "Options:\n"
        "  -d, --model-dir PATH   MiniMax-H3-Image local directory\n"
        "  -p, --prompt TEXT      Image prompt\n"
        "  -o, --output PATH      Output PNG (default: h3img-output.png)\n"
        "      --width N          Output width, multiple of 32 in [32, 8192]\n"
        "      --height N         Output height, multiple of 32 in [32, 8192]\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --ref-image PATH   Edit reference image, repeatable (max 8)\n"
        "      --latents FILE     Inject fp32 initial latents (A/B harness)\n"
        "      --steps N          Accepted and ignored (distilled 6-step model)\n"
        "  -h, --help             Show this help\n",
        program, program, program);
}

static int parse_int(const char *value, const char *label) {
    /* strtol("") returns 0 with end == value and errno == 0; reject empty
     * input explicitly instead of treating it as a valid zero. Reject '+' for
     * the same reason parse_u64 does (consistent CLI numeric syntax). */
    if (!value || !*value || value[0] == '+') {
        fprintf(stderr, "h3img: invalid %s: %s\n", label, value ? value : "");
        exit(2);
    }
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3img: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)parsed;
}

static uint64_t parse_u64(const char *value, const char *label) {
    /* strtoull accepts a leading '-' and wraps it (e.g. "-1" -> ULLONG_MAX)
     * with errno == 0; reject signs so CLI and serve/repl agree. An empty
     * string would otherwise silently parse as seed 0. */
    if (!value || !*value || value[0] == '-' || value[0] == '+') {
        fprintf(stderr, "h3img: invalid %s: %s\n", label, value ? value : "");
        exit(2);
    }
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || end == value || *end) {
        fprintf(stderr, "h3img: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
} cli_state;

static void cli_progress(const char *phase, int completed, int total,
                         void *opaque) {
    cli_state *state = opaque;
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return;
    if (strcmp(state->phase, phase)) {
        if (state->active) fputc('\n', stderr);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    state->completed = completed;
    state->total = total;
    state->active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!state->active) fputc('\n', stderr);
    fflush(stderr);
}

int main(int argc, char **argv) {
    /* serve/repl are dispatched before the one-shot getopt loop: they take
     * only -d and share no flags with image generation. */
    if (argc > 1 && (!strcmp(argv[1], "serve") || !strcmp(argv[1], "repl"))) {
        const char *dir = NULL;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--model-dir")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "h3img: %s requires a value\n", argv[i]);
                    return 2;
                }
                if (dir) {
                    fprintf(stderr, "h3img: -d given twice for %s\n", argv[1]);
                    return 2;
                }
                dir = argv[++i];
            } else {
                fprintf(stderr, "h3img: unknown option for %s: %s\n", argv[1],
                        argv[i]);
                return 2;
            }
        }
        if (!dir) {
            fprintf(stderr, "h3img: %s needs -d MODEL_DIR\n", argv[1]);
            return 2;
        }
        /* Resolve before h3img_ensure_shader_dir possibly chdirs to the
         * engine directory — a relative model dir must keep its original
         * meaning. */
        char *absolute = realpath(dir, NULL);
        if (!absolute) {
            fprintf(stderr, "h3img: cannot resolve model directory %s: %s\n",
                    dir, strerror(errno));
            return 2;
        }
        int status = !strcmp(argv[1], "serve") ? h3img_cmd_serve(absolute)
                                               : h3img_cmd_repl(absolute);
        free(absolute);
        return status;
    }

    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_SEED, OPT_REF_IMAGE,
           OPT_LATENTS, OPT_STEPS };
    static const struct option options[] = {
        {"model-dir", required_argument, NULL, 'd'},
        {"prompt", required_argument, NULL, 'p'},
        {"output", required_argument, NULL, 'o'},
        {"width", required_argument, NULL, OPT_WIDTH},
        {"height", required_argument, NULL, OPT_HEIGHT},
        {"seed", required_argument, NULL, OPT_SEED},
        {"ref-image", required_argument, NULL, OPT_REF_IMAGE},
        {"latents", required_argument, NULL, OPT_LATENTS},
        {"steps", required_argument, NULL, OPT_STEPS},
        {"t2i", no_argument, NULL, 't'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = "h3img-output.png";
    int width = 0, height = 0;
    uint64_t seed = 42;
    const char *latents_path = NULL;
    const char *ref_images[8] = {0};
    size_t ref_count = 0;
    cli_state cli = {{0}, 0, -1, -1};
    int option;
    while ((option = getopt_long(argc, argv, "d:p:o:th", options, NULL)) != -1) {
        switch (option) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 't': break; /* accepted for backward compatibility */
            case OPT_WIDTH: width = parse_int(optarg, "width"); break;
            case OPT_HEIGHT: height = parse_int(optarg, "height"); break;
            case OPT_SEED: seed = parse_u64(optarg, "seed"); break;
            case OPT_REF_IMAGE:
                if (ref_count >= 8) {
                    fprintf(stderr, "h3img: at most 8 reference images\n");
                    return 2;
                }
                ref_images[ref_count++] = optarg;
                break;
            case OPT_LATENTS: latents_path = optarg; break;
            case OPT_STEPS: (void)parse_int(optarg, "steps"); break;
            case 'h': usage(argv[0]); return 0;
            default: usage(argv[0]); return 2;
        }
    }
    if (!model_dir || !prompt) {
        usage(argv[0]);
        return 2;
    }
    if (!width || !height) {
        fprintf(stderr, "h3img: --width and --height are required\n");
        return 2;
    }

    char error[512] = {0};
    char *absolute = realpath(model_dir, NULL);
    if (!absolute) {
        fprintf(stderr, "h3img: cannot resolve model directory %s: %s\n",
                model_dir, strerror(errno));
        return 1;
    }
    if (!h3img_ensure_shader_dir(error, sizeof(error)) ||
        h3img_engine_lock(error, sizeof(error)) < 0) {
        fprintf(stderr, "h3img: %s\n", error);
        free(absolute);
        return 1;
    }
    int steps = 0;
    int ok = h3img_generate_ex(absolute, prompt,
                               (const char *const *)ref_images, ref_count,
                               width, height, seed, latents_path, output,
                               &steps, cli_progress, &cli, error,
                               sizeof(error));
    free(absolute);
    if (!ok) {
        fprintf(stderr, "h3img: %s\n", error);
        return 1;
    }
    fprintf(stderr, "h3img: wrote %s (%d steps)\n", output, steps);
    return 0;
}
