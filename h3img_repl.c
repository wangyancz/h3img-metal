#include "h3img_serve.h"

#include "h3img_pipeline.h"
#include "linenoise.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3IMG_REPL_MAX_IMAGES 8

typedef struct {
    char phase[64];
    int completed;
    int total;
    int active;
} repl_cli;

static void repl_progress(const char *phase, int completed, int total,
                          void *opaque) {
    repl_cli *cli = opaque;
    if (!strcmp(cli->phase, phase) && cli->completed == completed &&
        cli->total == total) return;
    if (strcmp(cli->phase, phase)) {
        if (cli->active) fputc('\n', stderr);
        snprintf(cli->phase, sizeof(cli->phase), "%s", phase);
    }
    cli->completed = completed;
    cli->total = total;
    cli->active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!cli->active) fputc('\n', stderr);
    fflush(stderr);
}

typedef struct {
    int width, height;
    uint64_t seed;
    char out_prefix[256];
    char images[H3IMG_REPL_MAX_IMAGES][1024];
    size_t n_images;
    unsigned long counter;
} repl_state;

static void repl_help(void) {
    printf("  <prompt>      generate one image (edit when references set)\n");
    printf("  /width N      output width, multiple of 32 (default 1024)\n");
    printf("  /height N     output height, multiple of 32 (default 1024)\n");
    printf("  /seed N       next seed (auto-increments after every image)\n");
    printf("  /out PREFIX   output file prefix (default \"out\")\n");
    printf("  /image PATH   add an edit reference (max %d; the prompt may "
           "refer to <Picture i>)\n", H3IMG_REPL_MAX_IMAGES);
    printf("  /image clear  remove all references (bare /image also "
           "clears)\n");
    printf("  /refs         list the current references\n");
    printf("  /show         show current settings\n");
    printf("  /help         this list\n");
    printf("  /quit         leave the session\n");
}

static void repl_show(const repl_state *st) {
    printf("  size %dx%d  seed %llu  out %s-<n>.png  refs %zu\n",
           st->width, st->height, (unsigned long long)st->seed,
           st->out_prefix, st->n_images);
    for (size_t i = 0; i < st->n_images; i++)
        printf("    Picture %zu: %s\n", i + 1, st->images[i]);
}

static int repl_parse_u32(const char *text, uint32_t *out) {
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (!text[0] || !end || *end || parsed > 4294967295ull) return 0;
    *out = (uint32_t)parsed;
    return 1;
}

static int repl_parse_u64(const char *text, uint64_t *out) {
    /* Reject signs: strtoull("-1") silently wraps to ULLONG_MAX. */
    if (text[0] == '-' || text[0] == '+') return 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (!text[0] || !end || *end) return 0;
    *out = (uint64_t)parsed;
    return 1;
}

static void repl_generate(repl_state *st, const char *dir,
                          const char *prompt, const char *refs[],
                          size_t n_refs) {
    char error[512] = {0};
    char output[1400];
    snprintf(output, sizeof(output), "%s-%lu.png", st->out_prefix,
             st->counter);
    int steps = 0;
    repl_cli cli = {{0}, -1, -1, 0};
    int ok = h3img_generate_ex(dir, prompt, refs, n_refs, st->width,
                               st->height, st->seed, NULL, output, &steps,
                               repl_progress, &cli, error, sizeof(error));
    if (cli.active) fputc('\n', stderr);
    if (!ok) {
        fprintf(stderr, "h3img: %s\n", error);
        return;
    }
    printf("wrote %s (%d steps, seed %llu)\n", output, steps,
           (unsigned long long)st->seed);
    st->counter++;
    st->seed++;
}

int h3img_cmd_repl(const char *dir) {
    char error[512] = {0};
    if (!h3img_ensure_shader_dir(error, sizeof(error))) {
        fprintf(stderr, "h3img: %s\n", error);
        return 1;
    }
    if (h3img_engine_lock(error, sizeof(error)) < 0) {
        fprintf(stderr, "h3img: %s\n", error);
        return 1;
    }
    if (!h3img_engine_ready(dir, error, sizeof(error))) {
        fprintf(stderr, "h3img: %s\n", error);
        return 1;
    }
    repl_state st = {1024, 1024, 42, "out", {{0}}, 0, 0};
    fprintf(stderr, "h3img REPL — type a prompt, /help for commands\n");
    linenoiseHistorySetMaxLen(500);
    linenoiseSetMultiLine(1);
    char *line;
    while ((line = linenoise("h3img> ")) != NULL) {
        char *input = line;
        while (*input == ' ' || *input == '\t') input++;
        size_t len = strlen(input);
        while (len && (input[len - 1] == ' ' || input[len - 1] == '\t'))
            input[--len] = 0;
        if (!len) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(input);
        if (input[0] != '/') {
            const char *refs[H3IMG_REPL_MAX_IMAGES];
            for (size_t i = 0; i < st.n_images; i++) refs[i] = st.images[i];
            repl_generate(&st, dir, input, refs, st.n_images);
            linenoiseFree(line);
            continue;
        }
        char *arg = strpbrk(input, " \t");
        if (arg) {
            *arg = 0;
            arg++;
            while (*arg == ' ' || *arg == '\t') arg++;
        }
        if (!strcmp(input, "/quit") || !strcmp(input, "/q")) {
            linenoiseFree(line);
            break;
        } else if (!strcmp(input, "/help")) {
            repl_help();
        } else if (!strcmp(input, "/show")) {
            repl_show(&st);
        } else if (!strcmp(input, "/refs")) {
            if (!st.n_images) printf("  no references (t2i mode)\n");
            for (size_t i = 0; i < st.n_images; i++)
                printf("  Picture %zu: %s\n", i + 1, st.images[i]);
        } else if (!strcmp(input, "/width") || !strcmp(input, "/height")) {
            uint32_t value = 0;
            if (!arg || !repl_parse_u32(arg, &value) || value < 32 ||
                value > 8192 || value % 32) {
                printf("  %s needs a multiple of 32 in [32, 8192]\n", input);
            } else if (!strcmp(input, "/width")) {
                st.width = (int)value;
            } else {
                st.height = (int)value;
            }
        } else if (!strcmp(input, "/seed")) {
            uint64_t value = 0;
            if (!arg || !repl_parse_u64(arg, &value)) {
                printf("  /seed needs a non-negative integer\n");
            } else {
                st.seed = value;
                printf("  next seed %llu\n", (unsigned long long)value);
            }
        } else if (!strcmp(input, "/out")) {
            if (!arg || !arg[0] || strlen(arg) > 200 ||
                strchr(arg, '/')) {
                printf("  /out needs a short file-name prefix (no '/')\n");
            } else {
                snprintf(st.out_prefix, sizeof(st.out_prefix), "%s", arg);
            }
        } else if (!strcmp(input, "/image")) {
            if (!arg || !arg[0] || !strcmp(arg, "clear")) {
                st.n_images = 0;
                printf("  references cleared (t2i mode)\n");
            } else if (st.n_images >= H3IMG_REPL_MAX_IMAGES) {
                printf("  at most %d references\n", H3IMG_REPL_MAX_IMAGES);
            } else if (strlen(arg) >= sizeof(st.images[0])) {
                printf("  path too long\n");
            } else {
                snprintf(st.images[st.n_images], sizeof(st.images[0]), "%s",
                         arg);
                st.n_images++;
                printf("  Picture %zu: %s\n", st.n_images, arg);
            }
        } else {
            printf("  unknown command %s — /help lists commands\n", input);
        }
        linenoiseFree(line);
    }
    fprintf(stderr, "h3img: bye\n");
    return 0;
}
