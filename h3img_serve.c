#include "h3img_serve.h"

#include "h3img_json.h"
#include "h3img_pipeline.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define H3IMG_SERVE_MAX_IMAGES 8
#define H3IMG_SERVE_LINE_MAX 65538

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static pthread_mutex_t g_stdout_mu = PTHREAD_MUTEX_INITIALIZER;

/* Every protocol line fits comfortably in 4096 bytes: paths are validated
 * to 512 bytes and the JSON escape can inflate each byte to 6 (control
 * bytes → \u00xx), so the worst-case done line is ~3.1 KB. */
static void emit(const char *fmt, ...) {
    char line[4096];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    if (n < 0) return;
    if ((size_t)n >= sizeof(line)) n = (int)sizeof(line) - 1;
    pthread_mutex_lock(&g_stdout_mu);
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    pthread_mutex_unlock(&g_stdout_mu);
}

static void emit_state(const char *state, double load_seconds) {
    char escaped[128];
    h3img_json_escape(state, escaped, sizeof(escaped));
    if (load_seconds >= 0.0)
        emit("{\"type\":\"state\",\"state\":\"%s\",\"loadSeconds\":%.1f}",
             escaped, load_seconds);
    else
        emit("{\"type\":\"state\",\"state\":\"%s\"}", escaped);
}

static void emit_error(const char *id, const char *message) {
    char rid[488], msg[3076];
    h3img_json_escape(id ? id : "", rid, sizeof(rid));
    h3img_json_escape(message && message[0] ? message : "generation failed",
                      msg, sizeof(msg));
    emit("{\"type\":\"error\",\"id\":\"%s\",\"message\":\"%s\"}", rid, msg);
}

static void emit_crashed(const char *message) {
    char msg[3076];
    h3img_json_escape(message, msg, sizeof(msg));
    emit("{\"type\":\"state\",\"state\":\"crashed\",\"message\":\"%s\"}",
         msg);
}

/* ------------------------------------------------------------------ */
/* Progress mapping                                                    */
/* ------------------------------------------------------------------ */

static struct {
    pthread_mutex_t mu;
    char rid[80];
    char stage[16];
    double percent;
    double last_emit;
} g_serve = {PTHREAD_MUTEX_INITIALIZER, {0}, {0}, 0.0, 0.0};

/* Total-progress spans per stage, calibrated on M5 Max request profiles
 * (encode = text + AdaLN + lazy weight load, sample = the six Euler
 * evaluations, decode = the tiled ViT VAE). */
static void serve_span(const char *stage, double *lo, double *hi) {
    if (!strcmp(stage, "encode")) { *lo = 0.0;  *hi = 0.30; return; }
    if (!strcmp(stage, "sample")) { *lo = 0.30; *hi = 0.65; return; }
    if (!strcmp(stage, "decode")) { *lo = 0.65; *hi = 0.97; return; }
    if (!strcmp(stage, "save"))   { *lo = 0.97; *hi = 1.0;  return; }
    *lo = 0.0;
    *hi = 1.0;
}

/* Ratio within the encode stage. Phase names come from h3img_generate_ex
 * and the forwarded h3_dit load reports. */
static double encode_ratio(const char *phase, int completed, int total) {
    double frac = total > 0 ? (double)completed / (double)total : 0.0;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    if (!strcmp(phase, "tokenize")) return 0.03;
    if (!strcmp(phase, "text")) return 0.03 + 0.36 * frac;
    if (!strcmp(phase, "vision")) return 0.39 + 0.11 * frac;
    if (!strcmp(phase, "refine text")) return 0.50 + 0.05 * frac;
    if (!strcmp(phase, "precompute AdaLN")) return 0.55 + 0.30 * frac;
    if (!strcmp(phase, "load transformer core")) return 0.85 + 0.15 * frac;
    /* The outer "dit" marker (0/50) and any unknown phase. */
    return 0.50;
}

static void serve_progress(const char *phase, int completed, int total,
                           void *opaque) {
    (void)opaque;
    const char *stage;
    double ratio;
    if (!strcmp(phase, "denoise")) {
        stage = "sample";
        ratio = total > 0 ? (double)completed / (double)total : 0.0;
    } else if (!strcmp(phase, "vae")) {
        stage = "decode";
        ratio = 1.0;
    } else if (!strcmp(phase, "png")) {
        stage = "save";
        ratio = 1.0;
    } else {
        stage = "encode";
        ratio = encode_ratio(phase, completed, total);
    }
    double lo, hi;
    serve_span(stage, &lo, &hi);
    double percent = lo + (hi - lo) * ratio;

    double now = now_seconds();
    pthread_mutex_lock(&g_serve.mu);
    int force = strcmp(g_serve.stage, stage) != 0;
    snprintf(g_serve.stage, sizeof(g_serve.stage), "%s", stage);
    g_serve.percent = percent;
    if (!force && now - g_serve.last_emit < 0.25) {
        pthread_mutex_unlock(&g_serve.mu);
        return;
    }
    g_serve.last_emit = now;
    char rid[80];
    snprintf(rid, sizeof(rid), "%s", g_serve.rid);
    pthread_mutex_unlock(&g_serve.mu);
    if (!rid[0]) return;
    char rid_esc[488];
    h3img_json_escape(rid, rid_esc, sizeof(rid_esc));
    emit("{\"type\":\"progress\",\"id\":\"%s\",\"stage\":\"%s\","
         "\"percent\":%.3f}",
         rid_esc, stage, percent);
}

static void *serve_heartbeat(void *arg) {
    (void)arg;
    for (;;) {
        struct timespec pause = {4, 0};
        nanosleep(&pause, NULL);
        pthread_mutex_lock(&g_serve.mu);
        if (!g_serve.rid[0]) {
            pthread_mutex_unlock(&g_serve.mu);
            continue;
        }
        char rid_esc[488];
        h3img_json_escape(g_serve.rid, rid_esc, sizeof(rid_esc));
        /* Emit while still holding g_serve.mu so a request ending (rid cleared
         * by serve_handle) cannot race between the snapshot and the write and
         * leave a stale heartbeat — or a torn final line at process exit.
         * emit() takes only g_stdout_mu; no other path takes it before
         * g_serve.mu, so the lock order is consistent. */
        emit("{\"type\":\"heartbeat\",\"id\":\"%s\"}", rid_esc);
        pthread_mutex_unlock(&g_serve.mu);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Request handling                                                    */
/* ------------------------------------------------------------------ */

/* Type-strict numeric fields: a present-but-wrong-typed value ("abc",
 * true, null) must fail the request instead of silently falling back to
 * the default — a silent 1024 default on a typo once cost a full 2-minute
 * generation (the yue overlay_sampling lesson). Fractional values are
 * rejected too: 42.9 silently truncating to seed 42 is the same trap. */
static int serve_json_u32(const h3img_json *req, const char *key,
                          uint32_t def, uint32_t *out) {
    const h3img_json *item = h3img_json_get(req, key);
    if (!item) {
        *out = def;
        return 1;
    }
    if (h3img_json_type(item) != H3IMG_JSON_NUMBER) return 0;
    double v = h3img_json_number(item);
    if (!(v >= 0.0 && v <= 4294967295.0) || v != floor(v)) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int serve_json_u64(const h3img_json *req, const char *key,
                          uint64_t def, uint64_t *out) {
    const h3img_json *item = h3img_json_get(req, key);
    if (!item) {
        *out = def;
        return 1;
    }
    if (h3img_json_type(item) != H3IMG_JSON_NUMBER) return 0;
    double v = h3img_json_number(item);
    /* A double cannot represent integers above 2^53 exactly; accepting them
     * here silently changes the seed (breaking reproducibility). The JSON
     * parser stores numbers as double, so cap at the exact-integer range —
     * which matches the host's seed <= 2^53-1 contract — instead of 2^64,
     * where UINT64_MAX rounds up to 2^64 and gets spuriously rejected. */
    if (!(v >= 0.0 && v <= 9007199254740992.0) || v != floor(v)) return 0;
    *out = (uint64_t)v;
    return 1;
}

/* String fields follow the same rule: a present non-string must fail the
 * request with a precise message instead of degrading to "" (which would
 * misreport as "prompt is required" and send the host chasing the wrong
 * field). */
static int serve_json_string(const h3img_json *req, const char *key,
                             const char **out, char *error,
                             size_t error_size) {
    const h3img_json *item = h3img_json_get(req, key);
    if (!item) {
        *out = "";
        return 1;
    }
    if (h3img_json_type(item) != H3IMG_JSON_STRING) {
        snprintf(error, error_size, "%s must be a string", key);
        return 0;
    }
    *out = h3img_json_string(item);
    return 1;
}

static void serve_handle(const char *dir, const h3img_json *req) {
    char error[512] = {0};
    /* progress/heartbeat store the id in an 80-byte buffer; done/error
     * reuse the same truncated value so every event stays keyed to one
     * id (the Go host matches progress and the terminal event by id). */
    char rid[80];
    const char *rid_in = NULL;
    double t0 = now_seconds();

    const char *prompt = NULL;
    const char *out = NULL;
    uint32_t width = 1024, height = 1024;
    uint64_t seed = 42;
    int bad = 0;
    if (!serve_json_string(req, "id", &rid_in, error, sizeof(error)) ||
        !serve_json_string(req, "prompt", &prompt, error, sizeof(error)) ||
        !serve_json_string(req, "output", &out, error, sizeof(error)) ||
        !serve_json_u32(req, "width", 1024, &width) ||
        !serve_json_u32(req, "height", 1024, &height) ||
        !serve_json_u64(req, "seed", 42, &seed)) {
        if (!error[0])
            snprintf(error, sizeof(error),
                     "numeric parameter out of range (width/height/seed)");
        bad = 1;
    }
    snprintf(rid, sizeof(rid), "%s", rid_in ? rid_in : "");
    if (!bad) {
        pthread_mutex_lock(&g_serve.mu);
        snprintf(g_serve.rid, sizeof(g_serve.rid), "%s", rid);
        snprintf(g_serve.stage, sizeof(g_serve.stage), "encode");
        g_serve.percent = 0.0;
        g_serve.last_emit = 0.0;
        pthread_mutex_unlock(&g_serve.mu);
    }
    const h3img_json *imgs = h3img_json_get(req, "images");
    const char *images[H3IMG_SERVE_MAX_IMAGES] = {0};
    size_t n_images = 0;
    if (!bad && imgs) {
        /* A present-but-wrong-typed images field must fail the request
         * instead of silently degrading to t2i (same lesson as the
         * numeric fields). */
        if (h3img_json_type(imgs) != H3IMG_JSON_ARRAY) {
            snprintf(error, sizeof(error), "images must be an array");
            bad = 1;
        } else {
            size_t count = h3img_json_count(imgs);
            if (count > H3IMG_SERVE_MAX_IMAGES) {
                snprintf(error, sizeof(error),
                         "at most %d reference images (got %zu)",
                         H3IMG_SERVE_MAX_IMAGES, count);
                bad = 1;
            } else {
                for (size_t i = 0; i < count; i++) {
                    const h3img_json *el = h3img_json_at(imgs, i);
                    if (h3img_json_type(el) != H3IMG_JSON_STRING) {
                        snprintf(error, sizeof(error),
                                 "images[%zu] must be a string", i);
                        bad = 1;
                        break;
                    }
                    images[n_images++] = h3img_json_string(el);
                }
            }
        }
    }
    /* Sanity bounds: 0 slips through the %32 gate and huge values would
     * burn tens of GB in allocations before failing. */
    if (!bad &&
        (width < 32 || width > 8192 || height < 32 || height > 8192)) {
        snprintf(error, sizeof(error),
                 "width/height must be in [32, 8192] (got %ux%u)", width,
                 height);
        bad = 1;
    }
    /* done.path is escaped into a 1024-byte buffer inside a 2048-byte
     * line; an unbounded output string would be silently truncated there
     * while the PNG is written to the full path. */
    if (!bad && strlen(out) > 512) {
        snprintf(error, sizeof(error),
                 "output path too long (max 512 bytes)");
        bad = 1;
    }

    int steps = 0;
    int ok = 0;
    if (bad) {
        /* error already set */
    } else if (!rid[0]) {
        snprintf(error, sizeof(error), "id is required");
    } else if (!prompt[0]) {
        snprintf(error, sizeof(error), "prompt is required");
    } else if (!out[0]) {
        snprintf(error, sizeof(error), "output path is required");
    } else if (width % 32 || height % 32) {
        snprintf(error, sizeof(error), "size must be multiples of 32");
    } else {
        ok = h3img_generate_ex(dir, prompt, images, n_images, (int)width,
                               (int)height, seed, NULL, out, &steps,
                               serve_progress, NULL, error, sizeof(error));
    }

    /* Clearing rid stops the heartbeat (worker finally-clause semantics). */
    pthread_mutex_lock(&g_serve.mu);
    g_serve.rid[0] = 0;
    pthread_mutex_unlock(&g_serve.mu);

    if (ok) {
        char rid_esc[488], out_esc[3076];
        h3img_json_escape(rid, rid_esc, sizeof(rid_esc));
        h3img_json_escape(out, out_esc, sizeof(out_esc));
        emit("{\"type\":\"done\",\"id\":\"%s\",\"path\":\"%s\","
             "\"width\":%u,\"height\":%u,\"steps\":%d,\"elapsed\":%.1f}",
             rid_esc, out_esc, width, height, steps, now_seconds() - t0);
    } else {
        emit_error(rid, error);
    }
}

/* ------------------------------------------------------------------ */
/* Main loop                                                           */
/* ------------------------------------------------------------------ */

int h3img_cmd_serve(const char *dir) {
    char error[512] = {0};
    if (!h3img_ensure_shader_dir(error, sizeof(error))) {
        fprintf(stderr, "serve: %s\n", error);
        emit_crashed(error);
        return 1;
    }
    if (h3img_engine_lock(error, sizeof(error)) < 0) {
        fprintf(stderr, "serve: %s\n", error);
        emit_crashed(error);
        return 1;
    }
    emit_state("loading", -1.0);
    double t0 = now_seconds();
    if (!h3img_engine_ready(dir, error, sizeof(error))) {
        fprintf(stderr, "serve: %s\n", error);
        emit_crashed(error);
        return 1;
    }
    fprintf(stderr,
            "serve: model %s present (%.1fs) — weights load lazily per "
            "request\n", dir, now_seconds() - t0);
    emit_state("ready", now_seconds() - t0);

    pthread_t beat;
    if (pthread_create(&beat, NULL, serve_heartbeat, NULL) != 0) {
        fprintf(stderr, "serve: cannot start heartbeat thread\n");
        emit_error("", "cannot start heartbeat thread");
        return 1;
    }
    /* Never joined: stdin EOF exits the process while the heartbeat sleeps. */
    pthread_detach(beat);

    /* 65536 payload bytes + '\n' + NUL: with a 65536-byte buffer a legal
     * line of exactly 65535 bytes plus its newline filled the buffer and
     * was misdetected as an over-long request. */
    static char line[H3IMG_SERVE_LINE_MAX];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        /* fgets silently splits an over-long request into two "lines":
         * drain the remainder so the stream stays line-aligned and
         * answer with an error event instead of leaving the host
         * waiting on a terminal event. */
        if (len == sizeof(line) - 1 && line[len - 1] != '\n' &&
            line[len - 1] != '\r') {
            int ch;
            while ((ch = fgetc(stdin)) != EOF && ch != '\n') {}
            fprintf(stderr, "serve: request line exceeds %zu bytes\n",
                    sizeof(line) - 1);
            emit_error("", "request line too long");
            continue;
        }
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (!len) continue;
        h3img_json *req = NULL;
        char perr[128];
        if (!h3img_json_parse(line, &req, perr, sizeof(perr))) {
            fprintf(stderr, "serve: bad request line: %s\n", perr);
            emit_error("", "malformed request line");
            continue;
        }
        if (!strcmp(h3img_json_get_string(req, "cmd", ""), "ping")) {
            /* Echo the id so the host can key the readiness probe to its
             * request (serve_handle emits the same state line keyed by id
             * at startup, where no id exists). */
            const char *ping_id = h3img_json_get_string(req, "id", "");
            char ping_esc[488];
            h3img_json_escape(ping_id, ping_esc, sizeof(ping_esc));
            if (ping_id[0])
                emit("{\"type\":\"state\",\"state\":\"ready\","
                     "\"id\":\"%s\"}",
                     ping_esc);
            else
                emit_state("ready", -1.0);
        } else {
            serve_handle(dir, req);
        }
        h3img_json_free(req);
    }
    fprintf(stderr, "serve: stdin closed — bye\n");
    return 0;
}
