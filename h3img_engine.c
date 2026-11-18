#include "h3img_serve.h"

#include <fcntl.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

/* Run from the engine directory: every h3img module compiles
 * h3_shaders.metal relative to the process cwd. A serve/repl launched
 * from elsewhere re-anchors to the executable's directory; relative
 * output paths follow the same anchor (documented behavior). */
int h3img_ensure_shader_dir(char *error, size_t error_size) {
    if (!access("h3_shaders.metal", R_OK)) return 1;
    char path[4096];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        char *slash = strrchr(path, '/');
        if (slash) {
            *slash = 0;
            if (!chdir(path) && !access("h3_shaders.metal", R_OK)) return 1;
        }
    }
    snprintf(error, error_size,
             "h3_shaders.metal not found — start from the engine directory");
    return 0;
}

/* GPU single-instance lock: ~6 GB live footprint plus heavy pipeline
 * compilation make two concurrent engines pure thrash. */
int h3img_engine_lock(char *error, size_t error_size) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/h3img-engine-%d.lock", (int)getuid());
    int fd = open(path, O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
    if (fd < 0) {
        snprintf(error, error_size, "cannot open %s", path);
        return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        snprintf(error, error_size, "another h3img engine holds %s", path);
        return -1;
    }
    /* Deliberately never closed: the lock lives for the process. */
    return fd;
}

/* Cheap presence check over the four pipeline components; the weights
 * themselves stay mmap-lazy and load per request. */
int h3img_engine_ready(const char *dir, char *error, size_t error_size) {
    static const char *components[] = {"tokenizer/tokenizer.json",
                                       "text_encoder", "transformer",
                                       "vae_diffusers"};
    for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); i++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, components[i]);
        struct stat st;
        if (stat(path, &st) != 0) {
            snprintf(error, error_size,
                     "model directory %s is missing %s", dir, components[i]);
            return 0;
        }
    }
    return 1;
}

/* Escapes the payload of a caller-supplied string (ids, paths, messages)
 * so one quote or control byte cannot corrupt the protocol line. */
void h3img_json_escape(const char *in, char *out, size_t out_size) {
    if (!out || out_size == 0) return;
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)(in ? in : "");
         *p && o + 7 < out_size; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c < 0x20) {
            o += (size_t)snprintf(out + o, out_size - o, "\\u%04x", c);
        } else {
            out[o++] = (char)c;
        }
    }
    out[o] = '\0';
}
