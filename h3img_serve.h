#ifndef H3IMG_SERVE_H
#define H3IMG_SERVE_H

#include <stddef.h>

/* JSONL worker protocol (`h3 serve`) and interactive session (`h3 repl`)
 * over the MiniMax-H3-Image pipeline. Both run one generation at a time:
 * the pipeline is self-contained per request (mmap-lazy weights make the
 * reload cost a few seconds against 30 s+ of generation). */

/* JSONL worker: requests on stdin, events on stdout, logs on stderr.
 * Requests:  {"id","cmd":"ping"} (answered with the id echoed) or
 *            {"id","prompt","width","height","seed","output",
 *             "images":["path", ...]}
 * Events:    state loading/ready/crashed, progress, heartbeat, done, error.
 * All numeric fields reject non-number types and fractional values; id/
 * prompt/output/images reject non-string types. An over-long request line
 * and a malformed line are answered with an id-less error event (the id
 * cannot be recovered from a truncated line).
 * The checkpoint is guidance-distilled: six sampler steps are fixed and
 * there is no CFG/negative prompt — request steps/negativePrompt fields
 * are ignored. */
int h3img_cmd_serve(const char *model_dir);

/* linenoise interactive session. Plain lines are prompts; /-commands
 * manage size, seed, output prefix, and edit references. */
int h3img_cmd_repl(const char *model_dir);

/* Shared serve/repl helpers (h3img_engine.c). */
int h3img_ensure_shader_dir(char *error, size_t error_size);
int h3img_engine_lock(char *error, size_t error_size);
int h3img_engine_ready(const char *model_dir, char *error,
                       size_t error_size);
void h3img_json_escape(const char *in, char *out, size_t out_size);

#endif
