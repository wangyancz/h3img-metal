#ifndef H3IMG_JSON_H
#define H3IMG_JSON_H

#include <stddef.h>

/* Minimal JSON reader, shared by the serve request loop and configuration
 * readers. Objects preserve insertion order so duplicate keys resolve to the
 * first entry. */

typedef enum {
    H3IMG_JSON_NULL = 0,
    H3IMG_JSON_BOOL,
    H3IMG_JSON_NUMBER,
    H3IMG_JSON_STRING,
    H3IMG_JSON_ARRAY,
    H3IMG_JSON_OBJECT
} h3img_json_kind;

typedef struct h3img_json h3img_json;

/* Parse NUL-terminated text. Returns 1 and fills *out on success. */
int h3img_json_parse(const char *text, h3img_json **out, char *error,
                     size_t error_size);
void h3img_json_free(h3img_json *value);

h3img_json_kind h3img_json_type(const h3img_json *value);
const h3img_json *h3img_json_get(const h3img_json *value, const char *key);
const h3img_json *h3img_json_at(const h3img_json *value, size_t index);
size_t h3img_json_count(const h3img_json *value);
/* True bool, or number != 0, or string "true"/"1". */
int h3img_json_bool(const h3img_json *value);
double h3img_json_number(const h3img_json *value);
/* NUL-terminated UTF-8 payload of a string value (or ""). */
const char *h3img_json_string(const h3img_json *value);
/* Convenience: object member as double/string with fallback. */
double h3img_json_get_number(const h3img_json *value, const char *key,
                             double fallback);
const char *h3img_json_get_string(const h3img_json *value, const char *key,
                                  const char *fallback);
int h3img_json_get_bool(const h3img_json *value, const char *key,
                        int fallback);

#endif
