"""Compare C-side vision stage dumps against the official fixture.

Usage: check_vision_ab.py [c_dump_dir] [fixture]
Defaults: /tmp/vision_c misc/fixtures/h3img_vision_ab.safetensors
"""

import sys

import numpy as np
from safetensors import safe_open

C_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision_c"
FIXTURE = sys.argv[2] if len(sys.argv) > 2 else \
    "misc/fixtures/h3img_vision_ab.safetensors"


def read_bf16(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def rel(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = max(np.linalg.norm(b), 1e-12)
    overall = np.linalg.norm(a - b) / denom
    rows_a = a.reshape(a.size // b.reshape(-1).size and -1 or 1, -1) \
        if False else None
    return overall


def stage(name, official, c_raw, shape):
    official = official.astype(np.float64).reshape(shape)
    c = c_raw.astype(np.float64).reshape(shape)
    diff = c - official
    denom = np.linalg.norm(official)
    overall = np.linalg.norm(diff) / max(denom, 1e-12)
    rows = shape[0]
    flat_o = official.reshape(rows, -1)
    flat_c = c.reshape(rows, -1)
    row_rel = np.linalg.norm(flat_c - flat_o, axis=1) / np.maximum(
        np.linalg.norm(flat_o, axis=1), 1e-12)
    print(f"{name:<16} rel-L2 {overall:.5f}  row rel min "
          f"{row_rel.min():.4f} med {np.median(row_rel):.4f} "
          f"max {row_rel.max():.4f}  |o| {denom:.2f} |c| "
          f"{np.linalg.norm(c):.2f}")
    return overall


def main() -> int:
    fixture = safe_open(FIXTURE, framework="np")
    expected_tokens = None
    worst = 0.0

    # Stage 0: processor pixel rows vs C patchified input.
    official = fixture.get_tensor("x.pixel_values")
    c = read_bf16(f"{C_DIR}/c_patch_input.bf16")
    if official.size != c.size:
        print(f"patch_input SIZE MISMATCH official {official.shape} "
              f"c {c.size}")
        return 1
    worst = max(worst, stage("patch_input", official, c, official.shape))

    tokens = official.shape[0]
    hidden = 1152
    merged_tokens = tokens // 4

    for name in ["embed_pos"] + [f"block_{i}" for i in (0, 7, 8, 16, 24, 26)]:
        official = fixture.get_tensor(
            f"x.{name.replace('block_', 'block')}")
        c = read_bf16(f"{C_DIR}/c_{name}.bf16")
        if official.size != c.size:
            print(f"{name} SIZE MISMATCH official {official.shape} "
                  f"c {c.size}")
            return 1
        worst = max(worst, stage(name, official, c,
                                 (tokens, hidden)))

    for name in ["merged", "deepstack0", "deepstack1", "deepstack2"]:
        official = fixture.get_tensor(f"x.{name}")
        c = read_bf16(f"{C_DIR}/c_{name.replace('deepstack', 'deepstack_')}.bf16")
        if official.size != c.size:
            print(f"{name} SIZE MISMATCH official {official.shape} "
                  f"c {c.size}")
            return 1
        worst = max(worst, stage(name, official, c,
                                 (merged_tokens, 5120)))

    print(f"\nworst stage rel-L2: {worst:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
