# h3img-metal

Native **MiniMax-H3-Image** inference for Apple Silicon, written from scratch
in C11 and Objective-C Metal — **no PyTorch, no MLX, no Python at runtime**.
One binary covers text-to-image and reference-based editing (`--t2i`, `serve`,
`repl`). Every stage is validated against the official diffusers pipeline with
checked-in fixtures, down to bit-level parity on the deterministic parts and
semantic gates on the full generation. Current work is incremental
H3-Image-specific Metal performance and memory optimization on M3 Max and M5 Max.

[中文文档](README-ZH.md)

## MiniMax-H3-Image: native image generation

A from-scratch native engine for the MiniMax-H3-Image checkpoint (`--t2i`,
`serve`, `repl`), validated stage-for-stage against the released diffusers
pipeline (`/Volumes/Work/diffusers-h3img/.venv`).

### Pipeline

1. **Text** — verbatim-prompt tokenization (no template, no special tokens,
   512-token cap), then the first 50 Qwen3-VL language layers; the pre-norm
   final state becomes the conditioning rows.
2. **Vision (edit only)** — each reference PNG is composited onto white
   (alpha semantics of the reference stack), resized so the short edge lands
   on 1024 (round to 32), and encoded by the Qwen3-VL vision tower into
   merged 5120-wide rows ("Picture i: " blocks prepended to the prompt).
   The same reference is separately VAE-encoded (short edge clamped to
   [1024, 2048]), normalized with the per-channel latent statistics, noised
   to the t=0.999 conditioning level, and patchified into 96-wide rows.
3. **DiT** — the packed-sequence FL2VA transformer (50 blocks, text rows +
   one latent frame + reference rows). The sampler is the pdd-distilled
   six-step Euler/velocity grid from `pdd_config.json`, consumed verbatim;
   the terminal sigma is zero. There is **no CFG and no negative prompt** —
   the checkpoint is guidance-distilled, and the step count is fixed.
4. **VAE** — the ViT decoder runs per-tile (256 px tiles, 64 px overlap,
   linear cross-fade stitching), mirroring the released `_decode` tiling.
   Latents are channel-wise denormalized before the decode.

### Generation canvas (quality vs speed)

The model is trained on 2048-class canvases: small direct sizes are out of
distribution. Like the released pipeline, the engine generates on a canvas of
at least `min_generation_area` (default 2048×2048) at the requested aspect
ratio and bicubic-resizes the decoded 8-bit image back to the requested size:

| Request size | Canvas | Typical wall time (M5 Max) | Peak memory |
|---|---|---|---|
| 256×256 | 2048×2048 (default floor) | ~2 min | ~6.6 GB |
| 512×512 | 2048×2048 (default floor) | ~2 min (~63 s warm in a resident session) | ~6.6 GB |
| any ≤1024² | 1024×1024 (`H3IMG_MIN_AREA=1048576`, fast档) | ~22 s | ~6.1 GB |
| 2048-class | 2048×2048 | ~108 s | ~6.6 GB |
| 4096×4096 | 4096×4096 | ~284 s | ~6.4 GB |
| 6272×2688 (21:9, official maximum) | 6272×2688 | ~296 s | ~5.9 GB |

The supported output sizes follow the released size table: every bucket is
~16.8 MP with sides up to 6272 px (`4096×4096`, `4736×3552`, `5024×3360`,
`5472×3072`, `6272×2688` and their aspect mirrors; all multiples of 32).
Anything beyond 6272 px per side is rejected up front. Peak memory is the
`/usr/bin/time` "peak memory footprint" of a one-shot run; a `repl` session
generating 10 images in a row peaked at ~8.5 GB with no per-image ratchet.

`H3IMG_MIN_AREA` overrides the floor (values below 1024² clamp to 1024²;
the released pipeline accepts 1024²..4096²). The 2048² default is the
official quality behavior; the 1024² floor is the fast档. `--latents` (the
A/B harness) pins the fixture geometry instead and skips canvas logic.
Sizes must be multiples of 32.

#### What does "peak memory" actually mean? (measured)

The table above is the **process** footprint (`/usr/bin/time` peak RSS).
There are three different numbers in play; they must not be added together:

| Metric (one-shot 6-step run, M5 Max) | 1024² fast | 2048² default |
|---|---|---|
| System memory increase, peak (`active+wired+compressed` vs. idle baseline) | **+38.9 GB** | **+39.5 GB** |
| of which file-backed pages (mmap weights faulted in) | +40.8 GB | +37.1 GB |
| Process `phys_footprint` peak (Activity Monitor "Memory" column) | 5.7 GB | 5.6 GB |
| Engine `peak_live_bytes`, DiT (logical MTLBuffer ledger, mmap-backed) | 36.2 GiB | 36.9 GiB |
| Real device allocation at teardown (`MTLDevice.currentAllocatedSize`) | ~2.4–4.0 GiB | ~2.4–4.4 GiB |

The ~39 GB **system increase** is the number a web UI / Activity Monitor
shows while a job runs, and is what people quote as "just under 40 GB". It is
dominated by the 62 GB `transformer` safetensors mapped with `mmap` +
`newBufferWithBytesNoCopy`: as the GPU reads each weight page it is faulted
into the system file cache (`File-backed pages`). These pages are clean and
purgeable — the kernel reclaims them under pressure — and they are **not**
charged to the process footprint. The 62 GB `text_encoder` is streamed with
pread and contributes a smaller transient.

The engine's `peak_live_bytes` (~36.9 GiB) is a *logical* counter: creating a
zero-copy Metal buffer over an mmap'd weight counts its full size immediately,
even though only touched pages are physically resident. It is an attribution
signal (and the reason the naive "sum of weights" number looks huge), not
physical usage. Actual GPU device allocations stay at a few GiB.

The peak is essentially **independent of resolution**: it is set by which
weights get mapped/faulted in, not by the canvas size. net system increase
+38.9 GB (1024²) vs +39.5 GB (2048²) differ by <1 GB. Session footprint does
not ratchet: a `repl` run of 10 images peaked at ~8.5 GB and returned to
baseline.

Reproduce (needs the GPU lock free — stop any `serve`/`repl` first):

```sh
# sample system used + process phys_footprint at 4 Hz during a one-shot run
python3 /tmp/h3img_mem_probe.py -- \
  ./h3img --t2i -d /Volumes/Work/models/MiniMax-H3-Image \
  -p "A red apple on a wooden table" --width 2048 --height 2048 -o /tmp/out.png
# add H3_PROFILE=1 H3_DEBUG_GPU_MEMORY=1 for the logical/real GPU ledger
```

### One-shot CLI

```sh
make -j8
H3IMG_MIN_AREA=1048576 ./h3img -d /Volumes/Work/models/MiniMax-H3-Image --t2i \
  -p "A red apple on a wooden table" \
  --width 512 --height 512 --seed 7 -o outputs/apple.png

# Edit: repeatable --ref-image; the prompt may refer to <Picture i>.
./h3img -d /Volumes/Work/models/MiniMax-H3-Image --t2i \
  -p "Turn the apple into a green lime while keeping the composition" \
  --ref-image apple.png --width 512 --height 512 -o outputs/lime.png
```

`--latents FILE` injects fp32 initial latents (`x.latents` [24, H/16, W/16],
plus `x.ref_latents_<i>`/`x.ref_noise_<i>` for edit A/B) so both stacks
replay identical draws — the engine's own noise RNG cannot be matched
RNG-for-RNG with the reference.

### serve: JSONL worker

`h3img serve -d MODEL_DIR` speaks the same JSONL worker protocol family as the
other native engines: requests on stdin, events on stdout (line-buffered),
diagnostics on stderr. Requests are handled sequentially; the weights are
mmap-lazy and load per request (a few seconds against 30 s+ of generation),
so there is no separate loading phase beyond a component presence check.

Requests:

```json
{"id":"t1","cmd":"ping"}
{"id":"t2","prompt":"A red apple on a wooden table","width":512,
 "height":512,"seed":7,"output":"/tmp/apple.png"}
{"id":"t3","prompt":"Turn the apple into a lime","width":512,"height":512,
 "seed":7,"output":"/tmp/lime.png","images":["/path/apple.png"]}
```

Events: `state` (`loading`/`ready` with `loadSeconds`, or `crashed` with
`message`), `progress` (`stage` ∈ encode/sample/decode/save, global
`percent` 0..1, rate-limited to ~4 Hz), `heartbeat` (every 4 s while a
request is active, keyed by `id`), `done` (`path`, `width`, `height`,
`steps` = 6, `elapsed`), `error` (`id`, `message`). `progress`,
`heartbeat`, `done`, and `error` all carry the request `id` (truncated to
80 bytes), so a host matches every event to its job.

Validation: `id`, `prompt`, and `output` (≤512 bytes) are required;
`width`/`height` must be numbers in [32, 8192] and multiples of 32; `seed`
is an unsigned 64-bit number (default 42); `images` is an array of at most
8 path strings. Present-but-wrong-typed numeric fields and `images` fail
the request instead of silently falling back to defaults. `steps` and
`negativePrompt` fields are accepted and ignored (distilled checkpoint).

Operational notes:

- A GPU single-instance lock (`/tmp/h3img-engine-<uid>.lock`, non-blocking
  flock) refuses a second serve/repl on the same machine.
- Every module compiles `h3_shaders.metal` relative to the process cwd;
  serve/repl re-anchors to the engine directory automatically. Relative
  output paths resolve against that anchor (documented behavior).
- stdin EOF exits cleanly; cancellation is the host killing the process
  group (there is no in-process cancel).
- Per-request memory lines are not implemented; see "What does 'peak memory'
  actually mean?" above for the measured system-increase (~39 GB) vs.
  process-footprint (~6 GB) distinction.

### repl: interactive session

`h3img repl -d MODEL_DIR` keeps a session of generation settings (linenoise
prompt `h3img>`). Plain lines are prompts; each generation writes
`PREFIX-<n>.png` and auto-increments the seed.

```text
/width N /height N   output size, multiples of 32 in [32, 8192] (default 1024)
/seed N              next seed (auto-increments after every image)
/out PREFIX          output file prefix (default "out")
/image PATH          add an edit reference (max 8); bare /image or /image
                     clear removes them all (back to t2i)
/refs                list the current references
/show                current settings
/help /quit
```

### Tests

`make test` gates every image stage on the official oracle (fixtures under
`misc/fixtures/`, reference weights under
`/Volumes/Work/models/MiniMax-H3-Image`; items skip with a notice when
either is absent):

| Stage | Gate |
|---|---|
| text parity | layer-49 rel-L2 ≤ ~0.05 (measured 0.040) |
| DiT parity | three velocity steps, rel-L2 ≤ ~1.3 chaotic band (measured ~1.1) |
| VAE parity | latents/pixels rel-L2 ≤ 5e-4 (measured ~1e-6) |
| t2i e2e | semantic gates (hist corr, LapVar ratio) vs official |
| edit e2e | final latents rel-L2 ≤ 0.35 (measured ~0.13) + image gates |

Diagnostic-only (not part of `make test`): `make h3img-text-fixture`,
`make h3img-vision-fixture` (regenerate fixtures with the reference venv)
and `make h3img-vision-ab` (per-stage vision-tower A/B, ~2 min).

### Environment variables

| Variable | Effect |
|---|---|
| `H3IMG_MIN_AREA` | generation canvas floor (see table above) |
| `H3IMG_VAE_CPU` | force the CPU GEMM (Accelerate) VAE path |
| `H3IMG_VAE_SCALAR` | force the pure-scalar VAE reference path |
| `H3IMG_DUMP_LATENTS` | dump vision merged rows, layout, text embeds, final latents to `/tmp` |
| `H3IMG_VISION_DUMP=DIR` | per-stage vision-tower dumps (pairs with `check_vision_ab.py`) |
| `H3IMG_VAE_DUMP` | per-stage VAE decode dumps |
| `H3IMG_DIT_DUMP` | per-stage DiT dumps (block 0/24 tensors, schedule) |
| `H3IMG_DIT_STEP0_DUMP` | dump step-0 DiT tensors |
| `H3IMG_EDIT_TRACE` | edit-path trace lines (layout segments, ref loading) |
| `H3IMG_TIMING=1` | print cumulative wall seconds per pipeline phase (text+vision, DiT load, denoise, VAE decode, PNG) to stderr |
| `H3_DISABLE_FUSED_MLP`, `H3_DIT_F32_FINAL` | set by the pipeline itself: the fused-MLP fast path has a known Metal failure on this checkpoint, and the final head stays F32 |

`H3_PROFILE` also carries into the shared DiT internals; on the image
path it adds a per-component GPU ledger (wall/encode/wait/device-busy,
submissions, direct dispatches, linear encodes). A 2048² run with both flags
shows roughly: text 5.1 s, DiT load+AdaLN 7.9 s, 6-step denoise 36 s
(GPU-bound), VAE decode 8.4--8.6 s, PNG 0.1 s. The first two are fixed
per-request costs; `H3IMG_TIMING` makes them visible.

### Image decode: command chaining

The VAE ViT decoder issues its five stages per block (QKV, attention,
attention-out, FF1, FF2) as one Metal command chain: stages 1--4 use
`h3_gpu_continue` (commit without waiting) and only stage 5 submits, so each
block has a single CPU/GPU sync instead of five. Scratch tensors stay alive
until that terminal submit avoids the qwen-metal P5 hazard of freeing a buffer
still referenced by an in-flight command. This cut 2048² VAE decode from
~9.9 s to ~8.4--8.6 s with bit-identical output.

A fused Q/K/V linear (three `[1024,1024]` matmuls concatenated into one
`[1024,3072]` plus a split kernel) was implemented and validated bit-exact,
but **reverted**: although it removed a third of the MPSGraph linear encodes
(17545 → 11737), the larger matmul plus the extra split dispatches made wall
time ~5% *worse*. The 17545 linear encodes are 24 layers × 6 linears × 121
tiles of 1024² MPS GEMM; reducing their count does not help when each is
already at hardware efficiency.

### Known limits

- Six sampler steps are baked in (`pdd_config.json`); no CFG, no negative
  prompt, no per-request step control.
- The pipeline is self-contained per request: text encoding, AdaLN
  precompute, and lazy weight mmap reload cost ~10 s per request (measured
  on M5 Max). Compiled Metal shaders and pipeline states are cached at the
  process level, so repeated module contexts within one serve/repl session
  do not recompile. A resident-GPU refactor (weights held across requests,
  ~60 GB residency) is possible but not started.
- VAE decode tiles are 256 px and the decoder's weights are uploaded once
  per request and reused across tiles; extreme aspect ratios beyond the
  released buckets are not specifically validated.


## Build and test

```sh
make -j8          # builds the h3img binary
make test         # all image-stage parity gates (see table above)
```

`make test` runs the deterministic host suite and the image-specific Oracle
comparisons. Each stage gates on fixtures under `misc/fixtures/` and reference
weights under `/Volumes/Work/models/MiniMax-H3-Image`; a stage prints `skip`
when its fixture or weights are absent.

Prerequisites: Apple Silicon with Metal 3+, an OpenMP-free C11 toolchain
(`clang` shipped with Xcode), and the H3-Image model checkpoint. No Python,
PyTorch, or MLX runtime is needed — the reference diffusers venv is used only
to regenerate fixtures via the diagnostic `make h3img-*-fixture` targets.

Diagnostic targets (not part of `make test`): `make h3img-text-fixture`,
`make h3img-vision-fixture`, `make h3img-vision-ab` (per-stage vision-tower A/B,
~2 min).

## License

MIT — see [LICENSE](LICENSE). Third-party notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
