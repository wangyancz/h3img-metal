CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE \
	-DACCELERATE_NEW_LAPACK
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm -lz

# MiniMax-H3-Image only. The video/audio engine (h3.c, h3_video_*, h3_audio_*,
# h3_ffmpeg, h3_terminal, h3_cli) was removed; shared primitives (RNG, layout,
# schedule, weights, GPU runtime, Qwen encoders, DiT) stay because the image
# pipeline depends on them.
LIB_C := h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c \
	h3img_vae.c h3img_pipeline.c h3img_png.c h3img_imgproc.c \
	h3_vision_encoder.c h3_multimodal.c \
	h3img_json.c h3img_engine.c h3img_serve.c
LIB_M := h3_gpu.m h3_tokenizer.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)
# h3img_repl links linenoise, which lives with the CLI objects.
CLI_OBJ := main.o h3img_repl.o linenoise.o

.PHONY: all test parity real-parity h3img-text-fixture h3img-text-tests \
	h3img-vision-fixture h3img-vision-ab clean

all: h3img libh3.a

h3img: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3img_text_test: tests/test_real_h3img_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3img_dit_test: tests/test_real_h3img_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Regenerates the H3-Image t2i text fixture with the reference venv when it is
# installed; falls back to the system python (which likely lacks diffusers).
h3img_vae_test: tests/test_real_h3img_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3img_e2e_test: tests/test_real_h3img_e2e.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3img_edit_test: tests/test_real_h3img_edit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3img-text-fixture:
	@if test -x /Volumes/Work/diffusers-h3img/.venv/bin/python; then \
		/Volumes/Work/diffusers-h3img/.venv/bin/python tests/gen_text_fixture.py; \
	else \
		echo "skip: reference venv /Volumes/Work/diffusers-h3img/.venv is not installed"; \
	fi

h3img-text-tests: h3img_text_test
	@if test -f misc/fixtures/h3img_text_t2i.safetensors && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/text_encoder; then \
		./h3img_text_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures/h3img_text_t2i.safetensors; \
	else \
		echo "skip: H3-Image text fixture/weights are not installed"; \
	fi

# Regenerates the per-stage vision-tower A/B fixture (official side only; the
# C side is dumped by the engine run in h3img-vision-ab below).
h3img-vision-fixture:
	@if test -x /Volumes/Work/diffusers-h3img/.venv/bin/python; then \
		/Volumes/Work/diffusers-h3img/.venv/bin/python \
			tests/gen_vision_ab.py; \
	else \
		echo "skip: reference venv /Volumes/Work/diffusers-h3img/.venv is not installed"; \
	fi

# Full vision-tower A/B: run the engine once with H3IMG_VISION_DUMP and
# compare every stage against the fixture. Diagnostic only (the tower takes
# about two minutes), not part of `make test`.
h3img-vision-ab: h3img
	@if test -f misc/fixtures/h3img_vision_ab.safetensors && \
	         test -x /Volumes/Work/diffusers-h3img/.venv/bin/python; then \
		rm -rf /tmp/h3img_vision_ab_c && \
		mkdir -p /tmp/h3img_vision_ab_c && \
		H3IMG_VISION_DUMP=/tmp/h3img_vision_ab_c ./h3img -d \
			/Volumes/Work/models/MiniMax-H3-Image --t2i \
			--ref-image misc/fixtures/h3img_edit_ab_reference.png \
			-p "Turn the apple into a green lime while keeping the composition" \
			--width 256 --height 256 --latents \
			misc/fixtures/h3img_edit_ab_fixtures.safetensors \
			--seed 20260718 -o /tmp/h3img_vision_ab.png && \
		/Volumes/Work/diffusers-h3img/.venv/bin/python \
			tests/check_vision_ab.py /tmp/h3img_vision_ab_c \
			misc/fixtures/h3img_vision_ab.safetensors; \
	else \
		echo "skip: H3-Image vision A/B fixture/venv is not installed"; \
	fi

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

test: h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3img_text_test h3img_dit_test h3img_vae_test h3img_e2e_test \
	h3img_edit_test

	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	@if test -f misc/fixtures/h3img_text_t2i.safetensors && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/text_encoder; then \
		./h3img_text_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures/h3img_text_t2i.safetensors; \
	else \
		echo "skip: H3-Image text fixture/weights are not installed"; \
	fi
	@if test -f misc/fixtures/h3img_dit_t2i.safetensors && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/transformer; then \
		./h3img_dit_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures/h3img_dit_t2i.safetensors; \
	else \
		echo "skip: H3-Image DiT fixture/weights are not installed"; \
	fi
	@if test -f misc/fixtures/h3img_vae_t2i.safetensors && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/vae_diffusers; then \
		./h3img_vae_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures/h3img_vae_t2i.safetensors; \
	else \
		echo "skip: H3-Image VAE fixture/weights are not installed"; \
	fi
	@if test -f misc/fixtures/h3img_e2e_latents.safetensors && \
	         test -f misc/fixtures/h3img_e2e_reference.png && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/vae_diffusers; then \
		./h3img_e2e_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures; \
	else \
		echo "skip: H3-Image e2e fixture/weights are not installed"; \
	fi
	@if test -f misc/fixtures/h3img_edit_ab_fixtures.safetensors && \
	         test -f misc/fixtures/h3img_edit_ab_reference_out.png && \
	         test -d /Volumes/Work/models/MiniMax-H3-Image/vae_diffusers; then \
		./h3img_edit_test /Volumes/Work/models/MiniMax-H3-Image \
			misc/fixtures; \
	else \
		echo "skip: H3-Image edit A/B fixture/weights are not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3img h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_dit_bench h3_dit_bench_864 \
		h3img_text_test h3img_dit_test h3img_vae_test h3img_e2e_test \
		h3img_edit_test \
		libh3.a *.o *.d tests/*.o tests/*.d
