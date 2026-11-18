"""Generate the MiniMax-H3-Image DiT single-step parity fixture.

Runs the official MiniMax-H3ImageTransformer2DModel (transformer_diffusers/)
on a tiny 64x64 canvas (latent 4x4, 2x2 patch rows) with the M1 text fixture's
prompt, for two distilled steps, and dumps every quantity the native engine
needs for parity: prompt embeds, patchified latent rows, the float64 rotary
grid, modality tags, per-step timesteps, intermediate hidden states (post
refiner, after blocks 0/24) and the per-step pdd-head velocity output.

Usage:
    gen_dit_fixture.py [model_dir] [output_path]

Defaults:
    model_dir    /Volumes/Work/models/MiniMax-H3-Image
    output_path  misc/fixtures/h3img_dit_t2i.safetensors
"""

import json
import sys

import torch
from safetensors.torch import save_file

from diffusers import MiniMaxH3ImageTransformer2DModel
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_video_latents,
)

DEFAULT_MODEL = "/Volumes/Work/models/MiniMax-H3-Image"
DEFAULT_OUT = "misc/fixtures/h3img_dit_t2i.safetensors"

PROMPT = (
    "A photorealistic overhead food photograph of a rustic farm-to-table "
    "feast on a weathered dark brown wooden table: a woven jute placemat "
    "holding a freshly baked sourdough loaf with a deeply scored golden "
    "crust, a hammered copper bowl overflowing with purple kale, orange "
    "carrots and red beets, a wedge of bloomy rind brie on a dark cheese "
    "board, heirloom tomatoes on the vine, and a glass of dark red wine "
    "beside a brass candle holder, warm natural sunlight from the left, "
    "dappled shadows, shallow depth of field."
)

TEXT_TAG, IMAGE_TAG = 1, 0
PATCH = (1, 2, 2)
SPATIAL_RATIO = 16
# Distilled steps compared in the fixture: first, second and last.
FIXTURE_STEPS = (0, 1, 5)


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(f"{model_dir}/pdd_config.json") as stream:
        sample_sigmas = json.load(stream)["sigmas"]
    steps = len(sample_sigmas) - 1
    print(f"pdd sigmas: {steps} steps")

    # 64x64 canvas with min_generation_area=1024 stays at the requested size:
    # latent 4x4 -> 2x2 patch rows per frame.
    canvas_h, canvas_w = get_image_canvas_size(64, 64, 1024, 32)
    latent_h, latent_w = canvas_h // SPATIAL_RATIO, canvas_w // SPATIAL_RATIO
    print(f"canvas {canvas_h}x{canvas_w}, latent {latent_h}x{latent_w}")
    assert (canvas_h, canvas_w) == (64, 64)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    token_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    print(f"prompt tokens: {len(token_ids)}")

    print(f"loading Qwen3-VL conditioner on {device} (bf16)...")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder.eval()
    with torch.no_grad():
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=torch.zeros_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[50][0]
    del text_encoder, outputs, input_ids
    if device == "mps":
        torch.mps.empty_cache()
    print(f"prompt_embeds: {tuple(prompt_embeds.shape)}")

    # Official packed sequence: [text | generated image], float64 rotary grid.
    text_token_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    position_ids, token_tags, image_indices, text_indices, _ = (
        build_image_packed_sequence(
            text_token_tags, [], latent_h, latent_w, PATCH, IMAGE_TAG
        )
    )
    assert token_tags.shape[0] == len(token_ids) + (latent_h // 2) * (latent_w // 2)
    assert image_indices[0] == len(token_ids)
    assert text_indices[-1] == len(token_ids) - 1
    assert position_ids[len(token_ids), 0] == float(len(token_ids)), (
        "the generated image must take the rotary slot right after the text")

    torch.manual_seed(42)
    latents = torch.randn(
        (1, 24, 1, latent_h, latent_w), dtype=torch.bfloat16, device=device
    )
    video_rows = patchify_video_latents(latents, PATCH).float().cpu()
    print(f"video rows: {tuple(video_rows.shape)}")

    print(f"loading DiT on {device} (bf16)...")
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        model_dir, subfolder="transformer_diffusers", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    num_blocks = len(transformer.transformer_blocks)
    assert num_blocks == 50, num_blocks

    captures = {}

    def capture(name):
        def hook(_module, _inputs, output):
            if name not in captures:
                value = output[0] if isinstance(output, tuple) else output
                captures[name] = value.detach()[0].float().cpu().contiguous()
        return hook

    transformer.token_refiner.register_forward_hook(capture("refined"))
    transformer.transformer_blocks[0].register_forward_hook(capture("block0"))
    transformer.transformer_blocks[24].register_forward_hook(capture("block24"))

    tensors = {
        "x.ids": torch.tensor([token_ids], dtype=torch.int32),
        "x.text": prompt_embeds.contiguous(),
        "x.video_rows": video_rows.contiguous(),
        "x.position_ids": position_ids.contiguous(),
        "x.token_tags": token_tags.to(torch.int32),
    }

    with torch.no_grad():
        for step in FIXTURE_STEPS:
            timestep_value = torch.tensor(
                1.0 - sample_sigmas[step], dtype=torch.float32
            )
            hidden_states = latents.reshape(
                1, (latent_h // 2) * (latent_w // 2), 96
            )
            result = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds[None],
                timestep=timestep_value.to(device)[None],
                timestep_indices=torch.zeros(
                    position_ids.shape[0], dtype=torch.long, device=device
                ),
                token_tags=token_tags.to(device),
                # The official rope casts the float64 grid to float32 before
                # computing angles (its MPS-compatible path and released
                # semantics); mirror that cast exactly.
                position_ids=position_ids.to(torch.float32).to(device),
                image_indices=image_indices.to(device),
                text_indices=text_indices.to(device),
                pdd_step_index=step,
            )
            velocity = result.sample[0].float().cpu()
            assert velocity.shape == video_rows.shape, velocity.shape
            tensors[f"x.timestep_{step}"] = timestep_value[None]
            tensors[f"want.velocity_{step}"] = velocity.contiguous()
            print(f"step {step}: timestep {timestep_value.item():.6f}, "
                  f"velocity rms {velocity.pow(2).mean().sqrt().item():.4f}")

    for name, value in captures.items():
        tensors[f"x.{name}"] = value.contiguous()
        print(f"x.{name}: {tuple(value.shape)}")

    save_file(tensors, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
