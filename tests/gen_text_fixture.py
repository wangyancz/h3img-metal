"""Generate the MiniMax-H3-Image t2i text-encoding parity fixture.

Runs the released Qwen3-VL conditioner (text_encoder/) on a verbatim prompt —
no chat template, no special tokens, exactly what the official MiniMax-H3-Image
pipeline does for text-to-image — and dumps the token ids plus several layer
hidden states (0-based: x.layer_49 == hidden_states[50], the pre-norm state
after 50 decoder layers that the DiT conditions on).

Usage:
    gen_text_fixture.py [model_dir] [output_path] [prompt]

Defaults:
    model_dir    /Volumes/Work/models/MiniMax-H3-Image
    output_path  misc/fixtures/h3img_text_t2i.safetensors
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

DEFAULT_MODEL = "/Volumes/Work/models/MiniMax-H3-Image"
DEFAULT_OUT = "misc/fixtures/h3img_text_t2i.safetensors"
DEFAULT_PROMPT = (
    "A photorealistic overhead food photograph of a rustic farm-to-table "
    "feast on a weathered dark brown wooden table: a woven jute placemat "
    "holding a freshly baked sourdough loaf with a deeply scored golden "
    "crust, a hammered copper bowl overflowing with purple kale, orange "
    "carrots and red beets, a wedge of bloomy rind brie on a dark cheese "
    "board, heirloom tomatoes on the vine, and a glass of dark red wine "
    "beside a brass candle holder, warm natural sunlight from the left, "
    "dappled shadows, shallow depth of field."
)

# 0-based decoder layer indices captured in the fixture. layer_49 is the
# conditioning state (hidden_states[50] in HF output_hidden_states terms).
LAYERS = (0, 7, 23, 48, 49)
MAX_TOKENS = 512


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    prompt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PROMPT

    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    print(f"prompt tokens: {len(token_ids)}")
    if len(token_ids) > MAX_TOKENS:
        token_ids = token_ids[:MAX_TOKENS]
        print(f"truncated to {MAX_TOKENS} tokens")

    try:
        processor = AutoProcessor.from_pretrained(model_dir, subfolder="processor")
        mm_token_type_ids = processor.create_mm_token_type_ids([token_ids])
        unexpected = {t for row in mm_token_type_ids for t in row if t != 0}
        if unexpected:
            raise ValueError(f"t2i fixture expects a pure text presentation, got {unexpected}")
        print("mm_token_type_ids: all text (0)")
    except Exception as error:  # noqa: BLE001 - fixture generation convenience
        print(f"processor unavailable ({error}); assuming all-text presentation")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
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

    tensors = {
        "x.ids": torch.tensor([token_ids], dtype=torch.int32),
    }
    for layer in LAYERS:
        hidden = outputs.hidden_states[layer]
        assert hidden.shape == (1, len(token_ids), 5120), hidden.shape
        tensors[f"x.layer_{layer}"] = hidden[0].to("cpu").contiguous()
        print(f"x.layer_{layer}: shape {tuple(hidden.shape)} "
              f"rms {hidden.float().pow(2).mean().sqrt().item():.4f}")

    save_file(tensors, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
