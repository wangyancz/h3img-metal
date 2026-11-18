"""Official-side vision tower dump for the edit A/B parity harness.

Replays the exact edit preprocessing from gen_edit_ab.py (PIL bicubic resize
to the bound_short_edge size, then processor.image_processor), runs the
Qwen3-VL visual tower, and stores every stage the C harness compares against:

  x.pixel_values   [patches, 1536]  f32  processor output (already x*2-1)
  x.patch_embed    [patches, 1152]  f32  after patch_embed, before pos add
  x.embed_pos      [patches, 1152]  f32  after pos add (rotary input)
  x.block{n}       [patches, 1152]  f32  after block n (n in 0,7,8,16,24,26)
  x.merged         [merged, 5120]   f32  final merger output (pooler)
  x.deepstack{i}   [merged, 5120]   f32  deepstack merger outputs

The merged/deepstack outputs come from get_image_features()'s
pooler_output / deepstack_features — no hooks required there. Block stages
use forward hooks on the block modules.

Usage: gen_vision_ab.py [model_dir] [fixture_path_prefix]
"""

import sys

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Work/models/MiniMax-H3-Image"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "misc/fixtures/h3img_vision_ab"
REFERENCE = "misc/fixtures/h3img_edit_ab_reference.png"
BLOCK_STAGES = (0, 7, 8, 16, 24, 26)


def bound_short_edge(width, height, min_edge, max_edge, multiple):
    short = min(width, height)
    scale = 1.0
    if short < min_edge:
        scale = min_edge / short
    elif short > max_edge:
        scale = max_edge / short
    out_w = max(multiple, round(width * scale / multiple) * multiple)
    out_h = max(multiple, round(height * scale / multiple) * multiple)
    return out_w, out_h


def main() -> int:
    device = "mps"
    image = Image.open(REFERENCE).convert("RGB")
    vw, vh = bound_short_edge(*image.size, 1024, 1024, 32)
    vision_img = image.resize((vw, vh), Image.BICUBIC)
    print(f"reference {image.size} -> vision {vw}x{vh}")

    processor = AutoProcessor.from_pretrained(MODEL, subfolder="processor")
    vision = processor.image_processor(images=[vision_img], return_tensors="pt")
    pixel_values = vision["pixel_values"]
    grid_thw = vision["image_grid_thw"]
    print("pixel_values", tuple(pixel_values.shape), "grid_thw",
          grid_thw.tolist())

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    encoder.eval()
    visual = encoder.model.visual

    caps = {}

    def hook(name):
        def grab(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            caps[name] = tensor.detach().float().cpu()
        return grab

    visual.patch_embed.register_forward_hook(hook("patch_embed"))
    fired = []
    for index in BLOCK_STAGES:
        visual.blocks[index].register_forward_hook(hook(f"block{index}"))
        fired.append(f"block{index}")
    # Input of the rotary module = hidden after patch embed + pos add.
    visual.rotary_pos_emb.register_forward_pre_hook(
        lambda module, inputs: caps.__setitem__(
            "embed_pos", inputs[0].detach().float().cpu()) or fired.append(
            "embed_pos"))

    with torch.no_grad():
        vision_out = encoder.get_image_features(
            pixel_values.to(device, torch.bfloat16),
            grid_thw.to(device))
    caps["merged"] = torch.cat(
        [t for t in vision_out.pooler_output]).float().cpu()
    caps["deepstack"] = [torch.cat([t for t in feats]).float().cpu()
                         for feats in vision_out.deepstack_features]
    print("hooks fired:", fired)
    for name in ("patch_embed", "embed_pos", "merged"):
        if name not in caps:
            raise RuntimeError(f"stage {name} was not captured")

    payload = {"x.pixel_values": pixel_values.float().cpu().contiguous()}
    for name, tensor in caps.items():
        if name == "deepstack":
            continue
        payload[f"x.{name}"] = tensor.contiguous()
    for index, tensor in enumerate(caps["deepstack"]):
        payload[f"x.deepstack{index}"] = tensor.contiguous()
    save_file(payload, f"{PREFIX}.safetensors")
    print("vision A/B fixture saved:", f"{PREFIX}.safetensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
