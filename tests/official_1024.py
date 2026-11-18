"""Official-stack 1024-generation A/B for the grid artifact question.

Usage: official_1024.py <output_png> [prompt] [seed]
Runs the released modular pipeline at 1024x1024 with
min_generation_area=1024 (generate at the requested size, no upscaling).
"""

import sys

import torch
from diffusers.modular_pipelines import ModularPipeline

MODEL = "/Volumes/Work/models/MiniMax-H3-Image"

def main() -> int:
    out = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else \
        "A red apple on a wooden table, studio lighting"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    pipe = ModularPipeline.from_pretrained(
        MODEL, macro="minimax-h3-image", trust_remote_code=True)
    pipe.to("mps")
    image = pipe(
        prompt=prompt, height=1024, width=1024,
        min_generation_area=1024,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="pil")[0][0]
    image.save(out)
    print("saved", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
