"""Official-stack per-phase timing at the 1024-class canvas (fast档 A/B).

Mirrors tests/gen_e2e_ab.py's validated component assembly (the same one the
fixtures pin against the engine) at 1024x1024: Qwen3-VL layer-50 pre-norm
text encoding (bf16 MPS), the distilled six-step transformer loop (bf16 MPS),
and the ViT VAE decode (fp32 MPS — the modular pipeline's .to("mps") target).
Usage: h3img_official_perf.py [out_png]
"""

import math
import sys
import time

import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler
from diffusers.models import MiniMaxH3ImageTransformer2DModel
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

MODEL = "/Volumes/Work/models/MiniMax-H3-Image"
PROMPT = "A red apple on a wooden table"
SEED = 42
SIZE = 1024
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
          0.5327739119529724, 0.23200887441635132, 0.0]
MEAN = torch.tensor([
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264])
STD = torch.tensor([
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.772995293140411, 1.8955937623977661, 0.946841835975647,
    0.7996807489047049, 0.44988900423049927, 0.7197399735450745,
    0.6936295211276511, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523])

t0 = time.time()
def mark(label):
    print(f"[{time.time()-t0:7.1f}s] {label}", flush=True)

def save_png(path, pixels):
    import struct, zlib
    # The VAE decodes into ImageNet-normalized RGB; revert like the released
    # DecodeStep (the old [-1,1] mapping was overexposed).
    rgb = ((pixels * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) +
            torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
           .clamp(0, 1) * 255.0).to(torch.uint8)
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    def chunk(tag, data):
        block = tag + data
        return (struct.pack(">I", len(data)) + block +
                struct.pack(">I", zlib.crc32(block) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        handle.write(chunk(b"IEND", b""))


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/h3img_official_perf.png"
    device = "mps"

    mark("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    mark(f"tokenizer ready ({len(ids)} tokens)")

    t = time.time()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    encoder.eval()
    mark(f"text encoder load+to(mps): {time.time()-t:.1f}s")

    t = time.time()
    with torch.no_grad():
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        text = encoder.model(
            input_ids=ids_t,
            attention_mask=torch.ones_like(ids_t),
            mm_token_type_ids=torch.zeros_like(ids_t),
            use_cache=False,
            output_hidden_states=True).hidden_states[50]
    text = text.to(torch.bfloat16)
    torch.mps.synchronize()
    mark(f"text encode: {time.time()-t:.1f}s")
    del encoder

    scheduler = MiniMaxH3Scheduler.from_pretrained(MODEL, subfolder="scheduler")
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]],
                             dtype=torch.float32)
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    scheduler.set_timesteps(sigmas=sigmas, device=device)

    t = time.time()
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer_diffusers",
        torch_dtype=torch.bfloat16).to(device)
    transformer.eval()
    mark(f"transformer load+to(mps): {time.time()-t:.1f}s")

    tokens = len(ids)
    latent_h, latent_w = SIZE // 16, SIZE // 16
    rows_h, rows_w = latent_h // 2, latent_w // 2
    image_rows = rows_h * rows_w
    seq_len = tokens + image_rows
    token_tags = torch.cat([torch.ones(tokens),
                            torch.zeros(image_rows)]).long()
    timestep_indices = torch.zeros(seq_len, dtype=torch.long)
    text_indices = torch.arange(tokens)
    image_indices = torch.arange(tokens, seq_len)
    pos = torch.zeros(seq_len, 3)
    pos[:tokens, 0] = torch.arange(tokens, dtype=torch.float32)
    image_t = float(tokens)
    area = math.sqrt(latent_h * latent_w)
    h_axis = (torch.arange(0.5, rows_h, dtype=torch.float32) / rows_h *
              (latent_h / area) * 32.0)
    w_axis = (torch.arange(0.5, rows_w, dtype=torch.float32) / rows_w *
              (latent_w / area) * 32.0)
    grid = torch.stack(torch.meshgrid(
        torch.arange(1, dtype=torch.float32) + image_t,
        h_axis, w_axis, indexing="ij"), dim=-1).reshape(-1, 3)
    pos[tokens:] = grid

    generator = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, 24, 1, latent_h, latent_w,
                          generator=generator, dtype=torch.float32)
    latents_rows = latents[0, :, 0].view(24, rows_h, 2, rows_w, 2).permute(
        1, 3, 0, 2, 4).reshape(image_rows, 96).to(device, torch.bfloat16)

    step_times = []
    with torch.no_grad():
        rows = latents_rows
        for i, tstep in enumerate(timesteps.tolist()):
            t = time.time()
            unique_t = torch.tensor([tstep], dtype=torch.float32,
                                    device=device)
            noise = transformer(
                hidden_states=rows[None],
                encoder_hidden_states=text,
                timestep=unique_t,
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags.to(device),
                position_ids=pos.to(device),
                image_indices=image_indices.to(device),
                text_indices=text_indices.to(device),
                pdd_step_index=i,
                return_dict=False)[0][0]
            sigma = SIGMAS[i]
            sigma_next = SIGMAS[i + 1]
            rows = (rows.float() + (sigma - sigma_next) *
                    noise.float()).to(torch.bfloat16)
            torch.mps.synchronize()
            step_times.append(time.time() - t)
            mark(f"dit step {i}: {step_times[-1]:.2f}s")
    for i, dt in enumerate(step_times):
        print(f"    step{i} {dt:.2f}s", flush=True)
    latents = rows.float().reshape(rows_h, rows_w, 24, 2, 2).permute(
        2, 0, 3, 1, 4).reshape(24, latent_h, latent_w).cpu()
    save_file({"x.final_latents": latents.contiguous()},
              "/tmp/h3img_official_final_latents.safetensors")
    del transformer

    latents = latents * STD.view(-1, 1, 1) + MEAN.view(-1, 1, 1)
    t = time.time()
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        MODEL, subfolder="vae_diffusers",
        torch_dtype=torch.float32).to(device)
    vae.eval()
    mark(f"vae load+to(mps): {time.time()-t:.1f}s")

    t = time.time()
    with torch.no_grad():
        pixels = vae.decode(latents.unsqueeze(0).to(device),
                            return_dict=False)[0][0]
    torch.mps.synchronize()
    mark(f"vae decode: {time.time()-t:.1f}s")
    save_png(out, pixels)
    mark(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
