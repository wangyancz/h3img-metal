"""Generate the end-to-end edit A/B fixture: shared noise draws, reference
image, and the official reference output.

Draw order mirrors the released pipeline on one seeded generator:
  1. generated-image noise (prepare latents)
  2. one posterior sample per reference (edit VAE encode)
  3. one noising noise per reference (edit prepare)

Everything the C side needs to replay exactly is saved: the initial noise,
the sampled/normalized reference latents, the reference noising noise, and
the reference image itself. The official output PNG is written alongside.

Usage:
    gen_edit_ab.py [model_dir] [fixture_path_prefix]
"""

import math
import struct
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler
from diffusers.models import MiniMaxH3ImageTransformer2DModel
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Work/models/MiniMax-H3-Image"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "misc/fixtures/h3img_edit_ab"
SEED = 20260718
PROMPT = "Turn the apple into a green lime while keeping the composition"
REF_SIZE = (512, 512)   # (width, height) of the synthetic reference
TARGET = (256, 256)     # generated image size
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
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523])
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])


def save_png(path, pixels):
    # The VAE decodes into ImageNet-normalized RGB; the released DecodeStep
    # reverts with pixel_mean/pixel_std, clamps to [0,1], and quantizes by
    # truncation. (The old [-1,1] mapping here made the reference PNG as
    # overexposed as the engine's, so the A/B never saw the bug.)
    rgb = ((pixels * PIXEL_STD.view(-1, 1, 1) + PIXEL_MEAN.view(-1, 1, 1))
           .clamp(0, 1) * 255.0).to(torch.uint8)
    rgb = rgb.permute(1, 2, 0).numpy()
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


def spatial_grid(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32.0
    return torch.from_numpy(grid).to(torch.float64)


def frame_grid(latent_h, latent_w):
    hg = spatial_grid(latent_h, 2, math.sqrt(latent_h * latent_w))
    wg = spatial_grid(latent_w, 2, math.sqrt(latent_h * latent_w))
    grids = torch.meshgrid(hg, wg, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def patchify(latents):
    # [C, H, W] -> rows [(h//2)(w//2), C*4], taps (c, ph, pw)
    c, h, w = latents.shape[-3:]
    lat = latents.reshape(-1, c, h, w)[0] if latents.dim() > 3 else latents
    out = lat.view(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4)
    return out.reshape((h // 2) * (w // 2), c * 4)


def main() -> int:
    device = "mps"
    generator = torch.Generator().manual_seed(SEED)

    # 1. The synthetic reference image (deterministic, saved for both sides).
    image = Image.new("RGB", REF_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse([REF_SIZE[0] // 4, REF_SIZE[1] // 4,
                  REF_SIZE[0] * 3 // 4, REF_SIZE[1] * 3 // 4],
                 fill=(200, 30, 30))
    image.save(f"{PREFIX}_reference.png")

    # 2. The two consumer resizes (PIL bicubic, like VaeImageProcessor).
    vision_w, vision_h = bound_short_edge(*REF_SIZE, 1024, 1024, 32)
    vae_w, vae_h = bound_short_edge(*REF_SIZE, 1024, 2048, 32)
    vision_img = image.resize((vision_w, vision_h), Image.BICUBIC)
    vae_img = image.resize((vae_w, vae_h), Image.BICUBIC)
    print(f"reference {REF_SIZE} -> vision {vision_w}x{vision_h}, "
          f"vae {vae_w}x{vae_h}")

    # 3. Generated-image noise (first draw on the generator).
    latent_h, latent_w = TARGET[1] // 16, TARGET[0] // 16
    init_noise = torch.randn(1, 24, 1, latent_h, latent_w,
                             generator=generator, dtype=torch.float32)

    # 4. The edit presentation + text encoding.
    tokenizer = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    processor = AutoProcessor.from_pretrained(MODEL, subfolder="processor")
    vision = processor.image_processor(images=[vision_img], return_tensors="pt")
    merge_size = processor.image_processor.merge_size ** 2
    token_ids, token_tags = [], []
    text_tag, image_tag = 1, 0
    for index, grid in enumerate(vision["image_grid_thw"]):
        label_ids = tokenizer(f"Picture {index + 1}: ",
                              add_special_tokens=False)["input_ids"]
        vision_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")]
            * (int(grid.prod()) // merge_size)
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        token_ids += label_ids + vision_ids
        token_tags += [text_tag] * len(label_ids) + [image_tag] * len(vision_ids)
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [text_tag] * len(prompt_ids)

    block_caps = {}
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    encoder.eval()
    mm = torch.tensor(processor.create_mm_token_type_ids([token_ids]),
                      dtype=torch.long, device=device)
    with torch.no_grad():
        vision_out = encoder.get_image_features(
            vision["pixel_values"].to(device, torch.bfloat16),
            vision["image_grid_thw"].to(device))
        lm = vision_out.last_hidden_state if hasattr(vision_out, "last_hidden_state") else \
            (vision_out[0] if isinstance(vision_out, tuple) else vision_out)
        block_caps["vision_merged"] = lm.detach().float().cpu()
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
        text = encoder.model(
            input_ids=ids_t,
            attention_mask=torch.ones_like(ids_t),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
            pixel_values=vision["pixel_values"].to(device, torch.bfloat16),
            image_grid_thw=vision["image_grid_thw"].to(device),
        ).hidden_states[50]
    text_token_tags = torch.tensor(token_tags)
    del encoder

    # 5. Reference VAE encode: posterior sample, fp16 round, crop, normalize.
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        MODEL, subfolder="vae_diffusers", torch_dtype=torch.float32)
    vae.eval()
    pixels = torch.from_numpy(np.array(vae_img)).permute(2, 0, 1)[None].float()
    pixels = pixels[..., : pixels.shape[-2] // 16 * 16,
                    : pixels.shape[-1] // 16 * 16]
    pixels = (pixels / 255.0 - PIXEL_MEAN.view(1, -1, 1, 1)) / \
        PIXEL_STD.view(1, -1, 1, 1)
    with torch.no_grad():
        posterior = vae.encode(pixels, return_dict=False)[0]
        ref_latent = posterior.sample(generator=generator).to(torch.float16).float()
    ref_latent = ref_latent[..., : ref_latent.shape[-2] // 2 * 2,
                            : ref_latent.shape[-1] // 2 * 2]
    ref_latent = (ref_latent[0] - MEAN.view(-1, 1, 1)) / STD.view(-1, 1, 1)
    del vae
    ref_lh, ref_lw = ref_latent.shape[-2:]
    ref_noise = torch.randn(1, 24, 1, ref_lh, ref_lw, generator=generator,
                            dtype=torch.float32)[0]

    # 6. The packed layout [text | references | generated].
    _, patch_h, patch_w = (1, 2, 2)
    num_text = text.shape[1]
    num_ref_rows = (ref_lh // patch_h) * (ref_lw // patch_w)
    num_target_rows = (latent_h // patch_h) * (latent_w // patch_w)
    seq_len = num_text + num_ref_rows + num_target_rows
    position_ids = torch.zeros(seq_len, 3, dtype=torch.float32)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float32)
    cursor = num_text
    rotary_time = float(num_text)
    rows = slice(cursor, cursor + num_ref_rows)
    cursor = rows.stop
    position_ids[rows, 0] = rotary_time
    position_ids[rows, 1:] = frame_grid(ref_lh, ref_lw).to(torch.float32)
    rotary_time += 1.0
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid(latent_h, latent_w).to(torch.float32)
    token_tags_seq = torch.empty(seq_len, dtype=torch.long)
    token_tags_seq[:num_text] = text_token_tags
    token_tags_seq[num_text:] = image_tag
    text_indices = torch.arange(num_text)
    image_indices = torch.arange(num_text, seq_len)

    # 7. The conditioning rows (t = 0.999, held for every step) and the
    # generated rows.
    noised = 0.999 * ref_latent + 0.001 * ref_noise
    condition_rows = patchify(noised[:, None])
    target_rows = patchify(init_noise[0, :, 0])
    # latents holds ONLY the image rows (references first), like the
    # official pipeline's block_state.latents.
    latents = torch.cat([condition_rows, target_rows]).to(device,
                                                          torch.bfloat16)

    # 8. The distilled six-step loop; only the generated rows step.
    scheduler = MiniMaxH3Scheduler.from_pretrained(MODEL, subfolder="scheduler")
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]],
                             dtype=torch.float32)
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer_diffusers",
        torch_dtype=torch.bfloat16).to(device)
    transformer.eval()
    def cap_refiner(mod, inp, out):
        block_caps["refined"] = (out[0] if isinstance(out, tuple) else out
                                 ).detach()[0].float().cpu()
    transformer.token_refiner.register_forward_hook(cap_refiner)
    for bi in (0, 24):
        def grab(mod, inp, out, bi=bi):
            block_caps[bi] = out.detach()[0].float().cpu()
        transformer.transformer_blocks[bi].register_forward_hook(grab)
    # Per-row timesteps, matching the released pipeline's row_timestep_plan:
    # every row steps with the step's t; ONLY the reference rows are held at
    # the 0.999 conditioning level for every step (before_denoise.py:
    # `row_timesteps[image_indices[:num_reference_rows]] = 0.999`).
    with torch.no_grad():
        for i, t in enumerate(timesteps.tolist()):
            row_t = torch.full((seq_len,), t, dtype=torch.float32)
            row_t[image_indices[:num_ref_rows]] = 0.999
            unique_t, timestep_indices = torch.unique(
                row_t, sorted=True, return_inverse=True)
            noise = transformer(
                hidden_states=latents[None],
                encoder_hidden_states=text,
                timestep=unique_t.to(device, torch.bfloat16),
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags_seq.to(device),
                position_ids=position_ids.to(device),
                image_indices=image_indices.to(device),
                text_indices=text_indices.to(device),
                pdd_step_index=i,
                return_dict=False)[0][0]
            if i == 0:
                block_caps["step0_noise"] = noise.detach().float().cpu()
                time_input = transformer.time_proj(unique_t)
                if isinstance(time_input, tuple):
                    time_input = time_input[0]
                temb = transformer.time_embedder(time_input.to(device))
                if isinstance(temb, tuple):
                    temb = temb[0]
                block_caps["temb"] = temb.detach()[0].float().cpu()
                with torch.no_grad():
                    mod = transformer.transformer_blocks[0].adaln_proj(
                        F.silu(temb.to(transformer.transformer_blocks[0]
                                       .adaln_proj.linear.weight.dtype)))
                if isinstance(mod, tuple):
                    mod = mod[0]
                block_caps["adaln0"] = mod.detach()[0].float().cpu()
            sigma, sigma_next = SIGMAS[i], SIGMAS[i + 1]
            stepped = (latents[num_ref_rows:].float() +
                       (sigma - sigma_next) * noise[num_ref_rows:].float())
            latents[num_ref_rows:] = stepped.to(torch.bfloat16)
            print(f"step {i}: t {t:.4f}")
    target_latents = latents[num_ref_rows:].float().reshape(
        latent_h // 2, latent_w // 2, 24, 2, 2).permute(
        2, 0, 3, 1, 4).reshape(24, latent_h, latent_w).cpu()
    save_file({
        "x.step0_noise": block_caps["step0_noise"].contiguous(),
        "x.temb": block_caps["temb"].contiguous(),
        "x.adaln0": block_caps["adaln0"].contiguous(),
        "x.vision_merged": block_caps["vision_merged"].contiguous(),
        "x.refined": block_caps["refined"].contiguous(),
        "x.block0": block_caps[0].contiguous(),
        "x.block24": block_caps[24].contiguous(),
        "x.latents": init_noise[0, :, 0].contiguous(),
        "x.ref_latents": ref_latent.cpu().contiguous(),
        "x.ref_noise": ref_noise.contiguous(),
        "x.text_embeds": text[0].float().cpu().contiguous(),
        "x.position_ids": position_ids.float().contiguous(),
        "x.token_tags": token_tags_seq.float().contiguous(),
        "x.condition_rows": condition_rows.float().cpu().contiguous(),
        "x.final_latents": target_latents.contiguous(),
    }, f"{PREFIX}_fixtures.safetensors")
    print("fixtures saved")
    del transformer

    # 9. Decode.
    denorm = target_latents * STD.view(-1, 1, 1) + MEAN.view(-1, 1, 1)
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        MODEL, subfolder="vae_diffusers", torch_dtype=torch.float32)
    vae.eval()
    with torch.no_grad():
        pixels = vae.decode(denorm.unsqueeze(0), return_dict=False)[0][0]
    save_png(f"{PREFIX}_reference_out.png", pixels)
    print(f"reference edit PNG written ({TARGET[0]}x{TARGET[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
