"""Generate the MiniMax-H3-Image VAE parity fixture (encoder + ViT decoder).

Runs the official `AutoencoderKLMiniMaxH3Image` (from `vae_diffusers/`) in
float32 on a small deterministic image and dumps every stage boundary:

    x.image            the input image, [3, 64, 64]
    x.enc_conv_in      after the first causal conv3d (image = one frame)
    x.enc_b0..b5       after each down block (spatial /16 total)
    x.enc_normout      after the final GroupNorm + SiLU
    x.enc_convout      after conv_out (the causal video head, T=1)
    want.latent        posterior mode = quant_conv[..., :24], [24, 4, 4]
    want.z_denorm      latent * latents_std + latents_mean (decode input)
    x.dec_postquant    after post_quant_conv
    x.dec_projin       after proj_in (patch tokens, [16, 1024])
    x.dec_b0..b1       after the first two ViT blocks (register/cls appended)
    x.dec_b23          after the last ViT block
    x.dec_normout      after the final LayerNorm
    x.dec_projout      after proj_out, [16, 768] (pre-unpatchify)
    want.pixels        the decoded image, [3, 64, 64]

Usage:
    gen_vae_fixture.py [model_dir] [output_path]
"""

import sys

import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLMiniMaxH3Image

DEFAULT_MODEL = "/Volumes/Work/models/MiniMax-H3-Image"
DEFAULT_OUT = "misc/fixtures/h3img_vae_t2i.safetensors"


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    print("loading VAE (float32, cpu)...")
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        model_dir, subfolder="vae_diffusers", torch_dtype=torch.float32
    )
    vae.eval()

    generator = torch.Generator().manual_seed(20260717)
    image = torch.rand(1, 3, 64, 64, generator=generator) * 2.0 - 1.0

    tensors = {"x.image": image[0].contiguous()}
    captures = {}

    def hook(name):
        def grab(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            captures[name] = output.detach()[0].float().clone()
        return grab

    encoder = vae.encoder
    encoder.conv_in.register_forward_hook(hook("enc_conv_in"))
    for i, block in enumerate(encoder.down_blocks):
        block.register_forward_hook(hook(f"enc_b{i}"))
    encoder.norm_out.register_forward_hook(hook("enc_normout"))
    encoder.conv_out.register_forward_hook(hook("enc_convout"))
    vae.quant_conv.register_forward_hook(hook("enc_quant"))

    decoder = vae.decoder
    vae.post_quant_conv.register_forward_hook(hook("dec_postquant"))
    decoder.proj_in.register_forward_hook(hook("dec_projin"))
    for i, block in enumerate(decoder.transformer_blocks):
        if i in (0, 1, 23):
            block.register_forward_hook(hook(f"dec_b{i}"))
    decoder.norm_out.register_forward_hook(hook("dec_normout"))
    decoder.proj_out.register_forward_hook(hook("dec_projout"))

    with torch.no_grad():
        posterior = vae.encode(image)
        latents = posterior.latent_dist.mode()  # [1, 24, 4, 4]
        tensors["want.latent"] = latents[0].contiguous()
        z_denorm = latents * torch.tensor(vae.latents_std, dtype=latents.dtype).view(1, -1, 1, 1) + \
            torch.tensor(vae.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1)
        tensors["want.z_denorm"] = z_denorm[0].contiguous()
        decoded = vae.decode(z_denorm, return_dict=False)[0]
        tensors["want.pixels"] = decoded[0].contiguous()

    # The rope table (cos) for the decoder grid, for C-side reference.
    with torch.no_grad():
        height = width = 4
        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        suffix = position_ids.new_zeros((decoder.num_register_tokens + 1, 2))
        position_ids = torch.cat([position_ids, suffix], dim=0)
        cos, sin = decoder.rope(position_ids.unsqueeze(0))
        tensors["x.dec_rope_cos"] = cos[0, :, 0, :].contiguous()
        tensors["x.dec_rope_sin"] = sin[0, :, 0, :].contiguous()

    for name, value in captures.items():
        tensors[f"x.{name}"] = value.contiguous()

    for name, value in sorted(tensors.items()):
        print(f"{name:22s} {tuple(value.shape)}")
    save_file(tensors, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
