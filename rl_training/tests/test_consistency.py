"""Consistency check: training model vs inference pipeline.

Compares text encoding, condition preparation, and transformer forward pass
to ensure the training code faithfully replicates the pipeline behavior.

Uses two GPUs: cuda:0 for pipeline, cuda:1 for training model.

Usage:
    python -m tests.test_consistency
"""

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

from src.models.wan_i2v import WanI2VForTraining

MODEL_PATH = "storage/models/Wan2.2-I2V-A14B-Diffusers"
DEV_P = torch.device("cuda:0")  # pipeline
DEV_T = torch.device("cuda:1")  # training model
NUM_FRAMES = 81
HEIGHT = 480
WIDTH = 832
PROMPT = "A robotic arm manipulates a Rubik cube, performing one rotation step."


def check(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-4):
    a, b = a.float().cpu(), b.float().cpu()
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if not ok:
        print(f"         shapes: {a.shape} vs {b.shape}")
    return ok


def main():
    all_pass = True

    # ====================================================================
    # Load pipeline on cuda:0
    # ====================================================================
    print("Loading pipeline on cuda:0 ...")
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe.to(DEV_P)

    print(f"  image_dim = {pipe.transformer.config.image_dim}")
    print(f"  boundary_ratio = {pipe.config.boundary_ratio}")
    print(f"  expand_timesteps = {pipe.config.expand_timesteps}")

    # ====================================================================
    # Load training model on cuda:1
    # ====================================================================
    print("Loading training model on cuda:1 ...")
    train = WanI2VForTraining(MODEL_PATH, train_experts="both")
    train.text_encoder.to(DEV_T)
    train.vae.to(DEV_T)
    train.transformer.to(DEV_T).eval()
    train.transformer_2.to(DEV_T).eval()

    # ====================================================================
    # Test 1: Text Encoding
    # ====================================================================
    print("\n=== Test 1: Text Encoding ===")

    pipe_embeds, _ = pipe.encode_prompt(
        PROMPT,
        do_classifier_free_guidance=False,
        device=DEV_P,
        max_sequence_length=512,
    )
    train_embeds = train.encode_text([PROMPT], DEV_T)

    ok = check("text_embeds", pipe_embeds, train_embeds, atol=1e-3)
    all_pass = all_pass and ok

    # ====================================================================
    # Test 2: VAE Encoding (video latents)
    # ====================================================================
    print("\n=== Test 2: VAE Encoding ===")

    torch.manual_seed(42)
    fake_video_cpu = torch.randn(1, 3, NUM_FRAMES, HEIGHT, WIDTH).clamp(-1, 1)

    with torch.no_grad():
        # Pipeline
        v_p = fake_video_cpu.to(DEV_P)
        pipe_latents = pipe.vae.encode(v_p.to(pipe.vae.dtype)).latent_dist.sample()
        p_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(pipe_latents)
        p_std_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std)).view(1, -1, 1, 1, 1).to(pipe_latents)
        pipe_latents_norm = ((pipe_latents - p_mean) * p_std_inv).to(torch.bfloat16)

        # Training
        v_t = fake_video_cpu.to(DEV_T)
        train_latents_norm = train.encode_video(v_t)

    ok = check("vae_latents", pipe_latents_norm, train_latents_norm, atol=1e-3)
    all_pass = all_pass and ok

    # ====================================================================
    # Test 3: Condition Preparation
    # ====================================================================
    print("\n=== Test 3: Condition Preparation ===")

    torch.manual_seed(123)
    fake_image_cpu = torch.randn(1, 3, HEIGHT, WIDTH).clamp(-1, 1)

    with torch.no_grad():
        # --- Pipeline condition (replicate prepare_latents logic) ---
        img_p = fake_image_cpu.to(DEV_P)
        image_5d = img_p.unsqueeze(2)
        video_cond = torch.cat([image_5d, image_5d.new_zeros(1, 3, NUM_FRAMES - 1, HEIGHT, WIDTH)], dim=2)
        pipe_cond_latents = retrieve_latents(pipe.vae.encode(video_cond.to(pipe.vae.dtype)), sample_mode="argmax")
        pipe_cond_norm = ((pipe_cond_latents - p_mean) * p_std_inv).to(torch.bfloat16)

        latent_h = HEIGHT // pipe.vae_scale_factor_spatial
        latent_w = WIDTH // pipe.vae_scale_factor_spatial
        mask = torch.ones(1, 1, NUM_FRAMES, latent_h, latent_w, device=DEV_P)
        mask[:, :, 1:] = 0
        first_frame_mask = torch.repeat_interleave(mask[:, :, 0:1], dim=2, repeats=pipe.vae_scale_factor_temporal)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, pipe.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2)
        pipe_condition = torch.cat([mask.to(pipe_cond_norm), pipe_cond_norm], dim=1)

        # --- Training condition ---
        img_t = fake_image_cpu.to(DEV_T)
        train_condition = train.prepare_condition(img_t, NUM_FRAMES, HEIGHT, WIDTH)

    vae_temporal = pipe.vae_scale_factor_temporal
    ok = check("condition_tensor", pipe_condition, train_condition, atol=1e-3)
    all_pass = all_pass and ok
    ok1 = check("  mask part", pipe_condition[:, :vae_temporal], train_condition[:, :vae_temporal])
    ok2 = check("  latent part", pipe_condition[:, vae_temporal:], train_condition[:, vae_temporal:], atol=1e-3)
    all_pass = all_pass and ok1 and ok2

    # ====================================================================
    # Test 4: Transformer Forward
    # ====================================================================
    print("\n=== Test 4: Transformer Forward ===")

    with torch.no_grad():
        torch.manual_seed(0)
        noise_cpu = torch.randn_like(train_latents_norm.cpu())

        # --- High-noise expert (t=950) ---
        t_val = 950
        sigma = t_val / 1000.0
        noisy_cpu = sigma * noise_cpu + (1.0 - sigma) * train_latents_norm.cpu()
        ts = torch.tensor([t_val])

        # Pipeline side
        inp_p = torch.cat([noisy_cpu.to(DEV_P), pipe_condition], dim=1)
        pipe_pred_high = pipe.transformer(
            hidden_states=inp_p,
            timestep=ts.to(DEV_P),
            encoder_hidden_states=pipe_embeds,
            return_dict=False,
        )[0]

        # Training side
        inp_t = torch.cat([noisy_cpu.to(DEV_T), train_condition], dim=1)
        train_pred_high = train.transformer(
            hidden_states=inp_t,
            timestep=ts.to(DEV_T),
            encoder_hidden_states=train_embeds,
            return_dict=False,
        )[0]

        ok = check("transformer (high, t=950)", pipe_pred_high, train_pred_high, atol=1e-2)
        all_pass = all_pass and ok

        # --- Low-noise expert (t=400) ---
        t_val = 400
        sigma = t_val / 1000.0
        noisy_cpu = sigma * noise_cpu + (1.0 - sigma) * train_latents_norm.cpu()
        ts = torch.tensor([t_val])

        inp_p = torch.cat([noisy_cpu.to(DEV_P), pipe_condition], dim=1)
        pipe_pred_low = pipe.transformer_2(
            hidden_states=inp_p,
            timestep=ts.to(DEV_P),
            encoder_hidden_states=pipe_embeds,
            return_dict=False,
        )[0]

        inp_t = torch.cat([noisy_cpu.to(DEV_T), train_condition], dim=1)
        train_pred_low = train.transformer_2(
            hidden_states=inp_t,
            timestep=ts.to(DEV_T),
            encoder_hidden_states=train_embeds,
            return_dict=False,
        )[0]

        ok = check("transformer_2 (low, t=400)", pipe_pred_low, train_pred_low, atol=1e-2)
        all_pass = all_pass and ok

    # ====================================================================
    # Test 5: Transformer Forward (pipe also cast to bf16)
    # ====================================================================
    print("\n=== Test 5: Transformer Forward (pipe also cast to bf16) ===")
    print("  (casts pipeline transformer to bf16 to match training model)")

    pipe.transformer.to(torch.bfloat16)
    pipe.transformer_2.to(torch.bfloat16)

    with torch.no_grad():
        # High-noise expert
        t_val = 950
        sigma = t_val / 1000.0
        torch.manual_seed(0)
        noise_cpu = torch.randn_like(train_latents_norm.cpu())
        noisy_cpu = sigma * noise_cpu + (1.0 - sigma) * train_latents_norm.cpu()
        ts = torch.tensor([t_val])

        inp_p = torch.cat([noisy_cpu.to(DEV_P), pipe_condition], dim=1)
        pipe_pred_high_bf16 = pipe.transformer(
            hidden_states=inp_p,
            timestep=ts.to(DEV_P),
            encoder_hidden_states=pipe_embeds,
            return_dict=False,
        )[0]

        inp_t = torch.cat([noisy_cpu.to(DEV_T), train_condition], dim=1)
        train_pred_high_2 = train.transformer(
            hidden_states=inp_t,
            timestep=ts.to(DEV_T),
            encoder_hidden_states=train_embeds,
            return_dict=False,
        )[0]

        ok = check("transformer bf16 (high, t=950)", pipe_pred_high_bf16, train_pred_high_2, atol=1e-4)
        all_pass = all_pass and ok

        # Low-noise expert
        t_val = 400
        sigma = t_val / 1000.0
        noisy_cpu = sigma * noise_cpu + (1.0 - sigma) * train_latents_norm.cpu()
        ts = torch.tensor([t_val])

        inp_p = torch.cat([noisy_cpu.to(DEV_P), pipe_condition], dim=1)
        pipe_pred_low_bf16 = pipe.transformer_2(
            hidden_states=inp_p,
            timestep=ts.to(DEV_P),
            encoder_hidden_states=pipe_embeds,
            return_dict=False,
        )[0]

        inp_t = torch.cat([noisy_cpu.to(DEV_T), train_condition], dim=1)
        train_pred_low_2 = train.transformer_2(
            hidden_states=inp_t,
            timestep=ts.to(DEV_T),
            encoder_hidden_states=train_embeds,
            return_dict=False,
        )[0]

        ok = check("transformer_2 bf16 (low, t=400)", pipe_pred_low_bf16, train_pred_low_2, atol=1e-4)
        all_pass = all_pass and ok

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 50)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
