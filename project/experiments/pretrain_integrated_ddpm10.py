# pretrain_integrated_ddpm.py (1.0 prior weight)

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import sys
sys.path.append('/kaggle/input/ddpm-dip-project/ddpm_dip_project')

import os
import torch
from config import get_config
from utils.dataset import load_dataset
from models.ddpm import UNet, DDPM
from models.dip import DeepImagePrior
from train import train_integrated_model, zip_output_dir
from utils.visualization import save_images
from utils.metrics import load_png_as_tensor

def pretrain_integrated_ddpm_10(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    print(f"[INFO] Using device: {device}")

    # CUDA memory management
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.8'
        print(f"[INFO] CUDA memory management initialized")

    print("[INFO] Loading dataset...")
    train_loader, val_loader, _ = load_dataset(args)

    # Load pretrained DIP
    dip = DeepImagePrior(args).to(args.device)
    if os.path.exists(args.dip_save_path):
        print(f"[INFO] Loading pretrained DIP model.")
        dip.load_state_dict(torch.load(args.dip_save_path, map_location=args.device))
    else:
        raise FileNotFoundError(f"Pretrained DIP model not found.")

    print(f"\n[INFO] DIP prior_weight -> {args.prior_weight} ===")


    if args.prior_path:
        prior_path = args.prior_path
        dip_prior = load_png_as_tensor(prior_path, device)
        print(f"\n[INFO] Loading DIP prior ===")
        print(f'DIP prior tensor shape: {dip_prior.shape}, device: {dip_prior.device}')

    else:
        # 取一張資料集圖像
        torch.manual_seed(args.seed)

        example_batch = next(iter(train_loader))[0]
        batch_size, channels, height, width = example_batch.shape
        target_shape = (1, channels, height, width)
        target_image = torch.randn(target_shape, device=args.device) # random noise

        dip_prior = dip.generate_prior(
            target_image=target_image,
            iterations=args.dip_train_steps,
            reg_noise_std=args.dip_reg_noise_std,
            device=args.device
        )

        # ===  save DIP prior ===

        save_dir = os.path.join(args.output_dir, "dip_prior")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dip_prior_10.png")

        if isinstance(dip_prior, torch.Tensor):
            dip_prior_np = dip_prior.detach().cpu().numpy()

        else:
            dip_prior_np = dip_prior

        if dip_prior_np.ndim == 3: # batch image
            dip_prior_np = dip_prior_np[None, ...]

        save_images(dip_prior_np, save_path, nrow=1)
        print(f"[INFO] DIP prior image saved to {save_path}")

    # === Train DIP-guided DDPM ===
    print(f"\n[INFO] Start DIP-Guided DDPM training for {args.epochs} epochs... (prior_weight={args.prior_weight})")
    train_integrated_model(args, train_loader, val_loader, f"{args.prior_weight}", dip, dip_prior=dip_prior)

    # backup
    zip_output_dir(args.output_dir, zip_name="integrated_ddpm_10.zip")

    print(f"[INFO] Done for training integrated ddpm with prior weight={args.prior_weight}.")

if __name__ == '__main__':
    args = get_config(from_cli=True)
    pretrain_integrated_ddpm_10(args)
