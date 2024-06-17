# -*- coding: utf-8 -*-
# main.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import sys
extract_dir = "/kaggle/input/ddpm-dip-project/ddpm_dip_project"
sys.path.append(extract_dir)

import os
import torch
import matplotlib.pyplot as plt
from config import get_config

# Import modules from project structure
from models.ddpm import UNet, DDPM
from models.dip import DeepImagePrior
from models.integration import DIP_Guided_DDPM
from experiments.evaluate import Evaluator
from experiments.train import zip_output_dir
from utils.dataset import load_dataset
from utils.visualization import plt_training_curve
from models.ema import EMA
import time

def main():
    args = get_config(from_cli=True)
    device = args.device
    print(f"[INFO] Using device: {device}")

    # CUDA memory management
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.8'
        print(f"[INFO] CUDA memory management initialized")

    # Load Dataset
    print(f"[INFO] Loading Dataset...")
    _, val_loader, test_loader = load_dataset(args)
    print(f"[INFO] Test dataset size: {len(test_loader)}")

    # Load pretrained DIP
    dip = DeepImagePrior(args).to(args.device)
    if os.path.exists(args.dip_save_path):
        print(f"[INFO] Loading pretrained DIP model.")
        dip.load_state_dict(torch.load(args.dip_save_path, map_location=args.device))
    else:
        raise FileNotFoundError(f"Pretrained DIP model not found.")

    # Initialize U-Net for DDPM model
    unet = UNet(
        in_channels=args.channels,
        out_channels=args.channels,
        hidden_size=args.ddpm_hidden_size,
        time_dim=args.ddpm_time_dim,
        num_res_blocks=args.ddpm_num_res_blocks,
        attention_resolutions=args.ddpm_attention_resolutions,
        dropout=args.ddpm_dropout
    ).to(args.device)

    # # Initialize DDPM
    ddpm = DDPM(unet=unet, args=args)
    baseline_ddpm = ddpm
    baseline_ema = EMA(baseline_ddpm, decay=0.999)

    # Load pretrained baseline DDPM weights
    if os.path.exists(args.baseline_ddpm_save_path):
        print(f"[INFO] Loading pretrained Baseline DDPM model.")
        baseline_checkpoint = torch.load(args.baseline_ddpm_save_path, map_location=args.device)

        # baseline_ddpm.load_state_dict(baseline_checkpoint['model_state_dict'])
        epoch = baseline_checkpoint['epoch'] + 1
        baseline_losses = baseline_checkpoint.get('losses', [])
        baseline_fids = baseline_checkpoint.get('fids', [])
        baseline_ema.ema_model.load_state_dict(baseline_checkpoint['ema_state_dict'])
        print(f"[INFO] Loaded pretrained Baseline DDPM model for {epoch} epochs.")
    else:
        raise FileNotFoundError(f"Pretrained Baseline DDPM model not found.")

    # Load pretrained DIP-Guided DDPM weights（1.0）
    # Initialize integrated model
    integrated_ddpm = DIP_Guided_DDPM(
        ddpm=baseline_ema.ema_model,
        dip=dip,
        prior_weight=args.prior_weight,
        args=args
    )

    # integrated_ema = EMA(integrated_ddpm, decay=0.999)

    # if os.path.exists(args.integrated_ddpm_save_path):
    #     print(f"[INFO] Loading pretrained Integrated DDPM model.")
    #     checkpoint = torch.load(args.integrated_ddpm_save_path, map_location='cpu')

    #     # integrated_ddpm.load_state_dict(checkpoint['model_state_dict'])
    #     integrated_losses = checkpoint.get('losses', [])
    #     integrated_fids = checkpoint.get('fids', [])
    #     integrated_ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])

    # else:
    #     raise FileNotFoundError(f"Pretrained Integrated DDPM model not found.")

    # === Plot Training Curves ===
    # print(f"\n[INFO] Plotting training curves...")
    # save_dir = os.path.join(args.output_dir, 'training_curves.png')

    # plt_training_curve(
    #     baseline_losses=baseline_losses,
    #     save_dir=save_dir,
    #     epochs=epoch
    # )

    # # === Initialize Evaluator ===
    evaluator = Evaluator(args) # Pass args to evaluator


    # # --- Calculate Baseline Metrics ---
    # print("\n" + "="*20 + " Evaluating Baseline Model " + "="*20)
    # baseline_results = evaluator.calculate_standard_metrics(
    #     model=baseline_ema.ema_model,
    #     dataloader=test_loader,
    #     num_fid_samples=args.num_fid_samples,
    #     evaluation_batch_size=args.evaluation_batch_size,
    #     mode="baseline"
    # )
    # print(f"-----------------------------------------------------")
    # print(f"[RESULT] Baseline DDPM Standard Metrics (Samples: {args.num_fid_samples}):")
    # print(f"  FID  : {baseline_results['fid']:.4f} (Lower is better)")
    # print(f"  LPIPS: {baseline_results['lpips']:.4f} (Lower is better)")
    # print(f"  PSNR : {baseline_results['psnr']:.4f} (Higher is better)")
    # print(f"  SSIM : {baseline_results['ssim']:.4f} (Higher is better)")
    # print(f"  Single Sample Time: {baseline_results.get('single_sample_time_baseline', float('nan')):.4f} seconds")
    # print(f"-----------------------------------------------------")

    # --- Calculate Integrated Metrics ---
    integrated_results = None
    if integrated_ddpm is not None:
        print("\n" + "="*20 + " Evaluating Integrated Model " + "="*20)
        integrated_results = evaluator.calculate_standard_metrics(
            model=integrated_ddpm,
            dataloader=test_loader,
            num_fid_samples=args.num_fid_samples,
            evaluation_batch_size=args.evaluation_batch_size,
            mode="integrated"
        )
        print(f"-----------------------------------------------------")
        print(f"[RESULT] Integrated DDPM Standard Metrics (Samples: {args.num_fid_samples}):")
        print(f"  FID  : {integrated_results['fid']:.4f} (Lower is better)")
        print(f"  LPIPS: {integrated_results['lpips']:.4f} (Lower is better)")
        print(f"  PSNR : {integrated_results['psnr']:.4f} (Higher is better)")
        print(f"  SSIM : {integrated_results['ssim']:.4f} (Higher is better)")
        print(f"  Single Sample Time: {integrated_results.get('single_sample_time_integrated', float('nan')):.4f} seconds")
        print(f"-----------------------------------------------------")
    else:
        print("\n[INFO] Skipping Integrated DDPM Standard Metrics calculation as the model was not loaded.")

    # === Plot Single Sample Comparison (if both models were evaluated) ===
    # if baseline_results is not None and integrated_results is not None:
    #     print("\n[INFO] Plotting comparison of saved single high-quality samples...")
    #     evaluator.plot_single_sample_comparison(baseline_results, integrated_results)

    # # === Done and Save (Keep Zip) ===
    print("\n[INFO] Zipping output directory...")
    zip_output_dir(args.output_dir, zip_name="evaluation_output.zip") # Changed zip name slightly
    print(f"[INFO] Evaluation script finished.")

if __name__ == '__main__':
    main()