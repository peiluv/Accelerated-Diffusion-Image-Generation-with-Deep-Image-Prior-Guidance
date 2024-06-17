# pretrain_dip.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image
from tqdm import tqdm
import sys
sys.path.append('/kaggle/input/ddpm-dip-project/ddpm_dip_project')

from models.dip import DeepImagePrior
from utils.dataset import load_dataset
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import show_images
from experiments.train import zip_output_dir
from config import get_config

def pretrain_dip(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print("[INFO] Loading dataset...")
    train_loader, val_loader, test_loader = load_dataset(args)

    target_image, _ = next(iter(train_loader))
    target_image = target_image.to(device)

    print("[INFO] Initializing DIP model...")
    dip = DeepImagePrior(args).to(device)
    dip.train()

    optimizer = torch.optim.Adam(dip.parameters(), lr=args.dip_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.dip_train_steps)

    input_noise = dip.get_random_input(target_image.shape, device)

    best_loss = float('inf')
    best_output = None
    total_loss = []
    total_psnr = []
    total_ssim = []
    denoised_images = []

    print(f"[INFO] Start DIP training for {args.dip_train_steps} iterations...")
    pbar = tqdm(range(args.dip_train_steps), desc="Pre-training PID: ")

    for i in pbar:
        if args.dip_reg_noise_std > 0:
            perturbed_input = input_noise + args.dip_reg_noise_std * torch.randn_like(input_noise)
        else:
            perturbed_input = input_noise

        output = dip(perturbed_input)

        # Downsample target image to match DIP output size
        target_resized = F.interpolate(target_image, size=output.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.mse_loss(output, target_resized)

        optimizer.zero_grad()
        loss.backward()
        psnr_val = calculate_psnr(output, target_resized)
        ssim_val = calculate_ssim(output, target_resized)
        optimizer.step()
        scheduler.step()

        if i % 20 == 0:
            # output, target_resized: [B, 3, H, W]，範圍 [-1,1]
            pred_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
            denoised_images.append(pred_np)

            print(f"\n[Step {i}] PSNR: {psnr_val:<.4f}  |  SSIM: {ssim_val:<.4f}")

            plt.figure(figsize=(8,4))

            plt.subplot(1,2,1)
            img = output.permute(0,2,3,1).detach().cpu().numpy()[0]
            img = (img / 2 + 0.5).clip(0, 1)
            plt.imshow(img)
            plt.title(f'Iter {i}')
            plt.axis('off')
            plt.subplot(1,2,2)
            tgt = target_resized.permute(0,2,3,1).cpu().numpy()[0]
            tgt = (tgt / 2 + 0.5).clip(0, 1)
            plt.imshow(tgt)
            plt.axis('off')
            plt.tight_layout()
            path = os.path.join(args.output_dir, f'iter_{i}.png')
            plt.savefig(path)
            plt.show()
            plt.close()

        current_loss = loss.item()
        total_loss.append(current_loss)
        total_psnr.append(psnr_val)
        total_ssim.append(ssim_val)

        if current_loss < best_loss:
            best_loss = current_loss
            best_output = output.detach().clone()

        if i > args.dip_early_stop_patience and current_loss > 1.1 * best_loss:
            print(f"\n[Early Stopping] Stopped at step {i} with best loss {best_loss:.3f}")
            break

        pbar.set_postfix({
            "loss": loss.item(),
            })

    save_path = os.path.join(args.output_dir, "pretrained_dip.pt")
    torch.save(dip.state_dict(), save_path)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(list(range(len(total_loss))), total_loss, label='DIP Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    ax1.legend(loc=0)


    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(list(range(len(total_loss))), total_psnr, 'b-', label='PSNR')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("PSNR")
    ax3=ax2.twinx()
    ax3.plot(list(range(len(total_loss))), total_ssim, 'r-', label='SSIM')
    ax2.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.5)
    ax3.set_ylabel("SSIM")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc=0)

    image_path = os.path.join(args.output_dir, 'pretrain_dip_curve.png')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()  # save memory

    show_images(denoised_images)
    zip_output_dir(args.output_dir, zip_name="pretrain_dip.zip")

    # # display
    # if os.path.exists(image_path):
    #     try:
    #         img = matplotlib.image.imread(image_path)
    #         plt.figure(figsize=(10, 6))
    #         plt.imshow(img)
    #         plt.axis('off')
    #         plt.show()
    #         print(f"[INFO] Displayed image from {image_path}")

    #     except Exception as e:
    #         print(f"[ERROR] Failed to display image: {e}")
    # else:
    #     print(f"[ERROR] Image file not found at {image_path}")

if __name__ == '__main__':
    args = get_config(from_cli=True)
    pretrain_dip(args)