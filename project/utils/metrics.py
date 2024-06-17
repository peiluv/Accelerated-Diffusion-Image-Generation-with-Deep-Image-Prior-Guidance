# -*- coding: utf-8 -*-
# metrics.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import warnings

# Filter torchvision and lpips warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global LPIPS model cache
_lpips_model = None

from config import get_config
args = get_config(from_cli=False)

def load_png_as_tensor(image_path, device='cuda'):
    # read image -> RGB
    pil_image = Image.open(image_path).convert('RGB')

    # -> to numpy array, shape -> (H, W, C) [0,255]
    np_image = np.array(pil_image).astype(np.float32) / 255.0  # 轉為 [0,1]
    # -> to tensor, shape (C, H, W)
    tensor_image = torch.from_numpy(np_image).permute(2, 0, 1)
    # -> normalize [-1, 1]
    tensor_image = tensor_image * 2 - 1
    # -> add batch shape
    tensor_image = tensor_image.unsqueeze(0).to(device)  # shape: [1, 3, H, W]
    return tensor_image


def calculate_psnr(img1, img2):
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    psnr_values = []
    for i in range(img1_np.shape[0]):
        data_range = img1_np[i].max() - img1_np[i].min()
        psnr_values.append(psnr(
            img1_np[i].transpose(1, 2, 0),
            img2_np[i].transpose(1, 2, 0),
            data_range=data_range
        ))
    return np.mean(psnr_values)

def calculate_ssim(img1, img2):
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    ssim_values = []
    for i in range(img1_np.shape[0]):
        data_range = img1_np[i].max() - img1_np[i].min()
        ssim_values.append(ssim(
            img1_np[i].transpose(1, 2, 0),
            img2_np[i].transpose(1, 2, 0),
            channel_axis=2,
            data_range=data_range
        ))
    return np.mean(ssim_values)

def calculate_lpips(img1, img2, device='cuda'):
    global _lpips_model
    if _lpips_model is None:
        print("Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]")
        _lpips_model = lpips.LPIPS(net='alex').to(device)
    # 只對 [0,1] 範圍影像做轉換
    def to_neg11(x):
        return x * 2 - 1 if x.min() >= 0 and x.max() <= 1 else x
    img1_norm = to_neg11(img1)
    img2_norm = to_neg11(img2)
    lpips_val = _lpips_model(img1_norm, img2_norm).mean().item()
    return lpips_val

def calculate_fid(real_images, generated_images, device='cuda'):
    """
    計算 FID 分數，輸入需為 [N, 3, H, W]，像素值範圍 [-1, 1] 或 [0, 1]
    會自動將資料轉到指定 device
    """
    # 確保圖片在 [0, 1]（torchmetrics 要求）
    def to_01(img):
        if img.min() < 0:  # [-1, 1] -> [0, 1]
            return (img + 1) / 2
        return img

    real_images = to_01(real_images).to(device)
    generated_images = to_01(generated_images).to(device)

    # torchmetrics FID 需要至少兩張圖像
    if real_images.shape[0] < 2 or generated_images.shape[0] < 2:
        raise ValueError("FID計算至少需要2張real和2張generated圖像")

    # 初始化 FID 指標
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    # 累積 real/fake 圖像特徵
    fid_metric.update(real_images, real=True)
    fid_metric.update(generated_images, real=False)

    fid_score = fid_metric.compute()
    return fid_score.item()

def calculate_reconstruction_metrics(original, reconstructed, device='cuda'):
    """Calculate all reconstruction metrics between original and reconstructed images"""
    metrics = {
        'psnr': calculate_psnr(original, reconstructed),
        'ssim': calculate_ssim(original, reconstructed),
        'lpips': calculate_lpips(original, reconstructed, device)
    }
    return metrics

def calculate_generation_metrics(real_samples, generated_samples, device='cuda'):
    """Calculate all generation metrics between real and generated samples"""
    metrics = {
        # 'fid': calculate_fid(real_samples, generated_samples, device),
        'lpips': calculate_lpips(real_samples, generated_samples, device)
    }
    return metrics

def compute_diffusion_step_metrics(model, image, steps, device='cuda'):
    """
    Compute quality metrics at different diffusion steps
    to analyze the impact of DIP prior on diffusion process
    """
    metrics = {'steps': [], 'psnr': [], 'ssim': [], 'lpips': []}

    # Ensure image is on the correct device
    image = image.to(device)

    # Normalize to [-1, 1] if needed
    if image.min() >= 0 and image.max() <= 1:
        image_normalized = 2 * image - 1
    else:
        image_normalized = image

    # For each diffusion step
    for step in steps:
        # Create a batch of the same timestep
        t = torch.full((image.shape[0],), step, device=device, dtype=torch.long)

        # Add noise according to the diffusion forward process
        x_t, _ = model.q_sample(image_normalized, t)

        # Try to reconstruct from this step
        reconstructed = model.p_sample_loop(image.shape, device, noise=x_t, start_step=step)

        # Compute metrics
        psnr_val = calculate_psnr(image, reconstructed)
        ssim_val = calculate_ssim(image, reconstructed)
        lpips_val = calculate_lpips(image, reconstructed, device)

        # Store results
        metrics['steps'].append(step)
        metrics['psnr'].append(psnr_val)
        metrics['ssim'].append(ssim_val)
        metrics['lpips'].append(lpips_val)

    return metrics