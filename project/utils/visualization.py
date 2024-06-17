# -*- coding: utf-8 -*-
# visualization.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils
import torchvision

from config import get_config
args = get_config(from_cli=False)

def unnormalize(tensor):
    """將影像從 [-1, 1] 還原到 [0, 1] 範圍"""
    return (tensor + 1) / 2

def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array for visualization"""
    tensor = tensor.detach().cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    np_array = tensor.numpy()
    np_array = np.transpose(np_array, (0, 2, 3, 1))
    if np_array.min() < 0:
        np_array = (np_array + 1) / 2
    return np.clip(np_array, 0, 1)

def save_images(images, filename, nrow=args.batch_size // 4):
    """
    images: torch.Tensor 或 np.ndarray，範圍 [-1,1] 或 [0,1]
    """
    # 若為 numpy，先轉成 Tensor
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    # 將 [-1,1] 轉到 [0,1]
    if images.min() < 0:
        images = (images + 1) / 2
    # make grid 並儲存
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    torchvision.utils.save_image(grid, filename)


def plot_time_curve(time_data, title="Inference Time Comparison", save_path=None):
    """Plot a line chart comparing inference times of two models"""
    iterations = list(range(1, len(time_data['baseline']) + 1))
    plt.figure(figsize=(8, 5))

    plt.plot(iterations, time_data['baseline'], label="Baseline DDPM", marker='o')
    plt.plot(iterations, time_data['integrated'], label="Integrated Model", marker='s')

    plt.xlabel("Generation Iteration")
    plt.ylabel("Inference Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plt_training_curve(baseline_losses, save_dir, epochs):
    """
    training curve：Loss、FID Lpips
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))

    # loss
    # plt.subplot(1, 2, 1)
    plt.plot(range(1,191), baseline_losses, label='Baseline DDPM')
    # plt.plot(epochs, integrated_losses, label='DIP-Guided DDPM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    # FID
    # plt.subplot(1, 2, 2)
    # plt.plot(metric_epochs, baseline_fids, 'b-', label='Baseline FID')
    # plt.plot(metric_epochs, integrated_fids, 'r-', label='DIP-Guided FID')
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.title('Metrics Curves')
    # plt.legend()
    # plt.grid(True)

    # save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Training Curves.png'))
    plt.close()

    print(f"[INFO] Training Curves saved to: {save_dir}")


def plot_time_comparison_boxplot(time_data, title="Inference Time Comparison", save_path=None):
    """
    畫出兩個模型的推論時間分布箱型圖
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 6))
    data = [time_data['baseline'], time_data['integrated']]
    labels = ['Baseline DDPM', 'DIP-Guided DDPM']
    plt.boxplot(data, labels=labels, showmeans=True, meanline=True)
    plt.ylabel("Inference Time (seconds)")
    plt.title(title)
    plt.grid(True, axis='y')

    # label mean
    means = [np.mean(times) for times in data]
    for i, mean in enumerate(means):
        plt.text(i + 1, mean, f"{mean:.3f}s", ha='center', va='bottom', fontsize=10, color='red')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_image(img1, img2, title1='Baseline', title2='Integrated', save_path=None):
    """
    將兩張圖像左右並排展示，適合單張 baseline/integrated 產圖品質比較
    img1, img2: torch.Tensor or numpy.ndarray, shape [C, H, W] or [H, W, C]
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu()
        if img1.dim() == 4:
            img1 = img1[0]
        img1 = (img1 + 1) / 2
        img1 = img1.permute(1, 2, 0).numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu()
        if img2.dim() == 4:
            img2 = img2[0]
        img2 = (img2 + 1) / 2
        img2 = img2.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(img1, 0, 1))
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(img2, 0, 1))
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def show_images(images):
    """Shows the provided images as sub-pictures in a square"""
    fig = plt.figure(figsize=(20, 8))
    rows = 1
    cols = round(len(images) / rows)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                ax.imshow(images[idx])
                ax.axis('off')
            idx += 1
    image_path = os.path.join(args.output_dir, 'dip_images.png')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)  # 明確釋放該 Figure，避免記憶體堆積
