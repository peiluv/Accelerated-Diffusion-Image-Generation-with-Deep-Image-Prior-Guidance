# -*- coding: utf-8 -*-
# integration.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from .ddpm import DDPM
from .dip import DeepImagePrior

from config import get_config
args = get_config(from_cli=False)

class DIP_Guided_DDPM(nn.Module):
    """Integration of Deep Image Prior and DDPM"""
    def __init__(self, ddpm, dip, prior_weight, args):
        super().__init__()
        self.ddpm = ddpm
        self.dip = dip
        self.prior_weight = prior_weight
        self.args = args

    def generate_dip_prior(self, target_shape, dip_train_steps, reg_noise_std=args.dip_reg_noise_std, device='cuda'):
        """Generate a prior using DIP for DDPM initialization"""
        print(f"Generating DIP prior with {dip_train_steps} training steps...")

        '''
        - 訓練步數可控（dip_train_steps）
        - 可重複利用，無需每 batch 都重新訓練
        '''

        # 建立一張 dummy target image
        dummy_image = torch.randn(target_shape, device=device)

        # Generate prior using DIP
        dip_prior = self.dip.generate_prior(
            target_image=dummy_image,
            iterations=dip_train_steps,
            reg_noise_std=args.dip_reg_noise_std,
            device=device
        )

        # Normalize prior to [-1, 1] range for DDPM
        dip_prior = 2 * (dip_prior - dip_prior.min()) / (dip_prior.max() - dip_prior.min() + 1e-8) - 1

        return dip_prior


    def generate_batch_dip_prior(self, target_images, dip_train_steps, reg_noise_std=args.dip_reg_noise_std, device='cuda'):
        """
        動態針對 batch 中每張 target image 產生 DIP prior
        target_images: [B, C, H, W]
        回傳: [B, C, H, W]
        """
        dip_priors = []
        for i in range(target_images.shape[0]):
            img = target_images[i:i+1]  # [1, C, H, W]
            dip_prior = self.dip.generate_prior(
                target_image=img,
                iterations=dip_train_steps,
                reg_noise_std=args.dip_reg_noise_std,
                device=device
            )
            dip_priors.append(dip_prior)
        return torch.cat(dip_priors, dim=0)


    @torch.no_grad() # for integrated model
    def sample(self, batch_size, image_size, channels, device="cuda",
            dip_train_steps=None, ddpm_steps=None, use_dip_prior=True, dip_prior=None, measure_sampling_time=False):
        '''
        從 DIP prior + DDPM p_sample_loop() 得到圖像
        - 純 DDPM：從高斯 noise 開始 (use_dip_prior=False)
        - integrated：從 DIP prior + 隨機 noise 加權組合
        '''
        shape = (batch_size, channels, image_size, image_size)

        # --- 根據 ddpm_steps 準備 start_step ---
        start_step = None
        steps_info = f"{self.ddpm.n_steps} (Full)"
        if ddpm_steps is not None and ddpm_steps < self.ddpm.n_steps:
            # 如果指定了較少的步數，則計算子集時間步
            step_interval = max(1, self.ddpm.n_steps // ddpm_steps)
            # 從 0 開始，以 step_interval 為間隔取步數
            start_step = list(range(0, self.ddpm.n_steps, step_interval))
            # 確保步數數量盡可能接近 ddpm_steps，並包含最後一步
            if self.ddpm.n_steps - 1 not in start_step:
                start_step.append(self.ddpm.n_steps - 1)
            # 按需截斷或調整
            # 可以直接使用 linspace 生成指定數量的點，再轉為整數，確保降序
            # 例如：使用 linspace 生成 ddpm_steps + 1 個點，從 n_steps-1 到 0
            ts = torch.linspace(self.ddpm.n_steps - 1, 0, ddpm_steps + 1).round().to(torch.long)
            start_step = sorted(list(set(ts.tolist())), reverse=True) # 去重並保證降序
            steps_info = f"{len(start_step)} (Specified: {ddpm_steps})"
        else:
            # 使用完整步數
            start_step = None # p_sample_loop 會使用預設 range

        mode_info = "DIP-Guided" if use_dip_prior else "Standard DDPM (within Integrated)"
        print(f"\n[{mode_info} Sampling]: DDPM Steps: {steps_info}, Prior Weight: {self.prior_weight if use_dip_prior else 'N/A'}")
        if use_dip_prior:
            print(f"\n -> Using DIP Prior (Steps: {dip_train_steps or args.dip_train_steps or 'Default'})")

        if use_dip_prior:
            # --- DIP Prior  ---
            if dip_prior is None:
                # 如果沒有提供 dip_prior，生成一個基於隨機目標的
                print("\n[Warning] No DIP prior provided, generating one based on random target.")
                dip_prior = self.generate_dip_prior(
                    target_shape=(1, channels, image_size, image_size),
                    dip_train_steps=dip_train_steps or args.dip_train_steps,
                    reg_noise_std=args.dip_reg_noise_std,
                    device=device)

            # 確保 dip_prior 批次大小匹配
            if dip_prior.shape[0] != batch_size:
                if dip_prior.shape[0] == 1:
                    dip_prior = dip_prior.repeat(batch_size, 1, 1, 1)
                else:
                    print(f"\n[Warning] DIP prior batch size ({dip_prior.shape[0]}) mismatch ({batch_size}).")
                    dip_prior = dip_prior[:batch_size] # 簡單截斷

            random_noise = torch.randn(shape, device=device)
            combined_prior = self.prior_weight * dip_prior + (1 - self.prior_weight) * random_noise
            initial_noise = combined_prior
        else:
            # Baseline DDPM case (or when explicitly not using prior)
            initial_noise = torch.randn(shape, device=device) # 從純噪聲開始

        # --- 開始計時 ---
        sampling_time = None
        if measure_sampling_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

        # ---  DDPM sampling  ---
        samples = self.ddpm.p_sample_loop(shape, device, noise=initial_noise, start_step=start_step)
        # --- 結束計時 ---
        if measure_sampling_time:
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize(device=device)
                sampling_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                end_time = time.time()
                sampling_time = end_time - start_time # second

        if measure_sampling_time:
            return samples, sampling_time
        else:
            return samples