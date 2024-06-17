# -*- coding: utf-8 -*-
# pretrain_baseline_ddpm.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import sys
sys.path.append('/kaggle/input/ddpm-dip-project/ddpm_dip_project')

import os
import torch
from torch.cuda.amp import GradScaler, autocast
from config import get_config
from utils.dataset import load_dataset
from models.ddpm import UNet, DDPM
from train import train_baseline_ddpm, zip_output_dir, validate_model

def pretrain_baseline_ddpm(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    print(f"[INFO] Using device: {device}")

    # CUDA 記憶體管理進階配置
    if device.startswith('cuda') and torch.cuda.is_available():
        # 取得裝置記憶體總量
        total_mem = torch.cuda.get_device_properties(0).total_memory

        # 動態設定分割閾值 (根據顯卡世代調整)
        if total_mem > 16 * 1024**3:  # 16GB+ 顯卡
            max_split_size = 128
            phase_ratio = 0.95
        else:  # 低於16GB
            max_split_size = 64
            phase_ratio = 0.85

        # 配置環境變數 (需在首次 CUDA 操作前設定)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size},expandable_segments:True,garbage_collection_threshold:0.9'

        # 分階段記憶體限制策略
        try:
            # 實驗性 API (PyTorch 1.12+)
            torch.cuda.set_per_process_memory_fraction(
                phase_ratio,
                device=torch.cuda.current_device()
            )
            torch._C._cuda_setMemoryFractionAdvanced(True)
        except AttributeError:
            print("[WARN] 當前 PyTorch 版本不支援進階記憶體管理")

        # 啟用 TF32 加速 (Ampere+ 架構)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 延遲初始化快取清理
        def delayed_cache_clear():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        delayed_cache_clear()

        print(f"[OPT] CUDA 記憶體優化啟用 (max_split={max_split_size}MB)")


    print("[INFO] Loading dataset...")
    train_loader, val_loader, _ = load_dataset(args)

    print("[INFO] Initializing Baseline DDPM model...")
    # initialize
    unet = UNet(
        in_channels=args.channels,
        out_channels=args.channels,
        hidden_size=args.ddpm_hidden_size,
        time_dim=args.ddpm_time_dim,
        num_res_blocks=args.ddpm_num_res_blocks,
        attention_resolutions=args.ddpm_attention_resolutions,
        dropout=args.ddpm_dropout
    ).to(args.device)

    ddpm = DDPM(unet=unet, args=args)
    ddpm = ddpm.to(device, memory_format=torch.channels_last)
    train_baseline_ddpm(args, train_loader, val_loader, ddpm)

    # backup
    zip_output_dir(args.output_dir, zip_name="baseline_ddpm.zip")

if __name__ == '__main__':
    args = get_config(from_cli=True)
    pretrain_baseline_ddpm(args)
