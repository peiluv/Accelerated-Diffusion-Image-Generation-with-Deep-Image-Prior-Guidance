# -*- coding: utf-8 -*-
# config.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import argparse
import torch

def get_config(from_cli=True):
    parser = argparse.ArgumentParser(description='DDPM-DIP Integration Configuration')

    # General settings
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--num_fid_samples', type=int, default=1024, help='sample for validation')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_warmup_steps', type=int, default=2000)
    parser.add_argument('--dip_prior_type', type=str, default='noise', choices=['noise', 'data'], help='DIP prior source : noise or data')
    parser.add_argument('--prior_weight', type=float, default=1.0)
    parser.add_argument('--integrated_eval_steps', type=int, default=200,
                        help='Number of DDPM steps for single sample evaluation in integrated mode. Default uses 200.')

    # DatasetS
    parser.add_argument('--evaluation_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8) # 8 -> 32
    parser.add_argument('--image_size', type=int, default=64) # 64 -> 32
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/celebahq-resized-256x256/')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--channels', type=int, default=3)

    # DDPM
    parser.add_argument('--ddpm_early_stop_patience', type=int, default=5)
    parser.add_argument('--ddpm_steps', type=int, default=1000)
    parser.add_argument('--ddpm_beta_start', type=float, default=1e-4)
    parser.add_argument('--ddpm_beta_end', type=float, default=0.02)
    parser.add_argument('--ddpm_beta_schedule', type=str, default='cosine', choices=['linear', 'quadratic', 'cosine'])
    parser.add_argument('--ddpm_hidden_size', type=int, default=128)
    parser.add_argument('--ddpm_time_dim', type=int, default=256)
    parser.add_argument('--ddpm_num_res_blocks', type=int, default=3)
    parser.add_argument('--ddpm_attention_resolutions', nargs='+', type=int, default=[8,16])
    parser.add_argument('--ddpm_dropout', type=float, default=0.1)

    # DIP
    parser.add_argument('--dip_reg_noise_std', type=float, default=0.05)
    parser.add_argument('--dip_train_steps', type=int, default=100)
    parser.add_argument('--dip_net_type', type=str, default='skip', choices=['skip'])
    parser.add_argument('--dip_in_channels', type=int, default=3)
    parser.add_argument('--dip_hidden_size', type=int, default=128)
    parser.add_argument('--dip_num_res_blocks', type=int, default=2)
    parser.add_argument('--dip_lr', type=float, default=1e-5)
    parser.add_argument('--dip_early_stop', action='store_true', default=True, help='Enable early stopping for DIP')
    parser.add_argument('--dip_early_stop_patience', type=int, default=50, help='Patience for DIP early stopping')

    # pre-train model
    parser.add_argument('--load_dip', action='store_true', default=True, help='Load pretrained DIP model')
    parser.add_argument('--dip_save_path', type=str, default='/kaggle/input/pretrained_dip/pytorch/default/1/pretrained_dip_64.pt', help='Path to DIP checkpoint')
    parser.add_argument('--baseline_ddpm_save_path', type=str, default='/kaggle/input/pretrained_baseline_ddpm/pytorch/default/1/pretrained_baseline_ddpm.pt')
    parser.add_argument('--integrated_ddpm_save_path', type=str, default='/kaggle/input/integrated_ddpm1.0/pytorch/default/1/integrated_ddpm1.0.pt')
    parser.add_argument('--resume_baseline_ddpm_checkpoint', type=str, default='/kaggle/input/pretrained_baseline_ddpm/pytorch/default/1/pretrained_baseline_ddpm.pt')
    parser.add_argument('--resume_integrated_ddpm_checkpoint', type=str, default='')
    parser.add_argument('--prior_path', type=str, default='/kaggle/input/dip-prior-05/dip_prior_05.png')

    return parser.parse_args() if from_cli else parser.parse_args([])