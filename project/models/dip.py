# -*- coding: utf-8 -*-
# dip.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DModel


from config import get_config
args = get_config(from_cli=False)

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2
    UNet 的基本 building block，對輸入的 feature 進行初步的特徵提取和轉換，改變 C 並引入非線性"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True), # BatchNorm2d 改為 InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True), # BatchNorm2d 改為 InstanceNorm2d
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv ->  U-Net 的 encoder"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 用 2x2 的 Pooling 和 stride 2，將 feature map 的寬度和高度都減半
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv ->  U-Net 的 decoder"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels -> 雙線性插值不改變 C
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 在 bilinear 模式下，需要額外的卷積層來調整通道數
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust the dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        '''
        torch.cat -> Skip-connection：
        將 Up-Sampeling 後的特徵圖 (x1) 與來自 Encoder 路徑中對應層的、具有更高分辨率的特徵圖 (x2) 在通道維度上連接起來
        這樣 Decoder 就能從 Encoder 中提取細節信息，有助於生成更精細的輸出
        '''
        return self.conv(x)

class OutConv(nn.Module):
    """U-Net 的輸出層 -> 將 U-Net 最終的特徵表示映射到所需的輸出圖像空間"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 1x1 kernel_size 可以用於改變 feature map 的 C，而不會改變其 W 和 H

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet architecture for DIP"""
    def __init__(self, input_channels, output_channels, hidden_size, bilinear=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear
        self.factor = 2 ** 5  # 因為有 5 個 downsampling：256 -> 8

        # Encoder
        self.inc = DoubleConv(input_channels, hidden_size)              # 3  -> 64
        self.down1 = Down(hidden_size, hidden_size * 2)                # 64 -> 128
        self.down2 = Down(hidden_size * 2, hidden_size * 4)            # 128 -> 256
        self.down3 = Down(hidden_size * 4, hidden_size * 8)            # 256 -> 512
        self.down4 = Down(hidden_size * 8, hidden_size * 16)           # 512 -> 1024

        # Decoder
        self.up1 = Up(hidden_size * 16 + hidden_size * 8, hidden_size * 8, bilinear)   # 1024+512=1536 -> 512
        self.up2 = Up(hidden_size * 8 + hidden_size * 4, hidden_size * 4, bilinear)    # 512+256=768  -> 256
        self.up3 = Up(hidden_size * 4 + hidden_size * 2, hidden_size * 2, bilinear)    # 256+128=384  -> 128
        self.up4 = Up(hidden_size * 2 + hidden_size, hidden_size, bilinear)            # 128+64=192   -> 64

        self.outc = OutConv(hidden_size, output_channels)

    def forward(self, x):
        x1 = self.inc(x)   # 64 @ 256
        x2 = self.down1(x1)  # 128 @ 128
        x3 = self.down2(x2)  # 256 @ 64
        x4 = self.down3(x3)  # 512 @ 32
        x5 = self.down4(x4)  # 1024 @ 16

        x = self.up1(x5, x4)  # -> 512 @ 32
        x = self.up2(x, x3)   # -> 256 @ 64
        x = self.up3(x, x2)   # -> 128 @ 128
        x = self.up4(x, x1)   # -> 64 @ 256
        return self.outc(x)


class HuggingFace_UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dip_model = UNet2DModel(
            sample_size=64,
            in_channels=args.dip_in_channels,
            out_channels=args.channels,
            layers_per_block=args.dip_num_res_blocks,         # 每個 block 2 個 ResNet 層
            block_out_channels=(256, 256, 256, 512, 1024),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
            act_fn="silu",
            norm_num_groups=32,         # GroupNorm（32是主流設計）
        )

class DeepImagePrior(nn.Module):
    """Deep Image Prior model wrapper"""
    def __init__(self, args):
        super(DeepImagePrior, self).__init__()
        self.args = args
        self.output_channels = args.channels
        self.net = HuggingFace_UNet(args)
        '''
        in_channels：RGB
        output_channels：RGB
        hidden_size：控制神經網路內部隱藏層的大小，影響模型的容量和複雜度
        '''

    def forward(self, noise):
        """通過神經網路將 noise 轉換為 image"""
        out = self.net.dip_model(noise, timestep=0).sample
        return out

    def get_random_input(self, target_shape, device='cuda'):
        """
        Generate random input noise for the DIP model
        """
        B, C, H, W = target_shape
        return torch.randn(B, args.dip_in_channels, H, W, device=device)

    def generate_prior(self, target_image, iterations, reg_noise_std=args.dip_reg_noise_std, device='cuda', optimizer_class=torch.optim.Adam, criterion=torch.nn.MSELoss()):
        self.eval()

        # print(f"[DIP] Generate-Prior target_image shape: {target_image.shape}")

        iterations = iterations if iterations is not None else args.dip_train_steps
        reg_noise_std = reg_noise_std if reg_noise_std is not None else args.dip_reg_noise_std

        target_image = target_image.to(device)

        input_noise = torch.randn(1, args.dip_in_channels, *target_image.shape[2:],
                                    device=device, requires_grad=True)

        optimizer = optimizer_class([input_noise], lr=getattr(self, 'lr', self.args.dip_lr))
        best_output = None
        best_loss = float('inf')

        with torch.enable_grad():
            # progress_bar = tqdm(range(iterations), desc="DIP Prior Optimization")
            progress_bar = range(iterations)

            for i in progress_bar:
                optimizer.zero_grad()

                if reg_noise_std > 0:
                    reg_noise = torch.randn_like(input_noise) * reg_noise_std
                    input_noise_perturbed = input_noise + reg_noise
                else:
                    input_noise_perturbed = input_noise

                # 前向傳播 DIP U-net
                output = self.net.dip_model(sample=input_noise_perturbed, timestep=0).sample
                loss = criterion(output, target_image)
                total_loss = loss

                # 反向傳播
                total_loss.backward()

                # 更新 input_noise
                optimizer.step()

                #
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    # 複製最佳輸出，並分離計算圖
                    best_output = output.detach().clone()

                # progress_bar.set_postfix({
                #     "loss": f"{current_loss:.4f}"
                # })

                # --- Early stopping 邏輯 ---
                if getattr(self.args, 'dip_early_stop', False) and \
                i > getattr(self.args, 'dip_early_stop_patience', 50) and \
                current_loss > 1.1 * best_loss:
                    print(f"\n[DIP] Early stopping triggered at iteration {i+1}")
                    break

                # --- 清理記憶體 ---
                # del loss, output, total_loss
                # torch.cuda.empty_cache()

        # --- 確保至少有一個輸出 ---
        if best_output is None:
            # 如果從未改善（例如只迭代1次或loss為NaN），則使用最後一次的結果
            with torch.no_grad(): # 在 no_grad 下執行最後一次前向傳播
                if reg_noise_std > 0: # 確保使用與迴圈中一致的噪聲
                    reg_noise = torch.randn_like(input_noise) * reg_noise_std
                    input_noise_perturbed = input_noise + reg_noise
                else:
                    input_noise_perturbed = input_noise
                best_output = self.net(input_noise_perturbed).detach().clone()

        # --- [-1, 1] ---
        best_output = torch.clamp(best_output, -1, 1)
        return best_output