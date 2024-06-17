# -*- coding: utf-8 -*-
# ddpm.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast

from config import get_config
args = get_config(from_cli=False)

def get_num_groups(channels, max_groups=8):
    for num_groups in reversed(range(1, max_groups + 1)):
        if channels % num_groups == 0:
            return num_groups
    return 1  # fallback to LayerNorm

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        '''
       【目的】：將標量的時間步 `t` 轉換成一個固定維度的向量表示，為 U-Net 提供一種方式來區分不同的時間步
       【運作原理】：
            1.【創建頻率】：計算一系列基於對數尺度均勻分佈的不同頻率
            2.【生成 embeddings】：將輸入的時間步與這些頻率相乘，得到不同頻率下的角度
            3.【正弦和餘弦】：對這些角度計算 sin 和 cos
            4.【拼接 torch.cat】：將相同頻率的 sin 和 cos 拼接在一起，形成最終的 time embeddings
            5.【Output】：一個形狀為 `(batch_size, dim)` 的 tensor，表示每個時間步的 embeddings
       【作用】：這種 embedding 方式在 Transformer 模型中被廣泛使用，因為它可以有效地表示序列中元素的位置信息
        '''

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff    = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # (B, C, H, W) → (B, N, C)
        x_seq = x.view(b, c, -1).permute(0, 2, 1)

        # 使用記憶體高效注意力
        # 1) Attention with Pre‐Norm
        y = self.norm1(x_seq)
        if hasattr(F, 'scaled_dot_product_attention'):  # PyTorch 2.0+ 支持
            q = k = v = y
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            y, _ = self.attn(y, y, y)

        x_seq = x_seq + y

        # 2) Feed‐Forward with Pre‐Norm
        y = self.norm2(x_seq)
        y = self.ff(y)
        x_seq = x_seq + y

        # (B, N, C) → (B, C, H, W)
        return x_seq.permute(0, 2, 1).view(b, c, h, w)
        '''
        `forward(self, x)`：
            【`x`】: 輸入 feature (batch_size, channels, height, width)

       【運作原理】：
            1.【flatten and transpose】：將 `x` 展平成一個 Seq，形狀為`(batch_size, height * width, channels)`
            2.【Multihead-Attention】：輸入的 `x` 同時作為查詢 (query)、鍵 (key) 和值 (value)
            3.【殘差連接 (`+ self.mha(...)`)】：將 Attention 的輸出加回到原始的 `x` 上（殘差連接）
            4.【LayerNor】：對殘差連接後的結果進行 LayerNor
            5.【前饋網路 (`self.ff_self`)】：通過一個包含兩層 Linear 和 GELU 的前饋網路進一步處理特徵
            6.【再次殘差連接 (`+ self.ff_self(...)`) 和層歸一化】：將前饋網路的輸出加回到注意力層的輸出上，再次進行 layerNor
            7.【重塑】：將處理後的序列重新塑造成原始的特徵圖形狀 `(batch_size, channels, height, width)`
            8.【Output】：經過 Self-Attention 處理後的 feature map

       【作用】：允許模型在處理圖像的每個位置時，考慮到所有其他位置的信息，從而更好地理解圖像的全局結構和上下文
        '''

class ResidualBlock(nn.Module):
    """Residual block with optional attention
    【目的】：在 Block 的基礎上增加了殘差連接和 optional attention，使其更適合構建深層網路並處理更複雜的依賴關係"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, attention=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(h)
        h = h + self.shortcut(x)
        h = self.attention(h)
        return h
        '''
       【運作原理】：
            1.【Conv1】：對輸入 `x` 進行 GroupNorm -> SiLU -> Conv2d
            2.【融入 time embeddings】：將 `t` 通過一個線性層轉換後，加到第一個卷積的輸出上（在空間維度上擴展）
            3.【Conv2】：對融合了時間資訊的 feature map 進行 GroupNorm -> SiLU -> Dropout -> Conv2d
            4.【殘差連接】：如果輸入和輸出 C 不同，則使用一個 1x1 卷積對輸入 `x` 進行通道數調整；否則，直接使用原始輸入
                - 將調整後的輸入加到第二個卷積的輸出上（殘差連接）
            5.【self-attention】：如果 `attention` 為 `True`，則對殘差連接後的結果應用 `SelfAttention`
           6. 【Output】：經過殘差連接和可選注意力處理後的 feature map
       【作用】：`ResidualBlock` 是 U-Net 中的主要特徵處理單元
           - 利用殘差連接促進梯度流動，使用 time embeddings 感知去噪進度，並可選地使用 Self-Attention 來捕獲空間上的長距離依賴
        '''

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, time_dim,
                 num_res_blocks, attention_resolutions, dropout):
        super().__init__()
        '''
        【in_channels、out_channels】：通常兩者 C 相同
        【time_dim (time embedding)】：
            - DDPM 中，時間步 t 會被 embed 成一個向量，作為 U-Net 的額外輸入，
            以告知模型當前處於擴散/去噪過程的哪個階段。

        【hidden_size】：U-Net 中間層數，控制網路的容量
        【num_res_blocks：在每個 Down-Sampeling 和 Up-Sampeling 使用的 ResidualBlock 的數量
            - ResidualBlock 是現代 CNN 中常用的結構，有助於訓練更深的網路

        【attention_resolutions】：一個元組，指定在哪些空間分辨率下使用注意力機制
            - attention 機制可以幫助模型捕獲圖像中的長距離依賴關係
            - 例如：(8, 16) 表示在特徵圖尺寸為 8x8 和 16x16 時使用 attention

        【dropout】：正則化，Dropout 概率
        '''

        """【Time embedding】-> 將時間步 t embed 成一個適合 U-Net 各個 ResidualBlock 使用的向量 """
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)

        # --- 準備記錄 encoder 通道與 decoder channels ---
        curr_channels = hidden_size
        num_downs = int(np.log2(args.image_size)) - 2
        # channels 用來記錄每次下採樣後的通道，第一層即 hidden_size
        channels = [curr_channels]
        self.downs = nn.ModuleList()
        self.encoder_channels = []
        self.resolutions = []        # 記錄每層的 feature map 大小

        # Encoder
        for i in range(num_downs):
            res = args.image_size // (2 ** i)
            self.resolutions.append(res)      # 記錄每層 feature map 大小
            # 1) 殘差塊群
            blocks = nn.ModuleList([
                ResidualBlock(curr_channels, curr_channels,
                              time_dim, dropout,
                              attention=(res in attention_resolutions))
                for _ in range(num_res_blocks)
            ])
            self.downs.append(blocks)
            self.encoder_channels.append(curr_channels)

            # 2) 下採樣 (除最後一層)
            if i < num_downs - 1:
                self.downs.append(
                    nn.Conv2d(curr_channels, curr_channels * 2,
                              kernel_size=4, stride=2, padding=1)
                )
                curr_channels *= 2
                channels.append(curr_channels)

        # Middle
        self.mid = nn.ModuleList([
            ResidualBlock(curr_channels, curr_channels, time_dim, dropout, True),
            ResidualBlock(curr_channels, curr_channels, time_dim, dropout, False)
        ])

        # Decoder（nearest‐neighbor upsampling + 卷積）
        self.ups = nn.ModuleList()
        dec_chs = channels[::-1]
        skip_chs = self.encoder_channels[::-1]  # 正確對應 Encoder 記錄
        curr_c = dec_chs[0]
        skip_idx = 0
        for i, res in enumerate(self.resolutions[::-1]):
            use_attn = res in attention_resolutions
            # 每層先拼接 skip, 再過 ResidualBlock
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                if j == 0:
                    in_c = curr_c + skip_chs[skip_idx]
                    skip_idx += 1
                else:
                    in_c = curr_c
                out_c = dec_chs[i]
                blocks.append(ResidualBlock(in_c, out_c, time_dim, dropout, attention=use_attn))
                curr_c = out_c
            self.ups.append(blocks)

            # 上採樣（最後一層不做）
            if i < len(self.resolutions) - 1:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(curr_c, dec_chs[i+1], kernel_size=3, padding=1)
                ))
                curr_c = dec_chs[i+1]

        self.conv_out = nn.Sequential(
            nn.GroupNorm(get_num_groups(curr_c), curr_c),
            nn.SiLU(),
            nn.Conv2d(curr_c, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        '''
        【`x`】：輸入的帶噪聲圖像 (`x_t`)。
        【`t`】：當前的時間步（一個批量的時間步）。
        【運作流程】：
            1. 【Time embedding】：將輸入的時間步 `t` 轉換為向量
            2. 【Input】：輸入圖像 `x` 通過 `self.conv_in` 得到初始特徵 `h`
            3. 【Encoder】：`h` 依次通過 `self.downs` 中 -> ResidualBlock -> Down-Sampling
                - 每個 ResidualBlock 都會以 `h` 和 `t` 作為輸入。
                - Encoder 每一層的輸出都會被保存在 `residuals` list 中，用於 Skip-connection
            4. 【Middle】：`h` 通過 `self.mid` 中的 ResidualBlock
            5. 【Decoder】：`h` 依次通過 `self.ups` 中 -> Up-Sampling -> ResidualBlock
                - 在每個 ResidualBlock 之前，`h` 會與 `residuals` list 中對應 Encoder 輸出進行 C 連接
            6. 【Final layer】：最終的 feature map `h` 通過 `self.conv_out` 得到預測的 noise
            7. 【Output】：返回預測的 noise tensor
        '''
        t = t.float()
        t = self.time_mlp(t)
        h = self.conv_in(x)
        residuals = []

        # Encoder
        for module in self.downs:
            if isinstance(module, nn.ModuleList):
                for block in module:
                    h = block(h, t)
                residuals.append(h)
            else:
                h = module(h)

        # Middle
        for res_block in self.mid:
            h = res_block(h, t)

        # Decoder
        for module in self.ups:
            if isinstance(module, nn.ModuleList):
                for idx, res_block in enumerate(module):
                    if idx == 0:
                        skip = residuals.pop()

                        # spatial size 對齊
                        if h.shape[2:] != skip.shape[2:]:
                            skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
                            '''
                            參考主流 UNet/DDPM 設計:
                            Stable Diffusion、HuggingFace Diffusers、OpenAI Improved DDPM 等主流 diffusion 實作
                            都會在 decoder concat skip 之前，確保 spatial size 完全一致，否則就 interpolate skip 使其對齊
                            '''
                        h = torch.cat([h, skip], dim=1)
                    h = res_block(h, t)
            else:
                h = module(h)  # upsample
        return self.conv_out(h)


class DDPM(nn.Module):
    def __init__(self, unet, args):
        super().__init__()
        self.unet = unet
        self.n_steps = args.ddpm_steps # 1000

        beta_start = args.ddpm_beta_start
        beta_end = args.ddpm_beta_end
        beta_schedule = args.ddpm_beta_schedule

        # Define beta schedule -> 選 linear 通常沒問題
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, self.n_steps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.n_steps) ** 2
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = self.n_steps + 1
            x = torch.linspace(0, self.n_steps, steps)
            alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        '''
        beta_schedule：定義 β 在 T 個時間步上的變化方式
        beta_start, beta_end：noise schedule 中 β 的起始和結束值，β 控制每一步添加到圖像中的噪聲量
        n_steps：擴散過程的總步數 T，noise 會逐步添加到圖像中 T 次。
        '''

        """內部預計算，可以提高效率"""
        # Precompute values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)


    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process (adding noise to the image)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        '''
        運作原理：
        根據 DDPM 的論文，在時間步 t，
        帶噪聲的圖像 x_t 可以通過原始圖像 x_0 和一個與累積的 alphas 相關縮放因子以及一個與 1- alphas 相關的噪聲項直接計算得到
        這個方法就是利用這個公式來快速得到任意時間步的帶噪聲圖像
        '''
        # Get alphas for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Compute noisy image at timestep t
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


    def _extract(self, a, t, shape):
        """Extract coefficients at specified timesteps t and reshape to match input shape"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)


    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise


    def q_posterior(self, x_0, x_t, t):
        """Compute parameters for posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        # Compute mean for posterior
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t

        # Compute variance for posterior
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """Predict mean and variance for reverse process p(x_{t-1} | x_t)"""

        '''
        p_mean_variance(...)：反向去噪過程的核心
            - 使用訓練好的 unet 模型來預測在時間步 t 需要去除的噪聲，
            基於這個預測的噪聲，估計前一個時間步的圖像 x_{t-1} 的 mean 和 variance
        '''
        # Predict noise using model
        predicted_noise = self.unet(x_t, t)

        # Predict x_0 from x_t and predicted noise
        x_0_predicted = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Clip predicted x_0 if requested
        if clip_denoised:
            x_0_predicted = torch.clamp(x_0_predicted, -1., 1.)

        # Compute parameters for posterior q(x_{t-1} | x_t, x_0)
        model_mean, model_variance, model_log_variance = self.q_posterior(
            x_0_predicted, x_t, t
        )

        return model_mean, model_variance, model_log_variance

    def p_sample(self, x_t, t, clip_denoised=True):
        """Sample from reverse process p(x_{t-1} | x_t)"""

        '''
        p_sample(...) 運作原理：
            1. 獲取 mean 和 variance：使用 `p_mean_variance` 預測當前時間步 `x_t` 的 denoise mean 和 variance
            2. 採樣 noise：如果當前時間步 `t > 0`，則從標準高斯分佈中採樣一個噪聲張量，在最後一步（`t = 0`），因為要得到最終的生成圖像，通常不添加噪聲
            3. 計算 x_{t-1}：將預測的 mean 加上與預測 variance 和採樣噪聲相關的項，得到去噪後的圖像 `x_{t-1}`
        '''
        # Get mean and variance for p(x_{t-1} | x_t)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(
            x_t, t, clip_denoised
        )

        # per-sample mask: 只有 t>0 才加 noise
        mask  = (t > 0).view(-1, *[1]*(x_t.dim()-1)).float()
        noise = torch.randn_like(x_t) * mask

        # Only add noise if t > 0
        variance = torch.exp(0.5 * model_log_variance)
        return model_mean + variance * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device, noise=None, start_step=None):
        """Run the entire reverse process to generate samples 純噪聲開始，逐步去除噪聲，最终生成圖像"""
        img = torch.randn(shape, device=device) if noise is None else noise

        # 嚴格按降序處理
        timesteps = start_step if start_step is not None else range(self.n_steps-1, -1, -1)

        for i in timesteps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img


    @torch.no_grad() # for integrated model
    def sample(self, batch_size, image_size, channels, device="cuda",
            dip_train_steps=None, ddpm_steps=None, use_dip_prior=True, dip_prior=None):
        '''
        從 DIP prior + DDPM p_sample_loop() 得到圖像
        - 純 DDPM：從高斯 noise 開始 (實際上此模式下 use_dip_prior 應為 False)
        - integrated：從 DIP prior + 隨機 noise 加權組合（`prior_weight` 控制）
        '''
        shape = (batch_size, channels, image_size, image_size)

        # --- 根據 ddpm_steps 準備 start_step ---
        start_step = None
        if ddpm_steps is not None and ddpm_steps < self.n_steps:
            # 如果指定了較少的步數，則計算子集時間步
            step_interval = max(1, self.n_steps // ddpm_steps)
            start_step = list(range(0, self.n_steps, step_interval))
            # 確保包含最後一個時間步 (或接近的步數)，並確保數量接近 ddpm_steps
            # 這裡的邏輯可以參考 ddpm.py sample 方法的實現以保持一致
            if self.n_steps - 1 not in start_step:
                start_step.append(self.n_steps - 1) # 添加最後一步
            # 可能需要截斷以確保步數不超過 ddpm_steps
            if len(start_step) > ddpm_steps:
                start_step = start_step[:ddpm_steps]

            start_step = sorted(list(set(start_step)), reverse=True) # 去重、排序（必須降序）
            print(f"\n[INFO] Sampling using {len(start_step)} custom timesteps.")
        else:
            # 如果 ddpm_steps 為 None 或 >= n_steps，則使用完整步數
            start_step = None # p_sample_loop 會使用預設 range
            print("\n[INFO] Sampling using full timesteps.")

        if use_dip_prior:
            print(f"\n[DIP-Guided Sampling]: DIP Steps: {dip_train_steps or 'N/A'}, DDPM Steps: {ddpm_steps or self.n_steps}, Prior Weight: {args.prior_weight}")

            if dip_prior is None:
                # 如果沒有提供 dip_prior，根據設定生成一個
                print("\n[Warning] No DIP prior provided to integrated sample, generating one based on random target.")
                dip_prior = self.generate_dip_prior(
                    target_shape=(1, channels, image_size, image_size), # 注意：生成隨機目標
                    dip_train_steps=dip_train_steps or args.dip_train_steps,
                    reg_noise_std=args.dip_reg_noise_std,
                    device=device)

            # 確保 dip_prior 批次大小匹配
            if dip_prior.shape[0] != batch_size:
                if dip_prior.shape[0] == 1:
                    dip_prior = dip_prior.repeat(batch_size, 1, 1, 1) # 如果 prior 只有 1 個，則重複
                else:
                    # 其他情況（例如 prior > 1 但不等於 batch_size）可能需要報錯或更複雜的處理
                    print(f"\n[Warning] DIP prior batch size ({dip_prior.shape[0]}) mismatch with required batch size ({batch_size}). Trying to adapt.")
                    dip_prior = dip_prior[:batch_size] # 簡單截斷，可能不是最佳方案

            random_noise = torch.randn(shape, device=device)
            combined_prior = args.prior_weight * dip_prior + (1 - args.prior_weight) * random_noise

            samples = self.p_sample_loop(shape, device, noise=combined_prior, start_step=start_step)
        else:
            samples = self.p_sample_loop(shape, device, start_step=start_step)

        return samples


    def training_losses(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion to get x_t
        x_t, noise = self.q_sample(x_0, t, noise=noise)

        # 使用正確的 autocast 上下文
        with autocast(dtype=torch.float16):
            predicted_noise = self.unet(x_t, t)

        # Predict noise
        predicted_noise = self.unet(x_t, t)

        # 1. 原始逐像素MSE損失 (使用reduction='none'保留維度)
        pixel_loss = F.mse_loss(predicted_noise, noise, reduction='none')

        # 2. 計算噪聲統計量損失
        noise_std_loss = F.mse_loss(
            predicted_noise.std(dim=(1,2,3)), noise.std(dim=(1,2,3)))
        noise_mean_loss = F.mse_loss(
            predicted_noise.mean(dim=(1,2,3)), noise.mean(dim=(1,2,3)))

        # 3. 整合損失 (可調整權重係數)
        loss = pixel_loss.mean() + 0.1 * noise_std_loss + 0.1 * noise_mean_loss

        return loss


    def forward(self, x, t=None):
        """Forward pass (used for prediction during training)"""
        if t is None:
            # Sample random timesteps for batch
            t = torch.randint(0, self.n_steps, (x.shape[0],), device=x.device)

        # Return predicted noise
        return self.unet(x, t)