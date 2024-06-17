# -*- coding: utf-8 -*-
# train.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from functools import partial
import math
from tqdm import tqdm
import shutil
from utils.visualization import save_images
from utils.metrics import calculate_generation_metrics
from experiments.evaluate import Evaluator
from models.ddpm import DDPM, UNet
from models.dip import DeepImagePrior
from models.integration import DIP_Guided_DDPM
from models.ema import EMA

from config import get_config

args = get_config(from_cli=False)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, losses, fids, lpips, ema, tag=""):
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'losses': losses,
        'fids': fids,
        'lpips':lpips,
        'ema_state_dict': ema.ema_model.state_dict(),
    }

    path = f"{args.output_dir}/checkpoints/{tag}.pt"
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved to {path}")


def zip_output_dir(output_dir, zip_name="results_backup.zip"):
    shutil.make_archive(zip_name.replace(".zip", ""), 'zip', output_dir)
    print(f"[INFO] Output directory zipped as {zip_name}")


def lr_lambda(args, num_training_steps, current_step):

    num_warmup_steps = args.num_warmup_steps

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_baseline_ddpm(args, train_loader, val_loader, model):
    """Train baseline DDPM model without DIP integration"""
    '''
    訓練標準 DDPM（無 DIP）baseline model
    模型：`UNet` ➤ `DDPM` class()
    '''
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    total_loss = []
    total_fid = []
    total_lpips = []
    best_fid = float('inf')
    patience = args.ddpm_early_stop_patience
    patience_counter = 0

    # Set up warmup steps
    lr = args.lr
    num_training_steps = args.epochs * len(train_loader)
    lr_lambda_partial = partial(lr_lambda, args, num_training_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_partial)
    ema = EMA(model, decay=0.999)
    '''
    現代 diffusion model 主流都採用這類 scheduler:
    Warmup 防止初期梯度爆炸，提升穩定性 +
    Cosine decay 平滑降低學習率，防止後期 loss 震盪，提升生成細節
    '''
    # 初始化混合精度訓練
    # scaler = GradScaler(device='cuda')
    scaler = torch.cuda.amp.GradScaler()
    # 梯度累積步數
    gradient_accumulation_steps = args.gradient_accumulation_steps

    if args.resume_baseline_ddpm_checkpoint and os.path.exists(args.resume_baseline_ddpm_checkpoint):
        print(f"[INFO] Resuming from checkpoint: {args.resume_baseline_ddpm_checkpoint}")

        checkpoint = torch.load(args.resume_baseline_ddpm_checkpoint, map_location=args.device)

        for key in checkpoint['model_state_dict']:
            if checkpoint['model_state_dict'][key].is_floating_point():
                checkpoint['model_state_dict'][key] = checkpoint['model_state_dict'][key].to(dtype=torch.float16)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        total_loss = checkpoint.get('losses', [])
        total_fid = checkpoint.get('fids', [])
        total_lpips = checkpoint.get('lpips', [])
    else:
        print("[INFO] Training from scratch.")
        start_epoch = 0
        total_loss = []
        total_fid = []
        total_lpips = []

    print(f"[INFO] Start Baseline DDPM training from {start_epoch} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # Training
        epoch_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(args.device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                t = torch.randint(0, args.ddpm_steps, (images.size(0),), device=args.device)
                loss = model.training_losses(images, t)

            # 反向傳播與梯度累積
            scaler.scale(loss).backward()

            epoch_loss += loss.item() / gradient_accumulation_steps

            # 梯度累積條件判斷
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪與參數更新
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        total_loss.append(avg_loss)
        # pbar.set_postfix({
        #     "epoch": f"{epoch+1}/{args.epochs}",
        #     })

        # empty non-used tensor
        del loss, t
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (epoch + 1) % 5 == 0:
            generate_samples(ema.ema_model, args, epoch=epoch+1, mode="baseline")

        # Validating & Early Stopping
        if (epoch+1) % 100 == 0:
            model.eval()
            val_metrics = validate_model(args, ema.ema_model, val_loader, epoch, num_fid_samples=args.num_fid_samples, mode="baseline")
            current_fid = val_metrics['fid']
            current_lpips = val_metrics.get('lpips', float('nan'))
            total_fid.append(current_fid)
            total_lpips.append(current_lpips)

            print(f"Epoch {epoch+1}/{args.epochs:<2} Train Loss: {avg_loss:<.3f}  |  Val FID: {current_fid:<.3f}\n")

            if current_fid < best_fid:
                best_fid = current_fid
                patience_counter = 0
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_loss, total_fid , total_lpips, ema, tag="pretrained_baseline_ddpm")

            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}.")

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        else:
            print(f"Epoch {epoch+1}/{args.epochs:<2} Train Loss: {avg_loss:<.3f}")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_loss, total_fid , total_lpips, ema, tag="pretrained_baseline_ddpm")

    return model, total_loss, total_fid, total_lpips


def train_integrated_model(args, train_loader, val_loader, mode=f"{args.prior_weight}", pretrained_dip=None, dip_prior=None):
    """Train the integrated DDPM-DIP model"""
    '''
    Step 1. | 初始化：UNet → DDPM、DIP → 整合為 `DIP_Guided_DDPM` |
    Step 2. | 使用 DIP prior 進行整合訓練 |
    '''
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize UNet for DDPM model
    unet = UNet(
        in_channels=args.channels,
        out_channels=args.channels,
        hidden_size=args.ddpm_hidden_size,
        time_dim=256,
        num_res_blocks=args.ddpm_num_res_blocks,
        attention_resolutions=args.ddpm_attention_resolutions,
        dropout=args.ddpm_dropout
    ).to(args.device)

    # Initialize DDPM
    ddpm = DDPM(
        unet=unet,
        args=args
    )

    # Initialize DIP model
    dip = pretrained_dip if pretrained_dip is not None else DeepImagePrior(args).to(args.device)

    # Initialize integrated model
    integrated_model = DIP_Guided_DDPM(
        ddpm=ddpm,
        dip=dip,
        prior_weight=args.prior_weight,
        args=args
    )

    if dip_prior is None:
        raise ValueError("dip_prior must be provided for integrated model training")

    # Get example batch for shape inference
    example_batch = next(iter(train_loader))[0]
    batch_size, channels, height, width = example_batch.shape

    # Training loop
    total_loss = []
    total_fid = []
    total_lpips = []
    best_fid = float('inf')
    patience = args.ddpm_early_stop_patience
    patience_counter = 0

    # Set up warmup steps
    lr=args.lr
    num_training_steps = args.epochs * len(train_loader)
    lr_lambda_partial = partial(lr_lambda, args, num_training_steps)
    optimizer = torch.optim.Adam(integrated_model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_partial)
    ema = EMA(integrated_model, decay=0.999)
    '''
    現代 diffusion model 主流都採用這類 scheduler:
    Warmup 防止初期梯度爆炸，提升穩定性 +
    Cosine decay 平滑降低學習率，防止後期 loss 震盪，提升生成細節
    '''
    # 初始化混合精度訓練
    scaler = GradScaler()

    # 梯度累積步數
    gradient_accumulation_steps = args.gradient_accumulation_steps

    if args.resume_integrated_ddpm_checkpoint and os.path.exists(args.resume_integrated_ddpm_checkpoint):
        print(f"[INFO] Resuming from checkpoint: {args.resume_integrated_ddpm_checkpoint}")

        checkpoint = torch.load(args.resume_integrated_ddpm_checkpoint, map_location=args.device)

        integrated_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        total_loss = checkpoint.get('losses', [])
        total_fid = checkpoint.get('fids', [])
        total_lpips = checkpoint.get('lpips', [])
    else:
        print("[INFO] Training from scratch.")
        start_epoch = 0
        total_loss = []
        total_fid = []
        total_lpips = []

    print(f"[INFO] Start DIP-Guieded DDPM training from {start_epoch} epochs...")
    for epoch in range(start_epoch, args.epochs):

        # Training
        epoch_loss = 0.0
        integrated_model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Training Integrated DDPM")
        for batch_idx, (images, _) in enumerate(pbar):
        # for batch_idx, (images, _) in enumerate(train_loader):
            # clean CUDA cache
            # if batch_idx % 10 == 0 and torch.cuda.is_available():
                # torch.cuda.empty_cache()

            images = images.to(args.device)
            batch_size = images.shape[0]

            # 動態切片
            dip_prior_batch = integrated_model.generate_batch_dip_prior(
                target_images=images,
                dip_train_steps=args.dip_train_steps,
                reg_noise_std=args.dip_reg_noise_std,
                device=args.device
            )

            # 使用混合精度訓練
            if torch.cuda.is_available() and scaler is not None:
                with autocast(dtype=torch.bfloat16):
                    # Sample random timesteps
                    t = torch.randint(0, ddpm.n_steps, (images.shape[0],), device=args.device)

                    # Create a blend of random noise and DIP prior for initialization
                    random_noise = torch.randn_like(images)
                    noise = args.prior_weight * dip_prior_batch + (1 - args.prior_weight) * random_noise

                    # Compute loss
                    loss = ddpm.training_losses(images, t, noise=noise)
            else:
                # Sample random timesteps
                t = torch.randint(0, ddpm.n_steps, (images.shape[0],), device=args.device)

                # Create a blend of random noise and DIP prior for initialization
                random_noise = torch.randn_like(images)
                noise = args.prior_weight * dip_prior_batch + (1 - args.prior_weight) * random_noise

                # Compute loss
                loss = ddpm.training_losses(images, t, noise=noise)


            # === detect NaN  ===
            if torch.isnan(loss).any():
                print(f"[ERROR] NaN detected in loss at batch {batch_idx}. Terminating training.")
                break

            # 使用混合精度訓練進行反向傳播
            if torch.cuda.is_available() and scaler is not None:
                # AMP mode
                scaler.scale(loss).backward() # 反向傳播（自動縮放梯度）
                scaler.unscale_(optimizer)  # 必須先 unscale 梯度才能裁剪
                torch.nn.utils.clip_grad_norm_(integrated_model.parameters(), max_norm=args.clip_grad) # 裁剪未縮放的梯度
                scaler.step(optimizer)  # 更新參數（內部會跳過包含 inf/NaN 的梯度）
                scaler.update()         # 更新 scaler（調整縮放因子）
                optimizer.zero_grad()
            else:
                # 非 AMP mode
                loss.backward()
                torch.nn.utils.clip_grad_norm_(integrated_model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            # 後續步驟（EMA、scheduler）
            ema.update(integrated_model)
            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({
                "epoch": f"{epoch+1}/{args.epochs}",
                })

            # empty non-used tensor
            del loss, t, random_noise, noise
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_loss = epoch_loss / len(train_loader)
        total_loss.append(avg_loss)

        # Validating & Early Stopping
        if (epoch+1) % 100 == 0:
            integrated_model.eval()
            val_metrics = validate_model(args, ema.ema_model, val_loader, epoch, args.num_fid_samples, mode="integrated", dip_prior=dip_prior)
            current_fid = val_metrics['fid']
            current_lpips = val_metrics.get('lpips', float('nan'))
            total_fid.append(current_fid)
            total_lpips.append(current_lpips)

            print(f"Epoch {epoch+1}/{args.epochs:<2} Train Loss: {avg_loss:<.3f}  |  Val FID: {current_fid:<.3f}\n")

            if current_fid < best_fid:
                best_fid = current_fid
                patience_counter = 0
                save_checkpoint(integrated_model, optimizer, scheduler, scaler, epoch, total_loss, total_fid, total_lpips, ema, tag=f"integrated_ddpm{mode}")

            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}.")

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        else:
            print(f"Epoch {epoch+1}/{args.epochs:<2} Train Loss: {avg_loss:<.3f}")
            save_checkpoint(integrated_model, optimizer, scheduler, scaler, epoch, total_loss, total_fid, total_lpips, ema, tag=f"integrated_ddpm{mode}")

    return integrated_model, total_loss, total_fid, total_lpips

def validate_model(args, model, val_loader, epoch, num_fid_samples, mode="integrated", dip_prior=None):
    model.eval()
    metrics = {'fid': float('nan'), 'lpips': float('nan')}
    all_real = []
    all_fake = []
    n_collected = 0

    try:
        for real_batch, _ in val_loader:
            if n_collected >= num_fid_samples:
                break

            real_batch = real_batch.to(args.device)
            batch_size = real_batch.shape[0]

            with torch.no_grad():
                if mode == "integrated":
                    samples = model.sample(
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        channels=args.channels,
                        device=args.device,
                        dip_train_steps=args.dip_train_steps,
                        ddpm_steps=100,
                        use_dip_prior=True,
                        dip_prior=dip_prior
                    )
                elif mode == "baseline":
                    samples = model.sample(
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        channels=args.channels,
                        device=args.device)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            all_real.append(real_batch.detach().cpu())
            all_fake.append(samples.detach().cpu())
            n_collected += batch_size

        # 合併所有 samples，僅取前 num_fid_samples
        real_images = torch.cat(all_real, dim=0)[:num_fid_samples].to(args.device)
        generated_images = torch.cat(all_fake, dim=0)[:num_fid_samples].to(args.device)

        # 計算 FID、LPIPS
        metrics = calculate_generation_metrics(real_images, generated_images, args.device)

        # --- 只保存第一個 batch 的 generation samples ---
        samples_dir = f"{args.output_dir}/samples/{mode}"
        os.makedirs(samples_dir, exist_ok=True)
        epoch_str = f"_epoch{epoch+1}" if epoch is not None else ""
        save_path = f"{samples_dir}/samples{epoch_str}.png"

        # 取 all_fake 的第一個 batch
        first_fake_batch = all_fake[0]
        # Denormalize from [-1, 1] to [0, 1]
        first_fake_batch = (first_fake_batch + 1) / 2

        # Convert to numpy and save
        samples_np = first_fake_batch.cpu().numpy()
        save_images(samples_np, save_path, nrow=4)

    except Exception as e:
        print(f"Validating Error: {e}")

    return metrics

def generate_samples(model, args,  epoch, mode="integrated", dip_prior=None):
    """Generate samples using the model"""
    '''
    產生圖片樣本並儲存為 `.png`
        - 使用 integrated model 或 baseline 模型的 `.sample(...)`
        - 輸出會自動轉為 `[0, 1]` 範圍並存圖
        - 可用來做訓練過程觀察或生成比較
    '''
    model.eval()
    samples = None
    try:
        # Generate samples
        with torch.no_grad():
            if mode == "integrated":
                # Generate samples using the integrated model
                samples = model.sample(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    channels=args.channels,
                    device=args.device,
                    dip_train_steps=args.dip_train_steps,
                    ddpm_steps=None, # None -> same as baseline DDPM
                    use_dip_prior=False, # -> same as baseline DDPM
                    dip_prior=dip_prior # Pass the pre-generated prior
                )
            elif mode == "baseline":
                # Generate samples using the baseline DDPM model
                samples = model.sample(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    channels=args.channels,
                    device=args.device
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

        samples_dir = f"{args.output_dir}/samples/{mode}"
        os.makedirs(samples_dir, exist_ok=True)

        # Save samples
        save_path = f"{samples_dir}/{epoch}_samples.png"

        samples = (samples + 1) / 2
        samples_np = samples.cpu().numpy()
        save_images(samples_np, save_path, nrow=4)

    except Exception as e:
        print(f"Sample Generation Error: {e}")
        return None

    return samples
