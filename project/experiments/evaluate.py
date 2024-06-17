# evaluate.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from inspect import signature
import time

# --- Metrics and Visualization Imports ---
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.visualization import save_images, unnormalize, plot_image
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips

from config import get_config
args = get_config(from_cli=False)

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        # Ensure output directories exist
        self.samples_dir = os.path.join(self.args.output_dir, 'evaluation_samples')
        os.makedirs(self.samples_dir, exist_ok=True)

    # --- Measure Single Sample Time & Save ---
    def measure_and_save_single_sample(self, model, mode, real_image_for_prior=None):
        """
        Generates ONE sample, measures time, saves the image.
        For 'integrated' mode, uses specified reduced steps (200) and measures ONLY the DDPM sampling time if possible.
        For 'baseline' mode, uses full steps and measures the entire sample call time.

        Args:
            model: The model (EMA usually) to use for sampling.
            mode: 'baseline' or 'integrated'.
            real_image_for_prior: A single real image tensor [1, C, H, W] needed for integrated mode.

        Returns:
            tuple: (inference_time, saved_image_path, generated_sample_tensor)
        """
        model.eval()
        print(f"Measuring single sample inference time and generating sample for mode: {mode}...")

        # Prepare inputs for sample function
        batch_size = 1
        image_size = args.image_size
        channels = args.channels
        device = self.device

        # --- sample_kwargs and DIP Prior ---
        # Start with basic kwargs, steps will be added conditionally
        sample_kwargs = {'batch_size': batch_size, 'image_size': image_size, 'channels': channels, 'device': device}
        dip_prior = None
        dip_train_steps_for_sample = None # 存儲 dip_train_steps

        if mode == "integrated":
            if real_image_for_prior is None:
                raise ValueError("real_image_for_prior is required for integrated mode single sample timing.")

            # Generate DIP prior for this single image (在計時之前完成)
            print(" -> Generating DIP prior (outside timing)...")
            if hasattr(model, 'generate_batch_dip_prior'):
                dip_prior = model.generate_batch_dip_prior(
                    target_images=real_image_for_prior.to(self.device),
                    dip_train_steps=self.args.dip_train_steps, # 使用 self.args
                    reg_noise_std=self.args.dip_reg_noise_std, # 使用 self.args
                    device=self.device )
            else:
                # Fallback or error if method doesn't exist, depends on your design
                raise AttributeError(f"Model for integrated mode does not have 'generate_batch_dip_prior' method.")

            dip_train_steps_for_sample = args.dip_train_steps # 記錄，以便傳給 sample

            # Update kwargs for integrated mode: SET ddpm_steps to 200
            print(f"\n -> Setting DDPM steps to 200 for Integrated mode single sample timing.")
            sample_kwargs.update({
                'ddpm_steps': args.integrated_eval_steps,
                'use_dip_prior': True,
                'dip_prior': dip_prior,
                'dip_train_steps': dip_train_steps_for_sample
            })
            print("\n -> Will attempt to measure internal DDPM sampling time.")

        elif mode == "baseline":
            # Update kwargs for baseline mode: KEEP ddpm_steps as None (full steps)
            sample_kwargs.update({
                'ddpm_steps': None,
                'use_dip_prior': False,
                'dip_prior': None
            })
            print("\n -> Will measure total sample call time for baseline.")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # --- Determine if the model's sample method supports internal timing ---
        # Check if 'measure_sampling_time' is a parameter in the model's sample method
        try:
            sig = signature(model.sample)
            use_internal_timer = 'measure_sampling_time' in sig.parameters
        except AttributeError:
            print(" -> Warning: Cannot inspect model.sample signature. Assuming external timer needed.")
            use_internal_timer = False


        # --- Timing Logic ---
        inference_time_seconds = None
        generated_sample = None

        if use_internal_timer:
            # Model supports internal timing (preferred, especially for integrated)
            print(f"   -> Using model's internal timer for {mode} mode.")
            # Use CUDA events for synchronization, even if model reports time internally
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start recording before the sample call
            start_event.record()

            # Call model.sample with measure_sampling_time=True and updated sample_kwargs
            # The model itself should handle the timing around the core loop
            generated_sample, sampling_time_ms = model.sample(**sample_kwargs, measure_sampling_time=True)

            # Record after the sample call finishes
            end_event.record()
            torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None # Ensure timing is accurate

            # Use the time reported by the model
            inference_time_seconds = sampling_time_ms / 1000.0
            print(f"    -> Internal timer reported: {inference_time_seconds:.4f} seconds.")

        else:
            # Fallback to external timing (for baseline or if integrated model lacks internal timer support)
            print(f"   -> Using external timer for {mode} mode.")
            if mode == "integrated":
                 # Note: DIP prior generation already happened outside this block
                 print("    -> DIP prior already generated.")

            torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start timing
            start_event.record()

            # Call model.sample without measure_sampling_time, using the modified sample_kwargs
            generated_sample = model.sample(**sample_kwargs)

            # End timing
            end_event.record()
            torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None

            # Calculate elapsed time
            inference_time_seconds = start_event.elapsed_time(end_event) / 1000.0 # Time in seconds
            print(f"    -> External timer measured: {inference_time_seconds:.4f} seconds.")


        # --- Save the generated sample ---
        save_path = os.path.join(self.samples_dir, f"standard_eval_sample_{mode}.png")
        # Denormalize [-1, 1] -> [0, 1] before saving
        save_images(((generated_sample.cpu().detach() + 1) / 2), save_path, nrow=1) # nrow=1 for single image
        print(f"Saved single high-quality sample to: {save_path}")
        print(f"Single sample inference time ({mode}, Steps: {sample_kwargs.get('ddpm_steps', 'Full')}): {inference_time_seconds:.4f} seconds")

        return inference_time_seconds, save_path, generated_sample.cpu().detach()


    # # --- Measure Single Sample Time & Save ---
    # def measure_and_save_single_sample(self, model, mode, real_image_for_prior=None):
    #     """
    #     Generates ONE sample with full steps, measures time, saves the image.
    #     For 'integrated' mode, measures ONLY the DDPM sampling time if the model supports it.
    #     For 'baseline' mode, measures the entire sample call time.

    #     Args:
    #         model: The model (EMA usually) to use for sampling.
    #         mode: 'baseline' or 'integrated'.
    #         real_image_for_prior: A single real image tensor [1, C, H, W] needed for integrated mode.

    #     Returns:
    #         tuple: (inference_time, saved_image_path, generated_sample_tensor)
    #     """
    #     model.eval()
    #     print(f"Measuring single sample inference time and generating sample for mode: {mode}...")

    #     # Prepare inputs for sample function
    #     batch_size = 1
    #     image_size = args.image_size
    #     channels = args.channels
    #     device = self.device

    #     # --- sample_kwargs and DIP Prior ---
    #     sample_kwargs = {'batch_size': batch_size, 'image_size': image_size, 'channels': channels, 'device': device}
    #     dip_prior = None
    #     dip_train_steps_for_sample = None # 存儲 dip_train_steps

    #     if mode == "integrated":
    #         print(f" -> Setting DDPM steps to 200 for Integrated mode single sample timing.")
    #         sample_kwargs['ddpm_steps'] = 200

    #         if real_image_for_prior is None:
    #             raise ValueError("real_image_for_prior is required for integrated mode single sample timing.")
    #         # Generate DIP prior for this single image (在計時之前完成)
    #         print(" -> Generating DIP prior (outside timing)...")
    #         # 假設 integrated_model 有 generate_batch_dip_prior 方法
    #         if hasattr(model, 'generate_batch_dip_prior'):
    #              dip_prior = model.generate_batch_dip_prior(
    #                 target_images=real_image_for_prior.to(device),
    #                 dip_train_steps=args.dip_train_steps,
    #                 reg_noise_std=args.dip_reg_noise_std,
    #                 device=device
    #             )
    #         else:
    #              # Fallback or error if method doesn't exist, depends on your design
    #              raise AttributeError(f"Model for integrated mode does not have 'generate_batch_dip_prior' method.")

    #         dip_train_steps_for_sample = args.dip_train_steps # 記錄，以便傳給 sample
    #         sample_kwargs.update({'ddpm_steps': None, 'use_dip_prior': True, 'dip_prior': dip_prior, 'dip_train_steps': dip_train_steps_for_sample})
    #         print(" -> Will attempt to measure internal DDPM sampling time.")

    #     elif mode == "baseline":
    #         sample_kwargs.update({'ddpm_steps': None, 'use_dip_prior': False, 'dip_prior': None})
    #         print(" -> Will measure total sample call time for baseline.")
    #     else:
    #         raise ValueError(f"Unknown mode: {mode}")

    #     # --- 條件化呼叫與計時 ---
    #     inference_time = None
    #     generated_sample = None

    #     if mode == "integrated":
    #         # 嘗試呼叫帶有 measure_sampling_time 的版本
    #         try:
    #             generated_sample, inference_time = model.sample(**sample_kwargs, measure_sampling_time=True)
    #             print(f" -> Successfully used internal timer for integrated mode.")
    #         except TypeError as e:
    #             # 如果 integrated model 實際上也沒有 measure_sampling_time (例如 fallback 到舊版)
    #             print(f" -> Warning: model.sample for integrated mode doesn't accept 'measure_sampling_time'. Falling back to external timer. ({e})")
    #             # Fallback to external timing
    #             torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
    #             start_event = torch.cuda.Event(enable_timing=True)
    #             end_event = torch.cuda.Event(enable_timing=True)
    #             start_event.record()

    #             generated_sample = model.sample(**sample_kwargs) # 呼叫不帶 measure_sampling_time

    #             end_event.record()
    #             torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
    #             inference_time = start_event.elapsed_time(end_event) / 1000.0 # 秒

    #     elif mode == "baseline":
    #         # Baseline 模式：使用外部計時器測量整個 sample 呼叫
    #         torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
    #         start_event = torch.cuda.Event(enable_timing=True)
    #         end_event = torch.cuda.Event(enable_timing=True)
    #         start_event.record()

    #         # *** 呼叫 baseline model 的 sample，不傳入 measure_sampling_time ***
    #         generated_sample = model.sample(**sample_kwargs)

    #         end_event.record()
    #         torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
    #         inference_time = start_event.elapsed_time(end_event) / 1000.0 # 秒

    #     # --- 保存圖像和打印時間 (使用正確獲取的 inference_time) ---
    #     save_path = os.path.join(self.samples_dir, f"standard_eval_sample_{mode}.png")
    #     # Denormalize [-1, 1] -> [0, 1] for saving
    #     save_images((generated_sample.cpu().detach() + 1) / 2, save_path)
    #     print(f"Saved single high-quality sample to: {save_path}")
    #     print(f"Single sample inference time ({mode}): {inference_time:.4f} seconds")

    #     return inference_time, save_path, generated_sample.cpu().detach()


    def calculate_standard_metrics(self, model, dataloader, num_fid_samples, evaluation_batch_size, mode):
        """
        Calculates standard metrics (FID, LPIPS, PSNR, SSIM) using batch processing.
        Also triggers single sample generation/timing and comparison plotting.
        """
        print(f"\n[INFO] Starting standard metrics calculation for {mode} model...")
        print(f"[INFO] Target samples: {num_fid_samples}, Evaluation batch size: {evaluation_batch_size}")

        model.eval()
        # --- Initialize Metrics ---
        fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        try:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False).to(self.device)
            print("[INFO] Using VGG network for LPIPS calculation (expects input in [-1, 1]).")
        except Exception as e:
            print(f"[WARN] Failed to initialize LPIPS with VGG ({e}). Falling back to AlexNet.")
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(self.device)
            print("[INFO] Using AlexNet network for LPIPS calculation (expects input in [-1, 1]).")

        # Reset metrics
        fid_metric.reset()
        psnr_metric.reset()
        ssim_metric.reset()
        lpips_metric.reset()

        pbar = tqdm(total=num_fid_samples, desc=f"Calculating Metrics ({mode})", unit="sample")
        num_processed = 0
        first_real_batch_for_prior = None # Store first batch for single sample prior gen

        # --- Batch Processing Loop ---
        for real_batch, _ in dataloader:
            if num_processed >= num_fid_samples:
                break

            # Store the very first real batch (needed for integrated single sample prior)
            if num_processed == 0 and mode == "integrated":
                 first_real_batch_for_prior = real_batch[:1].clone() # Take only the first image

            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)
            samples_to_process = min(batch_size, num_fid_samples - num_processed)
            real_batch = real_batch[:samples_to_process]

            # Generate fake batch (full steps)
            fake_batch = None
            if mode == "integrated":
                dip_prior_batch = model.generate_batch_dip_prior(
                    target_images=real_batch, # Use current batch for prior
                    dip_train_steps=args.dip_train_steps,
                    reg_noise_std=args.dip_reg_noise_std,
                    device=self.device
                )
                fake_batch = model.sample(
                    batch_size=samples_to_process, image_size=args.image_size, channels=args.channels,
                    device=self.device, ddpm_steps=args.ddpm_steps, use_dip_prior=True, dip_prior=dip_prior_batch
                )
            elif mode == "baseline":
                fake_batch = model.sample(
                    batch_size=samples_to_process, image_size=args.image_size, channels=args.channels,
                    device=self.device, ddpm_steps=args.ddpm_steps, use_dip_prior=False, dip_prior=None
                )
            else:
                 raise ValueError(f"Unknown mode: {mode}")

            # Preprocess for metrics
            real_batch_01 = (real_batch + 1) / 2
            fake_batch_01 = (fake_batch + 1) / 2
            real_batch_neg11 = real_batch
            fake_batch_neg11 = fake_batch

            # Update metrics
            fid_metric.update(real_batch_01, real=True)
            fid_metric.update(fake_batch_01, real=False)
            psnr_metric.update(fake_batch_01, real_batch_01)
            ssim_metric.update(fake_batch_01, real_batch_01)
            lpips_metric.update(fake_batch_neg11, real_batch_neg11)

            num_processed += samples_to_process
            pbar.update(samples_to_process)

            # Memory cleanup
            del real_batch, fake_batch, real_batch_01, fake_batch_01, real_batch_neg11, fake_batch_neg11
            if mode == "integrated": del dip_prior_batch
            torch.cuda.empty_cache()

        # --- Compute Final Batch Metrics ---
        final_metrics = {
            'fid': fid_metric.compute().item(),
            'psnr': psnr_metric.compute().item(),
            'ssim': ssim_metric.compute().item(),
            'lpips': lpips_metric.compute().item()
        }
        pbar.close()
        print(f"[INFO] Batch metrics calculation for {mode} completed.")

        # --- Perform Single Sample Generation & Timing ---
        # Pass the first real image if needed for integrated prior generation
        real_prior_img = first_real_batch_for_prior if mode == "integrated" else None
        single_sample_time, single_sample_path, single_sample_tensor = self.measure_and_save_single_sample(
            model, mode, real_prior_img
        )

        # Add single sample time to metrics
        final_metrics[f'single_sample_time_{mode}'] = single_sample_time
        # Store path and tensor for potential comparison plot later
        final_metrics[f'single_sample_path_{mode}'] = single_sample_path
        final_metrics[f'single_sample_tensor_{mode}'] = single_sample_tensor


        return final_metrics

    # --- Method to Plot Comparison of Saved Single Samples ---
    def plot_single_sample_comparison(self, baseline_metrics, integrated_metrics):
        """ Plots the saved single high-quality samples side-by-side """
        baseline_sample_tensor = baseline_metrics.get(f'single_sample_tensor_baseline')
        integrated_sample_tensor = integrated_metrics.get(f'single_sample_tensor_integrated')

        if baseline_sample_tensor is None or integrated_sample_tensor is None:
            print("[WARN] Could not find tensors for single sample comparison plot.")
            return

        # Assuming plot_image takes tensors in [-1, 1] range or handles denormalization
        # If plot_image expects [0, 1], denormalize here:
        # baseline_sample_tensor = (baseline_sample_tensor + 1) / 2
        # integrated_sample_tensor = (integrated_sample_tensor + 1) / 2

        # Make sure tensors have batch dimension [1, C, H, W] and are detached cpu tensors
        baseline_tensor_plot = baseline_sample_tensor.cpu().detach()
        integrated_tensor_plot = integrated_sample_tensor.cpu().detach()

        # Remove batch dim for plot_image if it expects [C, H, W]
        if baseline_tensor_plot.dim() == 4: baseline_tensor_plot = baseline_tensor_plot.squeeze(0)
        if integrated_tensor_plot.dim() == 4: integrated_tensor_plot = integrated_tensor_plot.squeeze(0)

        # Plotting
        comparison_save_path = os.path.join(self.samples_dir, "standard_eval_comparison.png")
        plot_image(
             baseline_tensor_plot, integrated_tensor_plot, # Pass tensors directly
             title1=f"Baseline Sample (Time: {baseline_metrics.get('single_sample_time_baseline', 0):.3f}s)",
             title2=f"Integrated Sample (Time: {integrated_metrics.get('single_sample_time_integrated', 0):.3f}s)",
             save_path=comparison_save_path
        )
        print(f"Saved single sample comparison plot to: {comparison_save_path}")


