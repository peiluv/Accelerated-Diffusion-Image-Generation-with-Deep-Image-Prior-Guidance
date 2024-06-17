# -*- coding: utf-8 -*-
# dataset.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

# import sys
# extract_dir = "/kaggle/input/ddpm-dip-project/ddpm_dip_project"
# sys.path.append(extract_dir)

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from config import get_config
args = get_config(from_cli=True)

def load_dataset(args):
    """
    Load and prepare dataset.

    - Applies augmentations (RandomHorizontalFlip, ColorJitter) only to the training set.
    - Validation and test sets only get resizing, cropping, tensor conversion, and normalization.
    - Uses separate ImageFolder instances with different transforms and index splitting.
    - Sets NumPy random seed based on args.seed for reproducible data splitting.

    DDPM 的預設要求：圖片都會轉成 `[-1, 1]` 範圍
    """
    # --- Define Transforms ---
    # Transforms for the training set (including data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(p=0.5), # Apply random flip only for training
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Apply color jitter only for training
        transforms.ToTensor(), # Converts PIL image [0, 255] to FloatTensor [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalizes tensor to range [-1, 1]
    ])

    # Transforms for the validation and test sets (no data augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), # Converts PIL image [0, 255] to FloatTensor [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalizes tensor to range [-1, 1]
    ])

    # --- Prepare Datasets with Specific Transforms ---
    # Create an ImageFolder instance for training data WITH augmentations
    # ImageFolder finds images and assigns class indices based on subfolder names
    try:
        train_dataset_base = ImageFolder(root=args.data_dir, transform=train_transform)

        # Create an ImageFolder instance for validation/test data WITHOUT augmentations
        # This relies on ImageFolder sorting files/classes identically for the same directory
        val_test_dataset_base = ImageFolder(root=args.data_dir, transform=val_test_transform)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data directory not found or empty: {args.data_dir}")
    except Exception as e:
        raise RuntimeError(f"Error initializing ImageFolder at {args.data_dir}: {e}")


    # Important Check: Ensure both instances found the same number of items
    n = len(train_dataset_base)
    if n == 0:
        raise ValueError(f"No images found in the specified data directory: {args.data_dir}")
    if n != len(val_test_dataset_base):
        # This should theoretically not happen if ImageFolder logic is consistent
        raise ValueError(f"Mismatch ({n} vs {len(val_test_dataset_base)}) in dataset sizes between train and val/test base ImageFolders. Check data directory integrity or ImageFolder behavior.")

    # --- Split Indices (Reproducible) ---
    n_train = int(0.8 * n) # 80% for train
    n_val = int(0.1 * n)   # 10% for val
    # Ensure all data is used, handle potential rounding issues for the test set
    n_test = n - n_train - n_val # Remaining 10% (approx) for test

    if n_train + n_val + n_test != n:
         print(f"Warning: Split sizes do not perfectly match total size ({n_train}+{n_val}+{n_test} != {n}). Adjusting test set size.")
         n_test = n - n_train - n_val # Recalculate test size precisely

    print(f"Dataset split: Train={n_train}, Validation={n_val}, Test={n_test} (Total={n})")

    indices = np.arange(n)

    # Set the random seed for NumPy BEFORE shuffling for reproducibility
    if hasattr(args, 'seed') and args.seed is not None:
         try:
             seed_val = int(args.seed)
             np.random.seed(seed_val)
             print(f"Using random seed: {seed_val} for dataset splitting.")
         except ValueError:
             print(f"Warning: Invalid seed value '{args.seed}'. Using default random state.")
             np.random.seed(None) # Reset to default random state if seed is invalid
    else:
         print("Warning: No random seed provided (args.seed is missing or None). Dataset splitting will not be reproducible.")

    np.random.shuffle(indices) # Shuffle indices in place (now reproducible if seed is set)

    # Split the shuffled indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # --- Create Subsets using Indices and Correctly Transformed Base Datasets ---
    # Subset wraps the base dataset and provides access only to the specified indices
    train_set = Subset(train_dataset_base, train_indices)
    val_set = Subset(val_test_dataset_base, val_indices)
    test_set = Subset(val_test_dataset_base, test_indices)

    # --- Create DataLoaders ---
    # pin_memory=True helps speed up host-to-GPU transfers if using CUDA
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle training data each epoch
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True # Optional: Drop last incomplete batch if needed
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.evaluation_batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")

    return train_loader, val_loader, test_loader

# --- Example Usage (assuming have an 'args' object) ---
# class Args:
#     image_size = 64 # Example image size
#     data_dir = './path/to/your/dataset' # Path to the main data directory
#     batch_size = 32
#     num_workers = 4

# args = Args()
# train_loader, val_loader, test_loader = load_dataset(args)

# print(f"Train loader batches: {len(train_loader)}")
# print(f"Validation loader batches: {len(val_loader)}")
# print(f"Test loader batches: {len(test_loader)}")

# # inspect a batch
# for images, labels in train_loader:
#     print("Train batch shape:", images.shape)
#     print("Train batch min/max:", images.min(), images.max())
#     break
# for images, labels in val_loader:
#     print("Validation batch shape:", images.shape)
#     print("Validation batch min/max:", images.min(), images.max())
#     break
