"""
Preprocess Sentinel-2 Image Sequences

This script converts sequences of Sentinel-2 TIF patches (defined in .pkl files)
into pre-processed PyTorch tensors (.pt files) for faster loading during training.
It performs the necessary image loading, resampling, resizing, and normalization once.
Can optionally filter sequences based on a pre-generated list of allowed patch paths.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd # Added for reading filtered list CSV
import torch
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
import cv2
import argparse
from tqdm import tqdm
import sys
import traceback

def load_and_preprocess_image(img_path, bands, target_patch_size):
    """
    Loads and preprocesses a single Sentinel-2 patch directory.
    Adapted from main.py's _load_sentinel2_image.
    Handles resampling to a common resolution (preferably 10m) and resizing.

    Args:
        img_path (str): Path to the patch directory.
        bands (list): List of band names to load (e.g., ['B02', 'B03', 'B04', 'B08']).
        target_patch_size (int): The desired output size (height and width) for the patch.

    Returns:
        torch.Tensor: Preprocessed image tensor (C, H, W), normalized to [0, 1].
        None: If loading or processing fails.
    """
    try:
        # Load the specified bands
        band_data = []
        target_shape = None
        resampling_method = Resampling.bilinear

        # Determine target shape from a 10m band if possible
        preferred_10m_bands = ['B04', 'B03', 'B02', 'B08']
        available_bands_in_patch = [os.path.basename(f).split('_')[-1].split('.')[0]
                                  for f in glob.glob(os.path.join(img_path, "*.tif"))]

        for band_name in preferred_10m_bands:
            if band_name in bands and band_name in available_bands_in_patch:
                band_files = glob.glob(os.path.join(img_path, f"*_{band_name}.tif"))
                if band_files:
                    try:
                        with rasterio.open(band_files[0]) as src:
                            target_shape = (src.height, src.width)
                            break
                    except rasterio.RasterioIOError:
                        pass # Try next preferred band

        # Load specified bands (from config.bands), resampling as needed
        loaded_band_data = {}
        for band in bands:
            band_files = glob.glob(os.path.join(img_path, f"*_{band}.tif"))
            if not band_files:
                return None # Cannot proceed without all required bands

            try:
                with rasterio.open(band_files[0]) as src:
                    current_shape = (src.height, src.width)

                    # If target_shape wasn't determined from a 10m band, use the first band's shape
                    if target_shape is None:
                        target_shape = current_shape

                    # Read and resample if necessary
                    if current_shape == target_shape:
                        band_array = src.read(1)
                    else:
                        band_array = src.read(
                            1,
                            out_shape=target_shape,
                            resampling=resampling_method
                        )
                    loaded_band_data[band] = band_array
            except rasterio.RasterioIOError as e:
                 # Reduced verbosity: print(f"Error reading band {band} in {img_path}: {e}", file=sys.stderr)
                 return None

        # Stack bands in the specified order
        # Check if all requested bands were actually loaded
        if len(loaded_band_data) != len(bands):
             return None
             
        img_array_bands = [loaded_band_data[b] for b in bands]
        img_array = np.stack(img_array_bands, axis=0)

        # Reshape to image format (H, W, C) for resizing
        img_array_hwc = reshape_as_image(img_array)

        # Resize to target patch size using cv2
        if img_array_hwc.shape[:2] != (target_patch_size, target_patch_size):
             img_array_resized = cv2.resize(img_array_hwc, (target_patch_size, target_patch_size), interpolation=cv2.INTER_LINEAR)
             # Ensure channel dimension is last if it gets squeezed during resize (for single band)
             if img_array_resized.ndim == 2:
                 img_array_resized = np.expand_dims(img_array_resized, axis=-1)
        else:
            img_array_resized = img_array_hwc

        # Normalize to [0, 1]
        img_array_normalized = img_array_resized.astype(np.float32) / 10000.0
        img_array_normalized = np.clip(img_array_normalized, 0, 1)

        # Transpose back to (C, H, W) for PyTorch
        img_tensor_chw = torch.from_numpy(img_array_normalized.transpose((2, 0, 1))).float()

        return img_tensor_chw

    except FileNotFoundError as e:
        # Reduced verbosity: print(f"Error (FileNotFound): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error processing {img_path}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def preprocess_sequences(data_root, sequence_file, output_dir, bands, patch_size, sequence_length, filtered_list_csv=None):
    """Loads sequences from a pkl file, optionally filters them, and saves preprocessed tensors."""

    print(f"Loading sequences from: {sequence_file}")
    try:
        with open(sequence_file, 'rb') as f:
            sequences = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Sequence file not found at {sequence_file}. Skipping.", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error loading sequence file {sequence_file}: {e}", file=sys.stderr)
        return 0

    if not sequences:
        print(f"No sequences found in {sequence_file}.")
        return 0

    # --- Filtering Logic ---
    allowed_paths = None
    if filtered_list_csv:
        print(f"Loading allowed paths from: {filtered_list_csv}")
        try:
            df_filter = pd.read_csv(filtered_list_csv)
            if 'relative_path' not in df_filter.columns:
                print(f"Error: CSV file {filtered_list_csv} must contain a 'relative_path' column.", file=sys.stderr)
                return 0
            # Convert to set for fast lookup
            allowed_paths = set(df_filter['relative_path'].tolist())
            print(f"Loaded {len(allowed_paths)} allowed relative paths.")
            
            # Filter the sequences
            original_count = len(sequences)
            filtered_sequences = []
            for seq in tqdm(sequences, desc="Filtering sequences"):
                # Check if ALL paths in the sequence are in the allowed set
                if all(rel_path in allowed_paths for rel_path in seq):
                    filtered_sequences.append(seq)
            
            sequences = filtered_sequences # Replace original list with filtered list
            print(f"Filtered sequences: {len(sequences)} remaining (from {original_count})")
            if not sequences:
                 print("No sequences remaining after filtering.")
                 return 0
        except FileNotFoundError:
            print(f"Error: Filtered list CSV not found at {filtered_list_csv}", file=sys.stderr)
            return 0
        except Exception as e:
            print(f"Error reading or processing filtered list CSV {filtered_list_csv}: {e}", file=sys.stderr)
            return 0
    # --- End Filtering Logic ---

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(sequences)} sequences. Outputting to: {output_dir}")

    processed_count = 0
    error_count = 0
    skip_count = 0 # Count sequences skipped due to missing images during processing

    # Expected length: input sequence + target frame
    expected_len = sequence_length + 1

    # Use index from original enumeration if filtered, otherwise just enumerate
    for i, relative_path_sequence in enumerate(tqdm(sequences, desc=f"Preprocessing {os.path.basename(output_dir)}")):

        if len(relative_path_sequence) != expected_len:
            # This check should ideally be done during sequence preparation
            print(f"Warning: Sequence {i} has incorrect length {len(relative_path_sequence)}, expected {expected_len}. Skipping.", file=sys.stderr)
            error_count += 1
            continue

        input_frames = []
        target_frame = None
        valid_sequence = True

        # Process input frames
        for j in range(sequence_length):
            relative_patch_path = relative_path_sequence[j]
            full_patch_path = os.path.join(data_root, relative_patch_path)
            img_tensor = load_and_preprocess_image(full_patch_path, bands, patch_size)
            if img_tensor is None:
                valid_sequence = False
                break
            input_frames.append(img_tensor)

        if not valid_sequence:
            skip_count += 1
            continue

        # Process target frame
        target_relative_path = relative_path_sequence[-1]
        target_full_path = os.path.join(data_root, target_relative_path)
        target_frame = load_and_preprocess_image(target_full_path, bands, patch_size)

        if target_frame is None:
            skip_count += 1
            continue

        # Stack input frames: (T, C, H, W)
        try:
            input_sequence_tensor = torch.stack(input_frames)
        except RuntimeError as e:
            print(f"Error stacking tensors for sequence {i}. Check tensor shapes. Error: {e}", file=sys.stderr)
            error_count += 1
            continue

        # Save the preprocessed data using the index 'i' from the current loop
        output_filename = f"sequence_{i:06d}.pt"
        output_path = os.path.join(output_dir, output_filename)
        try:
            torch.save({
                'input': input_sequence_tensor, # Shape (T, C, H, W)
                'target': target_frame         # Shape (C, H, W)
            }, output_path)
            processed_count += 1
        except Exception as e:
            print(f"Error saving preprocessed file {output_path}: {e}", file=sys.stderr)
            error_count += 1

    print(f"Finished preprocessing {os.path.basename(output_dir)}.")
    print(f"  Successfully processed and saved: {processed_count}")
    print(f"  Skipped due to image processing errors: {skip_count}")
    print(f"  Other errors (length mismatch, saving errors): {error_count}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='Preprocess BigEarthNet-S2 sequences')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the original BigEarthNet-S2 dataset (containing TIF patches)')
    parser.add_argument('--sequences_dir', type=str, required=True,
                        help='Directory containing the sequence .pkl files (output of prepare_sequences.py)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the preprocessed .pt files')
    parser.add_argument('--filtered_list_csv', type=str, default=None,
                        help='Optional path to a CSV file containing a list of allowed relative_paths (output of filter_images.py). Only sequences whose patches are all in this list will be processed.')
    parser.add_argument('--bands', nargs='+', default=['B02', 'B03', 'B04', 'B08'],
                        help='List of bands to include in the preprocessed tensors')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Target size (height and width) for the image patches')
    parser.add_argument('--sequence_length', type=int, default=11,
                        help='Expected length of the input sequence (excluding target)')

    args = parser.parse_args()

    print("--- Starting Preprocessing ---")
    print(f"Original Data Root: {args.data_root}")
    print(f"Input Sequence Dir: {args.sequences_dir}")
    print(f"Output Preprocessed Dir: {args.output_dir}")
    if args.filtered_list_csv:
        print(f"Filtering using CSV: {args.filtered_list_csv}")
    print(f"Bands: {args.bands}")
    print(f"Patch Size: {args.patch_size}")
    print(f"Input Sequence Length: {args.sequence_length}")

    # Define sequence files and output directories
    sequence_files = {
        "train": os.path.join(args.sequences_dir, "train_sequences.pkl"),
        "val": os.path.join(args.sequences_dir, "val_sequences.pkl"),
        "test": os.path.join(args.sequences_dir, "test_sequences.pkl"),
    }

    output_dirs = {
        "train": os.path.join(args.output_dir, "train"),
        "val": os.path.join(args.output_dir, "val"),
        "test": os.path.join(args.output_dir, "test"),
    }

    total_processed = 0
    # Process each split (train, val, test)
    for split in ["train", "val", "test"]:
        seq_file = sequence_files[split]
        out_dir = output_dirs[split]

        if os.path.exists(seq_file):
            print(f"\nProcessing {split} split...")
            count = preprocess_sequences(
                args.data_root,
                seq_file,
                out_dir,
                args.bands,
                args.patch_size,
                args.sequence_length,
                args.filtered_list_csv # Pass the filter CSV path
            )
            total_processed += count
        else:
            print(f"\nSequence file for {split} split not found ({seq_file}). Skipping.")

    print(f"\n--- Preprocessing Complete ---")
    print(f"Total sequences processed and saved as .pt files: {total_processed}")
    print(f"Preprocessed data saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 