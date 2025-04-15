"""
Prepare Image Sequences for Spatiotemporal Forecasting

This script prepares sequences of consecutive satellite images for training the
spatiotemporal forecasting model. It identifies temporally consecutive images
from the same location and organizes them into sequences.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
import pickle
from itertools import groupby
from operator import itemgetter

def extract_date_tile_patch_info(patch_path):
    """Extract date, tile ID, and patch coordinates from patch directory path."""
    patch_name = os.path.basename(patch_path)
    tile_name = os.path.basename(os.path.dirname(patch_path))
    
    parts = patch_name.split('_')
    
    # Extract date (format: YYYYMMDDTHHMMSS)
    date_str = parts[2]
    try:
        date = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
    except ValueError:
        return None, None, None, None, None # Indicate error
    
    # Extract tile ID (e.g., T35VNL from S2A_MSIL2A_..._T35VNL)
    tile_id = tile_name.split('_')[-1]
    
    # Extract patch coordinates (e.g., 26, 57 from S2A_..._26_57)
    try:
        patch_row = int(parts[-2])
        patch_col = int(parts[-1])
    except (IndexError, ValueError):
        patch_row, patch_col = None, None
        
    return date, tile_id, patch_name, patch_row, patch_col

def find_all_patch_dirs(data_root):
    """Find all patch directories within the tile directories."""
    tile_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    patch_dir_paths = []
    print(f"Scanning {len(tile_dirs)} tile directories for patches...")
    for tile_dir in tqdm(tile_dirs, desc="Scanning tiles"):
        tile_path = os.path.join(data_root, tile_dir)
        try:
            for patch_name in os.listdir(tile_path):
                patch_path = os.path.join(tile_path, patch_name)
                if os.path.isdir(patch_path):
                    # Check if patch directory contains TIF files
                    if glob.glob(os.path.join(patch_path, "*.tif")):
                        # Store relative path from data_root
                        relative_patch_path = os.path.relpath(patch_path, data_root)
                        patch_dir_paths.append(relative_patch_path)
        except FileNotFoundError:
            print(f"Warning: Tile directory {tile_path} not found or inaccessible.")
            continue
    return patch_dir_paths

def group_by_location(patch_dir_paths, data_root):
    """Group patch directories by their tile location."""
    patch_info = []
    print("Extracting metadata from patch paths...")
    for relative_patch_path in tqdm(patch_dir_paths, desc="Extracting metadata"):
        full_patch_path = os.path.join(data_root, relative_patch_path)
        date, tile_id, patch_name, _, _ = extract_date_tile_patch_info(full_patch_path)
        if date and tile_id:
            patch_info.append({
                'relative_path': relative_patch_path, # Store relative path
                'date': date,
                'tile_id': tile_id # Group by tile ID
            })
    
    # Sort by tile_id and date
    patch_info.sort(key=lambda x: (x['tile_id'], x['date']))
    
    # Group by tile_id
    location_groups = {}
    for tile_id, group in groupby(patch_info, key=lambda x: x['tile_id']):
        location_groups[tile_id] = list(group)
    
    return location_groups

def create_sequences(location_groups, sequence_length=11, max_time_gap_days=60):
    """
    Create sequences of consecutive images for each location (tile).
    
    Args:
        location_groups: Dictionary of location groups (grouped by tile_id)
        sequence_length: Number of consecutive images in a sequence (input length)
        max_time_gap_days: Maximum allowed time gap between consecutive images
    
    Returns:
        List of sequences, where each sequence is a list of relative patch directory paths
    """
    sequences = []
    skipped_count = 0
    total_patches = sum(len(patches) for patches in location_groups.values())
    processed_patches = 0

    print("Creating sequences...")
    with tqdm(total=total_patches, desc="Creating sequences") as pbar:
        for tile_id, patches in location_groups.items():
            # Sort by date (should already be sorted, but double-check)
            patches.sort(key=lambda x: x['date'])
            
            # Create sequences for this tile
            for i in range(len(patches) - sequence_length):
                # Check if time gaps are within limit for the sequence (input + target)
                valid_sequence = True
                # Need sequence_length steps (sequence_length+1 images)
                for j in range(i, i + sequence_length):
                    time_diff = (patches[j + 1]['date'] - patches[j]['date']).days
                    if time_diff > max_time_gap_days or time_diff < 0: # Check for negative gaps too
                        valid_sequence = False
                        skipped_count += 1
                        break
                
                if valid_sequence:
                    # Create sequence of relative patch directory paths
                    # Sequence includes input frames + target frame
                    seq = [patches[j]['relative_path'] for j in range(i, i + sequence_length + 1)]
                    sequences.append(seq)
                
                # Update progress bar only after processing the starting patch i
                pbar.update(1)
                processed_patches += 1

            # Update progress for remaining patches in the group not starting a sequence
            remaining = len(patches) - max(0, len(patches) - sequence_length) 
            pbar.update(remaining)
            processed_patches += remaining

    # Final update in case of rounding or edge cases
    pbar.n = total_patches
    pbar.refresh()

    print(f"\nCreated {len(sequences)} sequences")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} potential sequences due to time gaps > {max_time_gap_days} days or inconsistent dates.")
    
    return sequences

def save_sequences(sequences, output_file):
    """Save sequences to a pickle file and a CSV file."""
    # Save to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(sequences, f)
    
    # Also save as CSV for easier inspection
    csv_file = output_file.replace('.pkl', '.csv')
    
    # Create a DataFrame with one row per sequence
    df_data = []
    if sequences:
        # Determine max sequence length from data (should be consistent)
        seq_len_plus_target = len(sequences[0]) 
        col_names = [f'input_{j}' for j in range(seq_len_plus_target - 1)] + ['target']
        
        for i, seq in enumerate(sequences):
            if len(seq) == seq_len_plus_target:
                row = {'sequence_id': i}
                for j, relative_patch_path in enumerate(seq):
                   row[col_names[j]] = relative_patch_path
                df_data.append(row)
            else:
                 print(f"Warning: Sequence {i} has unexpected length {len(seq)}, skipping in CSV.")

    if df_data:
        df = pd.DataFrame(df_data)
        # Order columns: sequence_id, input_0, ..., input_N, target
        ordered_cols = ['sequence_id'] + col_names
        df = df[ordered_cols]
        df.to_csv(csv_file, index=False)
        print(f"Saved sequences to {output_file} and {csv_file}")
    elif sequences: # Sequences exist but might have had wrong length for CSV
         print(f"Saved sequences to {output_file}. CSV file not created due to sequence length issues.")
    else: # No sequences
        print(f"No sequences to save. Files {output_file} and {csv_file} not created.")

def filter_by_location(patch_dir_paths, locations, data_root):
    """Filter patch directories by their tile location ID."""
    filtered_paths = []
    print(f"Filtering for tile locations: {locations}")
    for relative_path in tqdm(patch_dir_paths, desc="Filtering by location"):
        full_path = os.path.join(data_root, relative_path)
        _, tile_id, _, _, _ = extract_date_tile_patch_info(full_path)
        if tile_id in locations:
            filtered_paths.append(relative_path)
    print(f"Filtered to {len(filtered_paths)} patch directories.")
    return filtered_paths

def analyze_sequences(sequences, data_root):
    """Analyze created sequences (time gaps, lengths). Uses relative paths."""
    if not sequences:
        print("\nNo sequences to analyze.")
        return
        
    # Extract dates from sequences
    seq_dates = []
    invalid_seq_count = 0
    print("Analyzing sequence dates...")
    for seq in tqdm(sequences, desc="Analyzing sequences"):
        dates_in_seq = []
        valid = True
        for relative_path in seq:
            full_path = os.path.join(data_root, relative_path)
            date, _, _, _, _ = extract_date_tile_patch_info(full_path)
            if date:
                dates_in_seq.append(date)
            else:
                valid = False
                break
        if valid and len(dates_in_seq) == len(seq): # Ensure all dates were extracted
             seq_dates.append(dates_in_seq)
        else:
            invalid_seq_count += 1
            
    if invalid_seq_count > 0:
        print(f"Warning: Skipped {invalid_seq_count} sequences during analysis due to date extraction errors.")
        
    if not seq_dates:
        print("No valid sequences found for analysis after date extraction.")
        return

    # Calculate time gaps
    time_gaps = []
    for dates in seq_dates:
        seq_gaps = []
        for i in range(len(dates) - 1):
            gap_days = (dates[i + 1] - dates[i]).days
            # Filter out potentially negative gaps if sequences weren't perfectly sorted
            if gap_days >= 0:
                seq_gaps.append(gap_days)
        if seq_gaps: # Only add if there are valid gaps
            time_gaps.append(seq_gaps)
    
    # Compute statistics
    all_gaps = [gap for seq_gaps in time_gaps for gap in seq_gaps]
    
    if not all_gaps:
        print("\nNo valid time gaps found between consecutive images in sequences.")
    else:
        print("\nSequence Time Gap Statistics (days):")
        print(f"Mean: {np.mean(all_gaps):.2f}")
        print(f"Median: {np.median(all_gaps):.2f}")
        print(f"Min: {np.min(all_gaps)}")
        print(f"Max: {np.max(all_gaps)}")
        print(f"Std Dev: {np.std(all_gaps):.2f}")
    
    # Sequence lengths in days
    seq_lengths = [(dates[-1] - dates[0]).days for dates in seq_dates if dates] # Check dates not empty
    
    if not seq_lengths:
        print("\nCould not calculate sequence lengths.")
    else:
        print("\nSequence Length Statistics (days from first to last image):")
        print(f"Mean: {np.mean(seq_lengths):.2f}")
        print(f"Median: {np.median(seq_lengths):.2f}")
        print(f"Min: {np.min(seq_lengths)}")
        print(f"Max: {np.max(seq_lengths)}")

def create_train_val_test_split(sequences, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split sequences into train, validation, and test sets."""
    np.random.seed(seed)
    np.random.shuffle(sequences)
    
    n_train = int(len(sequences) * train_ratio)
    n_val = int(len(sequences) * val_ratio)
    
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:n_train + n_val]
    test_sequences = sequences[n_train + n_val:]
    
    return train_sequences, val_sequences, test_sequences

def main():
    parser = argparse.ArgumentParser(description='Prepare image sequences for spatiotemporal forecasting')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BigEarthNet-S2 dataset')
    parser.add_argument('--output_dir', type=str, default='./sequences', help='Output directory for sequence files')
    parser.add_argument('--sequence_length', type=int, default=11, help='Number of consecutive images in input sequence (e.g., 11 inputs to predict 12th)')
    parser.add_argument('--max_time_gap', type=int, default=60, help='Maximum allowed time gap (days) between consecutive images')
    parser.add_argument('--locations', type=str, nargs='+', help='Specific tile locations (e.g., T35VNL) to include. If not specified, all locations are used.')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='Train, validation, test split ratios (must sum to 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting sequences')

    args = parser.parse_args()
    
    # Validate split ratios
    if not np.isclose(sum(args.split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    train_ratio, val_ratio, test_ratio = args.split_ratios

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all patch directories
    print("Finding all patch directories...")
    patch_dir_paths = find_all_patch_dirs(args.data_root)
    print(f"Found {len(patch_dir_paths)} total patch directories")
    
    # Filter by location if specified
    if args.locations:
        patch_dir_paths = filter_by_location(patch_dir_paths, args.locations, args.data_root)
    
    if not patch_dir_paths:
        print("No patch directories remaining after filtering. Exiting.")
        return

    # Group by location (tile_id)
    print("Grouping patches by location...")
    location_groups = group_by_location(patch_dir_paths, args.data_root)
    print(f"Grouped patches into {len(location_groups)} unique locations (tiles)")
    
    # Create sequences
    print(f"Creating sequences with input length {args.sequence_length} (predicting frame {args.sequence_length + 1})...")
    sequences = create_sequences(location_groups, args.sequence_length, args.max_time_gap)
    
    if not sequences:
        print("No sequences were created. Check data and parameters. Exiting.")
        return
        
    # Analyze sequences
    analyze_sequences(sequences, args.data_root)
    
    # Split into train, validation, and test sets
    print("\nSplitting sequences...")
    train_sequences, val_sequences, test_sequences = create_train_val_test_split(
        sequences, train_ratio=train_ratio, val_ratio=val_ratio, seed=args.seed
    )
    
    print(f"\nSplit {len(sequences)} sequences into:")
    print(f"  Train: {len(train_sequences)} sequences ({len(train_sequences)/len(sequences)*100:.1f}%)")
    print(f"  Validation: {len(val_sequences)} sequences ({len(val_sequences)/len(sequences)*100:.1f}%)")
    print(f"  Test: {len(test_sequences)} sequences ({len(test_sequences)/len(sequences)*100:.1f}%)")
    
    # Save sequence files
    print("\nSaving sequence files...")
    save_sequences(train_sequences, os.path.join(args.output_dir, 'train_sequences.pkl'))
    save_sequences(val_sequences, os.path.join(args.output_dir, 'val_sequences.pkl'))
    save_sequences(test_sequences, os.path.join(args.output_dir, 'test_sequences.pkl'))
    
    # Save all sequences for reference (optional, can be large)
    # save_sequences(sequences, os.path.join(args.output_dir, 'all_sequences.pkl'))
    
    print("\nSequence preparation complete!")

if __name__ == "__main__":
    main() 