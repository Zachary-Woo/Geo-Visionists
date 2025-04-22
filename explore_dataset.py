"""
BigEarthNet-S2 Dataset Explorer

This script provides functions to explore and visualize the BigEarthNet-S2 dataset,
helping with understanding the data structure, band combinations, and metadata.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime
import sys
import re
from rasterio.enums import Resampling

def load_image_metadata(img_dir):
    """Load metadata from a BigEarthNet image directory."""
    metadata_files = glob.glob(os.path.join(img_dir, "*_metadata.json"))
    
    if not metadata_files:
        return None
    
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    return metadata

def load_bands(img_dir, bands=None):
    """
    Load specific bands from a BigEarthNet image directory, resampling to a common resolution.
    
    Args:
        img_dir: Path to the image directory
        bands: List of band names to load (e.g., ['B02', 'B03', 'B04'] for RGB)
               If None, loads all available bands found in the directory.
    
    Returns:
        Dictionary of band arrays with band names as keys (all resampled to target shape)
        List of missing bands
    """
    
    # If bands argument is None, find all available bands in the directory
    available_bands = []
    if bands is None:
        all_tif_files = glob.glob(os.path.join(img_dir, "*.tif"))
        for tif_file in all_tif_files:
            filename = os.path.basename(tif_file)
            band_match = re.search(r'_(B\d+A?)\.tif$', filename)
            if band_match:
                available_bands.append(band_match.group(1))
        if not available_bands:
             print(f"Warning: No band files found in {img_dir}", file=sys.stderr)
             return {}, [] # Return empty if no bands found
        bands = available_bands # Use all found bands
        print(f"No specific bands requested. Loading available bands: {', '.join(bands)}")


    band_data = {}
    missing_bands = []
    target_shape = None
    resampling_method = Resampling.bilinear

    # Define preferred 10m bands to determine target shape
    preferred_10m_bands = ['B04', 'B03', 'B02', 'B08']
    
    # First pass: Determine target shape from a 10m band if available
    for band in preferred_10m_bands:
        if band in bands:
            band_files = glob.glob(os.path.join(img_dir, f"*_{band}.tif"))
            if band_files:
                try:
                    with rasterio.open(band_files[0]) as src:
                        target_shape = (src.height, src.width)
                        print(f"Determined target shape {target_shape} from band {band}")
                        break # Target shape found
                except rasterio.RasterioIOError as e:
                    print(f"Warning: Could not read {band_files[0]} to determine shape: {e}", file=sys.stderr)

    # Second pass: Load all requested bands, resampling if necessary
    for band in bands:
        band_files = glob.glob(os.path.join(img_dir, f"*_{band}.tif"))
        
        if not band_files:
            missing_bands.append(band)
            # Don't print warning here, handle missing bands later if needed
            continue
        
        try:
            with rasterio.open(band_files[0]) as src:
                current_shape = (src.height, src.width)
                
                # If target_shape is not set yet, use the shape of the first valid band
                if target_shape is None:
                    target_shape = current_shape
                    print(f"Warning: No preferred 10m band found. Using shape {target_shape} from band {band} as target.")

                # Read band, resampling if shape differs from target
                if current_shape == target_shape:
                    band_array = src.read(1)
                else:
                    print(f"Resampling band {band} from {current_shape} to {target_shape}")
                    band_array = src.read(
                        1,
                        out_shape=target_shape,
                        resampling=resampling_method
                    )
                
                band_data[band] = band_array
        except rasterio.RasterioIOError as e:
             print(f"Error reading band {band} file {band_files[0]}: {e}", file=sys.stderr)
             missing_bands.append(band)


    # Check for completely missing essential bands after trying to load all
    essential_missing = [b for b in ['B02', 'B03', 'B04'] if b not in band_data]
    if essential_missing:
         print(f"Warning: Missing essential bands for RGB visualization: {essential_missing} in {img_dir}", file=sys.stderr)

    # Report all missing bands at the end
    if missing_bands:
        print(f"Warning: The following bands were requested but not found or couldn't be read in {img_dir}: {', '.join(set(missing_bands))}", file=sys.stderr)

    if not band_data:
         print(f"Error: No bands could be successfully loaded for {img_dir}", file=sys.stderr)

    return band_data, list(set(missing_bands)) # Return unique missing bands

def create_rgb_image(band_data, r_band='B04', g_band='B03', b_band='B02', scale=0.0001):
    """
    Create an RGB image from specified bands. Assumes bands are already resampled.
    
    Args:
        band_data: Dictionary of band arrays
        r_band, g_band, b_band: Band names for RGB channels
        scale: Scale factor for normalization (Sentinel-2 typically uses 0.0001)
    
    Returns:
        RGB image as numpy array (normalized to [0, 1])
    """
    missing = [band for band in [r_band, g_band, b_band] if band not in band_data]
    if missing:
        # Create a blank RGB image instead of raising an error
        print(f"Warning: Cannot create RGB image. Missing bands: {missing}", file=sys.stderr)
        # Get shape from any available band, or use default
        shape = (100, 100, 3)  # Default shape if no bands available
        for band in band_data:
            shape = (band_data[band].shape[0], band_data[band].shape[1], 3)
            break
        return np.zeros(shape)
    
    # Stack RGB bands
    rgb = np.stack([
        band_data[r_band],
        band_data[g_band],
        band_data[b_band]
    ], axis=0)
    
    # Reshape to image format (H, W, C)
    rgb = reshape_as_image(rgb)
    
    # Normalize to [0, 1]
    rgb = rgb.astype(np.float32) * scale
    rgb = np.clip(rgb, 0, 1)
    
    # Apply contrast stretching using percentile clipping for visualization
    stretched_rgb = np.zeros_like(rgb)
    for i in range(rgb.shape[2]): # Iterate through R, G, B channels
        channel = rgb[:, :, i]
        # Calculate 2nd and 98th percentiles
        p2, p98 = np.percentile(channel, (2, 98))
        # Clip and rescale
        stretched_channel = (channel - p2) / (p98 - p2)
        stretched_rgb[:, :, i] = np.clip(stretched_channel, 0, 1)
        
    return stretched_rgb

def create_ndvi_image(band_data, nir_band='B08', red_band='B04'):
    """
    Calculate NDVI (Normalized Difference Vegetation Index). Assumes bands are already resampled.
    
    NDVI = (NIR - Red) / (NIR + Red)
    """
    missing = [band for band in [nir_band, red_band] if band not in band_data]
    if missing:
        # Create a blank NDVI image instead of raising an error
        print(f"Warning: Cannot create NDVI image. Missing bands: {missing}", file=sys.stderr)
        # Get shape from any available band, or use default
        shape = (100, 100)  # Default shape if no bands available
        for band in band_data:
            shape = band_data[band].shape
            break
        return np.zeros(shape)
    
    nir = band_data[nir_band].astype(np.float32)
    red = band_data[red_band].astype(np.float32)
    
    # Avoid division by zero
    denominator = nir + red
    denominator[denominator == 0] = 1
    
    ndvi = (nir - red) / denominator
    
    return ndvi

def visualize_image(img_dir, output_dir=None, include_ndvi=True):
    """
    Create visualization for a single image directory.
    
    Args:
        img_dir: Path to image directory (either a tile directory or patch directory)
        output_dir: Directory to save visualizations
        include_ndvi: Whether to include NDVI visualization
    
    Returns:
        Output file path if saved, None otherwise
    """
    try:
        # First, check if this is a patch directory with band files or a tile directory
        band_files = glob.glob(os.path.join(img_dir, "*.tif"))
        
        if not band_files:
            # This might be a tile directory, look for patch directories
            subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            if not subdirs:
                print(f"Error: No subdirectories found in {img_dir}", file=sys.stderr)
                return None
                
            # Try to find a patch directory that contains band files
            patch_dir = None
            for subdir in subdirs:
                subdir_path = os.path.join(img_dir, subdir)
                if glob.glob(os.path.join(subdir_path, "*.tif")):
                    patch_dir = subdir_path
                    break
                    
            if not patch_dir:
                print(f"Error: No patch directories with band files found in {img_dir}", file=sys.stderr)
                return None
                
            print(f"Found patch directory: {os.path.basename(patch_dir)}")
            # Update img_dir to the patch directory
            img_dir = patch_dir
            
        # Load metadata
        metadata = load_image_metadata(img_dir)
        
        # List all .tif files to identify available bands
        all_tif_files = glob.glob(os.path.join(img_dir, "*.tif"))
        available_bands = []
        
        for tif_file in all_tif_files:
            # Extract band name (e.g., B02, B03) from filename
            filename = os.path.basename(tif_file)
            band_match = re.search(r'_(B\d+A?)\.tif$', filename)
            if band_match:
                available_bands.append(band_match.group(1))
        
        if not available_bands:
            print(f"Error: No band files found in {img_dir}", file=sys.stderr)
            return None
        
        print(f"Available bands: {', '.join(available_bands)}")
        
        # Load available bands
        bands_to_load = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        bands_to_load = [b for b in bands_to_load if b in available_bands]
        
        band_data, missing_bands = load_bands(img_dir, bands_to_load)
        
        # Check if we have enough bands to create visualizations
        essential_bands = ['B02', 'B03', 'B04']
        essential_missing = [band for band in essential_bands if band not in band_data]
        
        if essential_missing:
            print(f"Error: Cannot visualize {img_dir} - Missing essential bands: {essential_missing}", file=sys.stderr)
            # Create a message image instead
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            ax.text(0.5, 0.5, f"Missing essential bands: {essential_missing}\nCannot visualize this image.", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{os.path.basename(img_dir)}_error.png")
                plt.savefig(output_file, dpi=150)
                plt.close()
                return output_file
            else:
                plt.show()
                return None
        
        # Create figure
        if include_ndvi:
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # True Color (RGB)
        rgb = create_rgb_image(band_data, 'B04', 'B03', 'B02', scale=0.0001)
        axs[0, 0].imshow(rgb)
        axs[0, 0].set_title("True Color (RGB)")
        
        # False Color (NIR-R-G)
        if 'B08' in band_data and all(b in band_data for b in ['B04', 'B03']):
            false_color = create_rgb_image(band_data, 'B08', 'B04', 'B03', scale=0.0001)
            axs[0, 1].imshow(false_color)
            axs[0, 1].set_title("False Color (NIR-R-G)")
        else:
            axs[0, 1].text(0.5, 0.5, "Missing bands for False Color", ha='center', va='center')
            axs[0, 1].axis('off')
        
        if include_ndvi:
            # NDVI
            if 'B08' in band_data and 'B04' in band_data:
                ndvi = create_ndvi_image(band_data)
                ndvi_plot = axs[1, 0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                axs[1, 0].set_title("NDVI")
                plt.colorbar(ndvi_plot, ax=axs[1, 0], fraction=0.046, pad=0.04)
            else:
                axs[1, 0].text(0.5, 0.5, "Missing bands for NDVI", ha='center', va='center')
                axs[1, 0].axis('off')
            
            # Plot land cover labels if available in metadata
            if metadata and 'labels' in metadata:
                axs[1, 1].axis('off')
                land_cover = "\n".join(metadata['labels'])
                axs[1, 1].text(0.5, 0.5, f"Land Cover Classes:\n{land_cover}", 
                            ha='center', va='center', fontsize=12)
            else:
                # More bands visualization
                if all(b in band_data for b in ['B12', 'B8A', 'B04']):
                    swir = create_rgb_image(band_data, 'B12', 'B8A', 'B04', scale=0.0001)
                    axs[1, 1].imshow(swir)
                    axs[1, 1].set_title("SWIR Composite (B12-B8A-B04)")
                else:
                    axs[1, 1].text(0.5, 0.5, "Missing bands for SWIR Composite", ha='center', va='center')
                    axs[1, 1].axis('off')
        
        # Set title with acquisition date if available
        if metadata and 'acquisition_date' in metadata:
            date_str = metadata['acquisition_date']
            plt.suptitle(f"Sentinel-2 Image: {os.path.basename(img_dir)}\nAcquisition Date: {date_str}", 
                        fontsize=14)
        else:
            # Try to extract date from directory name
            date_str = os.path.basename(img_dir).split('_')[2]
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
                date_formatted = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                plt.suptitle(f"Sentinel-2 Image: {os.path.basename(img_dir)}\nAcquisition Date: {date_formatted}", 
                            fontsize=14)
            except:
                plt.suptitle(f"Sentinel-2 Image: {os.path.basename(img_dir)}", fontsize=14)
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.basename(img_dir)}.png")
            plt.savefig(output_file, dpi=150)
            plt.close()
            print(f"Successfully visualized and saved: {os.path.basename(img_dir)}")
            return output_file
        else:
            plt.show()
            print(f"Successfully visualized: {os.path.basename(img_dir)}")
            return None
            
    except Exception as e:
        print(f"Error visualizing {img_dir}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def explore_dataset(data_root, num_samples=5, output_dir=None, random_seed=None):
    """
    Explore the BigEarthNet-S2 dataset by visualizing sample images.
    
    Args:
        data_root: Path to the dataset
        num_samples: Number of sample images to visualize
        output_dir: Directory to save visualizations
        random_seed: Random seed for reproducibility
    """
    # Check if data_root exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset path '{data_root}' does not exist", file=sys.stderr)
        return
    
    # First, identify if we have a nested structure by checking a few directories
    top_level_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                     if os.path.isdir(os.path.join(data_root, d))]
    
    if not top_level_dirs:
        print(f"Error: No valid image directories found in {data_root}", file=sys.stderr)
        return
    
    # Check if we're dealing with a nested structure by looking for band files
    # vs looking for patch directories
    # We need to check multiple directories as some might be empty or malformed
    is_nested = False
    checked_dirs = 0
    for d in top_level_dirs:
        if checked_dirs >= 5:  # Check up to 5 directories
            break
        
        # Look for .tif files directly in the top-level dir
        band_files = glob.glob(os.path.join(d, "*.tif"))
        
        # If no .tif files, check for subdirectories that might contain patches
        if not band_files:
            subdirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            if subdirs:
                # Check if any subdirectory contains .tif files
                has_patch_with_tif = False
                for subdir in subdirs:
                    if glob.glob(os.path.join(d, subdir, "*.tif")):
                        has_patch_with_tif = True
                        break
                if has_patch_with_tif:
                    is_nested = True
                    break # Confirmed nested structure
            # If no subdirs, might be an empty top-level dir, continue checking others
        else:
            # Found .tif files directly, assume flat structure
            is_nested = False
            break # Confirmed flat structure
            
        checked_dirs += 1

    # Define image_dirs based on structure type
    image_dirs = []
    if is_nested:
        print("Detected nested dataset structure (tiles containing patches)")
        # Get all patch directories (these contain the actual band files)
        print("Scanning for patch directories...")
        for tile_dir in tqdm(top_level_dirs, desc="Scanning tiles"):
            # Look for patch directories inside each tile directory
            try:
                for d in os.listdir(tile_dir):
                    patch_path = os.path.join(tile_dir, d)
                    if os.path.isdir(patch_path):
                        # Verify this directory has band files
                        if glob.glob(os.path.join(patch_path, "*.tif")):
                            image_dirs.append(patch_path)
            except FileNotFoundError:
                print(f"Warning: Tile directory {tile_dir} not found or inaccessible.", file=sys.stderr)
                continue
        
        if not image_dirs:
            print(f"Error: No valid patch directories with band files found in {data_root}", file=sys.stderr)
            return False
        
        print(f"Found {len(image_dirs)} patch directories with band files")
    else:
        print("Detected flat dataset structure (directories with band files)")
        # Filter top_level_dirs to only include those with actual band files
        print("Verifying directories...")
        for dir_path in tqdm(top_level_dirs, desc="Verifying image directories"):
            if glob.glob(os.path.join(dir_path, "*.tif")):
                image_dirs.append(dir_path)
                
        if not image_dirs:
            print(f"Error: No valid image directories with band files found in {data_root}", file=sys.stderr)
            return False
            
        print(f"Found {len(image_dirs)} valid image directories in dataset")
    
    # Sample images AFTER identifying the correct directories
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if num_samples >= len(image_dirs):
        sampled_dirs = image_dirs
    else:
        # Ensure image_dirs is not empty before sampling
        if not image_dirs:
            print("Error: No image directories available for sampling.", file=sys.stderr)
            return False
        sampled_dirs = np.random.choice(image_dirs, num_samples, replace=False)
    
    print(f"Visualizing {len(sampled_dirs)} sample images...")
    
    # Create a progress bar with more descriptive information
    tqdm_bar = tqdm(sampled_dirs, desc="Processing images", unit="image", 
                   bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")
    
    visualized_files = []
    errors = 0
    success = 0
    
    for img_dir in tqdm_bar:
        tqdm_bar.set_description(f"Processing {os.path.basename(img_dir)}")
        output_file = visualize_image(img_dir, output_dir)
        if output_file:
            visualized_files.append(output_file)
            success += 1
        else:
            errors += 1
    
    if output_dir:
        print(f"Saved {success} visualizations to {output_dir}")
    
    if errors > 0:
        print(f"Warning: {errors}/{len(sampled_dirs)} images had errors and could not be visualized properly", file=sys.stderr)
    
    return success > 0

def analyze_temporal_coverage(data_root, output_dir=None):
    """Analyze temporal distribution of images in the dataset."""
    # Check if data_root exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset path '{data_root}' does not exist", file=sys.stderr)
        return
        
    # Get all image directories
    image_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    if not image_dirs:
        print(f"Error: No valid image directories found in {data_root}", file=sys.stderr)
        return
        
    print(f"Analyzing temporal distribution of {len(image_dirs)} images...")
    
    # Extract dates from directory names
    dates = []
    processing_bar = tqdm(image_dirs, desc="Extracting dates", unit="image")
    
    for img_dir in processing_bar:
        parts = img_dir.split('_')
        if len(parts) < 3:
            continue
            
        date_str = parts[2]  # E.g., 20180525T094029
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
            dates.append(date_obj)
        except ValueError:
            continue
    
    if not dates:
        print("No valid dates found in directory names.", file=sys.stderr)
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'date': dates,
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates]
    })
    
    # Plot temporal distribution
    plt.figure(figsize=(12, 6))
    
    # Monthly distribution
    monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_counts['year_month'] = monthly_counts.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
    
    plt.bar(monthly_counts['year_month'], monthly_counts['count'])
    plt.xticks(rotation=90)
    plt.title('Temporal Distribution of Images by Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=150)
    else:
        plt.show()
    
    # Print temporal coverage summary
    print("\nTemporal Coverage Summary:")
    print(f"Date Range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
    print(f"Total number of distinct dates: {df['date'].dt.date.nunique()}")
    print(f"Number of images per year:")
    print(df.groupby('year').size())

def main():
    parser = argparse.ArgumentParser(description='Explore BigEarthNet-S2 Dataset')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BigEarthNet-S2 dataset')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample images to visualize')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--temporal_analysis', action='store_true', help='Perform temporal analysis of the dataset')
    parser.add_argument('--img_dir', type=str, help='Visualize a specific image directory')
    
    args = parser.parse_args()
    
    # Check for required paths
    if not os.path.exists(args.data_root):
        print(f"Error: The specified data_root path '{args.data_root}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.img_dir:
            # Visualize specific image
            img_path = args.img_dir
            if not os.path.isdir(img_path) and os.path.isdir(os.path.join(args.data_root, args.img_dir)):
                img_path = os.path.join(args.data_root, args.img_dir)
            
            if not os.path.isdir(img_path):
                print(f"Error: Image directory '{img_path}' does not exist.", file=sys.stderr)
                sys.exit(1)
                
            print(f"Visualizing single image directory: {img_path}")
            visualize_image(img_path, args.output_dir)
        elif args.temporal_analysis:
            # Analyze temporal coverage
            analyze_temporal_coverage(args.data_root, args.output_dir)
        else:
            # Explore dataset with sample visualizations
            explore_dataset(args.data_root, args.num_samples, args.output_dir, args.random_seed)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 