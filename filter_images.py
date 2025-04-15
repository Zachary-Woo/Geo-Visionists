"""
Filter Geographic Images from BigEarthNet-S2 by Bounding Box

This script helps identify and filter Sentinel-2 images from the BigEarthNet-S2 dataset
that are located within a specified geographic bounding box.

It will:
1. Scan tile/patch directories.
2. Read the metadata or TIF file for each patch.
3. Extract the geographic coordinates.
4. Filter images that fall within the specified bounding box.
5. Create a list of selected images for use in training or analysis.
"""

import os
import json
import glob
import pandas as pd
import rasterio
import rasterio.warp
from tqdm import tqdm
import argparse
import shutil
from shapely.geometry import Polygon, box
import sys
import logging

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.ERROR)

# Track CRS warnings to avoid repetition
crs_warnings = set()

def extract_coordinates(metadata_file):
    """Extract coordinates from a BigEarthNet metadata file."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # BigEarthNet metadata contains a GeoJSON-like coordinates field
        if 'coordinates' in metadata:
            # Ensure it's a list of coordinate pairs
            coords_list = metadata['coordinates']
            if isinstance(coords_list, list) and len(coords_list) > 0 and isinstance(coords_list[0], list):
                 # Sometimes the coordinates are nested one level deeper
                 if len(coords_list) == 1 and isinstance(coords_list[0][0], list):
                     coords_list = coords_list[0]
                 
                 lats = [point[1] for point in coords_list if len(point) > 1]
                 lons = [point[0] for point in coords_list if len(point) > 1]
                 
                 if not lats or not lons:
                      print(f"Warning: Could not parse lat/lon from coordinates in {metadata_file}")
                      return None
                 
                 return {
                     'min_lat': min(lats),
                     'max_lat': max(lats),
                     'min_lon': min(lons),
                     'max_lon': max(lons),
                     'centroid_lat': sum(lats) / len(lats),
                     'centroid_lon': sum(lons) / len(lons)
                 }
            else:
                 print(f"Warning: Unexpected format for 'coordinates' in {metadata_file}")
                 return None
                 
    except (json.JSONDecodeError, KeyError, FileNotFoundError, TypeError) as e:
        print(f"Error processing metadata {metadata_file}: {e}")
    
    return None

def extract_from_tif(tif_file):
    """Extract coordinates from a GeoTIFF file using rasterio."""
    try:
        with rasterio.open(tif_file) as src:
            # Get the bounds in the file's CRS
            bounds = src.bounds
            
            # Check if CRS is geographic (like WGS84)
            if src.crs and src.crs.is_geographic:
                # Assume lat/lon order for geographic CRS
                return {
                    'min_lat': bounds.bottom,
                    'max_lat': bounds.top,
                    'min_lon': bounds.left,
                    'max_lon': bounds.right,
                    'centroid_lat': (bounds.bottom + bounds.top) / 2,
                    'centroid_lon': (bounds.left + bounds.right) / 2
                }
            else:
                # If it's a projected CRS, we need to transform bounds to lat/lon (WGS84)
                crs_str = str(src.crs)
                if crs_str not in crs_warnings:
                    crs_warnings.add(crs_str)
                    print(f"Info: Transforming images with CRS {crs_str} to WGS84 (only shown once per CRS)")
                
                try:
                    # Transform corner points
                    # Create a transformer to WGS84 (EPSG:4326)
                    xs = [bounds.left, bounds.right]
                    ys = [bounds.bottom, bounds.top]
                    wgs84_crs = {'init': 'epsg:4326'}
                    xs_t, ys_t = rasterio.warp.transform(src.crs, wgs84_crs, xs, ys)
                    
                    return {
                        'min_lat': min(ys_t),
                        'max_lat': max(ys_t),
                        'min_lon': min(xs_t),
                        'max_lon': max(xs_t),
                        'centroid_lat': sum(ys_t) / len(ys_t),
                        'centroid_lon': sum(xs_t) / len(xs_t)
                    }
                except Exception as transform_e:
                    print(f"Error transforming coordinates for {tif_file}: {transform_e}")
                    return None
            
    except Exception as e:
        print(f"Error processing TIF {tif_file}: {e}")
    
    return None

def is_in_region(coords, target_bbox, buffer_degrees=0.01):
    """Check if the given coordinates overlap with the target bounding box (with buffer)."""
    if not coords:
        return False
    
    # Check if any corner of the image intersects with the target region
    # This is a simpler check that avoids creating polygon objects for every patch
    image_min_lon = min(coords['min_lon'], coords['max_lon'])
    image_max_lon = max(coords['min_lon'], coords['max_lon'])
    image_min_lat = min(coords['min_lat'], coords['max_lat'])
    image_max_lat = max(coords['min_lat'], coords['max_lat'])
    
    # Add buffer to the target bbox
    target_min_lon = target_bbox['min_lon'] - buffer_degrees
    target_max_lon = target_bbox['max_lon'] + buffer_degrees
    target_min_lat = target_bbox['min_lat'] - buffer_degrees
    target_max_lat = target_bbox['max_lat'] + buffer_degrees
    
    # Check if bounding boxes overlap
    return (image_min_lon <= target_max_lon and image_max_lon >= target_min_lon and
            image_min_lat <= target_max_lat and image_max_lat >= target_min_lat)

def filter_region_images(data_root, target_bbox, output_file='filtered_images.txt', copy_to_folder=None):
    """
    Filter images from BigEarthNet-S2 that are in the specified region.
    Handles nested (tile -> patch) directory structure.
    
    Args:
        data_root: Path to BigEarthNet-S2 dataset
        target_bbox: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        output_file: Where to save the list of selected images
        copy_to_folder: Optional folder to copy selected images to
    """
    # Get all top-level tile directories
    tile_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    # Find all patch directories within the tile directories
    patch_dirs_info = []
    print(f"Scanning {len(tile_dirs)} tile directories for patches...")
    for tile_dir in tqdm(tile_dirs, desc="Scanning tiles"):
        tile_path = os.path.join(data_root, tile_dir)
        try:
            for patch_name in os.listdir(tile_path):
                patch_path = os.path.join(tile_path, patch_name)
                if os.path.isdir(patch_path):
                    # Check if patch directory contains TIF files to be sure
                    if glob.glob(os.path.join(patch_path, "*.tif")):
                         patch_dirs_info.append({'name': patch_name, 'path': patch_path})
        except FileNotFoundError:
            print(f"Warning: Tile directory {tile_path} not found or inaccessible.")
            continue
            
    if not patch_dirs_info:
        print(f"Error: No patch directories with TIF files found under {data_root}")
        return []

    # Create output directory if copying
    if copy_to_folder:
        os.makedirs(copy_to_folder, exist_ok=True)
    
    selected_images = []
    image_data = []
    
    print(f"Processing {len(patch_dirs_info)} patch directories...")
    
    # Create a bounding box for target region with buffer
    buffer_degrees = 0.01
    target_box = box(
        target_bbox['min_lon'] - buffer_degrees,
        target_bbox['min_lat'] - buffer_degrees,
        target_bbox['max_lon'] + buffer_degrees,
        target_bbox['max_lat'] + buffer_degrees
    )
    
    # Print the target coordinates for confirmation
    print(f"Target region: Lat [{target_bbox['min_lat']}, {target_bbox['max_lat']}], Lon [{target_bbox['min_lon']}, {target_bbox['max_lon']}]")
    
    # Track progress with tqdm
    for patch_info in tqdm(patch_dirs_info, desc="Filtering patches"):
        patch_path = patch_info['path']
        patch_name = patch_info['name'] # This is the directory name like S2*_XX_YY
        
        # First try to find metadata JSON file within the patch directory
        metadata_files = glob.glob(os.path.join(patch_path, "*_metadata.json"))
        
        coords = None
        if metadata_files:
            coords = extract_coordinates(metadata_files[0])
        else:
            # If no metadata file, try to get coordinates from the first TIF file
            # within the patch directory
            tif_files = glob.glob(os.path.join(patch_path, "*.tif"))
            if tif_files:
                # Try B04 first as it's likely WGS84 or similar
                b04_files = glob.glob(os.path.join(patch_path, "*_B04.tif"))
                if b04_files:
                    coords = extract_from_tif(b04_files[0])
                else:
                    coords = extract_from_tif(tif_files[0]) # Fallback to first TIF
        
        if coords and is_in_region(coords, target_bbox):
            selected_images.append(patch_name) # Store the patch name
            # Store all information
            # Use relative path for consistency with prepare_sequences
            relative_patch_path = os.path.relpath(patch_path, data_root)
            image_data.append({
                'directory': patch_name, # The patch directory name
                'tile_directory': os.path.basename(os.path.dirname(patch_path)), # Parent tile directory
                'relative_path': relative_patch_path,
                'date': patch_name.split('_')[2],  # Extract date from folder name
                'min_lat': coords['min_lat'],
                'max_lat': coords['max_lat'],
                'min_lon': coords['min_lon'],
                'max_lon': coords['max_lon'],
                'centroid_lat': coords['centroid_lat'],
                'centroid_lon': coords['centroid_lon']
            })
            
            # Copy to new folder if requested - copy the whole patch directory
            if copy_to_folder:
                # Create the parent tile directory in the destination if it doesn't exist
                dest_tile_dir = os.path.join(copy_to_folder, os.path.basename(os.path.dirname(patch_path)))
                os.makedirs(dest_tile_dir, exist_ok=True)
                # Copy the patch directory into the destination tile directory
                # Use dirs_exist_ok=True for robustness
                shutil.copytree(patch_path, os.path.join(dest_tile_dir, patch_name), dirs_exist_ok=True)
    
    # Save the list to a text file
    with open(output_file, 'w') as f:
        for img_dir in selected_images:
            f.write(f"{img_dir}\n")
    
    # Save more detailed information to CSV
    if image_data:
        df = pd.DataFrame(image_data)
        # Reorder columns for clarity
        cols = ['directory', 'tile_directory', 'relative_path', 'date', 'min_lat', 'max_lat', 'min_lon', 'max_lon', 'centroid_lat', 'centroid_lon']
        df = df[cols]
        df.to_csv(output_file.replace('.txt', '.csv'), index=False)
    
    print(f"Found {len(selected_images)} patch directories in the specified region.")
    if len(selected_images) > 0:
        print(f"Saved list to {output_file}")
        if copy_to_folder:
            print(f"Copied selected patch directories to {copy_to_folder}")
    else:
         print("No images found in the specified region. Check coordinates and dataset coverage.")
         print(f"List file {output_file} created but is empty.")
    
    return selected_images

def main():
    parser = argparse.ArgumentParser(description='Filter BigEarthNet-S2 images by geographic bounding box')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BigEarthNet-S2 dataset')
    parser.add_argument('--output_file', type=str, default='filtered_images.txt', help='Output file to save list of selected images')
    parser.add_argument('--copy_to', type=str, default=None, help='Optional path to copy selected images to a new directory structure')
    # Bounding box arguments
    parser.add_argument('--min_lat', type=float, required=True, help='Minimum latitude of the bounding box')
    parser.add_argument('--max_lat', type=float, required=True, help='Maximum latitude of the bounding box')
    parser.add_argument('--min_lon', type=float, required=True, help='Minimum longitude of the bounding box')
    parser.add_argument('--max_lon', type=float, required=True, help='Maximum longitude of the bounding box')
    
    args = parser.parse_args()
    
    # Basic validation for lat/lon
    if args.min_lat >= args.max_lat:
         print("Error: min_lat must be less than max_lat.", file=sys.stderr)
         sys.exit(1)
    if args.min_lon >= args.max_lon:
         print("Error: min_lon must be less than max_lon.", file=sys.stderr)
         sys.exit(1)
        
    target_bbox = {
        'min_lat': args.min_lat,
        'max_lat': args.max_lat,
        'min_lon': args.min_lon,
        'max_lon': args.max_lon
    }
    
    filter_region_images(args.data_root, target_bbox, args.output_file, args.copy_to)

if __name__ == "__main__":
    main() 