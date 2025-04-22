import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import DataLoader # Needed for DataLoader creation

# Helper function for contrast stretching
def stretch_contrast(img_array):
    """Apply contrast stretching using percentile clipping (2nd-98th)."""
    stretched = np.zeros_like(img_array, dtype=np.float32)
    for i in range(img_array.shape[2]): # Iterate through channels
        channel = img_array[:, :, i].astype(np.float32)
        p2, p98 = np.percentile(channel, (2, 98))
        if p98 - p2 > 1e-6: # Avoid division by zero/small range
            stretched_channel = (channel - p2) / (p98 - p2)
            stretched[:, :, i] = np.clip(stretched_channel, 0, 1)
        else:
            # If range is too small, just clip to [0, 1]
            stretched[:, :, i] = np.clip(channel, 0, 1)
    return stretched

# Evaluation and visualization functions
def visualize_predictions(model, test_loader, config, num_samples=5):
    """
    Visualize predictions on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        config: Configuration object
        num_samples: Number of samples to visualize
    """
    device = config.device
    model = model.to(device)
    model.eval()
    
    # Create visualization DataLoader with batch_size=1 if test_loader has a different batch size
    # Ensure test_loader is not None before accessing attributes
    if test_loader is not None and hasattr(test_loader, 'batch_size') and test_loader.batch_size != 1:
        viz_dataset = test_loader.dataset
        # Use shuffle=False for consistent visualization samples if desired
        viz_loader = DataLoader(viz_dataset, batch_size=1, shuffle=False, num_workers=0) 
    elif test_loader is not None:
        viz_loader = test_loader
    else:
        print("Error: test_loader is None, cannot perform visualization.")
        return

    # Create output directory for visualizations using config
    vis_dir = config.visualization_dir # Use path from config
    if not vis_dir:
        print("Error: config.visualization_dir not set. Cannot save visualizations.")
        return
    os.makedirs(vis_dir, exist_ok=True) # Ensure it exists
    
    # For computing NDVI change
    pixel_change_thresholds = {
        'low': 0.05,   # Minor change
        'medium': 0.15, # Moderate change
        'high': 0.25    # Significant change
    }
    
    with torch.no_grad():
        # Ensure we don't try to visualize more samples than available
        actual_samples_to_visualize = min(num_samples, len(viz_loader))
        if actual_samples_to_visualize == 0:
            print("No samples available in the loader for visualization.")
            return
            
        print(f"Generating {actual_samples_to_visualize} visualizations...")
        
        for i, (inputs, targets) in enumerate(viz_loader):
            if i >= actual_samples_to_visualize:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Convert tensors to numpy arrays for visualization
            # Take the first sample in the batch (batch_size=1)
            input_seq = inputs[0].cpu().numpy()
            target_img = targets[0].cpu().numpy()
            output_img = outputs[0].cpu().numpy()
            
            # Create figure for the sequence and prediction
            # Adjust figure size based on whether NDVI is plotted
            plot_ndvi = target_img.shape[0] >= 4
            rows, cols = (3, 4) if plot_ndvi else (2, 3) # Reduced columns as GIF is removed
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten() # Flatten for easier indexing
            ax_idx = 0
            
            # Plot input sequence (show only every 3rd frame to fit in figure)
            for j in range(4): # Show up to 4 input frames
                frame_idx = j * 3  # Show frames 0, 3, 6, 9
                if frame_idx >= input_seq.shape[0]:
                    break # Don't exceed sequence length
                frame = input_seq[frame_idx]
                
                # For RGB visualization, take the first 3 bands if available
                if frame.shape[0] >= 3:
                    rgb_frame = frame[:3].transpose(1, 2, 0)
                    # Apply contrast stretching for visualization
                    stretched_frame = stretch_contrast(rgb_frame)
                    axes[ax_idx].imshow(stretched_frame)
                else:
                    # Grayscale doesn't need stretching in the same way
                    axes[ax_idx].imshow(np.clip(frame[0], 0, 1), cmap='gray')
                
                axes[ax_idx].set_title(f"Input frame {frame_idx}")
                axes[ax_idx].axis('off')
                ax_idx += 1
            
            # Fill remaining input frame axes if less than 4 were shown
            while ax_idx < 4:
                axes[ax_idx].axis('off')
                ax_idx += 1

            # --- Plot target and prediction --- 
            target_ax = ax_idx
            pred_ax = ax_idx + 1
            rgb_err_ax = ax_idx + 2
            ax_idx += 3
            
            if target_img.shape[0] >= 3:
                # RGB visualization
                rgb_target = target_img[:3].transpose(1, 2, 0)
                rgb_output = output_img[:3].transpose(1, 2, 0)
                
                # Apply contrast stretching for visualization
                stretched_target = stretch_contrast(rgb_target)
                stretched_output = stretch_contrast(rgb_output)
                
                axes[target_ax].imshow(stretched_target)
                axes[pred_ax].imshow(stretched_output)
                axes[target_ax].set_title("Target Frame (RGB)")
                axes[pred_ax].set_title("Predicted Frame (RGB)")
                
                # RGB Error map (MSE per pixel) - Calculate on original outputs
                error_map = np.mean((np.clip(rgb_target, 0, 1) - np.clip(rgb_output, 0, 1)) ** 2, axis=2)
                im = axes[rgb_err_ax].imshow(error_map, cmap='hot')
                axes[rgb_err_ax].set_title("RGB Error Map (MSE)")
                plt.colorbar(im, ax=axes[rgb_err_ax], fraction=0.046, pad=0.04)
                
            else:
                # Grayscale visualization
                axes[target_ax].imshow(target_img[0], cmap='gray')
                axes[pred_ax].imshow(output_img[0], cmap='gray')
                
                # Error map
                error_map = (target_img[0] - output_img[0]) ** 2
                im = axes[rgb_err_ax].imshow(error_map, cmap='hot')
                plt.colorbar(im, ax=axes[rgb_err_ax], fraction=0.046, pad=0.04)
                
                axes[target_ax].set_title("Target Frame (Grayscale)")
                axes[pred_ax].set_title("Predicted Frame (Grayscale)")
                axes[rgb_err_ax].set_title("Error Map (MSE)")
            
            axes[target_ax].axis('off')
            axes[pred_ax].axis('off')
            axes[rgb_err_ax].axis('off')

            # --- Plot NDVI comparison if possible --- 
            if plot_ndvi:
                target_ndvi_ax = ax_idx
                pred_ndvi_ax = ax_idx + 1
                ndvi_err_ax = ax_idx + 2
                change_map_ax = ax_idx + 3 # Replaced GIF slot
                stats_ax = ax_idx + 4
                ax_idx += 5
                
                # NDVI = (NIR - Red) / (NIR + Red)
                # Check indices: B08 (NIR) is index 3, B04 (Red) is index 2 in default config bands
                nir_idx = config.bands.index('B08') if 'B08' in config.bands else -1
                red_idx = config.bands.index('B04') if 'B04' in config.bands else -1

                if nir_idx != -1 and red_idx != -1 and target_img.shape[0] > max(nir_idx, red_idx):
                    target_nir = target_img[nir_idx]
                    target_red = target_img[red_idx]
                    output_nir = output_img[nir_idx]
                    output_red = output_img[red_idx]
                    
                    target_ndvi = (target_nir - target_red) / (target_nir + target_red + 1e-8)
                    output_ndvi = (output_nir - output_red) / (output_nir + output_red + 1e-8)
                    
                    # NDVI visualizations
                    im1 = axes[target_ndvi_ax].imshow(target_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    im2 = axes[pred_ndvi_ax].imshow(output_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    plt.colorbar(im1, ax=axes[target_ndvi_ax], fraction=0.046, pad=0.04)
                    plt.colorbar(im2, ax=axes[pred_ndvi_ax], fraction=0.046, pad=0.04)
                    
                    axes[target_ndvi_ax].set_title("Target NDVI")
                    axes[pred_ndvi_ax].set_title("Predicted NDVI")
                    
                    # NDVI error map
                    ndvi_error = np.abs(target_ndvi - output_ndvi)
                    im3 = axes[ndvi_err_ax].imshow(ndvi_error, cmap='hot', vmin=0, vmax=0.5)
                    plt.colorbar(im3, ax=axes[ndvi_err_ax], fraction=0.046, pad=0.04)
                    axes[ndvi_err_ax].set_title("NDVI Error")
                    
                    # Change detection map (for environmental monitoring)
                    change_map = np.zeros_like(ndvi_error)
                    
                    # Categorize changes
                    change_map[ndvi_error < pixel_change_thresholds['low']] = 1      # No/Minor change
                    change_map[ndvi_error >= pixel_change_thresholds['low']] = 2     # Low change
                    change_map[ndvi_error >= pixel_change_thresholds['medium']] = 3  # Medium change
                    change_map[ndvi_error >= pixel_change_thresholds['high']] = 4    # High change
                    
                    # Create custom colormap for change detection
                    change_cmap = ListedColormap(['black', 'green', 'yellow', 'orange', 'red'])
                    
                    im4 = axes[change_map_ax].imshow(change_map, cmap=change_cmap, vmin=1, vmax=4) # Adjusted vmin
                    cbar = plt.colorbar(im4, ax=axes[change_map_ax], fraction=0.046, pad=0.04, ticks=[1, 2, 3, 4])
                    cbar.set_ticklabels(['No/Minor', 'Low', 'Medium', 'High']) # Updated label
                    axes[change_map_ax].set_title("NDVI Change Detection")
                    
                    # Calculate statistics on changes
                    total_pixels = change_map.size
                    change_percentages = {
                        'no_minor': np.sum(change_map == 1) / total_pixels * 100,
                        'low': np.sum(change_map == 2) / total_pixels * 100,
                        'medium': np.sum(change_map == 3) / total_pixels * 100,
                        'high': np.sum(change_map == 4) / total_pixels * 100
                    }
                    
                    # Display change statistics
                    axes[stats_ax].axis('off')
                    stats_text = (
                        f"Change Statistics:\n"
                        f" No/Minor: {change_percentages['no_minor']:.1f}%\n"
                        f" Low: {change_percentages['low']:.1f}%\n"
                        f" Medium: {change_percentages['medium']:.1f}%\n"
                        f" High: {change_percentages['high']:.1f}%\n\n"
                        f"NDVI Error (Mean): {np.mean(ndvi_error):.4f}\n"
                        f"RGB Error (MSE): {np.mean(error_map):.4f}"
                    )
                    axes[stats_ax].text(0.1, 0.5, stats_text, fontsize=10, va='center')
                    axes[stats_ax].set_title("Change Statistics")
                    
                    axes[target_ndvi_ax].axis('off')
                    axes[pred_ndvi_ax].axis('off')
                    axes[ndvi_err_ax].axis('off')
                    axes[change_map_ax].axis('off')
                else:
                    # Handle missing bands for NDVI plots
                    axes[target_ndvi_ax].text(0.5, 0.5, "NDVI N/A", ha='center', va='center')
                    axes[pred_ndvi_ax].text(0.5, 0.5, "NDVI N/A", ha='center', va='center')
                    axes[ndvi_err_ax].text(0.5, 0.5, "NDVI N/A", ha='center', va='center')
                    axes[change_map_ax].text(0.5, 0.5, "Change N/A", ha='center', va='center')
                    axes[stats_ax].text(0.5, 0.5, "Stats N/A", ha='center', va='center')
                    for k in range(5): axes[ax_idx - 5 + k].axis('off')

            # --- Turn off unused axes --- 
            while ax_idx < len(axes):
                 axes[ax_idx].axis('off')
                 ax_idx += 1
            
            # Set main title with metrics
            mse = np.mean(error_map) # Use the already calculated error map
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) # Avoid log10(0)
            plt.suptitle(f"Sample {i+1} - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB", fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            save_path = os.path.join(vis_dir, f'prediction_sample_{i+1}.png')
            plt.savefig(save_path, dpi=150)
            plt.close(fig) # Close the figure to free memory
            
            # --- GIF creation code removed here --- 
            
    print(f"Saved {actual_samples_to_visualize} visualizations to {vis_dir}") 