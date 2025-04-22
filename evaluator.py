import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM

# Import model architecture
from model import UNetSwinHybrid 

def evaluate_model(model_path, test_loader, config):
    """
    Evaluate a pre-trained model on the test dataset.
    
    Args:
        model_path: Path to the saved model
        test_loader: DataLoader for test data
        config: Configuration object (used for device and output paths)
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating model from {model_path}")
    
    # Initialize model
    # Ensure UNetSwinHybrid is imported or defined
    model = UNetSwinHybrid(config)
    
    # Load model weights
    # Use weights_only=True for security when loading state dicts
    state_dict = torch.load(model_path, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()
    
    # Initialize metrics
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'psnr': 0.0,
        'ssim': 0.0
    }
    
    # Initialize SSIM module
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=len(config.bands))
    
    # Evaluate on test set
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics
            mse = F.mse_loss(outputs, targets).item()
            mae = F.l1_loss(outputs, targets).item()
            
            # Fix: Convert mse to tensor before using torch.log10
            # Clamp mse to avoid log10(0) or negative values
            mse_clamped = max(mse, 1e-10)
            mse_tensor = torch.tensor(mse_clamped, device=config.device)
            psnr = 10 * torch.log10(1.0 / mse_tensor).item()
            
            # Ensure inputs are on the same device
            if outputs.device != targets.device:
                targets = targets.to(outputs.device)
            ssim_value = ssim_module(outputs, targets).item()
            
            # Accumulate metrics
            metrics['mse'] += mse
            metrics['mae'] += mae
            metrics['psnr'] += psnr
            metrics['ssim'] += ssim_value
    
    # Calculate average metrics
    num_batches = len(test_loader)
    if num_batches > 0:
        for key in metrics:
            metrics[key] /= num_batches
    else:
        print("Warning: Test loader is empty. Metrics cannot be calculated.")
        for key in metrics:
            metrics[key] = float('nan') # Set metrics to NaN
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Save metrics to file within the experiment directory
    if config.experiment_dir:
        metrics_path = os.path.join(config.experiment_dir, 'evaluation_metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True) # Ensure dir exists
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Evaluation metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error saving evaluation metrics to {metrics_path}: {e}")
    else:
        print("Warning: config.experiment_dir not set. Cannot save metrics file.")

    return metrics 