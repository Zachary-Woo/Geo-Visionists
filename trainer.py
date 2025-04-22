import os
import time
import json
import itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM # Needed for validation metrics calculation
import numpy as np

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
    
    Returns:
        Trained model and training history
    """
    device = config.device
    model = model.to(device)
    
    print(f"Using device: {device} {'(GPU)' if torch.cuda.is_available() else '(CPU)'}")
    if device.type == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
    
    # For mixed precision training
    scaler = torch.amp.GradScaler() if config.mixed_precision and device.type == 'cuda' else None
    
    # Create checkpoint and visualization directories if they don't exist
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {config.checkpoint_dir}")
    
    if not os.path.exists(config.visualization_dir):
        os.makedirs(config.visualization_dir, exist_ok=True)
        print(f"Created visualization directory: {config.visualization_dir}")
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    # Resume from checkpoint if specified
    if config.resume_training and config.checkpoint_path:
        if os.path.isfile(config.checkpoint_path):
            print(f"Loading checkpoint from {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=device) # Load to correct device
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load other training state variables
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            
            # Load history if available
            if 'history' in checkpoint:
                history = checkpoint['history']
            
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {config.checkpoint_path}, starting from scratch")
    
    # Track training start time to estimate completion
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Display more detailed progress
        epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(epoch_progress):
            # Check if the batch is None (due to safe_collate returning None for an empty batch)
            if inputs is None or targets is None:
                # print(f"Skipping empty/corrupted batch {batch_idx}") # Optional: Log skipped batches
                continue # Skip to the next batch
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training (only on CUDA)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Update progress bar with more information
            if batch_idx % 10 == 0:
                # Calculate elapsed time and estimate remaining time
                elapsed_time = time.time() - training_start_time
                progress = (epoch * len(train_loader) + batch_idx + 1) / (config.num_epochs * len(train_loader))
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    
                    # Format time as hours:minutes
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)
                    
                    # Update progress bar description
                    avg_loss = train_loss / batch_count if batch_count > 0 else 0
                    epoch_progress.set_description(
                        f"Epoch {epoch+1}/{config.num_epochs} - Loss: {avg_loss:.4f} - ETA: {remaining_hours}h {remaining_minutes}m"
                    )
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        history['train_loss'].append(avg_train_loss)
        
        # Save checkpoint after each epoch (or less frequently based on config)
        # Modified to save less frequently based on config
        if (epoch + 1) % config.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Validation phase (every eval_interval epochs)
        if (epoch + 1) % config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_mae = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            
            # Use a subset of validation data for faster evaluation if dataset is large
            # Using the full validation set now, adjust if needed
            # max_val_batches = min(len(val_loader), 100)  # Cap at 100 batches max for validation
            max_val_batches = len(val_loader)
            
            if max_val_batches == 0:
                print("Warning: Validation loader is empty. Skipping validation.")
                history['val_loss'].append(float('nan')) # Record NaN for skipped validation
                history['val_mse'].append(float('nan'))
                history['val_mae'].append(float('nan'))
                history['val_psnr'].append(float('nan'))
                history['val_ssim'].append(float('nan'))
                continue # Skip to next epoch
                
            with torch.no_grad():
                val_progress = tqdm(
                    # itertools.islice(val_loader, max_val_batches), # Removed slice to use full val set
                    val_loader,
                    total=max_val_batches,
                    desc="Validation"
                )
                for inputs, targets in val_progress:
                    # Check if the batch is None (can happen if val set also has corrupted files)
                    if inputs is None or targets is None:
                        # print(f"Skipping empty/corrupted validation batch") # Optional
                        max_val_batches -= 1 # Adjust total count if skipping
                        val_progress.total = max_val_batches # Update progress bar total
                        continue
                        
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
                    # Calculate other metrics
                    mse = F.mse_loss(outputs, targets).item()
                    mae = F.l1_loss(outputs, targets).item()
                    
                    # Calculate PSNR - fixing the tensor type issue
                    # Convert mse to tensor before using torch.log10
                    # Ensure mse is not zero or negative before log
                    mse_clamped = max(mse, 1e-10) # Clamp mse to avoid log10(0)
                    mse_tensor = torch.tensor(mse_clamped, device=device)
                    psnr = 10 * torch.log10(1.0 / mse_tensor).item()
                    
                    # Calculate SSIM
                    # Re-initialize SSIM module inside loop if needed, or ensure it handles varying batch sizes
                    # For simplicity, using the one initialized outside the loop
                    # Ensure inputs are on the same device
                    if outputs.device != targets.device:
                        targets = targets.to(outputs.device)
                    ssim_module = SSIM(data_range=1.0, size_average=True, channel=len(config.bands))
                    ssim_value = ssim_module(outputs, targets).item()
                    
                    val_mse += mse
                    val_mae += mae
                    val_psnr += psnr
                    val_ssim += ssim_value
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / max_val_batches
            avg_val_mse = val_mse / max_val_batches
            avg_val_mae = val_mae / max_val_batches
            avg_val_psnr = val_psnr / max_val_batches
            avg_val_ssim = val_ssim / max_val_batches
            
            history['val_loss'].append(avg_val_loss)
            history['val_mse'].append(avg_val_mse)
            history['val_mae'].append(avg_val_mae)
            history['val_psnr'].append(avg_val_psnr)
            history['val_ssim'].append(avg_val_ssim)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(config.experiment_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
                
            # Print metrics
            print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}")
            print(f"PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}\n")

            # Step the scheduler based on validation loss
            scheduler.step(avg_val_loss) 
    
    # Save final model
    final_model_path = os.path.join(config.experiment_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(config.experiment_dir, 'history.json')
    with open(history_path, 'w') as f:
        # Convert any non-serializable values to strings or safe floats
        history_json = {k: [float(val) if isinstance(val, (int, float, np.float32, np.float64)) and np.isfinite(val) else str(val) for val in v] 
                       for k, v in history.items()}
        json.dump(history_json, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    return model, history 