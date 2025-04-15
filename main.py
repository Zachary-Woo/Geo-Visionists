"""
U-Net + Swin Transformer Hybrid Model for Spatiotemporal Forecasting of Satellite Imagery

This script implements a deep learning model that combines U-Net for spatial encoding
and Swin Transformer for temporal reasoning to predict future satellite imagery frames
based on a sequence of previous frames.
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import rasterio
from rasterio.plot import reshape_as_image
import cv2
import argparse
import imageio  # Import imageio at the top level
import subprocess
import sys
import pickle # Added for loading sequences

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SwinConfig, SwinModel
from pytorch_msssim import SSIM

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Configuration
class Config:
    """Configuration for the model and training process."""
    # Data parameters
    data_root = "./dataset/BigEarthNet-S2"
    output_dir = "./output"
    jakarta_coords = {
        # Approximate bounding box for Jakarta, Indonesia
        # Will be used to filter relevant tiles if exact coordinates are known
        'min_lat': -6.4, 'max_lat': -5.9,
        'min_lon': 106.6, 'max_lon': 107.0
    }
    sequence_length = 11  # Input sequence length
    pred_horizon = 1      # Predict 1 frame ahead (12th frame)
    patch_size = 256      # Size of image patches
    bands = ['B02', 'B03', 'B04', 'B08']  # RGB + NIR bands
    
    # Model parameters
    backbone = "resnet50"
    hidden_dim = 256
    transformer_layers = 4
    transformer_heads = 8
    dropout = 0.3
    
    # Training parameters
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 50
    device = None  # Will be set during initialization
    mixed_precision = True
    
    # Evaluation parameters
    eval_interval = 5
    
    # Checkpointing parameters
    save_checkpoint_every = 5  # Save a checkpoint every N epochs
    resume_training = False    # Whether to resume training from checkpoint
    checkpoint_path = None     # Path to checkpoint to resume from
    
    # Experiment tracking
    experiment_name = None # Set after args parsing
    experiment_dir = None # Set after args parsing
    checkpoint_dir = None # Set after args parsing
    visualization_dir = None # Set after args parsing
    
    # Model evaluation
    model_path = None  # Path to model for evaluation only (bypassing training)
    
    # Image count thresholds
    min_train_samples = 100  # Minimum number of training samples to proceed
    
    # Set up directories when config is initialized - NOW DONE LATER in main()
    def __init__(self):
        # Set device - Initialize explicitly to ensure it's checked early
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available. Using CPU.")

    def set_experiment_paths(self, mode):
        """Set experiment name and paths based on mode."""
        # Find the next available run number for this mode
        run_number = 1
        while True:
            self.experiment_name = f"{mode}_{run_number}"
            potential_dir = os.path.join(self.output_dir, self.experiment_name)
            if not os.path.exists(potential_dir):
                break
            run_number += 1
        
        # Create experiment-specific output directory relative to the main output_dir
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create mode-specific directories only when needed
        if mode in ['train', 'eval']:
            # Create checkpoint directory inside experiment dir (for train mode)
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Create visualizations directory inside experiment dir (for train and eval modes)
            self.visualization_dir = os.path.join(self.experiment_dir, "visualizations")
            os.makedirs(self.visualization_dir, exist_ok=True)
        elif mode == 'explore':
            # Only create visualizations for explore mode
            self.visualization_dir = os.path.join(self.experiment_dir, "visualizations")
            os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Save configuration to JSON file inside experiment dir
        config_path = os.path.join(self.experiment_dir, "config.json")
        try:
            with open(config_path, 'w') as f:
                # Convert non-serializable objects to strings
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                             for k, v in self.__dict__.items()}
                json.dump(config_dict, f, indent=4)
        except Exception as e:
             print(f"Warning: Could not save config.json: {e}")

config = Config() # Instantiate config globally

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# Dataset preparation
class Sentinel2Sequence(Dataset):
    """Dataset for loading sequences of Sentinel-2 images from pre-computed sequence files."""
    
    def __init__(self, data_root, sequence_file, sequence_length, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory of BigEarthNet-S2 dataset (used to build full paths)
            sequence_file: Path to the .pkl file containing sequences (list of lists of relative paths)
            sequence_length: Expected number of frames in the input sequence (e.g., 11)
            transform: Image transformations
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load sequences from file
        print(f"Loading sequences from {sequence_file}...")
        try:
            with open(sequence_file, 'rb') as f:
                self.sequences = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Sequence file not found at {sequence_file}", file=sys.stderr)
            print("Please run 'python main.py --mode prepare ...' first.", file=sys.stderr)
            self.sequences = []
        except Exception as e:
            print(f"Error loading sequence file {sequence_file}: {e}", file=sys.stderr)
            self.sequences = []

        # Validate sequences
        if self.sequences:
            expected_len = self.sequence_length + 1 # Input + target
            original_count = len(self.sequences)
            self.sequences = [seq for seq in self.sequences if len(seq) == expected_len]
            if len(self.sequences) < original_count:
                print(f"Warning: Filtered out {original_count - len(self.sequences)} sequences with incorrect length.")
        
        if not self.sequences:
            print("Warning: No valid sequences loaded.")
            
    def _load_sentinel2_image(self, relative_patch_path):
        """Load and preprocess a Sentinel-2 image from a patch directory."""
        # Construct full path from data_root and relative_patch_path
        img_path = os.path.join(self.data_root, relative_patch_path)
        
        # Load the specified bands
        band_data = []
        target_shape = None # Target shape for resampling (e.g., 120x120)
        resampling_method = rasterio.enums.Resampling.bilinear
        
        # Determine target shape from a 10m band if possible
        preferred_10m_bands = ['B04', 'B03', 'B02', 'B08']
        available_bands_in_patch = [os.path.basename(f).split('_')[-1].split('.')[0] 
                                  for f in glob.glob(os.path.join(img_path, "*.tif"))]
        
        found_10m_shape = False
        for band_name in preferred_10m_bands:
            if band_name in available_bands_in_patch:
                band_files = glob.glob(os.path.join(img_path, f"*_{band_name}.tif"))
                if band_files:
                    try:
                        with rasterio.open(band_files[0]) as src:
                            target_shape = (src.height, src.width)
                            found_10m_shape = True
                            break
                    except rasterio.RasterioIOError:
                        pass # Try next preferred band
        
        # Load specified bands (from config.bands), resampling as needed
        loaded_band_data = {}
        for band in config.bands:
            # Find the band file
            band_files = glob.glob(os.path.join(img_path, f"*_{band}.tif"))
            if not band_files:
                # Handle missing band - return None or raise error?
                # For now, raise error as the model expects all configured bands
                raise FileNotFoundError(f"Required band {band} not found in {img_path}")
            
            try:
                with rasterio.open(band_files[0]) as src:
                    current_shape = (src.height, src.width)
                    
                    # Set target shape if not found yet
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
                 raise FileNotFoundError(f"Error reading band {band} in {img_path}: {e}")
        
        # Stack bands in the order specified by config.bands
        img_array_bands = [loaded_band_data[b] for b in config.bands]
        img_array = np.stack(img_array_bands, axis=0)
        
        # Reshape to image format (H, W, C)
        img_array = reshape_as_image(img_array)
        
        # Resize to patch size (this might be redundant if already 120x120 and patch_size is 120,
        # but ensures consistency if patch_size is different)
        if img_array.shape[:2] != (config.patch_size, config.patch_size):
             img_array = cv2.resize(img_array, (config.patch_size, config.patch_size), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 10000.0  # Sentinel-2 data range
        img_array = np.clip(img_array, 0, 1)
        
        return img_array
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a sequence of images and the target frame."""
        relative_path_sequence = self.sequences[idx]
        
        # Load input sequence (first sequence_length frames)
        input_frames = []
        try:
            for i in range(self.sequence_length):
                # relative_patch_path is like 'TileDirName/PatchDirName'
                relative_patch_path = relative_path_sequence[i]
                img = self._load_sentinel2_image(relative_patch_path)
                # Apply transform *after* loading and initial numpy processing
                if self.transform:
                    # Assuming transform expects PIL image or tensor, convert numpy array
                    # Example: Convert HWC numpy to tensor CHW
                    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
                    img = self.transform(img_tensor) # Apply transform to tensor
                else:
                    # Convert to tensor CHW format if no other transform
                    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
                    
                input_frames.append(img)
            
            # Load target frame (last frame)
            target_relative_path = relative_path_sequence[-1]
            target_frame = self._load_sentinel2_image(target_relative_path)
            # Convert to tensor CHW format
            target_frame_tensor = torch.from_numpy(target_frame.transpose((2, 0, 1))).float()
            # Apply transform to target if needed (usually only ToTensor for val/test)
            if self.transform:
                 # If validation transform is just ToTensor, it might already be done
                 # Check if target transform exists or is different
                 # For simplicity, assume val/test transform is just ToTensor (handled above)
                 pass 
                 
            target_frame = target_frame_tensor # Use the tensor

        except FileNotFoundError as e:
            print(f"Error loading sequence {idx}: {e}", file=sys.stderr)
            # Return dummy data or skip? Returning dummy data might hide issues.
            # For now, let it crash or return None and handle in DataLoader collate_fn
            # Simplest: return None, let DataLoader handle it (requires custom collate_fn)
            # Or: re-raise the error
            raise e 
        except Exception as e:
            print(f"Unexpected error loading sequence {idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise e

        # Stack input tensors
        # Input frames should already be tensors after transform or conversion
        input_sequence = torch.stack(input_frames)
        
        return input_sequence, target_frame

# Data augmentation
train_transform = transforms.Compose([
    # lambda x: transforms.ToTensor()(x), # ToTensor is now handled within __getitem__
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(10), # Rotation might be complex with multiple channels
    # transforms.ColorJitter(brightness=0.1, contrast=0.1) # ColorJitter might be complex
])

# For validation/test, usually only need to convert to tensor
val_transform = None # Handled in __getitem__

# Model architecture
class UNetEncoder(nn.Module):
    """U-Net encoder with ResNet-50 backbone for spatial feature extraction."""
    
    def __init__(self, in_channels=4):
        super(UNetEncoder, self).__init__()
        
        # Load pre-trained ResNet-50 as backbone
        from torchvision.models.resnet import ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify the first conv layer to accept the number of input channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                           stride=2, padding=3, bias=False)
        
        # We'll use the backbone as feature extractor up to different stages
        self.skip_connections = []
    
    def forward(self, x):
        """Extract features from input image at different scales for skip connections."""
        
        # Initial convolution and max pooling
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        skip1 = x  # 64 channels, 1/2 resolution
        
        x = self.backbone.maxpool(x)
        
        # Layer 1
        x = self.backbone.layer1(x)
        skip2 = x  # 256 channels, 1/4 resolution
        
        # Layer 2
        x = self.backbone.layer2(x)
        skip3 = x  # 512 channels, 1/8 resolution
        
        # Layer 3
        x = self.backbone.layer3(x)
        skip4 = x  # 1024 channels, 1/16 resolution
        
        # Layer 4
        x = self.backbone.layer4(x)  # 2048 channels, 1/32 resolution
        
        # Store skip connections
        self.skip_connections = [skip1, skip2, skip3, skip4]
        
        return x

class SwinTemporalTransformer(nn.Module):
    """Swin Transformer for temporal reasoning across frames."""
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.3):
        super(SwinTemporalTransformer, self).__init__()
        
        # Initial projection to transform ResNet features to transformer dimensions
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # Instead of using the Hugging Face Swin implementation which has compatibility issues,
        # let's create a simpler transformer that doesn't rely on position_embeddings
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4, 
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Final projection
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Process sequence of spatial features with temporal attention.
        
        Args:
            x: Tensor of shape [batch, sequence_length, channels, height, width]
        
        Returns:
            Temporally informed features for the predicted frame
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for projection: [batch*seq_len, channels, height, width]
        x_flat = x.reshape(batch_size * seq_len, channels, height, width)
        
        # Average pooling to reduce spatial dimensions
        x_pool = F.adaptive_avg_pool2d(x_flat, (8, 8))
        
        # Reshape to [batch*seq_len, channels, 8*8]
        x_reshape = x_pool.reshape(batch_size * seq_len, channels, -1)
        
        # Transpose to [batch*seq_len, 8*8, channels] for linear projection
        x_reshape = x_reshape.transpose(1, 2)
        
        # Project to hidden dimension
        x_proj = self.projection(x_reshape)  # [batch*seq_len, 8*8, hidden_dim]
        
        # Reshape to [batch, seq_len, 8*8, hidden_dim]
        x_seq = x_proj.reshape(batch_size, seq_len, 8*8, -1)
        
        # For pixel-wise temporal processing, we process each pixel location through time
        # Reshape for transformer to treat each pixel separately
        batch_pixels = batch_size * 8 * 8
        x_pixels = x_seq.permute(0, 2, 1, 3).reshape(batch_pixels, seq_len, -1)
        
        # Apply transformer encoder
        x_transformed = self.transformer_encoder(x_pixels)
        
        # Get the final temporal representation (last in sequence)
        x_final = self.final_projection(x_transformed[:, -1, :])
        
        # Reshape back to spatial representation [batch, hidden_dim, 8, 8]
        x_spatial = x_final.reshape(batch_size, 8, 8, -1).permute(0, 3, 1, 2)
        
        return x_spatial

class DecoderBlock(nn.Module):
    """Decoder block for upsampling features."""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetSwinHybrid(nn.Module):
    """U-Net + Swin Transformer hybrid model for spatiotemporal forecasting."""
    
    def __init__(self, config):
        super(UNetSwinHybrid, self).__init__()
        
        in_channels = len(config.bands)  # Number of input bands
        
        # U-Net encoder
        self.encoder = UNetEncoder(in_channels=in_channels)
        
        # Swin Temporal Transformer
        self.temporal_transformer = SwinTemporalTransformer(
            input_dim=2048,  # ResNet-50 final layer channels
            hidden_dim=config.hidden_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout
        )
        
        # Decoder blocks (upsampling path)
        self.decoder1 = DecoderBlock(config.hidden_dim, 1024)
        self.decoder2 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder4 = DecoderBlock(256, 64)
        
        # Additional upsampling to reach 256×256
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 128×128 -> 256×256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, sequence_length, channels, height, width]
        
        Returns:
            Predicted next frame of shape [batch, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each frame with U-Net encoder
        frame_features = []
        skip_connections = []
        
        for i in range(seq_len):
            # Extract features for current frame
            frame = x[:, i, :, :, :]
            features = self.encoder(frame)
            frame_features.append(features)
            
            # Save skip connections from the last frame (for decoder)
            if i == seq_len - 1:
                skip_connections = self.encoder.skip_connections
        
        # Stack frame features
        stacked_features = torch.stack(frame_features, dim=1)  # [batch, seq_len, channels, height, width]
        
        # Process temporal sequence with Swin Transformer
        temporal_features = self.temporal_transformer(stacked_features)
        
        # Decode the features with skip connections from the last frame
        x = self.decoder1(temporal_features, skip_connections[3])
        x = self.decoder2(x, skip_connections[2])
        x = self.decoder3(x, skip_connections[1])
        x = self.decoder4(x, skip_connections[0])
        
        # Additional upsampling to match the target size of 256×256
        x = self.final_upsample(x)
        
        # Final convolution to get output image
        output = self.final_conv(x)
        
        # Apply sigmoid to ensure output in range [0, 1]
        output = torch.sigmoid(output)
        
        return output

# Custom loss function combining L1 Loss and SSIM
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # Weight for L1 loss
        self.l1_loss = nn.L1Loss()
        # For SSIM, specify the number of channels from config.bands
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=4)  # 4 channels (RGB+NIR)
        
    def forward(self, pred, target):
        # Check for shape mismatch and notify
        if pred.shape != target.shape:
            print(f"Warning: Shape mismatch in loss calculation. Pred: {pred.shape}, Target: {target.shape}")
            # If shapes differ, resize pred to match target
            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
                print(f"Resized prediction to {pred.shape}")
        
        # Calculate L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Calculate SSIM (higher is better, so 1-SSIM for loss)
        ssim_value = self.ssim_loss(pred, target)
        ssim = 1 - ssim_value  # SSIM returns similarity, so we convert to loss
        
        # Combine losses
        loss = self.alpha * l1 + (1 - self.alpha) * ssim
        
        return loss

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
            checkpoint = torch.load(config.checkpoint_path)
            
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
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
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
            
        # Update learning rate
        scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase (every eval_interval epochs)
        if (epoch + 1) % config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_mae = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validation"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
                    # Calculate other metrics
                    mse = F.mse_loss(outputs, targets).item()
                    mae = F.l1_loss(outputs, targets).item()
                    
                    # Calculate PSNR
                    psnr = 10 * torch.log10(1.0 / mse).item()
                    
                    # Calculate SSIM
                    ssim_module = SSIM(data_range=1.0, size_average=True, channel=len(config.bands))
                    ssim_value = ssim_module(outputs, targets).item()
                    
                    val_mse += mse
                    val_mae += mae
                    val_psnr += psnr
                    val_ssim += ssim_value
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mse = val_mse / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            avg_val_psnr = val_psnr / len(val_loader)
            avg_val_ssim = val_ssim / len(val_loader)
            
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
            print(f"Epoch [{epoch+1}/{config.num_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}")
            print(f"PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}")
        
        # Save checkpoint
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
    
    # Save final model
    final_model_path = os.path.join(config.experiment_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(config.experiment_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(range(0, len(history['val_loss']) * config.eval_interval, config.eval_interval), 
             history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot MSE and MAE
    plt.subplot(2, 2, 2)
    plt.plot(range(0, len(history['val_mse']) * config.eval_interval, config.eval_interval), 
             history['val_mse'], label='MSE')
    plt.plot(range(0, len(history['val_mae']) * config.eval_interval, config.eval_interval), 
             history['val_mae'], label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Metrics')
    
    # Plot PSNR
    plt.subplot(2, 2, 3)
    plt.plot(range(0, len(history['val_psnr']) * config.eval_interval, config.eval_interval), 
             history['val_psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('Peak Signal-to-Noise Ratio')
    
    # Plot SSIM
    plt.subplot(2, 2, 4)
    plt.plot(range(0, len(history['val_ssim']) * config.eval_interval, config.eval_interval), 
             history['val_ssim'], label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('Structural Similarity Index')
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(config.experiment_dir, 'training_curves.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Training curves saved to {metrics_plot_path}")
    
    return model, history

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
    
    # Create output directory for visualizations using config
    vis_dir = config.visualization_dir # Use path from config
    # os.makedirs(vis_dir, exist_ok=True) # Already created by config.set_experiment_paths
    
    # For computing NDVI change
    pixel_change_thresholds = {
        'low': 0.05,   # Minor change
        'medium': 0.15, # Moderate change
        'high': 0.25    # Significant change
    }
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Convert tensors to numpy arrays for visualization
            # Take the first sample in the batch
            input_seq = inputs[0].cpu().numpy()
            target_img = targets[0].cpu().numpy()
            output_img = outputs[0].cpu().numpy()
            
            # Create figure for the sequence and prediction
            fig, axes = plt.subplots(3, 4, figsize=(20, 12))
            
            # Plot input sequence (show only every 3rd frame to fit in figure)
            for j in range(4):
                frame_idx = j * 3  # Show frames 0, 3, 6, 9
                frame = input_seq[frame_idx]
                
                # For RGB visualization, take the first 3 bands if available
                if frame.shape[0] >= 3:
                    rgb_frame = frame[:3].transpose(1, 2, 0)
                    axes[0, j].imshow(np.clip(rgb_frame, 0, 1))
                else:
                    axes[0, j].imshow(frame[0], cmap='gray')
                
                axes[0, j].set_title(f"Input frame {frame_idx}")
                axes[0, j].axis('off')
            
            # Plot target and prediction
            if target_img.shape[0] >= 3:
                # RGB visualization
                rgb_target = target_img[:3].transpose(1, 2, 0)
                rgb_output = output_img[:3].transpose(1, 2, 0)
                
                axes[1, 0].imshow(np.clip(rgb_target, 0, 1))
                axes[1, 1].imshow(np.clip(rgb_output, 0, 1))
                axes[1, 0].set_title("Target Frame (Ground Truth)")
                axes[1, 1].set_title("Predicted Frame")
                
                # RGB Error map (MSE per pixel)
                error_map = np.mean((rgb_target - rgb_output) ** 2, axis=2)
                im = axes[1, 2].imshow(error_map, cmap='hot')
                axes[1, 2].set_title("RGB Error Map")
                plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
                
                # NDVI comparison if we have NIR band (index 3)
                if target_img.shape[0] >= 4:
                    # NDVI = (NIR - Red) / (NIR + Red)
                    target_ndvi = (target_img[3] - target_img[0]) / (target_img[3] + target_img[0] + 1e-8)
                    output_ndvi = (output_img[3] - output_img[0]) / (output_img[3] + output_img[0] + 1e-8)
                    
                    # NDVI visualizations
                    im1 = axes[1, 3].imshow(target_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    im2 = axes[2, 0].imshow(output_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    plt.colorbar(im1, ax=axes[1, 3], fraction=0.046, pad=0.04)
                    plt.colorbar(im2, ax=axes[2, 0], fraction=0.046, pad=0.04)
                    
                    axes[1, 3].set_title("Target NDVI")
                    axes[2, 0].set_title("Predicted NDVI")
                    
                    # NDVI error map
                    ndvi_error = np.abs(target_ndvi - output_ndvi)
                    im3 = axes[2, 1].imshow(ndvi_error, cmap='hot', vmin=0, vmax=0.5)
                    plt.colorbar(im3, ax=axes[2, 1], fraction=0.046, pad=0.04)
                    axes[2, 1].set_title("NDVI Error")
                    
                    # Change detection map (for environmental monitoring)
                    change_map = np.zeros_like(ndvi_error)
                    
                    # Categorize changes
                    change_map[ndvi_error < pixel_change_thresholds['low']] = 1      # Minor change
                    change_map[ndvi_error >= pixel_change_thresholds['low']] = 2     # Low change
                    change_map[ndvi_error >= pixel_change_thresholds['medium']] = 3  # Medium change
                    change_map[ndvi_error >= pixel_change_thresholds['high']] = 4    # High change
                    
                    # Create custom colormap for change detection
                    from matplotlib.colors import ListedColormap
                    change_cmap = ListedColormap(['black', 'green', 'yellow', 'orange', 'red'])
                    
                    im4 = axes[2, 2].imshow(change_map, cmap=change_cmap, vmin=0, vmax=4)
                    cbar = plt.colorbar(im4, ax=axes[2, 2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
                    cbar.set_ticklabels(['No Data', 'No Change', 'Low', 'Medium', 'High'])
                    axes[2, 2].set_title("Change Detection Map")
                    
                    # Calculate statistics on changes
                    change_percentages = {
                        'no_change': np.sum(change_map == 1) / change_map.size * 100,
                        'low': np.sum(change_map == 2) / change_map.size * 100,
                        'medium': np.sum(change_map == 3) / change_map.size * 100,
                        'high': np.sum(change_map == 4) / change_map.size * 100
                    }
                    
                    # Display change statistics
                    axes[2, 3].axis('off')
                    stats_text = (
                        f"Change Statistics:\n\n"
                        f"No Change: {change_percentages['no_change']:.1f}%\n"
                        f"Low Change: {change_percentages['low']:.1f}%\n"
                        f"Medium Change: {change_percentages['medium']:.1f}%\n"
                        f"High Change: {change_percentages['high']:.1f}%\n\n"
                        f"NDVI Error (Mean): {np.mean(ndvi_error):.4f}\n"
                        f"RGB Error (MSE): {np.mean(error_map):.4f}"
                    )
                    axes[2, 3].text(0.1, 0.5, stats_text, fontsize=12, va='center')
                    axes[2, 3].set_title("Change Statistics")
            else:
                # Grayscale visualization
                axes[1, 0].imshow(target_img[0], cmap='gray')
                axes[1, 1].imshow(output_img[0], cmap='gray')
                
                # Error map
                error_map = (target_img[0] - output_img[0]) ** 2
                im = axes[1, 2].imshow(error_map, cmap='hot')
                plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
                
                axes[1, 0].set_title("Target Frame (Ground Truth)")
                axes[1, 1].set_title("Predicted Frame")
                axes[1, 2].set_title("Error Map")
            
            # Turn off unused axes
            for ax in axes.flatten():
                if not ax.has_data():
                    ax.axis('off')
            
            # Set main title with metrics
            if target_img.shape[0] >= 3:
                rgb_target = target_img[:3].transpose(1, 2, 0)
                rgb_output = output_img[:3].transpose(1, 2, 0)
                mse = np.mean((rgb_target - rgb_output) ** 2)
                psnr = 10 * np.log10(1.0 / max(mse, 1e-8))
                
                plt.suptitle(f"Sample {i+1} - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB", fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            plt.savefig(os.path.join(vis_dir, f'prediction_sample_{i+1}.png'), dpi=150)
            plt.close()
            
            # Create a before/after GIF to visualize change
            if target_img.shape[0] >= 3:
                try:
                    # Create frames for GIF (target and prediction)
                    frames = []
                    
                    # Add target frame
                    target_frame = (np.clip(rgb_target, 0, 1) * 255).astype(np.uint8)
                    frames.append(target_frame)
                    
                    # Add prediction frame
                    pred_frame = (np.clip(rgb_output, 0, 1) * 255).astype(np.uint8)
                    frames.append(pred_frame)
                    
                    # Save GIF
                    gif_path = os.path.join(vis_dir, f'before_after_{i+1}.gif')
                    imageio.mimsave(gif_path, frames, duration=1.0, loop=0)
                except ImportError:
                    print("imageio not installed. Skipping GIF creation.")
    
    print(f"Saved {min(num_samples, len(test_loader))} visualizations to {vis_dir}")

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
    model = UNetSwinHybrid(config)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=config.device))
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
            psnr = 10 * torch.log10(1.0 / mse).item()
            ssim_value = ssim_module(outputs, targets).item()
            
            # Accumulate metrics
            metrics['mse'] += mse
            metrics['mae'] += mae
            metrics['psnr'] += psnr
            metrics['ssim'] += ssim_value
    
    # Calculate average metrics
    for key in metrics:
        metrics[key] /= len(test_loader)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(config.experiment_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    return metrics

def main():
    """Main function to execute the training pipeline."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='U-Net + Swin Transformer Hybrid for Spatiotemporal Forecasting')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'eval', 'explore', 'filter', 'prepare'],
                        help='Operation mode (train, eval, explore, filter, prepare)')
    parser.add_argument('--data_root', type=str, default=config.data_root,
                        help='Path to the dataset root directory')
    parser.add_argument('--output_dir', type=str, default=config.output_dir,
                        help='Directory to save outputs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pre-trained model (for evaluation mode)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading (0 for single-process)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    # Parameters for explore_dataset mode
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize in explore mode')
    parser.add_argument('--temporal_analysis', action='store_true',
                        help='Perform temporal analysis in explore mode')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Specific image directory to explore')
    
    # Parameters for prepare_sequences mode
    parser.add_argument('--sequence_length', type=int, default=11,
                        help='Length of input sequence for training')
    parser.add_argument('--max_time_gap', type=int, default=60,
                        help='Maximum time gap (in days) between consecutive images')
    parser.add_argument('--locations', nargs='+', default=None,
                        help='Specific tile locations to use (e.g., T35VNL)')
    
    # Parameters for filter_images mode
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save filtered image list')
    parser.add_argument('--copy_to', type=str, default=None,
                        help='Directory to copy filtered images to')
    
    # Parameters for filter_region_images mode (from filter_images.py)
    parser.add_argument('--min_lat', type=float, help='Minimum latitude for filtering')
    parser.add_argument('--max_lat', type=float, help='Maximum latitude for filtering')
    parser.add_argument('--min_lon', type=float, help='Minimum longitude for filtering')
    parser.add_argument('--max_lon', type=float, help='Maximum longitude for filtering')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments BEFORE setting paths
    config.data_root = args.data_root
    config.output_dir = args.output_dir # Main output dir
    config.batch_size = args.batch_size
    config.resume_training = args.resume
    config.checkpoint_path = args.checkpoint_path
    config.model_path = args.model_path
    
    # Set experiment-specific paths including the mode
    config.set_experiment_paths(args.mode)
    
    # Print the experiment directory being used
    print(f"Using experiment directory: {config.experiment_dir}")
    
    # Create main output directory if it doesn't exist (might be redundant but safe)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Execute the requested mode
    if args.mode == 'explore':
        # Call explore_dataset.py
        cmd = [
            sys.executable, 'explore_dataset.py',
            '--data_root', args.data_root,
            '--output_dir', config.visualization_dir, # Save viz in experiment subdir
            '--num_samples', str(args.num_samples)
        ]
        
        if args.temporal_analysis:
            cmd.append('--temporal_analysis')
            
        if args.img_dir:
            cmd.extend(['--img_dir', args.img_dir])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        # Check if the exploration was successful
        if result.returncode != 0:
            print("\nNOTE: If you're seeing errors about missing bands, your dataset might have a nested structure.")
            print("The BigEarthNet dataset typically has this directory structure:")
            print("  /dataset/BigEarthNet-S2/")
            print("    ├── S2A_MSIL2A_YYYYMMDD_*_*_*/   (Tile directories)")
            print("    │   ├── S2A_MSIL2A_YYYYMMDD_*_*_*_XX_YY/   (Patch directories)")
            print("    │   │   ├── *_B01.tif")
            print("    │   │   ├── *_B02.tif")
            print("    │   │   └── ... (other band files)")
            print("\nTry running the explore command with a specific patch directory:")
            print(f"  python main.py --mode explore --data_root {args.data_root} --img_dir [SPECIFIC_PATCH_DIRECTORY]")
        
    elif args.mode == 'filter':
        # Call filter_images.py
        output_file = args.output_file or os.path.join(config.experiment_dir, 'filtered_images.txt') # Save in experiment dir
        
        # Check required args for filter mode
        if args.min_lat is None or args.max_lat is None or args.min_lon is None or args.max_lon is None:
             print("Error: --min_lat, --max_lat, --min_lon, --max_lon are required for filter mode.")
             return
             
        cmd = [
            sys.executable, 'filter_images.py',
            '--data_root', args.data_root,
            '--output_file', output_file,
            '--min_lat', str(args.min_lat),
            '--max_lat', str(args.max_lat),
            '--min_lon', str(args.min_lon),
            '--max_lon', str(args.max_lon)
        ]
        
        if args.copy_to:
            # Define copy_to path relative to experiment dir or absolute?
            # Let's make it relative to the main output dir for clarity
            copy_to_path = os.path.join(config.output_dir, args.copy_to)
            cmd.extend(['--copy_to', copy_to_path])
            print(f"Filtered images will be copied to: {copy_to_path}")
            
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'prepare':
        # Call prepare_sequences.py
        # Save directly to the standard sequences directory
        sequences_dir = os.path.join(args.output_dir, 'sequences')
        os.makedirs(sequences_dir, exist_ok=True)
        
        cmd = [
            sys.executable, 'prepare_sequences.py',
            '--data_root', args.data_root,
            '--output_dir', sequences_dir,  # Save directly to the standard location
            '--sequence_length', str(args.sequence_length),
            '--max_time_gap', str(args.max_time_gap)
        ]
        
        if args.locations:
            cmd.extend(['--locations'] + args.locations)
            
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'train':
        # Define sequence file paths based on the standard sequences directory
        # This is where prepare now saves its output
        prepared_sequence_dir = os.path.join(args.output_dir, 'sequences')
        train_seq_file = os.path.join(prepared_sequence_dir, 'train_sequences.pkl')
        val_seq_file = os.path.join(prepared_sequence_dir, 'val_sequences.pkl')
        test_seq_file = os.path.join(prepared_sequence_dir, 'test_sequences.pkl')

        # Check if sequence files exist
        if not os.path.exists(train_seq_file) or not os.path.exists(val_seq_file):
            print(f"Error: Training or validation sequence file not found in {prepared_sequence_dir}")
            print(f"Please run 'python main.py --mode prepare --output_dir {args.output_dir}' first to generate sequence files in the 'sequences' subdirectory.")
            return

        # Set device based on args
        if args.force_cpu:
            config.device = torch.device("cpu")
            print("Forcing CPU usage as requested")
        
        # Create datasets using sequence files
        print("Creating datasets from sequence files...")
        train_dataset = Sentinel2Sequence(config.data_root, train_seq_file, config.sequence_length, transform=train_transform)
        val_dataset = Sentinel2Sequence(config.data_root, val_seq_file, config.sequence_length, transform=val_transform)
        
        # Create test dataset loader only if the file exists (for final evaluation)
        test_loader = None
        if os.path.exists(test_seq_file):
            test_dataset = Sentinel2Sequence(config.data_root, test_seq_file, config.sequence_length, transform=val_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'))
        else:
            print(f"Warning: Test sequence file {test_seq_file} not found. Test evaluation will be skipped.")

        # Check if datasets loaded successfully
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print("Error: No valid sequences loaded into datasets. Exiting.")
             return

        # Check if we have sufficient training data
        if len(train_dataset) < config.min_train_samples:
            print(f"WARNING: Only {len(train_dataset)} training samples found. " +
                  f"This may be insufficient for good results (minimum recommended: {config.min_train_samples}).")
        
        # Print dataset information
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        if test_loader:
             print(f"Test samples: {len(test_dataset)}")
        
        # Create data loaders with appropriate num_workers (default 0 for Windows compatibility)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                 num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'))
        
        # Initialize model, loss function, optimizer, and scheduler
        print("Initializing model...")
        model = UNetSwinHybrid(config)
        criterion = HybridLoss(alpha=0.8)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        
        # Train the model
        print(f"Starting training for {config.num_epochs} epochs...")
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
        
        # Visualize predictions on validation set
        print("Generating prediction visualizations...")
        visualize_predictions(model, val_loader, config, num_samples=5)

        # Evaluate final model on test set (if test_loader exists)
        if test_loader:
            # Use best_model_path which is saved during the training process
            best_model_path = os.path.join(config.experiment_dir, 'best_model.pth') 
            if os.path.exists(best_model_path):
                print(f"\nEvaluating best model ({best_model_path}) on test set...")
                # Make sure evaluate_model saves its metrics inside the *current* experiment dir
                metrics = evaluate_model(best_model_path, test_loader, config)
            else:
                print(f"\nBest model file not found at {best_model_path}. Skipping evaluation.")
        else:
             print("\nSkipping final evaluation on test set as test sequences were not found.")
    
    elif args.mode == 'eval':
        # Define test sequence file path (look in standard sequences dir)
        prepared_sequence_dir = os.path.join(args.output_dir, 'sequences') 
        test_seq_file = os.path.join(prepared_sequence_dir, 'test_sequences.pkl')

        # Check if test sequence file exists
        if not os.path.exists(test_seq_file):
            print(f"Error: Test sequence file not found at {test_seq_file}")
            print(f"Please run 'python main.py --mode prepare --output_dir {args.output_dir}' first.")
            return
            
        # Check model path
        if not config.model_path:
            print("Error: --model_path is required for eval mode.")
            return
        if not os.path.exists(config.model_path):
            print(f"Error: Model file does not exist at {config.model_path}")
            return

        # Create test dataset
        print("Creating test dataset from sequence file...")
        test_dataset = Sentinel2Sequence(config.data_root, test_seq_file, config.sequence_length, transform=val_transform)
        if len(test_dataset) == 0:
            print("Error: No valid sequences loaded into test dataset. Exiting.")
            return
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        # Evaluate the model - results will be saved in the current eval experiment dir
        print(f"Evaluating model: {config.model_path}")
        metrics = evaluate_model(config.model_path, test_loader, config) # Pass config
    
    print(f"\nProcess for mode '{args.mode}' completed. Results saved to {config.experiment_dir}")

if __name__ == "__main__":
    main()
