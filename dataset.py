import os
import glob
import numpy as np
# import pickle # No longer needed for loading sequences
# import sys # Keep sys for potential stderr usage later
import sys 
# import cv2 # No longer needed for image processing
# import rasterio # No longer needed for image processing
# from rasterio.plot import reshape_as_image # No longer needed
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Import the global config object (might still be needed for transforms or other info)
from config import config

class PreprocessedSentinel2Sequence(Dataset):
    """Dataset for loading sequences of Sentinel-2 images from preprocessed .pt files."""
    
    def __init__(self, preprocessed_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            preprocessed_dir (str): Path to the directory containing the preprocessed .pt files 
                                    (e.g., ./output/preprocessed_sequences/train).
            transform: Transformations to apply to the loaded tensors (e.g., augmentations).
        """
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        self.sequences = []

        if not os.path.isdir(preprocessed_dir):
            print(f"Error: Preprocessed directory not found at {preprocessed_dir}", file=sys.stderr)
            # Optionally, you could try to create it or raise an error
            # os.makedirs(preprocessed_dir, exist_ok=True)
            # print(f"Created directory {preprocessed_dir}")
            # For now, just print error and return empty dataset
            return 
            
        # Find all existing .pt files
        # Sort them to ensure consistent order, assuming filenames are like sequence_000000.pt
        pt_files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "sequence_*.pt")))
        self.sequences = pt_files
        
        if not self.sequences:
            print(f"Warning: No preprocessed .pt files found in {self.preprocessed_dir}. Did you run preprocess_sequences.py?", file=sys.stderr)
        else:
             print(f"Found {len(self.sequences)} preprocessed sequences in {self.preprocessed_dir}")

    # _load_sentinel2_image method is no longer needed as we load tensors directly

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a preprocessed sequence (input tensor, target tensor)."""
        pt_file_path = self.sequences[idx]
        
        try:
            # Load the dictionary containing tensors
            data_dict = torch.load(pt_file_path)
            input_sequence = data_dict['input']  # Shape (T, C, H, W)
            target_frame = data_dict['target'] # Shape (C, H, W)
            
            # Apply transformations if specified
            # Note: Transformations like normalization should have been done during preprocessing.
            # This transform is typically for augmentations (RandomFlip, etc.)
            # We need to apply augmentations consistently to both input and target if they are spatial
            if self.transform:
                # --- Applying spatial transforms consistently --- 
                # 1. Stack input and target temporarily for consistent spatial transformation
                # Add batch dim temporarily for transform
                stacked_data = torch.cat([input_sequence, target_frame.unsqueeze(0)], dim=0) # Shape (T+1, C, H, W)
                
                # Apply the transform to the stack
                # Ensure transform handles multi-frame input correctly if needed, 
                # or apply frame-by-frame if necessary (less common for spatial transforms)
                # Assuming transforms like RandomHorizontalFlip work on (..., H, W)
                try:
                    # Note: Some transforms might expect (B, C, H, W) or (C, H, W)
                    # Adjust application logic based on your specific transforms
                    # Example: Applying transform to each frame individually if needed
                    # transformed_input = torch.stack([self.transform(frame) for frame in input_sequence])
                    # transformed_target = self.transform(target_frame)
                    
                    # Simpler: Apply to the whole stack (assumes transform handles it)
                    stacked_data_transformed = self.transform(stacked_data)
                    
                    # 2. Unstack them back
                    input_sequence = stacked_data_transformed[:-1, :, :, :] # Back to (T, C, H, W)
                    target_frame = stacked_data_transformed[-1, :, :, :]  # Back to (C, H, W)

                except Exception as e:
                     print(f"Warning: Could not apply transform to sequence {idx}. Error: {e}")
                     # Decide how to handle: return original or raise error?
                     # Returning original for now
                     pass 

        except FileNotFoundError:
            print(f"Error: Preprocessed file not found: {pt_file_path}", file=sys.stderr)
            # Return None or raise? Let dataloader handle it by raising error
            raise FileNotFoundError(f"Missing preprocessed file: {pt_file_path}")
        except KeyError as e:
            print(f"Error: Key {e} not found in preprocessed file: {pt_file_path}", file=sys.stderr)
            raise KeyError(f"Corrupted preprocessed file (missing key): {pt_file_path}")
        except Exception as e:
            print(f"Unexpected error loading or processing sequence {idx} ({pt_file_path}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise e

        return input_sequence, target_frame

# Data augmentation - Apply these *after* loading tensors if needed
# Normalization should be done in preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Add other augmentations if needed (e.g., RandomRotation, ColorJitter if applicable *before* preprocessing)
    # Note: ColorJitter applied here to tensors might not be standard.
])

# Validation/Test transforms usually don't include augmentation
val_transform = None 