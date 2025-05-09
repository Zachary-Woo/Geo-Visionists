import os
import glob
import sys 
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
            return 
            
        # Find all existing .pt files
        pt_files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "sequence_*.pt")))
        self.sequences = pt_files
        
        if not self.sequences:
            print(f"Warning: No preprocessed .pt files found in {self.preprocessed_dir}. Did you run preprocess_sequences.py?", file=sys.stderr)
        else:
             print(f"Found {len(self.sequences)} preprocessed sequences in {self.preprocessed_dir}")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a preprocessed sequence (input tensor, target tensor)."""
        pt_file_path = self.sequences[idx]
        
        try:
            # Load the dictionary containing tensors
            # Set weights_only=True for security as recommended by PyTorch.
            # Assumes .pt files contain only tensors and basic types.
            # --- Error Handling for torch.load ---
            try:
                data_dict = torch.load(pt_file_path, weights_only=True)
            except (EOFError, RuntimeError) as load_error:
                # Catch EOFError (incomplete file) and RuntimeError (e.g., zip archive errors)
                print(f"Warning: Skipping corrupted file {pt_file_path}. Error: {load_error}", file=sys.stderr)
                return None # Signal to collate_fn to skip this sample
            # --- End Error Handling ---

            input_sequence = data_dict['input']  # Shape (T, C, H, W)
            target_frame = data_dict['target'] # Shape (C, H, W)
            
            # Apply transformations if specified
            if self.transform:
                # --- Applying spatial transforms consistently --- 
                # 1. Stack input and target temporarily for consistent spatial transformation
                # Add batch dim temporarily for transform
                stacked_data = torch.cat([input_sequence, target_frame.unsqueeze(0)], dim=0) # Shape (T+1, C, H, W)
                
                # Apply the transform to the stack
                try:
                    # Note: Some transforms might expect (B, C, H, W) or (C, H, W)
                    stacked_data_transformed = self.transform(stacked_data)
                    
                    # 2. Unstack them back
                    input_sequence = stacked_data_transformed[:-1, :, :, :] # Back to (T, C, H, W)
                    target_frame = stacked_data_transformed[-1, :, :, :]  # Back to (C, H, W)

                except Exception as e:
                     print(f"Warning: Could not apply transform to sequence {idx}. Error: {e}")
                     pass 

        except FileNotFoundError:
            print(f"Error: Preprocessed file not found: {pt_file_path}", file=sys.stderr)
            # Return None to be skipped by collate_fn
            return None
        except KeyError as e:
            print(f"Error: Key {e} not found in preprocessed file: {pt_file_path}", file=sys.stderr)
            # Return None to be skipped by collate_fn
            return None
        except Exception as e:
            # Catch other potential errors during loading/processing
            print(f"Unexpected error processing sequence {idx} ({pt_file_path}): {e}", file=sys.stderr)
            return None

        return input_sequence, target_frame

# Custom collate function to handle None values returned by __getitem__
def safe_collate(batch):
    """Collate function that filters out None items (corrupted samples)."""
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None # Or return appropriate empty tensors if needed
    
    # Use default collate for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)

# Normalization should be done in preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

val_transform = None 