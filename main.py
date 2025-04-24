"""
Main script for U-Net + Swin Transformer Hybrid Model for Spatiotemporal Forecasting.
Handles dataset preparation, training, evaluation, and visualization.
"""

import os
import sys
import argparse
import subprocess

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import refactored components
from config import config # Import the global config instance
from utils import set_seed
from dataset import PreprocessedSentinel2Sequence, train_transform, val_transform, safe_collate
from model import UNetSwinHybrid
from loss import HybridLoss
from trainer import train_model
from evaluator import evaluate_model
from visualizer import visualize_predictions

# Set random seeds for reproducibility early
set_seed()

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
    parser.add_argument('--batch_size', type=int, default=config.batch_size, # Use config default
                        help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of worker processes for data loading (0 for single-process)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Enable fast mode with reduced dataset size for quick results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of training samples to use (for quick testing)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--eval_frequency', type=int, default=None,
                        help='Run validation every N epochs')
    
    # Parameters for explore_dataset mode
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize in explore mode')
    parser.add_argument('--temporal_analysis', action='store_true',
                        help='Perform temporal analysis in explore mode')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Specific image directory to explore')
    
    # Parameters for prepare_sequences mode
    parser.add_argument('--sequence_length', type=int, default=config.sequence_length, # Use config default
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
    
    # --- Update global config with command-line arguments ---
    config.data_root = args.data_root
    config.output_dir = args.output_dir # Main output dir
    config.batch_size = args.batch_size # Update config batch size from args
    config.sequence_length = args.sequence_length # Update sequence length from args
    config.resume_training = args.resume
    config.checkpoint_path = args.checkpoint_path
    config.model_path = args.model_path

    # Update config device based on force_cpu AFTER reading args
    if args.force_cpu:
        config.device = torch.device("cpu")
    else:
        # Re-evaluate device in case config was initialized before GPU became available
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # Apply fast mode settings if requested (small dataset, few epochs)
    if args.fast_mode:
        print("FAST MODE ENABLED - Using reduced dataset and training parameters")
        # Override max_samples if not manually set
        if args.max_samples is None:
            args.max_samples = 1000  # Use only 1000 training samples
        # Override epochs if not manually set
        if args.epochs is None:
            config.num_epochs = 3  # Train for just 3 epochs
        # Override eval frequency if not manually set
        if args.eval_frequency is None:
            config.eval_interval = 1  # Evaluate every epoch
    
    # Apply individual overrides if specified (these take precedence over fast_mode)
    if args.epochs is not None:
        config.num_epochs = args.epochs
        print(f"Overriding number of epochs to {config.num_epochs}")
    
    if args.eval_frequency is not None:
        config.eval_interval = args.eval_frequency
        print(f"Overriding validation frequency to every {config.eval_interval} epochs")
        
    # --- Set experiment-specific paths including the mode ---
    config.set_experiment_paths(args.mode)
    
    # Print the experiment directory being used
    print(f"Using experiment directory: {config.experiment_dir}")

    # Print device info ONCE here
    print(f"Using device: {config.device} {'(GPU)' if torch.cuda.is_available() else '(CPU)'}")
    if config.device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Create main output directory if it doesn't exist (might be redundant but safe)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- Execute the requested mode ---
    if args.mode == 'explore':
        # Call explore_dataset.py script via subprocess
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
        
        # Provide hint if exploration fails
        if result.returncode != 0:
            print("\nNOTE: If you're seeing errors about missing bands, check dataset structure.")
            print("See explore_dataset.py documentation for expected structure.")
            print(f"Try running explore with a specific patch directory if needed.")

    elif args.mode == 'filter':
        # Call filter_images.py script via subprocess
        output_file = args.output_file or os.path.join(config.experiment_dir, 'filtered_images.txt') # Save in experiment dir
        
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
            copy_to_path = os.path.join(config.output_dir, args.copy_to) # Relative to main output
            cmd.extend(['--copy_to', copy_to_path])
            print(f"Filtered images will be copied to: {copy_to_path}")
            
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'prepare':
        # Call prepare_sequences.py script via subprocess
        sequences_dir = os.path.join(args.output_dir, 'sequences') # Standard location
        os.makedirs(sequences_dir, exist_ok=True)
        
        cmd = [
            sys.executable, 'prepare_sequences.py',
            '--data_root', args.data_root,
            '--output_dir', sequences_dir, 
            '--sequence_length', str(config.sequence_length), # Use config value
            '--max_time_gap', str(args.max_time_gap)
        ]
        if args.locations:
            cmd.extend(['--locations'] + args.locations)
            
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'train':
        # --- Training Mode ---
        
        # Define preprocessed data directories
        preprocessed_base_dir = os.path.join(args.output_dir, 'preprocessed_sequences')
        train_preprocessed_dir = os.path.join(preprocessed_base_dir, 'train')
        val_preprocessed_dir = os.path.join(preprocessed_base_dir, 'val')
        test_preprocessed_dir = os.path.join(preprocessed_base_dir, 'test')

        # Check if preprocessed directories exist (basic check)
        if not os.path.isdir(train_preprocessed_dir) or not os.path.isdir(val_preprocessed_dir):
            print(f"Error: Preprocessed train or validation directory not found in {preprocessed_base_dir}")
            print(f"Please run 'python preprocess_sequences.py --output_dir {preprocessed_base_dir} ...' first.")
            return

        # Create datasets using PreprocessedSentinel2Sequence class from dataset.py
        print("Creating datasets from preprocessed files...")
        train_dataset = PreprocessedSentinel2Sequence(train_preprocessed_dir, transform=train_transform)
        val_dataset = PreprocessedSentinel2Sequence(val_preprocessed_dir, transform=val_transform) # Usually no transform for val
        
        # Apply max_samples limit if specified (works on the found .pt files)
        if args.max_samples is not None:
            if args.max_samples < len(train_dataset):
                train_indices = torch.randperm(len(train_dataset))[:args.max_samples]
                train_dataset = Subset(train_dataset, train_indices)
                print(f"⚡ Reduced training dataset to {len(train_dataset)} samples")
            
            # Reduce validation set proportionally or to a minimum
            val_samples = min(len(val_dataset), max(int(args.max_samples * 0.2), config.min_train_samples // 2)) # Ensure some val samples
            if val_samples < len(val_dataset):
                val_indices = torch.randperm(len(val_dataset))[:val_samples]
                val_dataset = Subset(val_dataset, val_indices)
                print(f"⚡ Reduced validation dataset to {len(val_dataset)} samples")

        # Create test dataset loader only if the directory exists
        test_loader = None
        test_dataset_len = 0
        if os.path.isdir(test_preprocessed_dir):
            test_dataset = PreprocessedSentinel2Sequence(test_preprocessed_dir, transform=val_transform)
            test_dataset_len = len(test_dataset)
            
            if test_dataset_len == 0:
                print(f"Warning: Preprocessed test directory {test_preprocessed_dir} exists but contains no .pt files.")
                test_loader = None # Ensure test_loader is None if dataset is empty
            else:
                # Reduce test set if max_samples is used
                if args.max_samples is not None:
                    test_samples = min(test_dataset_len, max(int(args.max_samples * 0.1), 50))
                    if test_samples < test_dataset_len:
                        # Need to create a subset based on indices of the *found* files
                        # Since PreprocessedSentinel2Sequence loads files dynamically, 
                        # Subset works directly on the instance.
                        test_indices = torch.randperm(test_dataset_len)[:test_samples]
                        test_dataset = Subset(test_dataset, test_indices)
                        test_dataset_len = len(test_dataset)
                        print(f"⚡ Reduced test dataset to {test_dataset_len} samples")
                
                # Use config.batch_size for test loader (evaluation can handle batches)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                                         num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'),
                                         collate_fn=safe_collate)
        else:
            print(f"Warning: Preprocessed test directory {test_preprocessed_dir} not found. Test evaluation will be skipped.")

        # Check if datasets loaded successfully
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print("Error: No valid sequences loaded into training/validation datasets. Exiting.")
             return
        if len(train_dataset) < config.min_train_samples:
            print(f"WARNING: Only {len(train_dataset)} training samples. Recommended minimum: {config.min_train_samples}.")
        
        # Print dataset information
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        if test_loader: print(f"Test samples: {test_dataset_len}")
             
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                 num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'),
                                 collate_fn=safe_collate)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'),
                               collate_fn=safe_collate)
        
        # Initialize model, loss, optimizer, scheduler
        print("Initializing model...")
        model = UNetSwinHybrid(config)
        criterion = HybridLoss(alpha=0.8) # Alpha could be configurable
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # Use ReduceLROnPlateau to adjust LR based on validation loss
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
        
        # --- Train the model ---
        print(f"Starting training for {config.num_epochs} epochs...")
        # Call train_model from trainer.py
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
        
        # --- Visualize predictions on validation set ---
        print("Generating prediction visualizations...")
        # Call visualize_predictions from visualizer.py
        # Pass the *trained* model state
        vis_model = UNetSwinHybrid(config) # Create a fresh instance for visualization
        vis_model.load_state_dict(model.state_dict()) # Load trained weights
        visualize_predictions(vis_model, val_loader, config, num_samples=5) # Use val_loader for consistency

        # --- Evaluate final model on test set ---
        if test_loader:
            best_model_path = os.path.join(config.experiment_dir, 'best_model.pth') 
            if os.path.exists(best_model_path):
                print(f"\nEvaluating best model ({best_model_path}) on test set...")
                # Call evaluate_model from evaluator.py
                metrics = evaluate_model(best_model_path, test_loader, config)
            else:
                print(f"\nBest model file not found at {best_model_path}. Evaluating final model instead.")
                final_model_path = os.path.join(config.experiment_dir, 'final_model.pth')
                if os.path.exists(final_model_path):
                     metrics = evaluate_model(final_model_path, test_loader, config)
                else:
                     print(f"Final model not found either. Skipping test evaluation.")
        else:
             print("\nSkipping final evaluation on test set as test sequences were not found.")
    
    elif args.mode == 'eval':
        # --- Evaluation Mode ---
        # Define preprocessed test directory path
        preprocessed_base_dir = os.path.join(args.output_dir, 'preprocessed_sequences') 
        test_preprocessed_dir = os.path.join(preprocessed_base_dir, 'test')

        # Check if preprocessed test directory exists
        if not os.path.isdir(test_preprocessed_dir):
            print(f"Error: Preprocessed test directory not found at {test_preprocessed_dir}")
            print(f"Run 'python preprocess_sequences.py --output_dir {preprocessed_base_dir} ...' first.")
            return

        if not config.model_path or not os.path.exists(config.model_path):
            print(f"Error: --model_path is required for eval mode and must exist.")
            print(f"Provided path: {config.model_path}")
            return

        print("Creating test dataset from preprocessed files...")
        test_dataset = PreprocessedSentinel2Sequence(test_preprocessed_dir, transform=val_transform)
        if len(test_dataset) == 0:
            print(f"Error: No valid sequences loaded from {test_preprocessed_dir}. Exiting.")
            return
            
        # Use config batch size for evaluation loader
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, pin_memory=(config.device.type == 'cuda'),
                                 collate_fn=safe_collate)
        
        print(f"Evaluating model: {config.model_path}")
        # Call evaluate_model from evaluator.py
        metrics = evaluate_model(config.model_path, test_loader, config) 
    
    print(f"\nProcess for mode '{args.mode}' completed. Results saved to {config.experiment_dir}")

if __name__ == "__main__":
    main()
