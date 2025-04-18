import os
import json
import torch

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
    batch_size = 24
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
        # Removed print statements from here to avoid repetition by workers

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
        if mode == 'train':
            # Training needs checkpoints and visualizations
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.visualization_dir = os.path.join(self.experiment_dir, "visualizations")
            os.makedirs(self.visualization_dir, exist_ok=True)
        elif mode == 'eval':
            # Standalone eval doesn't save checkpoints or visualizations by default
            # (Visualizations are done after training in 'train' mode)
            # We still define the paths in case they are needed later, but don't create folders
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
            self.visualization_dir = os.path.join(self.experiment_dir, "visualizations")
        elif mode == 'explore':
            # Explore mode only creates visualizations
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