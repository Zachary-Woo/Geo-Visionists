# U-Net + Swin Transformer Hybrid Model for Spatiotemporal Forecasting of Satellite Imagery

## Overview
This project implements a deep learning model that performs **spatiotemporal forecasting** of satellite imagery using Sentinel-2 data. Given a sequence of **11 consecutive satellite images** from a location, the model predicts the **12th image** in the sequence.

The model architecture combines:
- **U-Net**: For spatial encoding of individual frames
- **Swin Transformer**: For temporal reasoning across frame embeddings

This hybrid approach enables detailed spatial awareness while capturing temporal patterns in satellite imagery.

## Dataset
The project uses the BigEarthNet-S2 dataset, focusing on regions in Portugal (Lisbon and Porto Metropolitan Area). The filtering script can be used with any geographic bounding box.

## Quick Start Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore the Dataset (Optional)
View sample images to understand the data:
```bash
python main.py --mode explore --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --num_samples 10
```

### 3. Filter Images by Region (Optional)
Filter the dataset to focus on specific geographic areas:
```bash
# Example: Filter for Lisbon, Portugal
python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 38.6 --max_lat 38.9 --min_lon -9.3 --max_lon -9.0
```
Additional filter options:
```bash
# Copy filtered images to a new directory
python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 38.6 --max_lat 38.9 --min_lon -9.3 --max_lon -9.0 --copy_to Lisbon-S2
```

### 4. Prepare Image Sequences
Create temporal sequences (lists of image paths) for training:
```bash
python main.py --mode prepare --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --sequence_length 11 --max_time_gap 60
```
*   This creates `.pkl` sequence definition files in `./output/sequences/`.

### 5. Preprocess Image Sequences (Recommended for Speed)
Convert the raw image sequences defined in the `.pkl` files into preprocessed PyTorch tensors (`.pt` files). This significantly speeds up training by performing image loading and normalization only once.
```bash
python preprocess_sequences.py --data_root ./dataset/BigEarthNet-S2 --sequences_dir ./output/sequences --output_dir ./output/preprocessed_sequences --patch_size 256
```
*   This creates `train/`, `val/`, and `test/` subdirectories in `./output/preprocessed_sequences/` containing the `.pt` files.
*   If this process is interrupted, the dataset loader will use only the files that were successfully created.

### 6. Train the Model
Train the model using the preprocessed sequences:
```bash
# Default training run (uses settings from config.py)
python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
```
*   The `--data_root` argument is still needed for potential metadata access but the model primarily loads data from `./output/preprocessed_sequences/`.

#### Quick Training (Example: ~1 day)
```bash
# Train with reduced samples and epochs
python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --max_samples 20000 --epochs 5
```

#### Fast Mode (For code testing)
```bash
# Quick test run with minimal data
python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --fast_mode
```

#### Other Training Options
- `--batch_size`: Change batch size (affects memory usage)
- `--epochs`: Set number of training epochs
- `--eval_frequency`: Run validation every N epochs
- `--max_samples`: Limit training samples (useful for faster runs)
- `--resume`: Resume from the latest checkpoint in the experiment directory
- `--checkpoint_path`: Path to a specific checkpoint for resuming training

### 7. Evaluate the Model
Test the trained model's performance using preprocessed test data:
```bash
python main.py --mode eval --model_path ./output/train_X/best_model.pth --output_dir ./output
```
*   Replace `train_X` with your specific training run folder (e.g., `train_1`).
*   Evaluation also uses the preprocessed data from `./output/preprocessed_sequences/test/`.

## Project Structure

### Core Modules
- `main.py`: Entry point, argument parsing, orchestration.
- `config.py`: Configuration class.
- `utils.py`: Utility functions (e.g., `set_seed`).
- `dataset.py`: `PreprocessedSentinel2Sequence` class for loading `.pt` files.
- `model.py`: Model architecture (`UNetSwinHybrid`, etc.).
- `loss.py`: `HybridLoss` definition.
- `trainer.py`: Training loop function (`train_model`).
- `evaluator.py`: Evaluation function (`evaluate_model`).
- `visualizer.py`: Visualization function (`visualize_predictions`).

### Data Preparation Scripts
(Called via `subprocess` from `main.py` or runnable standalone)
- `explore_dataset.py`: Data exploration and visualization.
- `filter_images.py`: Geographic filtering.
- `prepare_sequences.py`: Create `.pkl` sequence definitions.
- `preprocess_sequences.py`: Convert `.pkl` sequences to `.pt` tensor files.

### Output Directory Structure
```
./output/
├── sequences/                  # Sequence definition files (*.pkl, *.csv)
├── preprocessed_sequences/     # Preprocessed tensor files (*.pt)
│   ├── train/
│   ├── val/
│   └── test/
├── filter_X/                   # Output from filter mode
├── train_X/                    # Training run results
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── checkpoints/            # Intermediate checkpoints
│   ├── visualizations/         # Prediction images (*.png)
│   ├── config.json             # Run configuration
│   └── history.json            # Training metrics history
└── eval_X/                     # Evaluation run results
    ├── evaluation_metrics.json
    └── config.json
```

## Model Architecture
The hybrid architecture combines:
1. **U-Net with ResNet50 backbone**: Extracts spatial features from each frame.
2. **Transformer**: Processes temporal relationships between frames.
3. **Decoder**: Reconstructs the predicted future frame.

## Performance Notes
- Preprocessing sequences into `.pt` files drastically reduces training time compared to loading TIFs on the fly.
- Training time depends heavily on the number of samples and epochs.
- Use `--max_samples` and `--epochs` to balance training time and model quality.

## Use Cases
- **Urban Expansion Monitoring**: Detect and forecast growth in urban infrastructure.
- **Environmental Change Detection**: Track deforestation, water body shifts, or vegetation health over time.
- **Disaster Preparedness**: Anticipate flood-prone or rapidly changing zones.
- **Forecasting and Simulation**: Predict future satellite scenes for planning and simulation.

## License
This project is for academic and research purposes only.