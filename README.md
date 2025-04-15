# U-Net + Swin Transformer Hybrid Model for Spatiotemporal Forecasting of Satellite Imagery

# Classification
# Future prediction

## Overview
This project focuses on building a deep learning model to perform **spatiotemporal forecasting** of satellite imagery using Sentinel-2 data. Specifically, given a sequence of **11 consecutive satellite images from a location**, the model is trained to **predict the 12th image** in the sequence. This represents a shift from traditional land cover classification to a **temporal image prediction problem** involving both spatial feature extraction and temporal reasoning.

**Note:** The provided BigEarthNet-S2 dataset sample covers parts of Europe, with our project focusing on regions in Portugal (Lisbon and Porto Metropolitan Area). The filtering script has been generalized to filter based on *any* provided geographic bounding box.

The model architecture combines a **U-Net** for spatial encoding of individual frames with a **Swin Transformer** for temporal attention across frame embeddings. This hybrid architecture enables fine-grained spatial awareness while learning high-level contextual evolution across time.

## Quick Start

### Setup
1. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Explore the dataset:
   ```
   python main.py --mode explore --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --num_samples 10
   ```
   
4. Filter images for specific regions in Portugal:
   ```bash
   # Example: Filter for Lisbon, Portugal
   python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 38.6 --max_lat 38.9 --min_lon -9.3 --max_lon -9.0

   # Example: Filter for Porto Metropolitan Area, Portugal
   python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 41.0 --max_lat 41.3 --min_lon -8.8 --max_lon -8.4
   ```
   *Note: This step creates `output/filter_1/filtered_images.txt` which is **not** currently used by the `prepare` step.*

5. Prepare image sequences (uses all data found in `data_root` unless `--locations` is specified):
   ```
   python main.py --mode prepare --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --sequence_length 11 --max_time_gap 60
   ```

6. Train the model:
   ```
   python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
   ```
   *Note: Training outputs go into `./output/train_1/`*

7. Evaluate a trained model:
   ```
   python main.py --mode eval --model_path ./output/train_1/best_model.pth --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
   ```
   *Note: Evaluation outputs go into `./output/eval_1/`*

## Project Components

### Data Processing Scripts
- `explore_dataset.py`: Visualize and analyze the BigEarthNet dataset
- `filter_images.py`: Identify images within a given geographic bounding box
- `prepare_sequences.py`: Create sequences of consecutive images for training

### Model Implementation
- `main.py`: Main script that handles all operations (training, evaluation, data processing)

### Features
- **Experiment Tracking**: Each run creates a numbered experiment directory named by mode (e.g., `train_1`, `filter_1`)
- **Model Checkpointing**: Auto-saves checkpoints during training
- **Resume Training**: Can resume from a saved checkpoint
- **Comprehensive Metrics**: MSE, MAE, PSNR, SSIM for evaluating performance
- **Visualizations**: Input sequences, predictions, error maps, change detection

## Output Directory Structure

Outputs are saved in subdirectories within the main `--output_dir` (default: `./output`):

```
./output/
├── sequences/                  # Output from 'prepare' mode (used by train/eval)
│   ├── train_sequences.pkl
│   ├── train_sequences.csv
│   ├── val_sequences.pkl
│   └── ...
├── explore_1/                  # Output from 'explore' mode
│   └── visualizations/
├── filter_1/                   # Output from 'filter' mode
│   ├── filtered_images.txt
│   └── filtered_images.csv
├── prepare_1/                  # Log/config output for the 'prepare' run (sequences saved separately)
│   └── config.json
├── train_1/                    # Output from 'train' mode
│   ├── best_model.pth
│   ├── checkpoints/
│   ├── config.json
│   ├── evaluation_metrics.json (from final test evaluation)
│   ├── history.json
│   ├── training_curves.png
│   └── visualizations/
├── eval_1/                     # Output from 'eval' mode
│   ├── config.json
│   ├── evaluation_metrics.json
│   └── visualizations/
└── ... (other experiment runs like train_2, eval_2, etc.)
```

## Command Reference

### Exploring the Dataset (Not needed to run but helpful for visualizations of the data)
```bash
# Basic dataset exploration
python main.py --mode explore --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --num_samples 10
```
*Output saved in `./output/explore_1/visualizations/`*

```bash
# Analyze temporal distribution of images
python main.py --mode explore --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --temporal_analysis
```
*Output saved in `./output/explore_1/visualizations/`*

```bash
# Visualize a specific image directory
python main.py --mode explore --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --img_dir S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB/S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_patch_name # Use patch path
```
*Output saved in `./output/explore_1/visualizations/`*

### Filtering Images by Region

```bash
# Filter images for Lisbon, Portugal
python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 38.6 --max_lat 38.9 --min_lon -9.3 --max_lon -9.0
```
*Output files `filtered_images.txt/.csv` saved in `./output/filter_1/`*

```bash
# Filter images for Porto Metropolitan Area, Portugal
python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 41.0 --max_lat 41.3 --min_lon -8.8 --max_lon -8.4
```
*Output files `filtered_images.txt/.csv` saved in `./output/filter_2/`*

```bash
# Filter and copy selected images to a new directory structure (for Lisbon)
python main.py --mode filter --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --min_lat 38.6 --max_lat 38.9 --min_lon -9.3 --max_lon -9.0 --copy_to Lisbon-S2
```
*Selected images copied to `./output/Lisbon-S2/` preserving tile/patch structure.*

### Preparing Sequences

```bash
# Create sequences of consecutive images (using all data)
python main.py --mode prepare --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --sequence_length 11 --max_time_gap 60
```
*Output sequence files saved in `./output/sequences/`. Log/config saved in `./output/prepare_1/`*

```bash
# Create sequences for specific locations
python main.py --mode prepare --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --locations T35VNL T35VNK
```
*Output sequence files saved in `./output/sequences/`*

### Training the Model

```bash
# Train the model with default parameters
python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
```
*Output saved in `./output/train_1/`*

```bash
# Train with custom batch size
python main.py --mode train --data_root ./dataset/BigEarthNet-S2 --output_dir ./output --batch_size 16
```
*Output saved in `./output/train_2/`*

```bash
# Resume training from a checkpoint
python main.py --mode train --resume --checkpoint_path ./output/train_1/checkpoints/checkpoint_epoch_XX.pth --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
```
*Continues saving to the specified experiment directory.*

### Evaluating the Model

```bash
# Evaluate a trained model
python main.py --mode eval --model_path ./output/train_1/best_model.pth --data_root ./dataset/BigEarthNet-S2 --output_dir ./output
```
*Output saved in `./output/eval_1/`*

```bash
# Evaluate with custom dataset path
python main.py --mode eval --model_path ./output/train_1/best_model.pth --data_root ./custom_dataset --output_dir ./output
```
*Output saved in `./output/eval_2/`*

## Use Cases
- **Urban Expansion Monitoring**: Detect and forecast growth in urban infrastructure.
- **Environmental Change Detection**: Track deforestation, water body shifts, or vegetation health over time.
- **Disaster Preparedness**: Anticipate flood-prone or rapidly changing zones.
- **Forecasting and Simulation**: Predict future satellite scenes for planning and simulation without waiting for satellite passes.

## Dataset

### Region of Interest
- **Geographic Focus**: Regions in Portugal (Lisbon and Porto Metropolitan Area)
  - **Lisbon Bounding Box**: Lat [38.6, 38.9], Lon [-9.3, -9.0]
  - **Porto Metropolitan Area Bounding Box**: Lat [41.0, 41.3], Lon [-8.8, -8.4]
- **Temporal Scope**: Time-series of Sentinel-2 imagery over months or years

### Source
- **Sentinel-2** imagery from BigEarthNet-S2 dataset

### Format & Preprocessing
- Bands: **RGB, NIR**, and NDVI (derived)
- Spatial resolution: Mixed (resampled to 10m/120px during processing)
- Patching: Images are cropped into uniform square patches (256×256)

### Sequence Construction
- For each training sample:
  - Input: Sequence of 11 consecutive images
  - Target: 12th image (forecast)
- Images are aligned by date and region

## Model Architecture

### U-Net (Spatial Encoder)
- Extracts high-resolution spatial features from each of the 11 input frames
- Shared-weight encoder with a **ResNet-50 backbone**
- Outputs a sequence of encoded spatial feature maps

### Swin Transformer (Temporal Processor)
- Processes the sequence of encoded features from U-Net
- Learns temporal dependencies using **shifted-window attention**
- Generates a temporally-informed representation of the predicted 12th frame

### Decoder
- Combines spatial and temporal embeddings
- Reconstructs the predicted 12th frame using a transposed convolution decoder

## Experiment Tracking

Each training run creates a numbered experiment directory:
```
./output/train_1/
├── best_model.pth         # Best model weights
├── checkpoints/           # Training checkpoints
├── config.json            # Configuration parameters
├── final_model.pth        # Final model weights
├── history.json           # Training metrics history
├── training_curves.png    # Learning curves visualization
└── visualizations/        # Prediction visualizations
```

## Output Files

- Model checkpoints and logs are saved to mode-specific subdirectories in `./output` (e.g., `./output/train_1/`)
- Visualizations are saved to `./output/[experiment_name]/visualizations/`
- Sequence data from `prepare` mode is saved to `./output/sequences/`

## Notes on BigEarthNet-S2

The BigEarthNet-S2 dataset consists of Sentinel-2 satellite imagery across Europe. Each directory (e.g., S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB) contains:

- Multiple band files (B02.tif, B03.tif, B04.tif, etc.)
- Metadata JSON files

Common bands used in the project:
- B02, B03, B04: RGB bands (10m)
- B08: Near-infrared (NIR) band

## License
This project is for academic and research purposes only.