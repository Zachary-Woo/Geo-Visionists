```markdown
# U-Net + Swin Transformer Hybrid Model for Spatiotemporal Forecasting of Satellite Imagery

## Overview
This project focuses on building a deep learning model to perform **spatiotemporal forecasting** of satellite imagery over **Jakarta, Indonesia** using Sentinel-2 data. Specifically, given a sequence of **11 consecutive satellite images**, the model is trained to **predict the 12th image** in the sequence. This represents a shift from traditional land cover classification to a **temporal image prediction problem** involving both spatial feature extraction and temporal reasoning.

The model architecture combines a **U-Net** for spatial encoding of individual frames with a **Swin Transformer** for temporal attention across frame embeddings. This hybrid architecture enables fine-grained spatial awareness while learning high-level contextual evolution across time.

## Use Cases
- **Urban Expansion Monitoring**: Detect and forecast growth in urban infrastructure.
- **Environmental Change Detection**: Track deforestation, water body shifts, or vegetation health over time.
- **Disaster Preparedness**: Anticipate flood-prone or rapidly changing zones.
- **Forecasting and Simulation**: Predict future satellite scenes for planning and simulation without waiting for satellite passes.

## Dataset

### Region of Interest
- **Geographic Focus**: Jakarta, Indonesia
- **Temporal Scope**: Time-series of Sentinel-2 imagery over months or years

### Source
- **Sentinel-2** imagery from:
  - Google Earth Engine
  - AWS Open Data Registry
  - Copernicus Open Access Hub

### Format & Preprocessing
- Bands: **RGB, NIR**, and optionally **NDVI**
- Spatial resolution: 10m
- Format: **PNG/JPEG** after preprocessing
- Patching: Images are cropped into uniform square patches (e.g., 256×256)

### Sequence Construction
- For each training sample:
  - Input: Sequence of 11 consecutive images (e.g., monthly or biweekly)
  - Target: 12th image (forecast)
- Images are aligned by date and region, filtered to remove clouds/shadows

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

### Fusion Strategy
- Two-stage: 
  1. **Spatial encoding** (U-Net applied frame-wise)
  2. **Temporal fusion** (Swin Transformer across frames)
- Output is a single 3-channel image (RGB prediction) or 4-channel (if using NIR)

## Training Configuration

### Hardware
- GPU: RTX 4090 (24GB VRAM)
- RAM: 64GB+
- Disk: 100GB+ for image storage

### Framework & Libraries
- `torch`, `torchvision`
- `transformers` (HuggingFace Swin)
- `numpy`, `scikit-learn`, `opencv-python`
- `matplotlib`, `earthengine-api`, `rasterio`

### Hyperparameters
- Optimizer: **AdamW**
- Learning Rate: **0.0001** (with scheduler)
- Batch Size: **8–16**
- Loss Function: **L1 Loss + SSIM** (or optionally MSE)
- Dropout: **0.3 – 0.5**
- Epochs: ~50
- Mixed Precision: Enabled

## Optimization Techniques
- **Data Augmentation**:
  - Temporal jittering
  - Brightness/contrast shift
  - Rotation/flipping
- **Hyperparameter Search**:
  - Grid search over learning rate, optimizer, and decoder depth
- **Cloud Filtering**:
  - Use Sentinel-2 QA bands or external cloud masks to filter input frames

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Overall pixel error
- **Mean Absolute Error (MAE)**: Simpler interpretability
- **Structural Similarity Index (SSIM)**: Captures perceived visual quality
- **Peak Signal-to-Noise Ratio (PSNR)**: Evaluates image fidelity
- **Temporal Consistency**: Measures visual smoothness across time

## Experimental Variations

### 1. Baseline Comparison
- Train a simple **3D CNN** or **ConvLSTM** to compare forecasting performance

### 2. Masked Frame Reconstruction
- Mask out a frame mid-sequence (e.g., frame 6) and test recovery quality

### 3. Multi-Step Prediction
- Train model to predict multiple future steps (e.g., frames 12–14) recursively

### 4. Change Detection Use Case
- Subtract predicted frame from ground truth to isolate urban or forest change regions

## Usage Guide

### 1. Data Preparation
- Export Sentinel-2 image sequences for Jakarta using Earth Engine or AWS
- Apply atmospheric correction, cloud masking, NDVI computation
- Convert and normalize bands; patchify into sequences

### 2. Training Workflow
- Construct dataset with 11-frame input and 12th-frame target
- Train U-Net encoder on individual frames
- Train hybrid model with temporal attention for 12th-frame reconstruction

### 3. Evaluation
- Visualize predicted vs. true 12th frames
- Quantify performance using SSIM, PSNR, and MSE
- Run on unseen dates or future ranges for forecasting

## Project Scope Adjustment Summary
- Previous Task: Static classification of land types using BigEarthNet
- Updated Task: **Temporal satellite image forecasting** (given 11 frames, predict the 12th)
- Region: Jakarta, Indonesia
- Architecture: Adapted **U-Net + Swin Transformer** for spatiotemporal modeling

## License
This project is for academic and research purposes only.
```