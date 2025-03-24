# U-Net + Swin Transformer Hybrid Model for Deforestation Classification

## Overview
This project implements a hybrid deep learning architecture that combines **U-Net**, a convolutional neural network for semantic segmentation, with the **Swin Transformer**, a hierarchical vision transformer, to classify and monitor land cover changes using satellite imagery from the **BigEarthNet** dataset.

The primary objective is to accurately classify various land cover types—including deforested, degraded, and healthy forest areas—and to enable reliable forecasting of geographical and environmental changes over time. This model is one of four used in a comparative study where other team members have implemented models based on CNN, MLP, and LSTM architectures.

## Use Cases
- **Urban Planning**: Detecting areas of urban sprawl and anticipating infrastructure needs.
- **Climate Change Monitoring**: Tracking forest degradation, loss, and ecosystem changes.
- **Disaster Preparedness**: Identifying zones prone to flooding, erosion, or other climate risks.
- **Land Management**: Mapping agricultural spread, deforestation rates, and conservation zones.

## Dataset
- Located in /dataset

### Source
- **BigEarthNet** (Sentinel-2 multispectral satellite image patches)

### Size & Format
- **Total Images**: ~590,000
- **Split**:
  - Training: ~472,000 (80%)
  - Validation: ~59,000 (10%)
  - Testing: ~59,000 (10%)
- **Format**: Originally in TIFF, converted to JPEG/PNG for CNN-based training pipelines.
- **Uncompressed Size**: ~59 GiB

### Channels Used
- **RGB** (Visible Spectrum)
- **NIR** (Near-Infrared)
- **NDVI** (Normalized Difference Vegetation Index — derived channel)

### Classification Categories
- Healthy Forest
- Degraded Forest
- Deforested Areas
- Urban Expansion
- Agricultural Land
- Water Bodies

## Model Architecture

### U-Net
- A deep CNN designed for semantic segmentation.
- **Encoder**:
  - Pretrained **ResNet-50** backbone for efficient and deep spatial feature extraction.
- **Decoder**:
  - Transposed convolution layers that upsample feature maps to generate high-resolution segmentation masks.
  - Skip connections bridge encoder and decoder at each resolution level to retain fine-grained spatial details.

### Swin Transformer
- A vision transformer architecture using shifted windows and hierarchical feature extraction.
- Processes the same image patches or U-Net feature maps to learn **global spatial dependencies**.
- Uses pretrained weights from **ImageNet-22k** for transfer learning and accelerated convergence.

### Fusion Strategy
- U-Net and Swin Transformer process each image in parallel.
- Outputs from both models are fused:
  - U-Net contributes high-resolution segmentation masks.
  - Swin Transformer contributes contextual embeddings.
- A **final classification head** combines these representations to produce a robust land-cover prediction for each image patch.

## Training Configuration

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 64GB DDR5+
- **Disk**: 100GB+ available for image storage and model checkpoints

### Framework & Libraries
- **Framework**: PyTorch (preferred) or TensorFlow
- **Key Libraries**:
  - `torch`, `torchvision`
  - `transformers` (for Swin Transformer)
  - `scikit-learn`, `numpy`, `opencv-python`
  - `matplotlib`, `albumentations`

### Hyperparameters
- **Optimizer**: AdamW (with weight decay)
- **Learning Rate**: 0.0001 (adaptive using ReduceLROnPlateau)
- **Batch Size**: 16 (also testing with 8, 32)
- **Loss Function**: Combined Dice Loss + Categorical Cross-Entropy
- **Dropout**: 0.3 – 0.5
- **Epochs**: ~30–50 (based on convergence and validation accuracy)
- **Mixed Precision Training**: Enabled (for memory efficiency and speed on 4090)

## Optimization Strategy
- **Data Augmentation**:
  - Random horizontal/vertical flipping
  - Rotation, brightness/contrast shifts
  - NDVI-specific normalization
- **Grid Search** for:
  - Learning rates
  - Optimizer types
  - Batch sizes
  - Dropout levels

## Evaluation Plan

### Metrics
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balance between precision and recall
- **Intersection over Union (IoU)**: Quality of segmentation masks
- **Confusion Matrix**: Visual breakdown of classification performance
- **Area Under ROC Curve (AUROC)**: Binary/one-vs-all robustness for each class

### Validation Method
- 10% held-out validation set
- Early stopping based on validation IoU or F1-score
- Model checkpoints saved at optimal validation performance

### Performance Monitoring
- Training and validation loss curves
- Metric tracking per epoch
- Visual inspection of segmentation overlays
- Evaluation across time-steps (for forecasting consistency)

## Usage Guide

### 1. Preprocess Dataset
- Convert all Sentinel-2 TIFF images to PNG or JPEG
- Normalize RGB and NIR channels
- Derive NDVI from NIR and Red bands
- Split into training, validation, and test sets

### 2. Train U-Net
- Use pretrained ResNet-50 encoder
- Train on pixel-wise segmentation masks for deforestation and land cover classes
- Save intermediate segmentation outputs

### 3. Fine-Tune Swin Transformer
- Input raw satellite patches or feature maps
- Fine-tune with multi-class cross-entropy loss
- Extract region-based contextual embeddings

### 4. Fuse and Train Final Classification Head
- Concatenate or attention-fuse U-Net and Swin outputs
- Train a shallow head model to output final class probabilities

### 5. Evaluate Model
- Run inference on test set
- Generate segmentation visualizations
- Produce confusion matrix and class-level scores

## Comparison to Other Models (Team-Wide Analysis)

### Models in Comparison
- ResNet-50 CNN (baseline spatial classifier)
- MLP (tabular feature classification)
- LSTM (sequence modeling of satellite time-series)
- U-Net + Swin Transformer (this model)

### Evaluation Points
- **Consistency Across Models**:
  - Do all models predict similar land changes over time?
  - Are some better at forecasting urban expansion or degraded forest regions?

- **Visual and Statistical Trends**:
  - Identify which models highlight the same regions as high-risk zones
  - Compare predictions over multiple time periods (spatial and temporal agreement)

- **Robustness and Generalization**:
  - Which model performs better under unseen or partially deforested conditions?
  - Which generalizes best across geography?

## License
This project is intended for academic use in CAP5610 (Machine Learning) and is not licensed for commercial distribution.