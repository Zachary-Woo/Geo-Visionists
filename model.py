import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from transformers import SwinConfig, SwinModel # Keep transformers import for now, though SwinModel is replaced

# Model architecture
class UNetEncoder(nn.Module):
    """U-Net encoder with ResNet-50 backbone for spatial feature extraction."""
    
    def __init__(self, in_channels=4):
        super(UNetEncoder, self).__init__()
        
        # Load pre-trained ResNet-50 as backbone
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