import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

# Import the global config object
from config import config

# Custom loss function combining L1 Loss and SSIM
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # Weight for L1 loss
        self.l1_loss = nn.L1Loss()
        # For SSIM, specify the number of channels from config.bands
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=len(config.bands)) 
        
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
        # Ensure inputs are on the same device for SSIM calculation
        if pred.device != target.device:
            target = target.to(pred.device)
            
        ssim_value = self.ssim_loss(pred, target)
        ssim = 1 - ssim_value  # SSIM returns similarity, so we convert to loss
        
        # Combine losses
        loss = self.alpha * l1 + (1 - self.alpha) * ssim
        
        return loss 