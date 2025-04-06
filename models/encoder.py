import torch.nn as nn
import torch
from arcface_model.iresnet import iresnet100
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torchvision import models


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 256 * 256, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class ArcFaceIdentityEncoder(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        self.arcface = iresnet100(pretrained=False)
        self.arcface.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.arcface.eval()

    def forward(self, x):
        with torch.no_grad():
            emb = self.arcface(x)               # Output shape: [B, 512]
            emb = F.normalize(emb, dim=1)       # L2 normalization across features
            return emb


class AttributeEncoder_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 112 * 112, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class AttributeEncoder_enc(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True, out_dim=512):
        super().__init__()
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=1  # not used, we discard decoder
        ).encoder

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # from B×512×H×W → B×512×1×1
    #    self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]  # shape: B×512×4×4
        x = self.pool(x).view(x.size(0), -1)  # shape: B×512
  #      x = self.fc(x)  # optional: map to 512 if needed
        return x


class AttributeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Encoder
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [64, 112, 112]
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # [64, 56, 56]
        self.enc3 = resnet.layer2                                         # [128, 28, 28]
        self.enc4 = resnet.layer3                                         # [256, 14, 14]
        self.enc5 = resnet.layer4                                         # [512, 7, 7]

        # Decoder with skip connections (U-Net style)
        self.dec1 = self._up_block(512 + 256, 256)  # Input: e5 + e4
        self.dec2 = self._up_block(256 + 128, 128)  # Input: d1 + e3
        self.dec3 = self._up_block(128 + 64, 64)    # Input: d2 + e2
        self.dec4 = self._up_block(64 + 64, 64)     # Input: d3 + e1

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [64, 112, 112]
        e2 = self.enc2(e1) # [64, 56, 56]
        e3 = self.enc3(e2) # [128, 28, 28]
        e4 = self.enc4(e3) # [256, 14, 14]
        e5 = self.enc5(e4) # [512, 7, 7]

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode='bilinear'), e4], dim=1))  # -> 256x14x14
        d2 = self.dec2(torch.cat([F.interpolate(d1, size=e3.shape[2:], mode='bilinear'), e3], dim=1))  # -> 128x28x28
        d3 = self.dec3(torch.cat([F.interpolate(d2, size=e2.shape[2:], mode='bilinear'), e2], dim=1))  # -> 64x56x56
        d4 = self.dec4(torch.cat([F.interpolate(d3, size=e1.shape[2:], mode='bilinear'), e1], dim=1))  # -> 64x112x112

        # Return 8 feature maps for AAD blocks
        return [e1, e2, e3, e4, e5, d1, d2, d4]
