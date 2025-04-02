import torch.nn as nn
import torch
from arcface_model.iresnet import iresnet100
import segmentation_models_pytorch as smp
import torch.nn.functional as F


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
    

def conv4x4(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def deconv4x4(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class AttributeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = conv4x4(3, 32)       # -> 32x128x128
        self.conv2 = conv4x4(32, 64)      # -> 64x64x64
        self.conv3 = conv4x4(64, 128)     # -> 128x32x32
        self.conv4 = conv4x4(128, 256)    # -> 256x16x16
        self.conv5 = conv4x4(256, 512)    # -> 512x8x8
        self.conv6 = conv4x4(512, 512)    # -> 512x4x4

        # Decoder
        self.deconv1 = deconv4x4(512, 512)  # -> 512x8x8
        self.deconv2 = deconv4x4(512, 256)  # -> 256x16x16
        self.deconv3 = deconv4x4(256, 128)  # -> 128x32x32
        self.deconv4 = deconv4x4(128, 64)   # -> 64x64x64
        self.deconv5 = deconv4x4(64, 32)    # -> 32x128x128
        self.deconv6 = deconv4x4(32, 16)    # -> 16x256x256 (if needed)

        self.to_8ch = nn.Conv2d(16, 8, kernel_size=3, padding=1)  # -> 8 channels for z_attr[7]

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)   # 32x128x128
        e2 = self.conv2(e1)  # 64x64x64
        e3 = self.conv3(e2)  # 128x32x32
        e4 = self.conv4(e3)  # 256x16x16
        e5 = self.conv5(e4)  # 512x8x8
        e6 = self.conv6(e5)  # 512x4x4

        # Decoder
        d1 = self.deconv1(e6)                 # 512x8x8
        d2 = self.deconv2(d1)                 # 256x16x16
        d3 = self.deconv3(d2)                 # 128x32x32
        d4 = self.deconv4(d3)                 # 64x64x64
        d5 = self.deconv5(d4)                 # 32x128x128
        d6 = self.deconv6(d5)                 # 16x256x256

        d7 = F.interpolate(d6, size=(112, 112), mode='bilinear', align_corners=False)
        d7 = self.to_8ch(d7)                  # 8x112x112

        # Return list of feature maps for AAD blocks
        return [e6, d1, d2, d3, d4, d5, d6, d7]

