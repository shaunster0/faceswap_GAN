import torch.nn as nn
import torch
from arcface_model.iresnet import iresnet100
import segmentation_models_pytorch as smp


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
            return self.arcface(x)


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


class AttributeEncoder(nn.Module):
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
