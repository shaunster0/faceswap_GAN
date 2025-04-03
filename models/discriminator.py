import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 112 * 112, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    
    class PatchDiscriminator(nn.Module):
        def __init__(self, in_channels=3, base_channels=64):
            super().__init__()

            def block(in_c, out_c, normalize=True):
                layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_c))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(in_channels, base_channels, normalize=False),   # -> [B, 64, 56, 56]
                *block(base_channels, base_channels * 2),              # -> [B, 128, 28, 28]
                *block(base_channels * 2, base_channels * 4),          # -> [B, 256, 14, 14]
                *block(base_channels * 4, base_channels * 8),          # -> [B, 512, 7, 7]
                nn.Conv2d(base_channels * 8, 1, kernel_size=3, padding=1)  # -> [B, 1, 7, 7]
            )

        def forward(self, x):
            return self.model(x) 
    