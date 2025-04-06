import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 112 * 112),
            nn.Tanh()
        )

    def forward(self, z_id, z_attr):
        z = torch.cat([z_id, z_attr], dim=1)
        out = self.fc(z)  # shape: [B, 3*112*112]
        out = out.view(z.size(0), 3, 112, 112)  # explicit batch dim
        return out


class AADLayer(nn.Module):
    def __init__(self, in_channels, id_channels, attr_channels, w_id=1.0, w_attr=0.8):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.in_channels = in_channels
        self.w_id = w_id
        self.w_attr = w_attr

        # Identity MLPs (with ReLU)
        self.mlp_id_gamma = nn.Sequential(
            nn.Linear(id_channels, in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp_id_beta = nn.Sequential(
            nn.Linear(id_channels, in_channels),
            nn.ReLU(inplace=True)
        )

        # Attribute MLPs (Conv2d)
        self.mlp_attr_gamma = nn.Conv2d(attr_channels, in_channels, kernel_size=1)
        self.mlp_attr_beta = nn.Conv2d(attr_channels, in_channels, kernel_size=1)

        # Learnable spatial mask M
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, z_id, z_attr):
        x_norm = self.norm(x)

        # Identity pathway
        gamma_id = self.mlp_id_gamma(z_id).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta_id = self.mlp_id_beta(z_id).unsqueeze(2).unsqueeze(3)
        I = gamma_id * x_norm + beta_id

        # Attribute pathway
        if z_attr.shape[2:] != x.shape[2:]:
            z_attr = F.interpolate(z_attr, size=x.shape[2:], mode='bilinear', align_corners=False)

        gamma_attr = self.mlp_attr_gamma(z_attr)
        beta_attr = self.mlp_attr_beta(z_attr)
        A = gamma_attr * x_norm + beta_attr

        # Adaptive blending mask
        M = torch.sigmoid(self.mask_conv(x_norm))  # [B, 1, H, W]

        # Blend the identity and attribute contributions
        out = (1 - M) * self.w_attr * A + M * self.w_id * I
        return out

class AADResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, z_id_dim, z_attr_channels):
        super().__init__()
        self.aad1 = AADLayer(in_channels, z_id_dim, z_attr_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.aad2 = AADLayer(out_channels, z_id_dim, z_attr_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, z_id, z_attr):
        residual = self.skip(x)
        out = F.relu(self.aad1(x, z_id, z_attr))
        out = self.conv1(out)
        out = F.relu(self.aad2(out, z_id, z_attr))
        out = self.conv2(out)
        return out + residual


class AADGenerator(nn.Module):
    def __init__(self, z_id_dim=512):
        super().__init__()
        self.fc = nn.Linear(z_id_dim, 512 * 4 * 4)

        self.blocks = nn.ModuleList([
            AADResBlock(512, 512, z_id_dim, 64),    # e1
            AADResBlock(512, 512, z_id_dim, 64),    # e2
            AADResBlock(512, 256, z_id_dim, 128),   # e3
            AADResBlock(256, 128, z_id_dim, 256),   # e4
            AADResBlock(128, 128, z_id_dim, 512),   # e5
            AADResBlock(128, 64, z_id_dim, 256),    # d1
            AADResBlock(64, 64, z_id_dim, 128),     # d2
            AADResBlock(64, 32, z_id_dim, 64),      # d4
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, z_id, z_attrs):
        x = self.fc(z_id).view(-1, 512, 4, 4)
        for i, block in enumerate(self.blocks):
            x = self.upsample(block(x, z_id, z_attrs[i]))

        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        return torch.tanh(self.out_conv(x))
    