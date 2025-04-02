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
    def __init__(self, in_channels, id_channels, attr_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.mlp_id_gamma = nn.Linear(id_channels, in_channels)
        self.mlp_id_beta = nn.Linear(id_channels, in_channels)
        self.mlp_attr_gamma = nn.Conv2d(attr_channels, in_channels, kernel_size=1)
        self.mlp_attr_beta = nn.Conv2d(attr_channels, in_channels, kernel_size=1)

    def forward(self, x, z_id, z_attr):
        x_norm = self.norm(x)

        gamma_id = self.mlp_id_gamma(z_id).unsqueeze(2).unsqueeze(3)
        beta_id = self.mlp_id_beta(z_id).unsqueeze(2).unsqueeze(3)

        # Resize attribute feature if needed
        if z_attr.shape[2:] != x.shape[2:]:
            z_attr = F.interpolate(z_attr, size=x.shape[2:], mode='bilinear', align_corners=False)

        gamma_attr = self.mlp_attr_gamma(z_attr)
        beta_attr = self.mlp_attr_beta(z_attr)

        return gamma_id * x_norm + beta_id + gamma_attr * x_norm + beta_attr


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
            AADResBlock(512, 512, z_id_dim, 64),    # z_attrs[0] from encoder
            AADResBlock(512, 512, z_id_dim, 64),    # z_attrs[1]
            AADResBlock(512, 256, z_id_dim, 128),   # z_attrs[2]
            AADResBlock(256, 128, z_id_dim, 256),   # z_attrs[3]
            AADResBlock(128, 64, z_id_dim, 512),    # z_attrs[4]
            AADResBlock(64, 32, z_id_dim, 256),     # z_attrs[5]
            AADResBlock(32, 16, z_id_dim, 128),     # z_attrs[6]
            AADResBlock(16, 8, z_id_dim, 64),       # z_attrs[7]
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, z_id, z_attrs):
        x = self.fc(z_id).view(-1, 512, 4, 4)
        for i, block in enumerate(self.blocks):
            x = self.upsample(block(x, z_id, z_attrs[i]))

        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        return torch.tanh(self.out_conv(x))
    