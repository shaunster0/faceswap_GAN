import torch.nn as nn
import torch


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
