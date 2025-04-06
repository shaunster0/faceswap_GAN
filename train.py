'''
Usage:

python train.py \
  --config config.yaml \
  --data_dir /path/to/lfw_funneled \
  --checkpoint_dir ./checkpoints \
  --checkpoint ./checkpoints/checkpoint_epoch_10.pth
'''

import os
import argparse
import yaml
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LFWDataset
from loss import LossFunction
from models.generator import AADGenerator
from models.encoder import ArcFaceIdentityEncoder, AttributeEncoder
from models.discriminator import Discriminator
from utils.log_utils import log_side_by_side_images

# ────────────────────────
# Config Loader
# ────────────────────────
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ────────────────────────
# Argument Parser
# ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config file')
parser.add_argument('--data_dir', type=str, help='Path to LFW dataset')
parser.add_argument('--checkpoint_dir', type=str, help='Where to save model checkpoints')
parser.add_argument('--checkpoint', type=str, help='Optional path to resume checkpoint')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--g_beta1', type=float, help='Beta1 for generator')
parser.add_argument('--d_beta1', type=float, help='Beta1 for discriminator')
parser.add_argument('--adv_weight', type=float, help='Adversarial loss weight')
parser.add_argument('--rec_weight', type=float, help='Reconstruction loss weight')
parser.add_argument('--id_weight', type=float, help='Identity loss weight')
args = parser.parse_args()

# ────────────────────────
# Merge config file with CLI overrides
# ────────────────────────
config = load_config(args.config)
for k, v in vars(args).items():
    if v is not None:
        config[k] = v

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────
# Init wandb
# ────────────────────────
wandb.init(project='faceswap_GAN', entity='shaunwerkhoven-i', config=config)

# ────────────────────────
# Transforms
# ────────────────────────
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ────────────────────────
# Dataset
# ────────────────────────
dataset = LFWDataset(config['data_dir'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# ────────────────────────
# Models
# ────────────────────────
generator = AADGenerator().to(DEVICE)
identity_encoder = ArcFaceIdentityEncoder('arcface_model/backbone.pth').to(DEVICE)
attribute_encoder = AttributeEncoder().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# ────────────────────────
# Optimizers
# ────────────────────────
g_optim = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['g_beta1'], 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['d_beta1'], 0.999))

# ────────────────────────
# Resume from Checkpoint
# ────────────────────────
start_epoch = 0
if config.get('checkpoint') and os.path.exists(config['checkpoint']):
    print(f"[✓] Resuming from checkpoint: {config['checkpoint']}")
    ckpt = torch.load(config['checkpoint'])
    generator.load_state_dict(ckpt['generator'])
    identity_encoder.load_state_dict(ckpt['identity_encoder'])
    attribute_encoder.load_state_dict(ckpt['attribute_encoder'])
    discriminator.load_state_dict(ckpt['discriminator'])
    g_optim.load_state_dict(ckpt['g_optim'])
    d_optim.load_state_dict(ckpt['d_optim'])
    start_epoch = ckpt['epoch']

# ────────────────────────
# Loss Function
# ────────────────────────
loss_fn = LossFunction(identity_model=identity_encoder, weights={
    'adv': config['adv_weight'],
    'rec': config['rec_weight'],
    'id': config['id_weight']
})

# ────────────────────────
# Training Loop
# ────────────────────────
os.makedirs(config['checkpoint_dir'], exist_ok=True)

for epoch in range(start_epoch, config['epochs']):
    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            z_id = identity_encoder(source)
        z_attrs = attribute_encoder(target)

        generated = generator(z_id, z_attrs)

        # Discriminator update
        d_real = discriminator(target)
        d_fake = discriminator(generated.detach())
        d_loss = loss_fn.discriminator_loss(d_real, d_fake)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Generator update
        d_fake = discriminator(generated)
        g_loss, losses_dict = loss_fn.generator_loss(generated, target, d_fake, z_id)

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Logging
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}] Step [{i}] D_loss: {d_loss.item():.4f} \
                    G_loss: {g_loss.item():.4f} | Detail: {losses_dict}")

        wandb.log({
            "step": epoch * len(dataloader) + i,
            "D_loss": d_loss.item(),
            "G_loss": g_loss.item(),
            "adv_loss": losses_dict['adv_loss'],
            "rec_loss": losses_dict['rec_loss'],
            "id_loss": losses_dict['id_loss']
        })

        if i % 50 == 0:
            log_side_by_side_images(source, target, generated, step=epoch * len(dataloader) + i)

    # Save checkpoint
    torch.save({
        'generator': generator.state_dict(),
        'identity_encoder': identity_encoder.state_dict(),
        'attribute_encoder': attribute_encoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'epoch': epoch + 1
    }, os.path.join(config['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth"))

wandb.finish()