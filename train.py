'''
Usage:

python train.py \
  --data_dir /path/to/lfw_funneled \
  --checkpoint_dir ./checkpoints \
  --epochs 15 \
  --batch_size 32 \
  --lr 2e-4 \
  --checkpoint ./checkpoints/checkpoint_epoch_10.pth

'''

import os
import argparse
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
# Argument Parser
# ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, 
                    default="/media/shaun/T7/personal/IDVerse/archive/lfw-funneled/lfw_funneled", help='Path to LFW dataset')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Where to save model checkpoints')
parser.add_argument('--checkpoint', type=str, default=None, help='Optional path to resume checkpoint')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--g_beta1', type=float, default=0.1)
parser.add_argument('--d_beta1', type=float, default=0.05)
parser.add_argument('--adv_weight', type=float, default=1.0)
parser.add_argument('--rec_weight', type=float, default=3.5)
parser.add_argument('--id_weight', type=float, default=14.0)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────
# Init wandb
# ────────────────────────
wandb.init(project='faceswap_GAN', entity='shaunwerkhoven-i', config=vars(args))

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
dataset = LFWDataset(args.data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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
g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.g_beta1, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.d_beta1, 0.999))

# ────────────────────────
# Resume from Checkpoint (if provided)
# ────────────────────────
start_epoch = 0
if args.checkpoint and os.path.exists(args.checkpoint):
    print(f"[✓] Resuming from checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint)
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
    'adv': args.adv_weight,
    'rec': args.rec_weight,
    'id': args.id_weight
})

# ────────────────────────
# Training Loop
# ────────────────────────
os.makedirs(args.checkpoint_dir, exist_ok=True)

for epoch in range(start_epoch, args.epochs):
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
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}] D_loss: {d_loss.item():.4f} \
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
    }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

wandb.finish()

