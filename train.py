# simplified_train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.generator import AADGenerator
# from models.encoder import IdentityEncoder, AttributeEncoder
from models.encoder import ArcFaceIdentityEncoder, AttributeEncoder
from models.discriminator import Discriminator
from dataset import LFWDataset
from loss import LossManager, LossFunction

# Config
DATA_DIR = "/media/shaun/T7/personal/IDVerse/archive/lfw-funneled/lfw_funneled"
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),             # ArcFace expects 112x112
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and Loader
dataset = LFWDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator = AADGenerator().to(DEVICE)
identity_encoder = ArcFaceIdentityEncoder('/media/shaun/T7/personal/IDVerse/arcface_model/backbone.pth').to(DEVICE)
attribute_encoder = AttributeEncoder().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Optimizers
g_optim = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Loss Manager
loss_fn = LossFunction(identity_model=identity_encoder, weights={
    'adv': 1.0,
    'rec': 10.0,
    'id': 5.0
})

# Training loop
for epoch in range(EPOCHS):
    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(DEVICE), target.to(DEVICE)

        # Encode identity and attributes
        with torch.no_grad():
            z_id = identity_encoder(source)
        z_attrs = attribute_encoder(target)

        # Generate face
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

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} | Detail: {losses_dict}")

    # Save model checkpoints after each epoch
    torch.save({
        'generator': generator.state_dict(),
        'identity_encoder': identity_encoder.state_dict(),
        'attribute_encoder': attribute_encoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'epoch': epoch + 1
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

