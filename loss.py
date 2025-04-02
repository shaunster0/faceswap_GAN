import torch.nn as nn
import torch
import torch.nn.functional as F


class LossManager:
    def __init__(self):
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def discriminator_loss(self, real_pred, fake_pred):
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        return self.bce(real_pred, real_labels) + self.bce(fake_pred, fake_labels)

    def generator_loss(self, fake_img, target_img, fake_pred, z_id):
        adv_loss = self.bce(fake_pred, torch.ones_like(fake_pred))
        rec_loss = self.l1(fake_img, target_img)
        total = adv_loss + rec_loss
        return total, {"adv_loss": adv_loss.item(), "rec_loss": rec_loss.item()}


class LossFunction(nn.Module):
    def __init__(self, identity_model=None, weights=None):
        super().__init__()
        self.adv_criterion = self.hinge_loss
        self.rec_criterion = nn.L1Loss() # nn.MSELoss()
        self.id_criterion = self.cosine_identity_loss

        self.identity_model = identity_model.eval() if identity_model else None
        self.weights = weights or {
            'adv': 1.0,
            'rec': 10.0,
            'id': 5.0
        }

    def hinge_loss(self, pred, is_real):
        if is_real:
            return torch.mean(F.relu(1.0 - pred))
        else:
            return torch.mean(F.relu(1.0 + pred))

    def cosine_identity_loss(self, embed_real, embed_fake):
        cos_sim = F.cosine_similarity(embed_real, embed_fake, dim=1)
        return 1 - cos_sim.mean()

    def discriminator_loss(self, real_pred, fake_pred):
        real_loss = self.adv_criterion(real_pred, is_real=True)
        fake_loss = self.adv_criterion(fake_pred, is_real=False)
        return 0.5 * (real_loss + fake_loss)

    def generator_loss(self, fake_img, target_img, fake_pred, z_id, identity_model=None):
        adv_loss = -torch.mean(fake_pred)

        rec_loss = self.rec_criterion(fake_img, target_img)

        id_loss = 0.0
        if self.identity_model:
            with torch.no_grad():
                embed_real = self.identity_model(target_img)
            embed_fake = self.identity_model(fake_img)
            id_loss = self.id_criterion(embed_real, embed_fake)

        total_loss = (
            self.weights['adv'] * adv_loss +
            self.weights['rec'] * rec_loss +
            self.weights['id'] * id_loss
        )

        return total_loss, {
            'adv_loss': adv_loss.item(),
            'rec_loss': rec_loss.item(),
            'id_loss': id_loss.item() if isinstance(id_loss, torch.Tensor) else 0.0
        }

