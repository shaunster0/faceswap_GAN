import torch.nn as nn
import torch


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

