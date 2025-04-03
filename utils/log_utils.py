import torchvision.utils as vutils
import torch
import wandb


def log_side_by_side_images(source, target, generated, step=None):
    # Denormalize from [-1, 1] to [0, 1]
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    # Take only the first sample from the batch
    source_img = denorm(source[0])
    target_img = denorm(target[0])
    generated_img = denorm(generated[0])

    # Stack side by side: [source | target | generated]
    grid = vutils.make_grid(
        torch.stack([source_img, target_img, generated_img]),
        nrow=3,
        padding=2
    )

    wandb.log({
        "comparison": wandb.Image(grid, caption="Source | Target | Generated"),
        "step": step
    })

