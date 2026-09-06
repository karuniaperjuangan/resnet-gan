import os
from constant import *
import torch
from torch import nn
from torchvision.utils import save_image

@torch.no_grad()
def save_samples(
    generator:nn.Module,
    fixed_noise:torch.Tensor,
    epoch:int
):
    generator.eval()

    out_fake = generator(fixed_noise)

    # Convert range [-1,1] from tanh generator to [0,1] for save image
    out_fake = (out_fake.clamp(-1,1) + 1) /2

    save_image(out_fake,fp=os.path.join(OUTPUT_DIR,f"epoch_{epoch:04d}.png"))

    generator.train()

def save_checkpoint(generator,
                    critic,
                    optimizer_g,
                    optimizer_c,
                    epoch
                    ):

    checkpoint = {
        "epoch":epoch,
        "critic":critic.state_dict(),
        "generator":generator.state_dict(),
        "optimizer_g":optimizer_g.state_dict(),
        "optimizer_c":optimizer_c.state_dict()
    }

    torch.save(checkpoint, f"{CHECKPOINT_DIR}/wgan_gp_{epoch:04d}.pt",)
    