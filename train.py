from pathlib import Path

import torch
from tqdm import tqdm
from torch import nn
from constant import *
from utils import save_checkpoint, save_samples

def set_requires_grad(
    model,
    enabled: bool,
):

    for parameter in model.parameters():
        parameter.requires_grad_(enabled)

def train(dataloader:torch.utils.data.DataLoader, generator:nn.Module, critic:nn.Module):

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LR,
        betas=BETAS,
        fused=DEVICE.type == "cuda"
    )

    optimizer_c = torch.optim.Adam(
        critic.parameters(),
        lr=LR,
        betas=BETAS,
        fused=DEVICE.type == "cuda"
    )

    fixed_noise = torch.randn(
        64,
        LATENT_DIM,
        device=DEVICE,
    )

    generator_step = 0

    for epoch in range(1, NUM_EPOCHS+1):
        # Ignore dataloader label as we only use x for unsupervised learning
        for batch_index, (real, _) in enumerate(progress:=tqdm(dataloader,desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            real = real.to(DEVICE, non_blocking=True)

            batch_size = real.size(0)

            # Train Critic model, freeze Generator model
            set_requires_grad(generator, False)
            set_requires_grad(critic, True)

            optimizer_c.zero_grad()

            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)

            with torch.no_grad():
                fake = generator(z)

            real_score = critic(real)
            fake_score = critic(fake)

            # Hinge loss separates real scores above 1 and fake scores below -1
            critic_loss = (
                torch.relu(1.0 - real_score).mean()
                + torch.relu(1.0 + fake_score).mean()
            )

            critic_loss.backward()

            optimizer_c.step()

            # Train Generator model, freeze Critic model

            generator_loss_value = None

            if(batch_index+1) % N_CRITIC == 0:

                set_requires_grad(critic, False)
                set_requires_grad(generator,True)

                optimizer_g.zero_grad()

                z = torch.randn(batch_size,LATENT_DIM, device=DEVICE)

                fake = generator(z)

                fake_score_for_g = critic(fake)

                generator_loss = -fake_score_for_g.mean()

                generator_loss.backward()

                optimizer_g.step()

                generator_step +=1

                generator_loss_value = generator_loss.item()

            postfix = {
                "C": f"{critic_loss.item():.3f}",
                "R": f"{real_score.mean().item():.3f}",
                "F": f"{fake_score.mean().item():.3f}",
            }

            if generator_loss_value is not None:
                postfix["G"]= f"{generator_loss_value:.3f}"
            progress.set_postfix(postfix)

        save_samples(generator=generator, fixed_noise=fixed_noise, epoch=epoch)

        # Checkpoint every 5 epoch
        if epoch % 5 == 0:
            save_checkpoint(
                generator=generator,
                critic=critic,
                optimizer_g=optimizer_g,
                optimizer_c=optimizer_c,
                epoch=epoch
            )



            
