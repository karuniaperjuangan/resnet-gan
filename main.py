from pathlib import Path

from models import Generator, Critic
from train import train
from constant import *
import random

from data import create_dataloader

if __name__ == "__main__":
    seed = SEED
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        
    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(CHECKPOINT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {DEVICE}")

    dataloader = create_dataloader()
    print(
        f"CelebA training images: "
        f"{len(dataloader.dataset):,}"
    )

    generator = Generator().to(DEVICE)
    print("Generator size:",sum(p.numel() for p in generator.parameters()))

    critic = Critic().to(DEVICE)
    print("Critic size:",sum(p.numel() for p in critic.parameters()))

    train(dataloader=dataloader, generator=generator, critic=critic)


