import torch
from argparse import ArgumentParser
from pathlib import Path

from models import Generator
from constant import *
from utils import save_image
import random


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        "-cp",
        required=True,
        help="Path to a training checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-path", "-o", default=None, help="Output image path")

    args = parser.parse_args()
    generator = Generator().to(DEVICE)
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    noise = torch.randn(
        64,
        LATENT_DIM,
        device=DEVICE,
    )

    with torch.no_grad():
        out_fake = generator(noise)

    # Convert range [-1,1] from tanh generator to [0,1] for save image
    out_fake = (out_fake.clamp(-1, 1) + 1) / 2

    if args.output_path is None:
        output_path = Path(OUTPUT_DIR) / f"generated_seed_{seed}.png"
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(out_fake, fp=output_path)
    print(f"Saved generated samples to {output_path}")

