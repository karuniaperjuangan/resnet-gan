import torch

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3

LATENT_DIM = 128

BATCH_SIZE = 64
NUM_EPOCHS = 50
MAX_SAMPLE = 20000

LR = 1e-4
BETAS = (0.0, 0.9)

N_CRITIC = 2
LAMBDA_GP = 10.0

NUM_WORKERS = 4
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
