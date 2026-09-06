from constant import *
from torchvision import transforms, datasets

def create_dataloader():

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,0.5,0.5),
            std=(0.5,0.5,0.5)
        )
    ])

    dataset = datasets.CelebA(
        root=DATA_DIR,
        split="train",
        target_type="attr",
        transform=transform,
        download=True
    )

    if MAX_SAMPLE is not None and MAX_SAMPLE < len(dataset):
        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:MAX_SAMPLE]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type=="cuda",
        persistent_workers=NUM_WORKERS>0,
        drop_last=True
    )

    return dataloader
