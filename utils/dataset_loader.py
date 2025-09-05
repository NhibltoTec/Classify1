import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dataset_loader(cfg, mode="train"):
    dataset_cfg = cfg["dataset"]

    root_dir = os.path.join(dataset_cfg["root"], dataset_cfg[f"{mode}_dir"])
    img_size = dataset_cfg["img_size"]

    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=dataset_cfg["batch_size"],
                            shuffle=(mode == "train"),
                            num_workers=dataset_cfg["num_workers"])

    return dataloader, dataset.classes