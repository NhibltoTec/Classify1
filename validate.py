import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.classifier import Classifier
import yaml
from tqdm import tqdm

with open("configs/config_train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier(cfg).to(device)

checkpoint = "runs/best_model.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder("Classify-Waste--1/valid", transform=transform)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()

val_loss, correct, total = 0, 0, 0
pbar = tqdm(val_loader, desc="Validating", unit="batch")

with torch.no_grad():
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{val_loss/ (total/labels.size(0)) :.4f}",
            "acc": f"{correct/total:.4f}"
        })

print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.4f}")
