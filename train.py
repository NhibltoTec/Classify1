from models.classifier import load_model
from utils.dataset_loader import dataset_loader
import torch
from utils.logger import get_logger, TensorboardLogger

logger = get_logger("test.log")
logger.info(f"Hello World")

import os
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.classifier import Classifier
from tqdm import tqdm
with open("configs/config_train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lấy giá trị từ config, nếu không có thì dùng mặc định
batch_size = cfg.get("batch_size", 32)
lr         = cfg.get("lr", 1e-3)
epochs     = cfg.get("epochs", 100)
backbone   = cfg.get("backbone", "shufflenet_v2_x1_0")
optimizer_name = cfg.get("optimizer", "Adam")

os.makedirs("runs", exist_ok=True)
log_path = "runs/train.log"

header = (
    f"\n===== Training started {datetime.datetime.now()} =====\n"
    f"Device: {device}\n"
    f"Backbone: {backbone}\n"
    f"Batch size: {batch_size}\n"
    f"Learning rate: {lr}\n"
    f"Epochs: {epochs}\n"
    f"Optimizer: {optimizer_name}\n"
)

with open(log_path, "a") as f:
    f.write(header)
    f.write("Full config:\n")
    for k, v in cfg.items():
        f.write(f"{k}: {v}\n")
    f.write("\n")

print(header)
print("===== Full config =====")
for k, v in cfg.items():
    print(f"{k}: {v}")
print("=======================\n")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder("Classify-Waste--1/train", transform=transform)
val_data   = datasets.ImageFolder("Classify-Waste--1/valid", transform=transform)

cfg["model"]["num_classes"] = len(train_data.classes)
print("✔ Classes:", train_data.classes)
print("✔ num_classes =", cfg["model"]["num_classes"])

model = Classifier(cfg).to(device)

print("Train classes:", train_data.classes)
print(" Val classes  :", val_data.classes)
print(" num_classes :", cfg["model"]["num_classes"])

#  Log param
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" Model Parameters: total={num_params:,} | trainable={num_trainable:,}")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0.0
save_path = "runs/best_model.pth"

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{train_loss/ (total/labels.size(0)) :.4f}",
            "acc": f"{correct/total:.4f}"
        })

    avg_loss = train_loss / len(train_loader)
    acc = correct / total
    print(f"\nEpoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")


    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f"Validation Acc: {val_acc:.4f}\n")


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f" Saved new best model (val_acc={val_acc:.4f}) to {save_path}")
