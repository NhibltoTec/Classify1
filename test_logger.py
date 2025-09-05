import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.logger import get_logger, TensorboardLogger  # đoạn bạn viết

# Fake dataset cho demo
X = torch.randn(500, 10)
y = torch.randint(0, 2, (500,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model đơn giản
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Khởi tạo logger
logger = get_logger("train.log")
tb_logger = TensorboardLogger("runs/exp1")

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # tqdm progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step, (inputs, targets) in pbar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Cập nhật tqdm giống YOLO
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

    avg_loss = train_loss / len(train_loader)

    # Log ra file và console
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Log TensorBoard
    tb_logger.log_metrics({"loss": avg_loss}, step=epoch, prefix="train")

tb_logger.close()
