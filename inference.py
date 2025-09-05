import torch
from torchvision import transforms
from PIL import Image
from models.classifier import Classifier
import yaml
import os
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), img
path = "Classify-Waste--1/test"
class_names = cfg.get("classes", None)

results = []

if os.path.isfile(path):
    pred, conf, img = predict(path)
    label = class_names[pred] if class_names else f"class {pred}"
    gt = os.path.basename(os.path.dirname(path))

    print(f"{path} -> GT: {gt}, Pred: {label} ({conf:.2f})")

    plt.imshow(img)
    plt.title(f"GT: {gt} | Pred: {label} ({conf:.2f})")
    plt.axis("off")
    plt.show()

    results.append({"file": os.path.basename(path), "ground_truth": gt, "prediction": label, "confidence": conf})

elif os.path.isdir(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                pred, conf, img = predict(img_path)
                label = class_names[pred] if class_names else f"class {pred}"
                gt = os.path.basename(root)

                print(f"{img_path} -> GT: {gt}, Pred: {label} ({conf:.2f})")

                plt.imshow(img)
                plt.title(f"GT: {gt} | Pred: {label} ({conf:.2f})")
                plt.axis("off")
                plt.show()

                results.append({
                    "file": os.path.relpath(img_path, path),
                    "ground_truth": gt,
                    "prediction": label,
                    "confidence": conf
                })
else:
    print(f" Path '{path}' not found!")
if results:
    os.makedirs("runs", exist_ok=True)
    save_path = "runs/inference_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Inference results saved to {save_path}")
