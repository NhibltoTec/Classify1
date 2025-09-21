# 🖼️ Image Classification Project

## 1. Introduction
This project implements a **general image classification model** using deep learning.  
It can be applied to:
- Classifying animals, objects, or scenes
- Academic research or machine learning coursework
- Fine-tuning on custom datasets for real-world applications

**Frameworks & Technologies Used:**
- [PyTorch](https://pytorch.org/) – for training and inference
- torchvision – for data augmentation and preprocessing
- matplotlib / tqdm – for visualization and progress tracking


---

## 2. Installation

### System Requirements
- **Python:** 3.10 or higher  
- **GPU:** (Optional) CUDA-enabled GPU recommended for faster training

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/classify-model.git
cd classify-model
pip install -r requirements.txt
```

## 3. Dataset

### Dataset Used
You can use any image dataset (CIFAR-10, ImageNet subset, or your custom dataset).

### Organize your dataset like this:

## 📂 Dataset Structure

Organize your dataset like this:

```bash
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/

    
You can also automatically download or prepare the dataset:
  !python download_dataset.py
```
## 4. Training

1. Edit `configs/config_train.yaml`:
   - Set `dataset.root` to your dataset folder (e.g., `Classify-Waste--1`)
   - Update `model.head.num_classes` to match the number of classes

2. Run training:
```bash
python train.py
```
## 5. Evaluation
run:
```bash
python validate.py
```

---

## 6. Inference
Run inference on a single image or a folder:
**Example (folder):**
```bash
python inference.py --path Classify-Waste--1/test --weights runs/best_model.pth
```
## 7. Results & Performance
Metrics reported:
Training loss & accuracy per epoch
Validation accuracy after each epoch
Final best model performance saved in train.log
Typical results (on Classify Waste dataset):
Accuracy: ~98%

## 8. Deploy
- Convert PyTorch model (`.pth`) to other formats:
  - **TorchScript**: `torch.jit.trace`
  - **ONNX**: `torch.onnx.export`
  - **TFLite**: via ONNX → TFLite converter
  - **NCNN**: via ONNX → ncnn converter
  - import torch
import os
import argparse
import yaml
import subprocess

from models.classifier import Classifier  # <-- import model của bạn

def export_model(pth_path, output_dir, input_shape):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open("configs/config_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Load model
    device = "cpu"
    model = Classifier(cfg)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(*input_shape)

    # 1️⃣ TorchScript
    traced = torch.jit.trace(model, dummy_input)
    torchscript_path = os.path.join(output_dir, "model_torchscript.pt")
    traced.save(torchscript_path)
    print(f"[✔] Saved TorchScript → {torchscript_path}")

    # 2️⃣ ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f"[✔] Saved ONNX → {onnx_path}")

    # 3️⃣ TFLite (tùy chọn)
    tflite_path = os.path.join(output_dir, "model.tflite")
    try:
        import onnx
        import onnx_tf.backend as backend
        import tensorflow as tf

        model_onnx = onnx.load(onnx_path)
        tf_rep = backend.prepare(model_onnx)
        tf_rep.export_graph(os.path.join(output_dir, "model_tf"))

        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(output_dir, "model_tf"))
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[✔] Saved TFLite → {tflite_path}")
    except Exception as e:
        print(f"[⚠] TFLite conversion failed: {e}")

    # 4️⃣ NCNN (tùy chọn)
    try:
        subprocess.run(["onnx2ncnn", onnx_path, 
                        os.path.join(output_dir, "model.param"), 
                        os.path.join(output_dir, "model.bin")], check=True)
        print(f"[✔] Saved NCNN model → model.param + model.bin")
    except FileNotFoundError:
        print("[⚠] NCNN conversion skipped (onnx2ncnn not found)")
### Cách chạy:
cd ClassifyModel
python tools/convert_model.py --pth runs/last_model.pth --out tools/converted --shape 1 3 224 224

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--out", type=str, default="converted", help="Output folder")
    parser.add_argument("--shape", type=int, nargs="+", default=[1, 3, 224, 224], help="Input shape")
    args = parser.parse_args()

    export_model(args.pth, args.out, tuple(args.shape))


This allows deploying the model on mobile or embedded devices.

---

## 9. Contact
- Author: [Lê Trương Uyển Nhi]  
- Email: [ltuyennhi11b1@gmail.com]  
- GitHub: [github.com/NhibltoTec]



