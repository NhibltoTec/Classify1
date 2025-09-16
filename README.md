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

**Quick Demo Result:**  
After training on a sample dataset, the model achieves **98% accuracy**<img width="514" height="389" alt="Ảnh màn hình 2025-09-16 lúc 10 29 26" src="https://github.com/user-attachments/assets/f3905346-efe6-48b1-a7bf-211d7e5ec43b" />


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

data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/

    
You can also automatically download or prepare the dataset:
  !python download_dataset.py
## 4.  Training

- Load training configuration from `configs/config_train.yaml`
- Initialize model (`Classifier`), optimizer (Adam / SGD), and loss function (CrossEntropyLoss)
- Create `DataLoader` for train/val datasets
- Run training loop for `epochs` with:
  - Forward pass
  - Loss calculation
  - Backpropagation & optimizer step
- Validate after each epoch and log results
- Save the best model weights to `runs/best_model.pth`
- Track training progress with `tqdm` and optionally TensorBoard logs


## 5. Evaluation

- Load trained model checkpoint (e.g. `runs/best_model.pth`)
- Create validation/test `DataLoader`
- Switch model to evaluation mode (`model.eval()`)
- Compute metrics:
  - Accuracy
  - Precision, Recall, F1-score (optional)
  - Confusion matrix (optional, visualize with matplotlib)
- Print and log results for analysis

---

## 6. Inference

Perform inference on single images or an entire folder.

**Steps:**
1. **Load Config & Model**
   - Load `config_train.yaml` for model params
   - Initialize model with correct `num_classes`
   - Load trained weights (`runs/best_model.pth`)
   - Set model to `eval` mode

2. **Preprocessing**
   - Open image with `PIL`
   - Resize to `(224, 224)`
   - Convert to tensor and normalize
   - Add batch dimension (`unsqueeze(0)`)

3. **Prediction**
   - Forward pass through model
   - Apply `softmax` to get probabilities
   - Select class with highest confidence
   - Map index → class name from config (if available)

4. **Visualization (Optional)**
   - Show image with predicted label and confidence
   - Print results to console

5. **Batch Mode**
   - If a folder is given, loop through all images
   - Store results (`file`, `ground_truth`, `prediction`, `confidence`) in a list

6. **Save Results**
   - Export predictions to `runs/inference_results.csv`
   - Use for later analysis or reporting
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

This allows deploying the model on mobile or embedded devices.

---

## 9. Contact
- Author: [Lê Trương Uyển Nhi]  
- Email: [ltuyennhi11b1@gmail.com]  
- GitHub: [github.com/NhibltoTec]
