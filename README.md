# AgriQPro - Quantum-Driven Precision Optimization for Smart Agriculture

This repository contains the implementation of a plant disease classification system based on the research paper **"AgriQPro"**. The system utilizes a **Swin Transformer V2** backbone enhanced with a **Quantum-Inspired Feature Interference (QIFI)** module for improved feature refinement and classification accuracy.

## Features

- **Advanced Architecture**: 
  - **Backbone**: Pretrained Swin Transformer V2 (`swinv2_tiny_window8_256`) adapted for 224x224 input.
  - **QIFI Module**: Custom Quantum-Inspired Feature Interference layers (Tanh -> Softmax -> Residual Attention).
  - **Head**: MLP with LayerNorm, GELU, and Dropout.
- **Robust Training Pipeline**:
  - CrossEntropyLoss with AdamW optimizer.
  - Cosine Annealing Learning Rate Scheduler.
  - Model checkpointing (Best & Latest).
- **Data Augmentation**:
  - Resize (256x256), RandomCrop (224x224).
  - AutoAugment (ImageNet Policy).
  - Normalization (ImageNet stats).
- **Inference Optimization**:
  - **Test-Time Augmentation (TTA)**: Averaging predictions from original and horizontally flipped images.
  - **INT8 Dynamic Quantization**: Post-training quantization of QIFI and MLP Head layers for efficient edge deployment.

## Installation

1. Clone this repository (if applicable) or navigate to the project directory.
2. Install the required dependencies:

```bash
pip install -r Trail1/requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch >= 1.12.0
- Torchvision >= 0.13.0
- Timm >= 0.6.12
- Scikit-learn
- Tqdm
- Pillow
- Numpy

## Project Structure

```
Trail1/
├── dataset.py       # Custom Dataset class and transforms
├── model.py         # AgriQPro model definition (SwinV2 + QIFI)
├── train.py         # Training script
├── evaluate.py      # Evaluation and Inference script
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## Usage

### 1. Training

To train the model from scratch (fine-tuning the backbone):

```bash
python Trail1/train.py \
  --train_dir "path/to/train_dataset" \
  --test_dir "path/to/test_dataset" \
  --batch_size 16 \
  --epochs 20
```

**Arguments:**
- `--train_dir`: Path to training dataset root (containing class subfolders). Automatically splits into Train (80%) and Val (20%).
- `--test_dir`: Path to test dataset root.
- `--backbone`: SwinV2 model name (default: `swinv2_tiny_window8_256`).
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`).

### 2. Evaluation

To evaluate a trained model on the test set:

```bash
python Trail1/evaluate.py \
  --checkpoint_path checkpoints/best_model.pth \
  --train_dir "path/to/train_dataset" \
  --test_dir "path/to/test_dataset"
```

**Note**: `--train_dir` is required during evaluation to correctly infer class names and structure.

### 3. Inference Optimizations

**Test-Time Augmentation (TTA):**
Use the `--tta` flag to enable TTA (Original + Horizontal Flip averaging):
```bash
python Trail1/evaluate.py --checkpoint_path checkpoints/best_model.pth --train_dir ... --test_dir ... --tta
```

**INT8 Dynamic Quantization:**
Use the `--quantize` flag to quantize the model (QIFI + Head) and save a quantized version for edge deployment:
```bash
python Trail1/evaluate.py --checkpoint_path checkpoints/best_model.pth --train_dir ... --test_dir ... --quantize
```
This will save a file named `*_quantized.pth` in the checkpoint directory.

### 4. Interactive UI

To launch the interactive Streamlit application for testing model predictions on images:

```bash
streamlit run Trail1/app.py
```

- Upload an image (JPG/PNG).
- Specify the checkpoint path. The app defaults to `models/best_model_quantized.pth` for deployment.
- View the predicted class and confidence score.

### 5. Deployment

To deploy the application (e.g., to GitHub/Streamlit Cloud):
1. Ensure the `models/` directory contains `best_model_quantized.pth`.
2. The `checkpoints/` folder is typically ignored in `.gitignore`, but ensure `models/best_model_quantized.pth` is tracked.
3. Push the repository to GitHub.
4. Connect to Streamlit Cloud and deploy!

## Benchmark Results

The following results were obtained from evaluating the model on the test dataset (2400 images):

| Metric | Standard Evaluation | Test-Time Augmentation (TTA) | INT8 Dynamic Quantization |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 98.75% | **98.88%** | 98.79% |
| **F1 Score (Weighted)** | 0.9875 | **0.9887** | 0.9879 |
| **Inference Time (2400 images)** | ~564s | ~696s | **~535s** |

### Per-Class Performance (Standard)

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Bacterial Leaf Disease | 0.99 | 0.99 | 0.99 |
| Dried Leaf | 1.00 | 1.00 | 1.00 |
| Fungal Brown Spot Disease | 0.99 | 0.99 | 0.99 |
| Healthy Leaf | 0.98 | 0.98 | 0.98 |
| Leaf Rot | 0.98 | 1.00 | 0.99 |
| Leaf Spot | 0.98 | 0.96 | 0.97 |

## Model Details

The **AgriQPro** model processes images as follows:
1. **Input**: (B, 3, 224, 224)
2. **Backbone**: Swin Transformer V2 extracts features.
3. **Global Average Pooling**: Converts spatial maps to feature vectors (B, D).
4. **QIFI Module (x2)**: 
   - Non-linear transformation (Tanh).
   - Attention weight calculation (Softmax).
   - Residual refinement ($x_{out} = x_{in} + x_{in} \odot \alpha$).
5. **Classifier**: 3-layer MLP projecting to class probabilities.

## Citations

- *AgriQPro: Quantum Driven Precision Optimization for Smart Agriculture*
- Swin Transformer V2: *Swin Transformer V2: Scaling Up Capacity and Resolution*
