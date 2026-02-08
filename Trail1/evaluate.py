import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import AgriQPro
from dataset import get_dataloaders

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    # For evaluation, we mainly need test_loader and classes. 
    # train_dir is needed by get_dataloaders to infer classes and create train/val splits (even if unused here).
    _, _, test_loader, classes = get_dataloaders(
        args.train_dir, 
        args.test_dir, 
        batch_size=args.batch_size
    )
    num_classes = len(classes)
    
    # Load Model
    print("Loading model...")
    model = AgriQPro(num_classes=num_classes, backbone_name=args.backbone)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Quantization (CPU only usually for dynamic quant via PyTorch)
    if args.quantize:
        print("Applying INT8 Dynamic Quantization...")
        # Move to CPU for quantization as PyTorch dynamic quantization is CPU optimized
        model.cpu()
        
        # We only quantize the QIFI modules and the head, as the timm SwinV2 backbone 
        # uses functional calls that are incompatible with dynamic quantization module replacement.
        model.qifi1 = torch.quantization.quantize_dynamic(
            model.qifi1, {nn.Linear}, dtype=torch.qint8
        )
        model.qifi2 = torch.quantization.quantize_dynamic(
            model.qifi2, {nn.Linear}, dtype=torch.qint8
        )
        model.mlp_head = torch.quantization.quantize_dynamic(
            model.mlp_head, {nn.Linear}, dtype=torch.qint8
        )
        
        quantized_model = model
        print("Model (QIFI + Head) quantized.")
        
        # Save quantized model
        quantized_model_path = args.checkpoint_path.replace(".pth", "_quantized.pth")
        torch.save(quantized_model.state_dict(), quantized_model_path)
        print(f"Quantized model saved to {quantized_model_path}")

        eval_device = torch.device("cpu")
        model_to_eval = quantized_model
    else:
        eval_device = device
        model_to_eval = model

    # Evaluation Loop
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)
            
            if args.tta:
                # Test-Time Augmentation: Original + Horizontal Flip
                # 1. Original
                out1 = model_to_eval(images)
                
                # 2. Horizontal Flip
                images_flipped = torch.flip(images, [3])
                out2 = model_to_eval(images_flipped)
                
                # Average probabilities
                probs1 = torch.softmax(out1, dim=1)
                probs2 = torch.softmax(out2, dim=1)
                avg_probs = (probs1 + probs2) / 2.0
                
                _, predicted = torch.max(avg_probs, 1)
            else:
                outputs = model_to_eval(images)
                _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f}s for {len(test_loader.dataset)} images")

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=classes)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(report_text)
    
    # Save Metrics to Text File
    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print("Metrics saved to metrics.txt")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved to confusion_matrix.png")
    plt.close()

    # Plot F1-Score per Class
    f1_per_class = {cls: report[cls]['f1-score'] for cls in classes}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(f1_per_class.keys()), y=list(f1_per_class.values()), palette='viridis')
    plt.title('F1 Score per Class')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('f1_per_class.png')
    print("Per-Class F1 Score plot saved to f1_per_class.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AgriQPro Model")
    parser.add_argument("--train_dir", type=str, default="G:\\My Drive\\Betel-Leaf\\Datasets", help="Path to training dataset (for classes)")
    parser.add_argument("--test_dir", type=str, default="G:\\My Drive\\Betel-Leaf\\Test_Dataset", help="Path to test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--backbone", type=str, default="swinv2_tiny_window8_256", help="Backbone model name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 Dynamic Quantization")
    
    args = parser.parse_args()
    
    if os.path.exists(args.test_dir) and os.path.exists(args.checkpoint_path):
         # Checking train_dir existence is good but technically if we just want test steps... 
         # But get_dataloaders currently loads it.
         if not os.path.exists(args.train_dir):
             print(f"Warning: Train dir {args.train_dir} not found. Classes might be incorrect if not loaded structure.")
         evaluate_model(args)
    else:
        print("Data directory or checkpoint path not found.")
