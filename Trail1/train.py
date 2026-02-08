


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from model import AgriQPro
from dataset import get_dataloaders

def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data Loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.train_dir,
        args.test_dir,
        batch_size=args.batch_size, 
        val_split=0.2
    )
    print(f"Data loaded. Classes: {classes}")

    # Model
    print("Initializing model...")
    model = AgriQPro(num_classes=len(classes), backbone_name=args.backbone).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc="Training", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
            
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        # Handle case where val_loader might be empty if dataset is very small, though unlikely
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Scheduler Step
        scheduler.step()
        
        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("Saved best model.")
            
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(args.checkpoint_dir, "latest_checkpoint.pth"))

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AgriQPro Model")
    parser.add_argument("--train_dir", type=str, default="G:\\My Drive\\Betel-Leaf\\Datasets", help="Path to training dataset")
    parser.add_argument("--test_dir", type=str, default="G:\\My Drive\\Betel-Leaf\\Test_Dataset", help="Path to test dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--backbone", type=str, default="swinv2_tiny_window8_256", help="Backbone model name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    if os.path.exists(args.train_dir) and os.path.exists(args.test_dir):
        train_model(args)
    else:
        print(f"Dataset paths not found. \nTrain: {args.train_dir}\nTest: {args.test_dir}")
