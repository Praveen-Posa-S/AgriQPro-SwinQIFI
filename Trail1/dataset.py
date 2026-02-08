import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image

class BetelVineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (assumed ImageFolder structure).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Use ImageFolder logic to find classes and images
        self.data = datasets.ImageFolder(root_dir)
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(is_train=True):
    """
    Returns transforms for training or validation/testing.
    Training: Resize(256), RandomCrop(224), AutoAugment, Normalize.
    Validation: Resize(256), CenterCrop(224), Normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(train_dir, test_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Creates and returns train, val, and test dataloaders.
    Splits train_dir into Train and Val.
    Uses test_dir for Test.
    """
    # Define transforms
    train_transform = get_transforms(is_train=True)
    val_test_transform = get_transforms(is_train=False)

    # 1. Handle Training and Validation from train_dir
    full_train_dataset = datasets.ImageFolder(root=train_dir)
    class_names = full_train_dataset.classes
    
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * val_split)
    train_size = total_train_size - val_size
    
    # Split train_dir into Train and Val
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # 2. Handle Test from test_dir
    # Note: We assume test_dir has the same class structure.
    test_dataset_raw = datasets.ImageFolder(root=test_dir)

    # Apply transforms using the module-level Dataset class
    train_data = TransformedSubset(train_subset, transform=train_transform)
    val_data = TransformedSubset(val_subset, transform=val_test_transform)
    # Wrap test dataset as well to ensure consistent transform application
    test_data = TransformedSubset(test_dataset_raw, transform=val_test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test the dataset implementation
    TRAIN_PATH = "G:\\My Drive\\Betel-Leaf\\Datasets"
    TEST_PATH = "G:\\My Drive\\Betel-Leaf\\Test_Dataset"
    
    try:
        if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
            train_loader, val_loader, test_loader, classes = get_dataloaders(TRAIN_PATH, TEST_PATH, batch_size=4)
            print(f"Classes: {classes}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            img, label = next(iter(train_loader))
            print(f"Batch shape: {img.shape}, Labels: {label}")
        else:
            print(f"Paths not found. Skipping dataset test.")
    except Exception as e:
        print(f"Error testing dataset: {e}")
