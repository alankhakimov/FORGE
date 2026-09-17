from data_loader import BUSBRAData, get_busbra_fold_loaders
from torchvision import transforms
import torch
import torch.nn as nn

def train_model():
    #TO-DO
    pass

def main():
    data_dir = r"C:\Documents\FORGE\BUSBRA\BUSBRA"
    standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
    train_loader, val_loader = get_busbra_fold_loaders(
        data_dir=data_dir,
        fold_num=1,
        cv_filename='5-fold-cv.csv',
        batch_size=32,
        transform=standard_transform,
        num_workers=2
    )



if __name__ == "__main__":
    main()