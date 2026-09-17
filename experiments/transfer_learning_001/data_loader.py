import os
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
from pathlib import Path

# First only focus on images - classification task
class BUSBRAData(Dataset):
    def __init__(self, df, image_dir, transform=None):
        # Reset index to ensure 0-based sequential indexing for self.df.iloc[idx]
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Map classes to numeric targets
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self.idx_to_class = {0: 'benign', 1: 'malignant'}

        print(f"Loaded {len(self.df)} samples from DataFrame")
        print(f"Class distribution:\n{self.df['Pathology'].value_counts()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct image path
        image_name = f"{row['ID']}.png"
        image_path = self.image_dir / image_name

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Map label string to integer
        label = self.class_to_idx[row['Pathology']]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_busbra_fold_loaders(
    data_dir, 
    fold_num=1, 
    cv_filename='5-fold-cv.csv', 
    batch_size=32, 
    transform=None,
    num_workers=2
):
    # Define Paths
    data_csv_path = os.path.join(data_dir, 'bus_data.csv')
    cv_csv_path = os.path.join(data_dir, cv_filename)
    image_dir = os.path.join(data_dir, 'Images')
    
    df_data = pd.read_csv(data_csv_path)
    df_cv = pd.read_csv(cv_csv_path)
    
    df_merged = pd.merge(df_data, df_cv, on=['ID', 'Pathology'])
    
    # Extract indices for specified fold (1.0 = Train, 0.0 = Val)
    fold_col = f'valid_{fold_num}'
    train_indices = df_merged[df_merged[fold_col] == 1.0].index.tolist()
    val_indices = df_merged[df_merged[fold_col] == 0.0].index.tolist()
    
    # Instantiate base dataset passing merged DataFrame
    dataset = BUSBRAData(
        df=df_merged,
        image_dir=image_dir,
        transform=transform
    )
    
    # Create PyTorch Subsets & DataLoaders
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"  Train: {len(train_indices)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_loader)} batches)")
    
    return train_loader, val_loader