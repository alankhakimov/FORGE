import os
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
from pathlib import Path

# First only focus on images - classification task
class BUSBRAData(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self.idx_to_class = {0: 'benign', 1: 'malignant'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['ID']}.png"
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row['Pathology']]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_busbra_fold_loaders(
    data_dir,
    fold_num=1,
    cv_filename='5-fold-cv.csv',
    batch_size=32,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    num_workers=2,
):
    data_csv_path = os.path.join(data_dir, 'bus_data.csv')
    cv_csv_path = os.path.join(data_dir, cv_filename)
    image_dir = os.path.join(data_dir, 'Images')

    df_data = pd.read_csv(data_csv_path)
    df_cv = pd.read_csv(cv_csv_path)
    df_merged = pd.merge(df_data, df_cv, on=['ID', 'Pathology'])

    fold_col = f'valid_{fold_num}'

    # Test set: this round's held-out fold (NaN in valid_col identifies these)
    test_indices = df_merged[df_merged['kFold'] == fold_num].index.tolist()
    # Train/val: split among the remaining rows
    train_indices = df_merged[df_merged[fold_col] == 1.0].index.tolist()
    val_indices = df_merged[df_merged[fold_col] == 0.0].index.tolist()

    # Sanity check: the three sets should be mutually exclusive and cover everything
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(df_merged), \
        "Index sets don't add up to full dataset — check for overlap or gaps"

    train_dataset = BUSBRAData(df_merged, image_dir, transform=train_transform)
    val_dataset = BUSBRAData(df_merged, image_dir, transform=val_transform)
    test_dataset = BUSBRAData(df_merged, image_dir, transform=test_transform)

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for name, idx in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        print(f"{name}: {len(idx)} samples")
        print(f"  Class balance: {df_merged.loc[idx, 'Pathology'].value_counts().to_dict()}")

    return train_loader, val_loader, test_loader