import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Block 1: learn low-level patterns (edges, blobs) — 3 input channels -> 16 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 224 -> 112

        # Block 2: combine low-level patterns into slightly more complex shapes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)                       # 112 -> 56

        # Block 3: higher-level, more abstract features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)                       # 56 -> 28

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(p=0.5)   # new
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))   
        x = self.pool2(self.relu(self.conv2(x)))   
        x = self.pool3(self.relu(self.conv3(x)))   
        x = self.flatten(x)                     
        x = self.relu(self.fc1(x))
        x = self.dropout(x)                # zeroes ~50% of these 128 values, randomly, each batch                 
        x = self.fc2(x)                             # (B, 2) — raw logits, no softmax, applied in loss function
        return x