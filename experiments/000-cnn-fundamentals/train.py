import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from model import SimpleCNN
from experiments.BUSBRA_data_loader import get_busbra_fold_loaders
from sklearn.metrics import classification_report

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # enables dropout/batchnorm training behavior (matters more once you add them)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)          # raw logits, shape (batch, 2)
        loss = criterion(outputs, labels)
        loss.backward()                  # backprop
        optimizer.step()                 # update weights

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if return_preds:
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if return_preds:
        return total_loss / total, correct / total, all_preds, all_labels
    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")  # CPU-only, per your setup

    # No pretrained normalization needed — this model has no pretrained weights to match
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = get_busbra_fold_loaders(
        data_dir=r"C:\Documents\FORGE\BUSBRA\BUSBRA",
        fold_num=1,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        num_workers=2,
    )

    model = SimpleCNN(num_classes=2).to(device)

    # Class-weighted loss, given the ~2:1 benign:malignant imbalance confirmed earlier
    class_weights = torch.tensor([1 / 812, 1 / 388])
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    best_model_path = "experiments/000-cnn-fundamentals/best_model.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_to_idx": {"benign": 0, "malignant": 1},
            }, best_model_path)   # save only on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    # Reload the BEST checkpoint, not whatever the loop ended on, before final test
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device, return_preds=True)
    print(f"Final test (best checkpoint): loss={test_loss:.4f} acc={test_acc:.4f}")
    print(classification_report(labels, preds, target_names=["benign", "malignant"]))


if __name__ == "__main__":
    main()