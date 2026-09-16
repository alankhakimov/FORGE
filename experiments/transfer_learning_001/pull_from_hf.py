from transformers import AutoImageProcessor, ResNetForImageClassification


def pull_pretrained_model():
    CHECKPOINT = "microsoft/resnet-18"

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = ResNetForImageClassification.from_pretrained(CHECKPOINT)

    print("Processor loaded:", type(processor).__name__)
    print("Model loaded:", type(model).__name__)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    SAVE_DIR = "experiments/transfer_learning_001/pretrained_resnet18"

    processor.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print(f"Saved processor and model to {SAVE_DIR}")

if __name__ == "__main__":
    pull_pretrained_model()