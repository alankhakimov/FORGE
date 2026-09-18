**Preface:**

*This experiment (000) is simply a learning exercise to understand a CNN and how to use PyTorch. Using the BUSBRA dataset we will construct a simple model and evaluate it, deriving why more complex architectures are necessary.*

**Methods:**

Use a simple 3 layer CNN with pooling, early dropout in training and early stopping.

**Outcomes:**

*Early stopping triggered at epoch #18 with the best model at epoch #15 (lowest validation loss)\
train\_loss=0.5000 train\_acc=0.7750 | val\_loss=0.5483 val\_acc=0.7425*

*Best model achieved testing accuracy of 0.74 and a "Malignant: recall of 0.71 meaning it accurately predicts malignancy 71% of the time (very meaningfull given the class imbalance). As a result of the class-weighted loss we have a false positive rate of 42%.*
