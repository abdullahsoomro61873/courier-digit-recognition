
import copy
import random
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import SmallModel, BigModel, RegularizedBigModel
from src.training_config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)


# --------------------------------------------------
# 1. Set random seeds
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# --------------------------------------------------
# 2. Select device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 3. Load prepared datasets
# --------------------------------------------------

data_dir = Path("data")

X_train = np.load(data_dir / "X_train.npy")
y_train = np.load(data_dir / "y_train.npy")

X_val = np.load(data_dir / "X_val.npy")
y_val = np.load(data_dir / "y_val.npy")


# Convert NumPy arrays into PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)


# --------------------------------------------------
# 4. Create DataLoaders
# --------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 5. Define loss function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 6. Training and validation function
# --------------------------------------------------

def run_epoch(model, loader, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# --------------------------------------------------
# 7. Train one model
# --------------------------------------------------

def train_model(
    model,
    model_name,
    weight_decay,
    use_early_stopping=False
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer
        )

        val_loss = run_epoch(
            model,
            val_loader
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"{model_name} | "
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        # Save best model and apply early stopping
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save checkpoint only for Model 3
            if use_early_stopping:

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": best_epoch,
                        "best_val_loss": best_val_loss
                    },
                    models_dir / "regularized_big_model.pth"
                )

        else:

            if use_early_stopping:

                patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:

                    print(
                        f"Early stopping at epoch {epoch}"
                    )
                    break

    # Restore best weights for Model 3
    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"{model_name} completed. "
        f"Best validation loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch}"
    )

    return history


# --------------------------------------------------
# 8. Train all three models
# --------------------------------------------------

models = [
    {
        "name": "Small Model",
        "model": SmallModel(),
        "weight_decay": 0.0,
        "early_stopping": False
    },
    {
        "name": "Big Model",
        "model": BigModel(),
        "weight_decay": 0.0,
        "early_stopping": False
    },
    {
        "name": "Regularized Big Model",
        "model": RegularizedBigModel(),
        "weight_decay": WEIGHT_DECAY,
        "early_stopping": True
    }
]

histories = {}

for item in models:

    set_seed(SEED)

    histories[item["name"]] = train_model(
        model=item["model"],
        model_name=item["name"],
        weight_decay=item["weight_decay"],
        use_early_stopping=item["early_stopping"]
    )



# --------------------------------------------------
# 8.5 Save training history to CSV
# --------------------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "training_history.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Epoch",
        "Training Loss",
        "Validation Loss"
    ])

    for model_name, history in histories.items():

        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_loss"], history["val_loss"]),
            start=1
        ):

            writer.writerow([
                model_name,
                epoch,
                train_loss,
                val_loss
            ])

print("Training history saved to", csv_path)

# --------------------------------------------------
# 9. Plot training and validation loss curves
# --------------------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 7))

for model_name, history in histories.items():

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(
        epochs,
        history["train_loss"],
        label=f"{model_name} - Training"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        linestyle="--",
        label=f"{model_name} - Validation"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(results_dir / "curves.png")

plt.close()

print("\nAll models training completed!")
print("Loss curves saved to results/curves.png")
