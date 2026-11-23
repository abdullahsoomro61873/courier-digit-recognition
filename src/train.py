
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
# 1. Set random seeds for reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# --------------------------------------------------
# 2. Select training device (GPU or CPU)
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 3. Load prepared training and validation datasets
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

    # If optimizer is provided, perform training.
    # Otherwise, perform validation.

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

        # Enable gradients only during training

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
# 7. Train one model and save its best checkpoint
# --------------------------------------------------

def train_model(
    model,
    model_name,
    weight_decay,
    use_early_stopping=False
):

    # Move model to selected device

    model = model.to(device)

    # AdamW optimizer

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    # Store training and validation losses

    history = {
        "train_loss": [],
        "val_loss": []
    }

    # Track the best validation performance

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    # Early stopping counter

    patience_counter = 0

    # Create models directory if it does not exist

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Checkpoint filenames for all three models

    checkpoint_names = {
        "Small Model": "small_model.pth",
        "Big Model": "big_model.pth",
        "Regularized Big Model": "regularized_big_model.pth"
    }

    # Training loop

    for epoch in range(1, MAX_EPOCHS + 1):

        # Calculate training loss

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer
        )

        # Calculate validation loss

        val_loss = run_epoch(
            model,
            val_loader
        )

        # Store losses for this epoch

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Display progress

        print(
            f"{model_name} | "
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        # Check whether validation loss improved

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            # Copy the best model weights

            best_state = copy.deepcopy(model.state_dict())

            # Reset early stopping counter

            patience_counter = 0

            # Save best checkpoint for this model

            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss
                },
                models_dir / checkpoint_names[model_name]
            )

        else:

            # Apply early stopping only to Regularized Big Model

            if use_early_stopping:

                patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:

                    print(
                        f"Early stopping at epoch {epoch}"
                    )

                    break

    # Restore the best weights for Regularized Big Model

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    # Display final training information

    print(
        f"{model_name} completed. "
        f"Best validation loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch}"
    )

    return history


# --------------------------------------------------
# 8. Define all three models
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


# --------------------------------------------------
# 9. Train all three models
# --------------------------------------------------

histories = {}

for item in models:

    # Reset seed before training each model

    set_seed(SEED)

    histories[item["name"]] = train_model(
        model=item["model"],
        model_name=item["name"],
        weight_decay=item["weight_decay"],
        use_early_stopping=item["early_stopping"]
    )


# --------------------------------------------------
# 10. Save training history to CSV
# --------------------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "training_history.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    # CSV column headings

    writer.writerow([
        "Model",
        "Epoch",
        "Training Loss",
        "Validation Loss"
    ])

    # Save each model's epoch-wise losses

    for model_name, history in histories.items():

        for epoch, (train_loss, val_loss) in enumerate(
            zip(
                history["train_loss"],
                history["val_loss"]
            ),
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
# 11. Plot training and validation loss curves
# --------------------------------------------------

plt.figure(figsize=(12, 7))

for model_name, history in histories.items():

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Plot training loss using solid line

    plt.plot(
        epochs,
        history["train_loss"],
        label=f"{model_name} - Training"
    )

    # Plot validation loss using dashed line

    plt.plot(
        epochs,
        history["val_loss"],
        linestyle="--",
        label=f"{model_name} - Validation"
    )


# Add graph labels and title

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss Curves"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

# Save final graph

plt.savefig(
    results_dir / "curves.png"
)

plt.close()


# --------------------------------------------------
# 12. Training completion message
# --------------------------------------------------

print("\nAll models training completed!")

print("Loss curves saved to results/curves.png")

print("All three best model checkpoints saved in models/")
