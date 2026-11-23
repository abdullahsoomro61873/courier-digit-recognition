
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models import SmallModel, BigModel, RegularizedBigModel
from src.training_config import BATCH_SIZE


# --------------------------------------------------
# 1. Select device (GPU or CPU)
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Load test dataset
# --------------------------------------------------

data_dir = Path("data")

X_test = np.load(data_dir / "X_test.npy")
y_test = np.load(data_dir / "y_test.npy")

# Convert NumPy arrays into PyTorch tensors

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create test DataLoader

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 3. Define model checkpoints
# --------------------------------------------------

models = [
    {
        "name": "Small Model",
        "model": SmallModel(),
        "checkpoint": "models/small_model.pth"
    },
    {
        "name": "Big Model",
        "model": BigModel(),
        "checkpoint": "models/big_model.pth"
    },
    {
        "name": "Regularized Big Model",
        "model": RegularizedBigModel(),
        "checkpoint": "models/regularized_big_model.pth"
    }
]


# --------------------------------------------------
# 4. Evaluate one model on test dataset
# --------------------------------------------------

def evaluate_model(model, test_loader):

    model.eval()

    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation during evaluation

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Generate model predictions

            outputs = model(inputs)

            # Select class with highest score

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Count correct predictions

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    # Calculate test accuracy

    accuracy = (
        correct_predictions / total_samples
    ) * 100

    return accuracy


# --------------------------------------------------
# 5. Load best checkpoints and evaluate all models
# --------------------------------------------------

evaluation_results = []

for item in models:

    print("\nEvaluating:", item["name"])

    model = item["model"].to(device)

    checkpoint_path = Path(item["checkpoint"])

    # Check whether checkpoint exists

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load saved best checkpoint

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Calculate test accuracy

    test_accuracy = evaluate_model(
        model,
        test_loader
    )

    print(
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    # Store evaluation results

    evaluation_results.append({
        "Model": item["name"],
        "Test Accuracy": test_accuracy,
        "Best Epoch": checkpoint["epoch"],
        "Best Validation Loss": checkpoint["best_val_loss"]
    })


# --------------------------------------------------
# 6. Load training history from CSV
# --------------------------------------------------

history_path = Path(
    "results/training_history.csv"
)

training_history = {}

with open(history_path, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        model_name = row["Model"]

        training_history.setdefault(
            model_name,
            []
        )

        training_history[model_name].append(row)


# --------------------------------------------------
# 7. Build final comparison table
# --------------------------------------------------

final_results = []

for result in evaluation_results:

    model_name = result["Model"]

    # Get final epoch's training loss

    last_record = training_history[model_name][-1]

    final_training_loss = float(
        last_record["Training Loss"]
    )

    final_results.append({
        "Model": model_name,
        "Final Training Loss": final_training_loss,
        "Best Validation Loss": float(
            result["Best Validation Loss"]
        ),
        "Best Epoch": result["Best Epoch"],
        "Test Accuracy (%)": result["Test Accuracy"]
    })


# --------------------------------------------------
# 8. Display final comparison table
# --------------------------------------------------

print("\n==============================================")
print("       FINAL MODEL EVALUATION RESULTS")
print("==============================================")

print(
    f"{'Model':<25}"
    f"{'Train Loss':<15}"
    f"{'Val Loss':<15}"
    f"{'Best Epoch':<12}"
    f"{'Test Accuracy':<15}"
)

print("-" * 82)

for result in final_results:

    print(
        f"{result['Model']:<25}"
        f"{result['Final Training Loss']:<15.6f}"
        f"{result['Best Validation Loss']:<15.6f}"
        f"{result['Best Epoch']:<12}"
        f"{result['Test Accuracy (%)']:.2f}%"
    )


# --------------------------------------------------
# 9. Save final results to CSV
# --------------------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

output_path = results_dir / "evaluation_results.csv"

with open(output_path, "w", newline="") as file:

    fieldnames = [
        "Model",
        "Final Training Loss",
        "Best Validation Loss",
        "Best Epoch",
        "Test Accuracy (%)"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(final_results)


# --------------------------------------------------
# 10. Evaluation completed
# --------------------------------------------------

print("\nEvaluation completed successfully!")

print(
    "Comparison table saved to:",
    output_path
)
