from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

# Fixed random seed
SEED = 42

# Load the digits dataset
digits = load_digits()

X = digits.data.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

# Split: 200 training images, 1597 remaining
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    train_size=200,
    random_state=SEED
)

# Split remaining data into 798 validation and 799 test images
X_val, X_test, y_val, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.5,
    random_state=SEED
)

# Create data folder if it does not exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Save all datasets
np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "y_train.npy", y_train)

np.save(data_dir / "X_val.npy", X_val)
np.save(data_dir / "y_val.npy", y_val)

np.save(data_dir / "X_test.npy", X_test)
np.save(data_dir / "y_test.npy", y_test)

# Display dataset information
print("Data preparation complete!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Pixel range:", X.min(), "to", X.max())
