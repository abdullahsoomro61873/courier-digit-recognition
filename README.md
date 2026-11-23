# Courier Digit Recognition

A deep learning project to recognize handwritten digits for a courier company's pincode-reading task.

## Project Overview

The goal is to train a neural network that can recognize handwritten digits using a limited amount of labelled data.

**Project challenge:**

We have only 200 labelled training images. We need to train models and use training and validation loss curves to determine whether they are learning or memorizing.

## Dataset

This project uses the `load_digits` dataset from Scikit-learn.

- Total images: 1,797
- Image size: 8 × 8 pixels
- Input features: 64 pixels per image
- Pixel values scaled from 0 to 1

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 200 |
| Validation | 798 |
| Testing | 799 |
| **Total** | **1,797** |

A fixed random seed (`42`) is used for reproducibility.

## Project Progress

### Phase 1 — Project Setup
- Project directory created.
- Python virtual environment configured.
- Required libraries installed.
- GitHub repository initialized and pushed.

### Phase 2 — Data Preparation
- Loaded the Scikit-learn digits dataset.
- Scaled pixel values to the range 0–1.
- Split data into training, validation, and testing sets.
- Saved processed datasets as NumPy `.npy` files.

### Phase 3 - Model Development
- Implemented Small Model with one linear layer.
- Implemented Big Model with two hidden layers (256 neurons each).
- Implemented Regularized Big Model with Dropout (0.3).
- Configured weight decay (0.05).
- Configured early stopping (patience: 30).
- Successfully tested all three models and configurations with output shape (4, 10).

### Phase 4 - Training
- Developed one common training loop for all three models.
- Trained all three models for up to 300 epochs.
- Recorded training and validation loss after every epoch.
- Saved training history in `results/training_history.csv`.
- Saved the best checkpoint of the third model in `models/regularized_big_model.pth`.
- Implemented early stopping with patience of 30 epochs.
- Generated loss curves in `results/curves.png`.

### Phase 5 — Learning Curves

- Plotted Small Model training and validation loss curves.
- Plotted Big Model training and validation loss curves.
- Plotted Regularized Big Model training and validation loss curves.
- Compared training and validation loss for all three models.
- Displayed all three experiments side by side in one image.
- Saved the final visualization as `results/curves.png`.

**Results**
- Small Model: Underfitting observed.
- Big Model: Overfitting observed.
- Regularized Big Model: Training stopped at epoch 80; best validation loss was 0.2473 at epoch 50.
- All three experiments successfully visualized in one image.

#### Training Results
| Model | Epochs |
|---|---:|
| Small Model | 300 |
| Big Model | 300 |
| Regularized Big Model | 80 |
| Total Records | 680 |

- Best validation loss: 0.2473
- Best epoch: 50
- Early stopping: Epoch 80

### Upcoming Work
- Build three neural network models.
- Train models for up to 300 epochs.
- Record training and validation loss after every epoch.
- Apply dropout, weight decay, and early stopping.
- Plot loss curves.
- Evaluate test accuracy.
- Select the best model for shipping.

## Project Structure

```text
courier-digit-recognition/
│
├── data/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── models/
├── results/
├── src/
│   └── prepare_data.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd courier-digit-recognition
```

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Data Preparation

```bash
python src/prepare_data.py
```

## Technologies

- Python
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib

## Status

**Currently completed:** Project setup and data preparation.

**Next milestone:** Neural network model development and training.
