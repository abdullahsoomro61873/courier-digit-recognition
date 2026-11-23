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

### Phase 0 — Project Setup
- Project directory created.
- Python virtual environment configured.
- Required libraries installed.
- GitHub repository initialized and pushed.

### Phase 1 — Data Preparation
- Loaded the Scikit-learn digits dataset.
- Scaled pixel values to the range 0–1.
- Split data into training, validation, and testing sets.
- Saved processed datasets as NumPy `.npy` files.

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
