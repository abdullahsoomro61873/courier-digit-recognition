
import torch.nn as nn


# Model 1: Small Model (Single linear layer)
# This model is used for Underfitting experiment which has One linear layer.
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


# Model 2: Big Model (2 hidden layers, 256 neurons each)
# This model is used for Overfitting experiment which have Two hidden layers, and Each hidden layer: 256 neurons. 
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


# Model 3: Big Model + Dropout (Big + Dropout (0.3) + Weight Decay (0.05))
# This model is used for Final candidate which have Two hidden layers of 256 neurons. Dropout: 0.3. Weight decay: 0.05. Early stopping: patience 30.
class RegularizedBigModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)
