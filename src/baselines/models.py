"""Two classifiers not directly available as sklearn/xgboost/lightgbm off-the-shelf
estimators, reproduced to match the companion repo's implementations
(external/Code-for-ICS-NAD-Dataset/Train_All_in_One_czh.ipynb)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ANNModel(nn.Module):
    """2-hidden-layer MLP, matches the paper's ANN cell exactly (64 units, ReLU, sigmoid out)."""

    def __init__(self, input_dim, hidden_units=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return torch.sigmoid(self.output(x))


class ANNClassifier:
    def __init__(self, input_dim, hidden_units=64, learning_rate=1e-3, epochs=15, batch_size=256):
        self.model = ANNModel(input_dim, hidden_units)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        Xt, yt = torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(Xt.size(0))
            for i in range(0, Xt.size(0), self.batch_size):
                idx = perm[i:i + self.batch_size]
                self.optimizer.zero_grad()
                out = self.model(Xt[idx])
                loss = self.criterion(out, yt[idx])
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            p = self.model(torch.FloatTensor(X)).squeeze(1).numpy()
        return np.column_stack([1.0 - p, p])


class ELMClassifier:
    """Extreme Learning Machine: fixed random input->hidden weights (sigmoid activation),
    output weights solved in closed form via Moore-Penrose pseudo-inverse. Reproduces the
    algorithm the paper's `hpelm`-based cell implements, without that C-extension dependency."""

    def __init__(self, hidden_units=200, seed=42):
        self.hidden_units = hidden_units
        self.rng = np.random.RandomState(seed)
        self.W = None
        self.b = None
        self.beta = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n_features = X.shape[1]
        self.W = self.rng.normal(size=(n_features, self.hidden_units)).astype(np.float32)
        self.b = self.rng.normal(size=(self.hidden_units,)).astype(np.float32)
        H = self._sigmoid(X @ self.W + self.b)
        # one-hot target for binary classification (2 output units, as in the reference impl)
        Y = np.zeros((len(y), 2), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        self.beta = np.linalg.pinv(H) @ Y
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        scores = H @ self.beta
        # raw pseudo-inverse outputs aren't probabilities (can be negative /
        # not sum to 1) - softmax normalize row-wise so they're usable for
        # threshold tuning the same way as every other classifier's proba.
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)
