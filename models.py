#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


# -------------------------------
# PyTorch Neural Network Classifier
# -------------------------------
class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_dim=32, output_dim=3,
                 epochs=10, batch_size=128, lr=0.001, dropout_rate=0.2,
                 verbose=True, random_state=None):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.classes_ = None
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _build_model(self):
        class CancerNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
                super(CancerNN, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim // 2, output_dim)
                )

            def forward(self, x):
                return self.net(x)

        return CancerNN(self.input_dim, self.hidden_dim, self.output_dim, self.dropout_rate)

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.classes_ = np.unique(y)
        self.output_dim = len(self.classes_)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, y
            X_val = scaler.transform(X_val)

        # Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                                batch_size=self.batch_size)

        self.model = self._build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                epoch_loss += loss.item()

            self.train_losses.append(epoch_loss / len(train_loader))
            self.train_accs.append(correct / total)

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == y_batch).sum().item()
                    val_total += y_batch.size(0)

            self.val_losses.append(val_loss / len(val_loader))
            self.val_accs.append(val_correct / val_total)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {self.train_losses[-1]:.4f} - "
                      f"Val Loss: {self.val_losses[-1]:.4f} - "
                      f"Train Acc: {self.train_accs[-1]:.4f} - "
                      f"Val Acc: {self.val_accs[-1]:.4f}")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.numpy()
