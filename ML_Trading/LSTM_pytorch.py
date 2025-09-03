import sys
import subprocess

# Import required libraries
import pandas as pd
import numpy as np
import os as os
# Import yahoo finance libraries
import yfinance as yf

# Import torch libraries
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Technical Indicators
from utils.technical_indicators import SMA, Momentum

from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, 2)  # 2 classes: up or down
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state (represents sequence info)
        last_hidden = h_n[-1]  # shape: [batch, hidden_size]
        
        out = self.dropout(self.relu(last_hidden))
        out = self.fc(out)
        return out
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

import torch.optim as optim

# Evaluation

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def build_features(OHLCV):
    Features = pd.DataFrame(data=None,index=OHLCV.index)

    Features['Returns'] = OHLCV['Close'].pct_change()
    Features['ReturnLag1'] = OHLCV['Close'].pct_change().shift(periods=1)
    Features['ReturnLag5'] = OHLCV['Close'].pct_change().shift(periods=5)
    Features['ReturnLag10'] = OHLCV['Close'].pct_change().shift(periods=10)

    Features['SMA_5Day'] = SMA(OHLCV['Close'].values.ravel(),5).values
    Features['SMA_10Day'] = SMA(OHLCV['Close'].values.ravel(),10).values
    Features['SMA_30Day'] = SMA(OHLCV['Close'].values.ravel(),30).values

    Features['Momentum3day'] = Momentum(OHLCV['Close'].values.ravel(),3).values
    Features['Momentum5day'] = Momentum(OHLCV['Close'].values.ravel(),5).values
    Features['Momentum10day'] = Momentum(OHLCV['Close'].values.ravel(),10).values

    Features['VWaP_5Day'] = SMA(OHLCV['Volume'].values.ravel(),5).values
    Features['VWaP_10Day'] = SMA(OHLCV['Volume'].values.ravel(),10).values

    Features.loc[Features['Returns'].between(-.004, .004) ,'Returns'] =0
    Features.loc[Features['Returns']>0 ,'MoveDirection'] = 1
    Features.loc[Features['Returns']<0 ,'MoveDirection'] = 0

    Features = Features.dropna()

    return Features


def main():
    # Load data
    OHLCV = yf.download('BTC-USD', start='2016-11-01', end='2022-11-01', progress=False)
    OHLCV = OHLCV[['Open','High','Low','Close','Volume']]

    # Build features
    Features = build_features(OHLCV)

    # Drop the target column before creating X
    feature_cols = [c for c in Features.columns if c != "MoveDirection"]

    X = Features[feature_cols].values
    y = Features["MoveDirection"].values

    # Parameters
    window_size = 30  # lookback window
    forecast_horizon = 1  # predict 1-day ahead

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size - forecast_horizon + 1):
        X_seq.append(X[i : i + window_size])  # 30-day window of features
        y_seq.append(y[i + window_size + forecast_horizon - 1])  # label is next day's MoveDirection

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print("X_seq shape:", X_seq.shape)  # (num_samples, window_size, num_features)
    print("y_seq shape:", y_seq.shape)  # (num_samples,)

    # First split train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    # Then split train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, shuffle=False
    )

    # Convert to torch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val     = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test   = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Hyperparameters
    input_size = X_train.shape[2]   # num_features
    hidden_size = 64
    num_layers = 2
    num_epochs = 20
    learning_rate = 0.001

    # Model, Loss, Optimizer
    model = LSTMClassifier(input_size, hidden_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        train_acc = 100 * correct / total
    
        # ---- Validation ----
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # ---- Early Stopping ----
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Run evaluation
    evaluate(model, test_loader)        
if __name__ == "__main__":
    main()