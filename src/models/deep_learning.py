"""
Deep Learning Forecasters:
PyTorch implementations of Two-Layer LSTM, GRU, CNN-LSTM, and TCN (Temporal Convolutional Network).
Trained with Huber Loss, AdamW optimizer, and reproducible seeding.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    torch.set_num_threads(1)
except Exception:
    pass

from src.models.base import BaseForecaster


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_sequence_tensor(X: np.ndarray, seq_len: int = 20) -> torch.Tensor:
    """
    Transforms 2D tabular features (N, D) into 3D sequential tensors (N, seq_len, D)
    using vectorized sliding window lookbacks with left zero-padding.
    """
    if X.ndim == 3:
        return torch.tensor(X, dtype=torch.float32)
        
    N, D = X.shape
    pad = np.zeros((seq_len - 1, D), dtype=np.float32)
    X_padded = np.vstack([pad, X])
    
    # Vectorized strided sequence extraction
    shape = (N, seq_len, D)
    strides = (X_padded.strides[0], X_padded.strides[0], X_padded.strides[1])
    sequences = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    return torch.tensor(sequences.copy(), dtype=torch.float32)


# =====================================================================
# PYTORCH NEURAL NETWORK MODULES
# =====================================================================

class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, input_dim: int, conv_filters: int = 32, lstm_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        # 1D Conv expects (batch, channels, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, seq_len, D) -> transpose to (N, D, seq_len)
        x_trans = x.transpose(1, 2)
        conv_out = self.conv(x_trans)  # (N, conv_filters, seq_len)
        conv_trans = conv_out.transpose(1, 2)  # (N, seq_len, conv_filters)
        lstm_out, _ = self.lstm(conv_trans)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class ChausalConv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res_match = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        x_padded = nn.functional.pad(x, (self.pad, 0))
        out = self.conv(x_padded)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.res_match is None else self.res_match(x)
        return out + res


class TCNNet(nn.Module):
    def __init__(self, input_dim: int, num_channels: Tuple[int, ...] = (32, 32, 32), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(ChausalConv1dBlock(in_c, out_c, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_c = out_c
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, seq_len, D) -> (N, D, seq_len)
        x_trans = x.transpose(1, 2)
        out = self.network(x_trans)  # (N, C_last, seq_len)
        last_step = out[:, :, -1]    # causal last time step
        return self.fc(last_step).squeeze(-1)


# =====================================================================
# PYTORCH FORECASTER WRAPPERS
# =====================================================================

class _BasePyTorchForecaster(BaseForecaster):
    def __init__(self, name: str, seq_len: int = 20, epochs: int = 20, batch_size: int = 64, lr: float = 0.003, seed: int = 42, device: str = "cpu"):
        super().__init__(name=name)
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.net: Optional[nn.Module] = None
        
    def _train_net(self, X: np.ndarray, y: np.ndarray) -> None:
        set_seed(self.seed)
        X_seq = to_sequence_tensor(X, self.seq_len)
        y_ten = torch.tensor(y, dtype=torch.float32)
        
        dataset = TensorDataset(X_seq, y_ten)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.net.to(self.device)
        self.net.train()
        
        criterion = nn.SmoothL1Loss(beta=0.005)  # Huber loss for financial log returns
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                
        self.is_fitted = True
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"{self.name} is not fitted.")
            
        self.net.eval()
        X_seq = to_sequence_tensor(X, self.seq_len)
        dataset = TensorDataset(X_seq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                batch_pred = self.net(batch_x).cpu().numpy()
                preds.append(batch_pred)
                
        return np.concatenate(preds).astype(np.float32)


class LSTMForecaster(_BasePyTorchForecaster):
    def __init__(self, seq_len: int = 20, hidden_dim: int = 32, num_layers: int = 2, epochs: int = 35, lr: float = 0.002, seed: int = 42):
        super().__init__(name="Two-Layer LSTM", seq_len=seq_len, epochs=epochs, lr=lr, seed=seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LSTMForecaster":
        input_dim = X.shape[-1]
        self.net = LSTMNet(input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self._train_net(X, y)
        return self


class GRUForecaster(_BasePyTorchForecaster):
    def __init__(self, seq_len: int = 20, hidden_dim: int = 32, num_layers: int = 2, epochs: int = 35, lr: float = 0.002, seed: int = 42):
        super().__init__(name="Gated Recurrent Unit (GRU)", seq_len=seq_len, epochs=epochs, lr=lr, seed=seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GRUForecaster":
        input_dim = X.shape[-1]
        self.net = GRUNet(input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self._train_net(X, y)
        return self


class CNNLSTMForecaster(_BasePyTorchForecaster):
    def __init__(self, seq_len: int = 20, conv_filters: int = 32, lstm_hidden: int = 32, epochs: int = 35, lr: float = 0.002, seed: int = 42):
        super().__init__(name="CNN-LSTM Hybrid", seq_len=seq_len, epochs=epochs, lr=lr, seed=seed)
        self.conv_filters = conv_filters
        self.lstm_hidden = lstm_hidden
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CNNLSTMForecaster":
        input_dim = X.shape[-1]
        self.net = CNNLSTMNet(input_dim=input_dim, conv_filters=self.conv_filters, lstm_hidden=self.lstm_hidden)
        self._train_net(X, y)
        return self


class TCNForecaster(_BasePyTorchForecaster):
    def __init__(self, seq_len: int = 20, num_channels: Tuple[int, ...] = (32, 32, 32), epochs: int = 35, lr: float = 0.002, seed: int = 42):
        super().__init__(name="Temporal ConvNet (TCN)", seq_len=seq_len, epochs=epochs, lr=lr, seed=seed)
        self.num_channels = num_channels
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TCNForecaster":
        input_dim = X.shape[-1]
        self.net = TCNNet(input_dim=input_dim, num_channels=self.num_channels)
        self._train_net(X, y)
        return self
