from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import ForecastSummary

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ---------------------------------------------------------------------------
# PatchTST — State-of-the-Art Transformer for Time Series
# Optimized for RTX 3050 (4 GB VRAM) with FP16 mixed precision
# ---------------------------------------------------------------------------

if nn is not None:

    class _PatchEmbedding(nn.Module):
        """Split a sequence into patches and project into d_model."""

        def __init__(self, patch_len: int, stride: int, d_model: int, seq_len: int, n_channels: int):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.n_patches = max((seq_len - patch_len) // stride + 1, 1)
            self.projection = nn.Linear(patch_len, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
            self.n_channels = n_channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_channels, seq_len)
            batch_size = x.shape[0]
            patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
            # patches: (batch, n_channels, n_patches, patch_len)
            patches = patches.reshape(batch_size * self.n_channels, self.n_patches, self.patch_len)
            embedded = self.projection(patches) + self.pos_encoding
            return embedded  # (batch * n_channels, n_patches, d_model)

    class _TransformerBlock(nn.Module):
        """Standard pre-norm Transformer encoder block."""

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x

    class _PatchTST(nn.Module):
        """Channel-independent PatchTST for time series forecasting.

        Architecture:
        - Patch embedding with learnable positional encoding
        - N Transformer encoder layers with pre-norm
        - Channel-independent (each feature treated separately)
        - Final head: flatten + linear -> 1 prediction
        """

        def __init__(self, seq_len: int, n_channels: int, patch_len: int, stride: int,
                     d_model: int, n_heads: int, n_layers: int, dropout: float):
            super().__init__()
            self.n_channels = n_channels
            self.patch_embed = _PatchEmbedding(patch_len, stride, d_model, seq_len, n_channels)
            self.encoder = nn.Sequential(*[
                _TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            n_patches = self.patch_embed.n_patches
            self.head = nn.Sequential(
                nn.Linear(n_channels * n_patches * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_channels, seq_len)
            batch_size = x.shape[0]
            embedded = self.patch_embed(x)
            encoded = self.encoder(embedded)
            encoded = self.norm(encoded)
            # Reshape: (batch*n_channels, n_patches, d_model) -> (batch, n_channels*n_patches*d_model)
            n_patches = encoded.shape[1]
            d_model = encoded.shape[2]
            encoded = encoded.reshape(batch_size, self.n_channels * n_patches * d_model)
            return self.head(encoded)

else:
    _PatchTST = None


class TorchForecaster:
    """PatchTST-based forecaster with FP16 mixed precision for 4GB GPU."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def forecast(self, close: pd.Series, features_df: pd.DataFrame | None = None) -> ForecastSummary:
        if not self.settings.enable_torch_model:
            return ForecastSummary(predicted_return=0.0, confidence=0.0, device="disabled",
                                  enabled=False, model_type="disabled")
        if torch is None or nn is None:
            return ForecastSummary(predicted_return=0.0, confidence=0.0, device="missing-torch",
                                  enabled=False, model_type="missing-torch")

        seq_len = self.settings.torch_sequence_length
        returns = close.pct_change().dropna()

        if len(returns) < seq_len + 60:
            return ForecastSummary(predicted_return=0.0, confidence=0.0, device="insufficient-data",
                                  enabled=False, model_type="insufficient-data")

        # Build multi-channel input: returns, volatility, momentum
        ret_vals = returns.to_numpy(dtype=np.float32)
        vol_vals = returns.rolling(20).std().bfill().to_numpy(dtype=np.float32)
        mom_vals = close.pct_change(20).fillna(0.0).iloc[1:].to_numpy(dtype=np.float32)

        # Truncate to same length
        min_len = min(len(ret_vals), len(vol_vals), len(mom_vals))
        ret_vals = ret_vals[-min_len:]
        vol_vals = vol_vals[-min_len:]
        mom_vals = mom_vals[-min_len:]

        # Normalize each channel
        channels = []
        channel_stats = []
        for vals in [ret_vals, vol_vals, mom_vals]:
            mean = float(vals.mean())
            std = float(vals.std()) or 1.0
            channels.append((vals - mean) / std)
            channel_stats.append((mean, std))

        # Create sliding windows
        n_channels = len(channels)
        x_data = []
        y_data = []
        for i in range(seq_len, min_len):
            window = np.stack([ch[i - seq_len:i] for ch in channels])  # (n_channels, seq_len)
            x_data.append(window)
            y_data.append(channels[0][i])  # predict normalized return

        if len(x_data) < 50:
            return ForecastSummary(predicted_return=0.0, confidence=0.0, device="insufficient-windows",
                                  enabled=False, model_type="insufficient-windows")

        x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.settings.torch_batch_size, len(dataset)), shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _PatchTST(
            seq_len=seq_len,
            n_channels=n_channels,
            patch_len=self.settings.patchtst_patch_len,
            stride=self.settings.patchtst_stride,
            d_model=self.settings.patchtst_d_model,
            n_heads=self.settings.patchtst_n_heads,
            n_layers=self.settings.patchtst_n_layers,
            dropout=self.settings.patchtst_dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.settings.torch_epochs)
        loss_fn = nn.HuberLoss(delta=1.0)  # More robust than MSE for financial data

        # FP16 mixed precision for 4GB VRAM
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        model.train()
        final_loss = 0.0
        for _epoch in range(self.settings.torch_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        predictions = model(batch_x)
                        loss = loss_fn(predictions, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(batch_x)
                    loss = loss_fn(predictions, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
            scheduler.step()
            final_loss = epoch_loss / max(len(loader), 1)

        # Inference
        model.eval()
        recent_window = np.stack([ch[-seq_len:] for ch in channels])  # (n_channels, seq_len)
        recent_tensor = torch.tensor(recent_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    prediction = model(recent_tensor).squeeze().item()
            else:
                prediction = model(recent_tensor).squeeze().item()

        # Denormalize (channel 0 = returns)
        ret_mean, ret_std = channel_stats[0]
        predicted_return = float((prediction * ret_std) + ret_mean)
        confidence = float(np.clip(abs(predicted_return) / (returns.std() + 1e-8), 0.0, 1.0))

        # Clean up GPU memory
        if device.type == "cuda":
            del model, optimizer, scaler
            torch.cuda.empty_cache()

        return ForecastSummary(
            predicted_return=predicted_return,
            confidence=confidence,
            device=str(device),
            enabled=True,
            model_type="PatchTST",
            training_loss=final_loss,
        )
