# -*- coding: utf-8 -*-
"""
HybridScalpInference v7

Supports both:
- legacy 3-horizon checkpoints (1/3/5)
- new 5-horizon checkpoints (1/3/5/8/10) with fused 8m/10m heads

New contract highlights
-----------------------
- FEAT30 default feature order
- Horizon-aware outputs: hybrid_1m/3m/5m/8m/10m when available
- Legacy compatibility: hybrid_total still computed from 1/3/5
- Checkpoint meta.features is trusted when present
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURES_V7: List[str] = [
    "gap_open",
    "high_ext",
    "low_ext",
    "body_ret",
    "r1",
    "r3",
    "r5",
    "r8",
    "r10",
    "atr1_rel",
    "atr3_rel",
    "atr5_rel",
    "atr8_rel",
    "atr10_rel",
    "vol_z_3",
    "vol_z_5",
    "vol_z_8",
    "vol_z_10",
    "vol_z_60",
    "spread_proxy",
    "taker_buy_ratio",
    "upper_wick_rel",
    "lower_wick_rel",
    "wick_ratio",
    "funding_diff",
    "session_vwap_dist",
    "session_vwap_slope",
    "session_range_pct",
    "bb_pctb_20",
    "efficiency_ratio_10",
]

LEGACY_FEATURES_V6: List[str] = [
    "gap_open",
    "high_ext",
    "low_ext",
    "body_ret",
    "r1",
    "r3",
    "r5",
    "r10",
    "atr1_rel",
    "atr3_rel",
    "atr5_rel",
    "atr10_rel",
    "vol_z_3",
    "vol_z_5",
    "vol_z_10",
    "vol_z_60",
    "spread_proxy",
    "taker_buy_ratio",
    "upper_wick_rel",
    "lower_wick_rel",
    "wick_ratio",
    "funding_diff",
]


def load_scaler(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(int(in_ch), int(out_ch), kernel_size=int(kernel_size), dilation=int(dilation), padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y


class TCNStack(nn.Module):
    def __init__(self, ch: int, kernel_size: int, n_blocks: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(*[TCNResidualBlock(ch=ch, kernel_size=kernel_size, dilation=(2 ** i), dropout=dropout) for i in range(int(n_blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiResTCNLSTMMultiHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 128,
        tcn_blocks: int = 6,
        kernel_size: int = 3,
        tcn_dropout: float = 0.10,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        pool_type: str = "avg",
        fusion_hidden: int = 128,
        fusion_dropout: float = 0.05,
        enable_extra_heads: bool = True,
    ):
        super().__init__()
        self.enable_extra_heads = bool(enable_extra_heads)
        self.in_proj = nn.Conv1d(int(input_dim), int(channels), kernel_size=1)
        if pool_type == "avg":
            self.pool3 = nn.AvgPool1d(kernel_size=3, stride=3, ceil_mode=False)
            self.pool5 = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=False)
        elif pool_type == "max":
            self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3, ceil_mode=False)
            self.pool5 = nn.MaxPool1d(kernel_size=5, stride=5, ceil_mode=False)
        else:
            raise ValueError(f"Unsupported pool_type={pool_type}")

        self.tcn1 = TCNStack(ch=int(channels), kernel_size=int(kernel_size), n_blocks=int(tcn_blocks), dropout=float(tcn_dropout))
        self.tcn3 = TCNStack(ch=int(channels), kernel_size=int(kernel_size), n_blocks=int(tcn_blocks), dropout=float(tcn_dropout))
        self.tcn5 = TCNStack(ch=int(channels), kernel_size=int(kernel_size), n_blocks=int(tcn_blocks), dropout=float(tcn_dropout))

        self.lstm1 = nn.LSTM(int(channels), int(lstm_hidden), num_layers=int(lstm_layers), dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0, batch_first=True)
        self.lstm3 = nn.LSTM(int(channels), int(lstm_hidden), num_layers=int(lstm_layers), dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0, batch_first=True)
        self.lstm5 = nn.LSTM(int(channels), int(lstm_hidden), num_layers=int(lstm_layers), dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0, batch_first=True)

        self.reg1 = nn.Linear(int(lstm_hidden), 1)
        self.cls1 = nn.Linear(int(lstm_hidden), 1)
        self.reg3 = nn.Linear(int(lstm_hidden), 1)
        self.cls3 = nn.Linear(int(lstm_hidden), 1)
        self.reg5 = nn.Linear(int(lstm_hidden), 1)
        self.cls5 = nn.Linear(int(lstm_hidden), 1)

        if self.enable_extra_heads:
            fusion_hidden = int(fusion_hidden)
            self.fusion = nn.Sequential(
                nn.Linear(int(lstm_hidden) * 3, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(float(fusion_dropout)),
            )
            self.reg8 = nn.Linear(fusion_hidden, 1)
            self.cls8 = nn.Linear(fusion_hidden, 1)
            self.reg10 = nn.Linear(fusion_hidden, 1)
            self.cls10 = nn.Linear(fusion_hidden, 1)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        x = self.in_proj(x)
        x1 = x
        x3 = self.pool3(x)
        x5 = self.pool5(x)
        z1 = self.tcn1(x1)
        z3 = self.tcn3(x3)
        z5 = self.tcn5(x5)
        o1, _ = self.lstm1(z1.transpose(1, 2).contiguous())
        o3, _ = self.lstm3(z3.transpose(1, 2).contiguous())
        o5, _ = self.lstm5(z5.transpose(1, 2).contiguous())
        h1 = o1[:, -1, :]
        h3 = o3[:, -1, :]
        h5 = o5[:, -1, :]
        reg1 = self.reg1(h1).squeeze(-1)
        logit1 = self.cls1(h1).squeeze(-1)
        reg3 = self.reg3(h3).squeeze(-1)
        logit3 = self.cls3(h3).squeeze(-1)
        reg5 = self.reg5(h5).squeeze(-1)
        logit5 = self.cls5(h5).squeeze(-1)
        if not self.enable_extra_heads:
            return reg1, logit1, reg3, logit3, reg5, logit5
        hf = self.fusion(torch.cat([h1, h3, h5], dim=1))
        reg8 = self.reg8(hf).squeeze(-1)
        logit8 = self.cls8(hf).squeeze(-1)
        reg10 = self.reg10(hf).squeeze(-1)
        logit10 = self.cls10(hf).squeeze(-1)
        return reg1, logit1, reg3, logit3, reg5, logit5, reg8, logit8, reg10, logit10


@dataclass
class ModelMeta:
    arch: str = "mr_tcn_lstm_multihead_v7_5h"
    seq_len: int = 300
    channels: int = 128
    tcn_blocks: int = 6
    kernel_size: int = 3
    tcn_dropout: float = 0.10
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    pool_type: str = "avg"
    fusion_hidden: int = 128
    fusion_dropout: float = 0.05
    y_scale: float = 1.0
    reg_target_mode: str = "signed_legacy"
    reg_activation: str = "identity"
    features: Optional[List[str]] = None
    horizons: Optional[List[int]] = None
    feature_contract_version: str = "feat30_state_v1"
    target_contract_version: str = "horizons_1_3_5_8_10_v1"

    @staticmethod
    def from_any(obj: Any) -> "ModelMeta":
        mm = ModelMeta()
        if not isinstance(obj, dict):
            return mm
        for k in mm.__dict__.keys():
            if k in obj:
                setattr(mm, k, obj[k])
        return mm


class HybridScalpInference:
    FEATURES: List[str] = list(FEATURES_V7)

    def __init__(
        self,
        scaler_path: str = "scaler_ethusdt.json",
        model_path: str = "model_hybrid_mr_tcnlstm_multihead_ethusdt.pt",
        seq_len: int = 300,
        w5: float = 0.1,
        w1: float = 0.1,
        w3: float = 0.8,
        device: Optional[str] = None,
        models_dir: Optional[str] = None,
        lazy: bool = False,
    ):
        self.models_dir = models_dir
        self.seq_len = int(seq_len)
        self.w5 = float(w5)
        self.w1 = float(w1)
        self.w3 = float(w3)
        self.scaler_path = str(scaler_path)
        self.model_path = str(model_path)
        if models_dir:
            md = str(models_dir)
            if (not os.path.isabs(self.scaler_path)) and (os.path.dirname(self.scaler_path) == ""):
                self.scaler_path = os.path.join(md, self.scaler_path)
            if (not os.path.isabs(self.model_path)) and (os.path.dirname(self.model_path) == ""):
                self.model_path = os.path.join(md, self.model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.scaler: Optional[Dict[str, Dict[str, float]]] = None
        self.features: List[str] = list(HybridScalpInference.FEATURES)
        self.horizons: List[int] = [1, 3, 5]
        self.model: Optional[nn.Module] = None
        self.meta: Dict[str, Any] = {}
        self.y_scale: float = 1.0
        self.reg_target_mode: str = "signed_legacy"
        self.reg_activation: str = "identity"
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._loaded = False
        self.buf = deque(maxlen=self.seq_len)
        if not bool(lazy):
            self.load()

    def load(self) -> "HybridScalpInference":
        if self._loaded:
            return self
        self.scaler = load_scaler(self.scaler_path)

        ckpt = torch.load(self.model_path, map_location=self.device)
        if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model" in ckpt):
            state = ckpt.get("state_dict", ckpt.get("model"))
            meta = ckpt.get("meta", {})
        elif isinstance(ckpt, dict):
            state = ckpt
            meta = {}
        else:
            raise ValueError(f"Unsupported checkpoint format type={type(ckpt)}")

        self.meta = meta if isinstance(meta, dict) else {}
        mm = ModelMeta.from_any(self.meta)
        self.y_scale = float(getattr(mm, "y_scale", 1.0) or 1.0)
        self.reg_target_mode = str(getattr(mm, "reg_target_mode", "signed_legacy") or "signed_legacy").strip().lower()
        default_activation = "softplus" if self.reg_target_mode == "magnitude_v2" else "identity"
        self.reg_activation = str(getattr(mm, "reg_activation", default_activation) or default_activation).strip().lower()

        if mm.features:
            self.features = list(mm.features)
        else:
            scaler_keys = set(self.scaler.keys())
            self.features = list(FEATURES_V7 if all(k in scaler_keys for k in FEATURES_V7) else LEGACY_FEATURES_V6)
        self.FEATURES = list(self.features)
        self._refresh_scaler_arrays(self.features)

        # checkpoint compatibility
        has_extra_heads = any(str(k).startswith("reg8.") for k in state.keys()) and any(str(k).startswith("reg10.") for k in state.keys())
        if isinstance(mm.horizons, list) and len(mm.horizons) >= 3:
            self.horizons = [int(h) for h in mm.horizons]
        else:
            self.horizons = [1, 3, 5, 8, 10] if has_extra_heads else [1, 3, 5]

        self.model = MultiResTCNLSTMMultiHead(
            input_dim=len(self.features),
            channels=int(mm.channels),
            tcn_blocks=int(mm.tcn_blocks),
            kernel_size=int(mm.kernel_size),
            tcn_dropout=float(mm.tcn_dropout),
            lstm_hidden=int(mm.lstm_hidden),
            lstm_layers=int(mm.lstm_layers),
            lstm_dropout=float(mm.lstm_dropout),
            pool_type=str(mm.pool_type),
            fusion_hidden=int(getattr(mm, "fusion_hidden", 128) or 128),
            fusion_dropout=float(getattr(mm, "fusion_dropout", 0.05) or 0.05),
            enable_extra_heads=bool(has_extra_heads),
        ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected state_dict keys: {unexpected}")
        # missing keys are acceptable only for legacy checkpoints without 8/10 heads
        legacy_allowed = {"fusion.0.weight", "fusion.0.bias", "reg8.weight", "reg8.bias", "cls8.weight", "cls8.bias", "reg10.weight", "reg10.bias", "cls10.weight", "cls10.bias"}
        bad_missing = [k for k in missing if k not in legacy_allowed]
        if bad_missing:
            raise RuntimeError(f"Missing critical state_dict keys: {bad_missing}")
        self.model.eval()
        self._loaded = True
        return self

    def reset(self) -> None:
        self.buf.clear()

    @property
    def scaler_mean(self) -> np.ndarray:
        self._ensure_loaded()
        assert self._mean is not None
        return self._mean

    @property
    def scaler_std(self) -> np.ndarray:
        self._ensure_loaded()
        assert self._std is not None
        return self._std

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _refresh_scaler_arrays(self, feats: Sequence[str]) -> None:
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Call load() first.")
        means, stds = [], []
        for c in feats:
            s = self.scaler.get(c)
            if not isinstance(s, dict):
                raise KeyError(f"Scaler missing key for feature: {c}")
            means.append(float(s.get("mean", 0.0)))
            std = float(s.get("std", 1.0))
            stds.append(std if std != 0.0 else 1.0)
        self._mean = np.asarray(means, dtype=np.float32)
        self._std = np.asarray(stds, dtype=np.float32)

    def scale_np(self, x: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        x = np.asarray(x, dtype=np.float32)
        return (x - self.scaler_mean) / self.scaler_std

    @staticmethod
    def _direction_from_logits(logit: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logit / 2.0)

    def _decode_reg_output(self, reg_raw: torch.Tensor) -> torch.Tensor:
        mode = str(self.reg_target_mode or "signed_legacy").strip().lower()
        if mode == "magnitude_v2":
            reg_eff = F.softplus(reg_raw)
        elif mode == "signed_legacy":
            reg_eff = reg_raw
        else:
            raise ValueError(f"Unsupported reg_target_mode={self.reg_target_mode}")
        if self.y_scale != 1.0:
            reg_eff = reg_eff / self.y_scale
        return reg_eff

    def _hybrid_from_logits(self, reg_raw: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        return self._decode_reg_output(reg_raw) * self._direction_from_logits(logit)

    def _decode_outputs(self, pred: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        horizon_names = [1, 3, 5] if len(pred) == 6 else [1, 3, 5, 8, 10]
        out: Dict[str, torch.Tensor] = {}
        for i, h in enumerate(horizon_names):
            reg_raw = pred[2 * i]
            logit = pred[2 * i + 1]
            out[f"hybrid_{h}m"] = self._hybrid_from_logits(reg_raw, logit)
        return out

    @torch.inference_mode()
    def forward_batch_scaled(self, xb_scaled: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()
        assert self.model is not None
        if xb_scaled.device != self.device:
            xb_scaled = xb_scaled.to(self.device, non_blocking=True)
        pred = self.model(xb_scaled)
        out = {k: v.detach().to("cpu") for k, v in self._decode_outputs(pred).items()}
        return out

    def update(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self.model is not None
        feat = np.asarray([float(row.get(c, 0.0)) for c in self.features], dtype=np.float32)
        self.buf.append(feat)
        if len(self.buf) < self.seq_len:
            return {"ready": False}
        x = np.stack(self.buf, axis=0)
        x = self.scale_np(x)
        xb = torch.from_numpy(x[None, :, :]).to(self.device)
        with torch.inference_mode():
            pred = self.model(xb)
            out = {k: float(v.item()) for k, v in self._decode_outputs(pred).items()}
        hy1 = out.get("hybrid_1m", 0.0)
        hy3 = out.get("hybrid_3m", 0.0)
        hy5 = out.get("hybrid_5m", 0.0)
        out.update(
            {
                "ready": True,
                "horizons_available": list(self.horizons),
                "hybrid_total": self.w5 * hy5 + self.w1 * hy1 + self.w3 * hy3,
            }
        )
        return out
