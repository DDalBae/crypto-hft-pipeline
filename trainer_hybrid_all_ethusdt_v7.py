# -*- coding: utf-8 -*-
"""
Trainer v7

Changes vs v6
-------------
- Feature contract: FEAT30 (adds r8 / atr8_rel / vol_z_8 / 5 session-state features)
- Target contract: 1/3/5/8/10 horizons
- Naming fix: y5_class replaces ambiguous y_class (legacy alias still accepted on input)
- Model: 1/3/5 multi-resolution paths + fused 8m/10m heads
- Contiguity requirement extended to horizon=10 by default

Checkpoint meta remains backward-friendly:
- features list stored in ckpt meta
- reg_target_mode / y_scale preserved
- horizons / target_contract_version added
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Contract constants
# =========================
FEATURES: List[str] = [
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

HORIZONS: List[int] = [1, 3, 5, 8, 10]
REG_TARGET_COLS: Dict[int, str] = {1: "y_next1", 3: "y_next3", 5: "y_next5", 8: "y_next8", 10: "y_next10"}
CLS_TARGET_COLS: Dict[int, str] = {1: "y1_class", 3: "y3_class", 5: "y5_class", 8: "y8_class", 10: "y10_class"}
LEGACY_CLS_ALIASES: Dict[str, str] = {"y_class": "y5_class"}


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_scaler(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    scaler: Dict[str, Dict[str, float]] = {}
    for c in cols:
        x = df[c].astype(np.float32).to_numpy()
        m = float(np.nanmean(x))
        s = float(np.nanstd(x))
        if s == 0.0 or not np.isfinite(s):
            s = 1.0
        scaler[c] = {"mean": m, "std": s}
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, Dict[str, float]], cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        m = scaler[c]["mean"]
        s = scaler[c]["std"]
        out[c] = (out[c].astype(np.float32) - m) / s
    return out


def build_contiguous_start_indices(times: np.ndarray, seq_len: int, horizon: int = 10) -> np.ndarray:
    total_len = int(seq_len) + int(horizon)
    if len(times) < total_len:
        return np.array([], dtype=np.int64)
    dt = np.diff(times.astype("datetime64[ns]").astype("int64"))
    one_min = 60 * 1_000_000_000
    ok = (dt == one_min)
    starts: List[int] = []
    run = 0
    for i, good in enumerate(ok):
        run = run + 1 if good else 0
        if run >= (total_len - 1):
            s = (i + 1) - (total_len - 1)
            starts.append(s)
    return np.asarray(starts, dtype=np.int64)


def resolve_target_columns(df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:
    cls_cols = dict(CLS_TARGET_COLS)
    if cls_cols[5] not in df.columns and "y_class" in df.columns:
        df[cls_cols[5]] = df["y_class"]
    miss_reg = [REG_TARGET_COLS[h] for h in HORIZONS if REG_TARGET_COLS[h] not in df.columns]
    miss_cls = [cls_cols[h] for h in HORIZONS if cls_cols[h] not in df.columns]
    if miss_reg:
        raise ValueError(f"Missing regression target columns: {miss_reg}")
    if miss_cls:
        raise ValueError(f"Missing classification target columns: {miss_cls}")
    return dict(REG_TARGET_COLS), cls_cols


# =========================
# Model blocks
# =========================
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
        blocks = [TCNResidualBlock(ch=ch, kernel_size=kernel_size, dilation=(2 ** i), dropout=dropout) for i in range(int(n_blocks))]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiResTCNLSTMMultiHead(nn.Module):
    """
    1m / 3m / 5m path heads + fused 8m / 10m heads.
    Returns tuple:
      reg1, logit1, reg3, logit3, reg5, logit5, reg8, logit8, reg10, logit10
    """

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
    ):
        super().__init__()
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
        hf = self.fusion(torch.cat([h1, h3, h5], dim=1))

        reg1 = self.reg1(h1).squeeze(-1)
        logit1 = self.cls1(h1).squeeze(-1)
        reg3 = self.reg3(h3).squeeze(-1)
        logit3 = self.cls3(h3).squeeze(-1)
        reg5 = self.reg5(h5).squeeze(-1)
        logit5 = self.cls5(h5).squeeze(-1)
        reg8 = self.reg8(hf).squeeze(-1)
        logit8 = self.cls8(hf).squeeze(-1)
        reg10 = self.reg10(hf).squeeze(-1)
        logit10 = self.cls10(hf).squeeze(-1)

        return reg1, logit1, reg3, logit3, reg5, logit5, reg8, logit8, reg10, logit10


# =========================
# Dataset
# =========================
class SeqMultiDataset(torch.utils.data.Dataset):
    """
    Returns
    -------
    x      : (seq_len, F)
    y_reg  : (H,) in horizon order [1,3,5,8,10]
    y_cls  : (H,) in [0,1], NaN -> neutral_target with weight neutral_weight
    w_cls  : (H,)
    """

    def __init__(
        self,
        X: np.ndarray,
        y_reg_by_h: Dict[int, np.ndarray],
        y_cls_by_h: Dict[int, np.ndarray],
        idx_list: np.ndarray,
        seq_len: int,
        neutral_weight: float,
        neutral_target: float = 0.5,
        horizons: Optional[Sequence[int]] = None,
    ):
        self.X = X.astype(np.float32)
        self.y_reg_by_h = {int(h): np.asarray(v, dtype=np.float32) for h, v in y_reg_by_h.items()}
        self.y_cls_by_h = {int(h): np.asarray(v, dtype=np.float32) for h, v in y_cls_by_h.items()}
        self.idx_list = idx_list.astype(np.int64)
        self.seq_len = int(seq_len)
        self.neutral_weight = float(neutral_weight)
        self.neutral_target = float(neutral_target)
        self.horizons = [int(h) for h in (horizons or HORIZONS)]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i: int):
        s = int(self.idx_list[i])
        e = s + self.seq_len - 1
        x = self.X[s:s + self.seq_len]

        y_reg = np.asarray([self.y_reg_by_h[h][e] for h in self.horizons], dtype=np.float32)
        cls = np.asarray([self.y_cls_by_h[h][e] for h in self.horizons], dtype=np.float32)
        w = np.ones_like(cls, dtype=np.float32)
        for k in range(len(self.horizons)):
            if not np.isfinite(cls[k]):
                cls[k] = self.neutral_target
                w[k] = self.neutral_weight
        return torch.from_numpy(x), torch.from_numpy(y_reg), torch.from_numpy(cls), torch.from_numpy(w)


# =========================
# Training
# =========================
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
    reg_target_mode: str = "magnitude_v2"
    reg_activation: str = "softplus"
    cls_balance_mode: str = "auto"
    cls_pos_weight: Optional[List[float]] = None
    features: Optional[List[str]] = None
    horizons: Optional[List[int]] = None
    reg_target_cols: Optional[Dict[str, str]] = None
    cls_target_cols: Optional[Dict[str, str]] = None
    feature_contract_version: str = "feat30_state_v1"
    target_contract_version: str = "horizons_1_3_5_8_10_v1"


def _compute_cls_pos_weight(
    label_arrays: Dict[int, np.ndarray],
    label_indices: np.ndarray,
    horizons: Sequence[int],
    mode: str = "auto",
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> Tuple[List[float], Dict[str, Dict[str, float]]]:
    weights: List[float] = []
    stats: Dict[str, Dict[str, float]] = {}
    mode = str(mode or "none").strip().lower()

    for h in horizons:
        arr = label_arrays[int(h)]
        vals = arr[label_indices]
        pos = int(np.sum(np.isfinite(vals) & (vals >= 0.999)))
        neg = int(np.sum(np.isfinite(vals) & (vals <= 0.001)))
        neu = int(np.sum(~np.isfinite(vals)))
        if mode == "auto" and pos > 0 and neg > 0:
            w = float(np.clip(neg / max(pos, 1), min_weight, max_weight))
        else:
            w = 1.0
        weights.append(w)
        stats[f"{h}m"] = {"pos": float(pos), "neg": float(neg), "neutral_or_missing": float(neu), "pos_weight": float(w)}
    return weights, stats


def _loss_multi(
    pred: Tuple[torch.Tensor, ...],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    w_cls: torch.Tensor,
    w_reg: float = 1.0,
    w_cls_loss: float = 1.0,
    w_hyb: float = 0.5,
    reg_target_mode: str = "magnitude_v2",
    cls_pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    regs_raw = torch.stack(pred[0::2], dim=1)
    logits = torch.stack(pred[1::2], dim=1)

    y_reg = y_reg.to(dtype=regs_raw.dtype)
    y_cls = y_cls.to(dtype=logits.dtype)

    mode = str(reg_target_mode or "signed_legacy").strip().lower()
    if mode == "magnitude_v2":
        regs_eff = F.softplus(regs_raw)
        reg_target = torch.abs(y_reg)
        hyb = regs_eff * torch.tanh(logits / 2.0)
    elif mode == "signed_legacy":
        regs_eff = regs_raw
        reg_target = y_reg
        hyb = regs_eff * torch.tanh(logits / 2.0)
    else:
        raise ValueError(f"Unsupported reg_target_mode={reg_target_mode}")

    reg_loss = F.smooth_l1_loss(regs_eff, reg_target, reduction="mean")

    bce = F.binary_cross_entropy_with_logits(logits, y_cls, reduction="none")
    w = w_cls.to(dtype=bce.dtype)
    if cls_pos_weight is not None:
        cpw = cls_pos_weight.to(device=bce.device, dtype=bce.dtype).view(1, -1)
        pos_mask = (y_cls >= 0.999).to(dtype=bce.dtype)
        balance_w = torch.ones_like(bce) + pos_mask * (cpw - 1.0)
        w = w * balance_w
    cls_loss = (bce * w).sum() / (w.sum() + 1e-8)

    hyb_loss = F.smooth_l1_loss(hyb, y_reg, reduction="mean")

    total = (w_reg * reg_loss) + (w_cls_loss * cls_loss) + (w_hyb * hyb_loss)
    return total, reg_loss.detach(), cls_loss.detach(), hyb_loss.detach()


@torch.no_grad()
def _eval_epoch(model, loader, device, w_reg, w_cls_loss, w_hyb, reg_target_mode: str, cls_pos_weight: Optional[torch.Tensor]):
    model.eval()
    total_sum = reg_sum = cls_sum = hyb_sum = 0.0
    n = 0
    for xb, yreg, ycls, wcls in loader:
        xb = xb.to(device)
        yreg = yreg.to(device)
        ycls = ycls.to(device)
        wcls = wcls.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            pred = model(xb)
            total, reg_l, cls_l, hyb_l = _loss_multi(
                pred, yreg, ycls, wcls,
                w_reg=w_reg,
                w_cls_loss=w_cls_loss,
                w_hyb=w_hyb,
                reg_target_mode=reg_target_mode,
                cls_pos_weight=cls_pos_weight,
            )
        bs = xb.size(0)
        total_sum += float(total.item()) * bs
        reg_sum += float(reg_l.item()) * bs
        cls_sum += float(cls_l.item()) * bs
        hyb_sum += float(hyb_l.item()) * bs
        n += bs
    if n == 0:
        return {"total": float("inf"), "reg": float("inf"), "cls": float("inf"), "hyb": float("inf")}
    return {"total": total_sum / n, "reg": reg_sum / n, "cls": cls_sum / n, "hyb": hyb_sum / n}


def train(
    model: nn.Module,
    train_loader,
    valid_loader,
    device,
    epochs: int,
    lr: float,
    patience: int,
    w_reg: float,
    w_cls_loss: float,
    w_hyb: float,
    optimizer: str,
    weight_decay: float,
    reg_target_mode: str,
    cls_pos_weight: Optional[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    optimizer = str(optimizer).lower().strip()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer={optimizer}")

    best_state = None
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_tot_sum = train_reg_sum = train_cls_sum = train_hyb_sum = 0.0
        n_train = 0
        for xb, yreg, ycls, wcls in train_loader:
            xb = xb.to(device)
            yreg = yreg.to(device)
            ycls = ycls.to(device)
            wcls = wcls.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                pred = model(xb)
                total, reg_l, cls_l, hyb_l = _loss_multi(
                    pred, yreg, ycls, wcls,
                    w_reg=w_reg,
                    w_cls_loss=w_cls_loss,
                    w_hyb=w_hyb,
                    reg_target_mode=reg_target_mode,
                    cls_pos_weight=cls_pos_weight,
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.size(0)
            train_tot_sum += float(total.item()) * bs
            train_reg_sum += float(reg_l.item()) * bs
            train_cls_sum += float(cls_l.item()) * bs
            train_hyb_sum += float(hyb_l.item()) * bs
            n_train += bs

        val = _eval_epoch(model, valid_loader, device, w_reg, w_cls_loss, w_hyb, reg_target_mode, cls_pos_weight)
        t_tot = train_tot_sum / max(1, n_train)
        t_reg = train_reg_sum / max(1, n_train)
        t_cls = train_cls_sum / max(1, n_train)
        t_hyb = train_hyb_sum / max(1, n_train)
        print(f"[epoch {ep:02d}] TRAIN tot={t_tot:.5f} reg={t_reg:.6f} cls={t_cls:.5f} hyb={t_hyb:.5f} | VALID tot={val['total']:.5f} reg={val['reg']:.6f} cls={val['cls']:.5f} hyb={val['hyb']:.5f}")

        if val["total"] < best_val - 1e-6:
            best_val = val["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[early stop]")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = _eval_epoch(model, valid_loader, device, w_reg, w_cls_loss, w_hyb, reg_target_mode, cls_pos_weight)
    return best_state, best_metrics


# =========================
# CLI
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="train_clean_final_v8.csv")
    ap.add_argument("--seq-len", type=int, default=300)
    ap.add_argument("--valid-tail", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--neutral-weight", type=float, default=0.25)
    ap.add_argument("--neutral-target", type=float, default=0.5)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--tcn-blocks", type=int, default=6)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--tcn-dropout", type=float, default=0.10)
    ap.add_argument("--lstm-hidden", type=int, default=64)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--lstm-dropout", type=float, default=0.0)
    ap.add_argument("--pool-type", type=str, default="avg", choices=["avg", "max"])
    ap.add_argument("--fusion-hidden", type=int, default=128)
    ap.add_argument("--fusion-dropout", type=float, default=0.05)

    ap.add_argument("--y-scale", type=float, default=1000.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-hyb", type=float, default=0.5)
    ap.add_argument("--reg-target-mode", type=str, default="magnitude_v2", choices=["signed_legacy", "magnitude_v2"])
    ap.add_argument("--cls-balance-mode", type=str, default="auto", choices=["none", "auto"])
    ap.add_argument("--cls-pos-weight-min", type=float, default=0.25)
    ap.add_argument("--cls-pos-weight-max", type=float, default=4.0)

    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--scaler-name", type=str, default="scaler_ethusdt.json")
    ap.add_argument("--model-name", type=str, default="model_hybrid_mr_tcnlstm_multihead_ethusdt.pt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    df = pd.read_csv(args.csv)
    if "time" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"time": "timestamp"}, inplace=True)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' or 'time' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    reg_cols, cls_cols = resolve_target_columns(df)

    n = len(df)
    split_row = max(args.seq_len + 10, n - args.valid_tail)
    print(f"[split] split_row={split_row} (n={n})")

    scaler = compute_scaler(df.iloc[:split_row], FEATURES)
    os.makedirs(args.out_dir, exist_ok=True)
    scaler_path = os.path.join(args.out_dir, args.scaler_name)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)
    print("[saved scaler]", scaler_path)

    df_norm = apply_scaler(df, scaler, FEATURES)
    X = df_norm[FEATURES].astype(np.float32).to_numpy()
    times = df_norm["timestamp"].to_numpy()

    y_reg_by_h: Dict[int, np.ndarray] = {}
    y_cls_by_h: Dict[int, np.ndarray] = {}
    ys = np.float32(args.y_scale)
    for h in HORIZONS:
        y = df_norm[reg_cols[h]].astype(np.float32).to_numpy()
        if args.y_scale != 1.0:
            y = y * ys
        y_reg_by_h[h] = y
        y_cls_by_h[h] = df_norm[cls_cols[h]].astype(np.float32).to_numpy()

    start_all = build_contiguous_start_indices(times, args.seq_len, horizon=max(HORIZONS))
    print("[starts] total (contiguous up to future horizon)", len(start_all))
    e_all = start_all + args.seq_len - 1
    ok_e = e_all < n
    start_all = start_all[ok_e]
    e_all = e_all[ok_e]

    reg_ok = np.ones_like(e_all, dtype=bool)
    for h in HORIZONS:
        reg_ok &= np.isfinite(y_reg_by_h[h][e_all])
    start_all = start_all[reg_ok]
    e_all = e_all[reg_ok]
    print("[starts] after reg finite", len(start_all))

    train_mask = e_all < split_row
    valid_mask = e_all >= split_row
    train_idx = start_all[train_mask]
    valid_idx = start_all[valid_mask]
    train_label_idx = e_all[train_mask]
    print(f"[split] train_starts={len(train_idx)} valid_starts={len(valid_idx)}")

    cls_pos_weight_list, cls_balance_stats = _compute_cls_pos_weight(
        label_arrays=y_cls_by_h,
        label_indices=train_label_idx,
        horizons=HORIZONS,
        mode=args.cls_balance_mode,
        min_weight=float(args.cls_pos_weight_min),
        max_weight=float(args.cls_pos_weight_max),
    )
    cls_pos_weight = torch.tensor(cls_pos_weight_list, dtype=torch.float32, device=device)
    print("[cls balance mode]", args.cls_balance_mode)
    print("[cls pos weight]", {k: round(v["pos_weight"], 4) for k, v in cls_balance_stats.items()})

    train_ds = SeqMultiDataset(X, y_reg_by_h, y_cls_by_h, train_idx, args.seq_len, neutral_weight=args.neutral_weight, neutral_target=args.neutral_target, horizons=HORIZONS)
    valid_ds = SeqMultiDataset(X, y_reg_by_h, y_cls_by_h, valid_idx, args.seq_len, neutral_weight=args.neutral_weight, neutral_target=args.neutral_target, horizons=HORIZONS)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = MultiResTCNLSTMMultiHead(
        input_dim=len(FEATURES),
        channels=args.channels,
        tcn_blocks=args.tcn_blocks,
        kernel_size=args.kernel_size,
        tcn_dropout=args.tcn_dropout,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        pool_type=args.pool_type,
        fusion_hidden=args.fusion_hidden,
        fusion_dropout=args.fusion_dropout,
    ).to(device)

    best_state, best_metrics = train(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        w_reg=args.w_reg,
        w_cls_loss=args.w_cls,
        w_hyb=args.w_hyb,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        reg_target_mode=args.reg_target_mode,
        cls_pos_weight=cls_pos_weight,
    )

    meta = ModelMeta(
        seq_len=args.seq_len,
        channels=args.channels,
        tcn_blocks=args.tcn_blocks,
        kernel_size=args.kernel_size,
        tcn_dropout=args.tcn_dropout,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        pool_type=args.pool_type,
        fusion_hidden=args.fusion_hidden,
        fusion_dropout=args.fusion_dropout,
        y_scale=float(args.y_scale),
        reg_target_mode=str(args.reg_target_mode),
        reg_activation=("softplus" if str(args.reg_target_mode) == "magnitude_v2" else "identity"),
        cls_balance_mode=str(args.cls_balance_mode),
        cls_pos_weight=[float(x) for x in cls_pos_weight_list],
        features=list(FEATURES),
        horizons=list(HORIZONS),
        reg_target_cols={str(h): reg_cols[h] for h in HORIZONS},
        cls_target_cols={str(h): cls_cols[h] for h in HORIZONS},
    )
    ckpt = {
        "state_dict": best_state,
        "meta": asdict(meta),
        "best_valid": best_metrics,
        "cls_balance_stats": cls_balance_stats,
    }

    out_path = os.path.join(args.out_dir, args.model_name)
    torch.save(ckpt, out_path)
    print("[saved model]", out_path)
    print("[best_valid]", best_metrics)


if __name__ == "__main__":
    main()
