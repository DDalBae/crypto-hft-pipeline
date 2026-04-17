# -*- coding: utf-8 -*-
"""
Shared helpers for the single-tier RL-mode v58 pipeline.

Authoritative v5 core with objective alignment, tp_window, entry_episode/rearm support, active_sparse lane v2, and the soft-SL/trail guard fix.

Goals:
- no low/mid/top tier logic in the runtime core
- flat config schema with backward migration from legacy top-only tiered configs
- same score logic for autotune and backtest worst-case path selection
- split hold parameters are fully independent
- gate and direction support weighted 1/3/5/8/10 mixtures with regime-adaptive profiles
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from numba import njit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hybrid_inference_scalp_v7 import HybridScalpInference


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        out = float(val)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) <= 1e-12:
        return float(default)
    return float(num) / den


def parse_float_list(text: str, default: Optional[List[float]] = None) -> List[float]:
    if text is None:
        return list(default or [])
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    if (not out) and default is not None:
        return list(default)
    return out


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out.get(k, {}), v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# weights / search-bounds aliases
# ---------------------------------------------------------------------------

def normalize_weight_triplet(val: Any, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> Dict[str, float]:
    """Return normalized non-negative weights dict {w1,w3,w5}. Legacy helper."""
    if isinstance(val, dict):
        w1 = max(0.0, safe_float(val.get("w1", fallback[0]), fallback[0]))
        w3 = max(0.0, safe_float(val.get("w3", fallback[1]), fallback[1]))
        w5 = max(0.0, safe_float(val.get("w5", fallback[2]), fallback[2]))
    elif isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
        w1 = max(0.0, safe_float(val[0], fallback[0]))
        w3 = max(0.0, safe_float(val[1], fallback[1]))
        w5 = max(0.0, safe_float(val[2], fallback[2]))
    else:
        w1 = max(0.0, safe_float(fallback[0], 0.0))
        w3 = max(0.0, safe_float(fallback[1], 0.0))
        w5 = max(0.0, safe_float(fallback[2], 1.0))
    s = w1 + w3 + w5
    if s <= 0.0:
        w1, w3, w5 = float(fallback[0]), float(fallback[1]), float(fallback[2])
        s = w1 + w3 + w5
        if s <= 0.0:
            w1, w3, w5, s = 0.0, 0.0, 1.0, 1.0
    return {"w1": w1 / s, "w3": w3 / s, "w5": w5 / s}


HORIZONS_ALL: List[int] = [1, 3, 5, 8, 10]
LEGACY_HORIZONS: List[int] = [1, 3, 5]


def _weight_key(h: int) -> str:
    return f"w{int(h)}"


def normalize_horizon_weights(
    val: Any,
    fallback: Optional[Dict[int, float]] = None,
    horizons: Sequence[int] = HORIZONS_ALL,
    available_horizons: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Return normalized non-negative weights dict over 1/3/5/8/10 horizons.

    Notes
    -----
    - missing keys default to 0.0
    - legacy {w1,w3,w5} is accepted; w8/w10 become 0
    - if available_horizons is supplied, only those horizons participate in normalization
      so new configs remain compatible with legacy 3-horizon checkpoints.
    """
    hz = [int(h) for h in horizons]
    avail = set(int(h) for h in (available_horizons or hz))
    fb = {int(h): 0.0 for h in hz}
    if fallback:
        for h, v in fallback.items():
            fb[int(h)] = float(v)
    if isinstance(val, dict):
        raw = {int(h): max(0.0, safe_float(val.get(_weight_key(h), fb.get(int(h), 0.0)), fb.get(int(h), 0.0))) for h in hz}
    elif isinstance(val, (list, tuple, np.ndarray)):
        arr = list(val)
        raw = {int(h): max(0.0, safe_float(arr[i] if i < len(arr) else fb.get(int(h), 0.0), fb.get(int(h), 0.0))) for i, h in enumerate(hz)}
    else:
        raw = {int(h): max(0.0, fb.get(int(h), 0.0)) for h in hz}

    # legacy triplet fallback if everything is zero
    if sum(raw.values()) <= 0.0 and isinstance(val, dict) and any(k in val for k in ("w1", "w3", "w5")):
        raw[1] = max(0.0, safe_float(val.get("w1", fb.get(1, 0.0)), fb.get(1, 0.0)))
        raw[3] = max(0.0, safe_float(val.get("w3", fb.get(3, 0.0)), fb.get(3, 0.0)))
        raw[5] = max(0.0, safe_float(val.get("w5", fb.get(5, 1.0)), fb.get(5, 1.0)))

    active = {h: raw.get(h, 0.0) for h in hz if h in avail}
    s = float(sum(active.values()))
    if s <= 0.0:
        active = {h: fb.get(h, 0.0) for h in hz if h in avail}
        s = float(sum(active.values()))
        if s <= 0.0:
            if 5 in active:
                active = {h: 0.0 for h in hz if h in avail}
                active[5] = 1.0
            else:
                first_h = min(avail) if avail else 5
                active = {h: 0.0 for h in hz if h in avail}
                active[first_h] = 1.0
            s = float(sum(active.values()))
    out = { _weight_key(h): 0.0 for h in hz }
    for h in hz:
        if h in active:
            out[_weight_key(h)] = float(active[h] / max(s, 1e-12))
    return out


def weights_from_self_mix(self_w: float, mix: float) -> Dict[str, float]:
    """Legacy 2-parameter representation kept for backward compatibility.
    Mass is allocated over 1/3/5 only; 8/10 default to zero.
    """
    self_w = float(np.clip(self_w, 0.0, 1.0))
    mix = float(np.clip(mix, 0.0, 1.0))
    rem = 1.0 - self_w
    w1 = rem * mix
    w3 = rem * (1.0 - mix)
    w5 = self_w
    return normalize_horizon_weights({"w1": w1, "w3": w3, "w5": w5, "w8": 0.0, "w10": 0.0}, fallback={1:0.0,3:0.0,5:1.0,8:0.0,10:0.0})


def weights_from_raw_vector(raw: Dict[int, float], horizons: Sequence[int] = HORIZONS_ALL) -> Dict[str, float]:
    return normalize_horizon_weights({ _weight_key(int(h)): safe_float(raw.get(int(h), 0.0), 0.0) for h in horizons }, fallback={5:1.0})


# ---------------------------------------------------------------------------
# regime-adaptive weight helpers
# ---------------------------------------------------------------------------

DEFAULT_REGIME_WEIGHT_CFG: Dict[str, Any] = {
    "enabled": 0,
    "source": "exo_stress",
    "stress_lo": 0.25,
    "stress_hi": 0.65,
    "alpha_ema": 0.15,
    "alpha_hysteresis": 0.03,
    "gate_calm_mix": 0.60,
    "gate_active_mix": 0.55,
    "dir_calm_mix": 0.35,
    "dir_active_mix": 0.50,
    "gate_calm_anchor": {"w1": 0.45, "w3": 0.40, "w5": 0.12, "w8": 0.03, "w10": 0.00},
    "gate_active_anchor": {"w1": 0.08, "w3": 0.28, "w5": 0.32, "w8": 0.22, "w10": 0.10},
    "dir_calm_anchor": {"w1": 0.30, "w3": 0.24, "w5": 0.24, "w8": 0.16, "w10": 0.06},
    "dir_active_anchor": {"w1": 0.08, "w3": 0.12, "w5": 0.32, "w8": 0.32, "w10": 0.16},
}


def _blend_base_to_anchor(
    base_weights: Dict[str, float],
    anchor_weights: Dict[str, float],
    mix: float,
    available_horizons: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    mix = float(np.clip(safe_float(mix, 0.0), 0.0, 1.0))
    raw: Dict[str, float] = {}
    for h in HORIZONS_ALL:
        k = _weight_key(int(h))
        raw[k] = (1.0 - mix) * float(base_weights.get(k, 0.0)) + mix * float(anchor_weights.get(k, 0.0))
    return normalize_horizon_weights(raw, fallback={5: 1.0}, available_horizons=available_horizons)

@njit(cache=True)
def _build_regime_alpha_exogenous_numba(
    atr_arr: np.ndarray,
    range_arr: np.ndarray,
    vol_arr: np.ndarray,
    funding_arr: np.ndarray,
    atr_high_th: float,
    range_cut: float,
    vol_low_th: float,
    funding_soft_min: float,
    stress_lo: float,
    stress_hi: float,
    alpha_ema: float,
    alpha_hysteresis: float,
    w_atr: float,
    w_rng: float,
    w_vol: float,
    w_fund: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(atr_arr))
    stress = np.zeros(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    bucket = np.zeros(n, dtype=np.int8)
    if n <= 0:
        return stress, alpha, bucket

    eps = 1e-12

    vol_enabled = (not np.isnan(vol_low_th)) and (vol_low_th > -1e8)
    vol_neutral = vol_low_th if vol_enabled else 0.0
    funding_soft = funding_soft_min if funding_soft_min > 0.0 else 0.0

    wa = w_atr if w_atr > 0.0 else 0.0
    wr = w_rng if w_rng > 0.0 else 0.0
    wv = w_vol if w_vol > 0.0 else 0.0
    wf = w_fund if w_fund > 0.0 else 0.0
    wsum = wa + wr + wv + wf
    if wsum <= 0.0:
        wa, wr, wv, wf = 0.35, 0.20, 0.30, 0.15
        wsum = wa + wr + wv + wf
    wa /= wsum
    wr /= wsum
    wv /= wsum
    wf /= wsum

    atr_high_enabled = (not np.isnan(atr_high_th)) and (atr_high_th > 0.0)
    range_cut_enabled = (not np.isnan(range_cut)) and (range_cut > 0.0)

    vol_norm = abs(vol_low_th) if abs(vol_low_th) > 1.0 else 1.0

    lo = stress_lo
    if lo < 0.0:
        lo = 0.0
    elif lo > 1.0:
        lo = 1.0

    hi = stress_hi
    if hi < 0.0:
        hi = 0.0
    elif hi > 1.0:
        hi = 1.0

    if hi <= lo:
        hi = min(1.0, lo + 0.05)
        if hi <= lo:
            lo = max(0.0, hi - 0.05)

    ema_k = alpha_ema
    if ema_k < 0.0:
        ema_k = 0.0
    elif ema_k > 1.0:
        ema_k = 1.0

    dead = alpha_hysteresis
    if dead < 0.0:
        dead = 0.0

    denom = hi - lo
    if denom < 1e-6:
        denom = 1e-6

    alpha_raw = np.zeros(n, dtype=np.float64)

    for i in range(n):
        a = atr_arr[i]
        if np.isnan(a) or np.isinf(a):
            a = 0.0

        r = range_arr[i]
        if np.isnan(r) or np.isinf(r):
            r = 0.0

        v = vol_arr[i]
        if np.isnan(v) or np.isinf(v):
            v = vol_neutral

        f = funding_arr[i]
        if np.isnan(f) or np.isinf(f):
            f = funding_soft if funding_soft > 0.0 else 0.0

        atr_stress = 0.0
        if atr_high_enabled:
            atr_stress = (a / max(atr_high_th, eps)) - 1.0
            if atr_stress < 0.0:
                atr_stress = 0.0
            elif atr_stress > 1.0:
                atr_stress = 1.0

        range_stress = 0.0
        if range_cut_enabled:
            range_stress = (r / max(range_cut, eps)) - 1.0
            if range_stress < 0.0:
                range_stress = 0.0
            elif range_stress > 1.0:
                range_stress = 1.0

        vol_deficit = 0.0
        if vol_enabled:
            vol_deficit = (vol_low_th - v)
            if vol_deficit < 0.0:
                vol_deficit = 0.0
            vol_deficit = vol_deficit / vol_norm
            if vol_deficit > 1.0:
                vol_deficit = 1.0

        funding_stress = 0.0
        if funding_soft > 0.0 and f < funding_soft:
            ratio = f / funding_soft
            if ratio < 0.0:
                ratio = 0.0
            elif ratio > 1.0:
                ratio = 1.0
            funding_stress = 1.0 - ratio

        s = wa * atr_stress + wr * range_stress + wv * vol_deficit + wf * funding_stress
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        stress[i] = s

        ar = (s - lo) / denom
        if ar < 0.0:
            ar = 0.0
        elif ar > 1.0:
            ar = 1.0
        alpha_raw[i] = ar

    prev = alpha_raw[0]
    alpha[0] = prev
    for i in range(1, n):
        raw_i = alpha_raw[i]
        ema_val = raw_i if ema_k <= 0.0 else (prev + ema_k * (raw_i - prev))
        if np.isnan(ema_val) or np.isinf(ema_val):
            ema_val = prev
        if ema_val < 0.0:
            ema_val = 0.0
        elif ema_val > 1.0:
            ema_val = 1.0

        if abs(ema_val - prev) < dead:
            alpha[i] = prev
        else:
            alpha[i] = ema_val
        prev = alpha[i]

    calm_cut = 0.5 - dead
    if calm_cut < 0.0:
        calm_cut = 0.0
    elif calm_cut > 1.0:
        calm_cut = 1.0

    active_cut = 0.5 + dead
    if active_cut < 0.0:
        active_cut = 0.0
    elif active_cut > 1.0:
        active_cut = 1.0

    for i in range(n):
        a = alpha[i]
        if a <= calm_cut:
            bucket[i] = 0
        elif a >= active_cut:
            bucket[i] = 2
        else:
            bucket[i] = 1

    return stress, alpha, bucket


def build_regime_alpha_exogenous(
    atr_arr: np.ndarray,
    range_arr: np.ndarray,
    vol_arr: np.ndarray,
    funding_arr: np.ndarray,
    *,
    atr_high_th: float,
    range_cut: float,
    vol_low_th: float,
    funding_soft_min: float,
    stress_lo: float,
    stress_hi: float,
    alpha_ema: float,
    alpha_hysteresis: float,
    w_atr: float,
    w_rng: float,
    w_vol: float,
    w_fund: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    atr_arr = np.asarray(atr_arr, dtype=np.float64)
    range_arr = np.asarray(range_arr, dtype=np.float64)
    vol_arr = np.asarray(vol_arr, dtype=np.float64)
    funding_arr = np.asarray(funding_arr, dtype=np.float64)

    return _build_regime_alpha_exogenous_numba(
        atr_arr,
        range_arr,
        vol_arr,
        funding_arr,
        float(atr_high_th),
        float(range_cut),
        float(vol_low_th),
        float(funding_soft_min),
        float(stress_lo),
        float(stress_hi),
        float(alpha_ema),
        float(alpha_hysteresis),
        float(w_atr),
        float(w_rng),
        float(w_vol),
        float(w_fund),
    )


_RANGE_KEY_ALIASES = {
    # flat canonical names
    "q_top_min": "q_entry_min",
    "q_top_max": "q_entry_max",
    "entry_q_min": "q_entry_min",
    "entry_q_max": "q_entry_max",
    "entry_th_floor_min": "entry_th_min",
    "entry_th_floor_max": "entry_th_max",
    "lev_min": "leverage_min",
    "lev_max": "leverage_max",
    "lev_mult_top_min": "lev_mult_min",
    "lev_mult_top_max": "lev_mult_max",
    "min_hold_top_min": "min_hold_min",
    "min_hold_top_max": "min_hold_max",
    "min_hold_tp_min": "min_hold_min",
    "min_hold_tp_max": "min_hold_max",
    "min_hold_trail_top_min": "min_hold_trail_min",
    "min_hold_trail_top_max": "min_hold_trail_max",
    "min_hold_soft_sl_top_min": "min_hold_soft_sl_min",
    "min_hold_soft_sl_top_max": "min_hold_soft_sl_max",
    "max_hold_top_min": "max_hold_min",
    "max_hold_top_max": "max_hold_max",
    "sl_mult_top_min": "sl_mult_min",
    "sl_mult_top_max": "sl_mult_max",
    "tp_sl_ratio_top_min": "tp_sl_ratio_min",
    "tp_sl_ratio_top_max": "tp_sl_ratio_max",
    "tp_mult_top_min": "tp_mult_min",
    "tp_mult_top_max": "tp_mult_max",
    "bep_ratio_top_min": "bep_ratio_min",
    "bep_ratio_top_max": "bep_ratio_max",
    "trail_ratio_top_min": "trail_ratio_min",
    "trail_ratio_top_max": "trail_ratio_max",
    "fee_bep_mult_min": "bep_arm_fee_mult_min",
    "fee_bep_mult_max": "bep_arm_fee_mult_max",
    "atr_entry_mult_top_min": "atr_entry_mult_min",
    "atr_entry_mult_top_max": "atr_entry_mult_max",
    "range_entry_mult_top_min": "range_entry_mult_min",
    "range_entry_mult_top_max": "range_entry_mult_max",
    "vol_low_th_top_min": "vol_low_th_min",
    "vol_low_th_top_max": "vol_low_th_max",
    "tune_dynamic_top": "tune_dynamic",
    "tune_pre_bep_top": "tune_pre_bep",
    "tune_margin_gate_top": "tune_margin_gate",
    "dw_top_self_min": "dir_self_min",
    "dw_top_self_max": "dir_self_max",
    "dw_mix_min": "dir_mix_min",
    "dw_mix_max": "dir_mix_max",
    # new 5-horizon explicit weights
    "gate_w1_min": "gate_w1_min",
    "gate_w1_max": "gate_w1_max",
    "gate_w3_min": "gate_w3_min",
    "gate_w3_max": "gate_w3_max",
    "gate_w5_min": "gate_w5_min",
    "gate_w5_max": "gate_w5_max",
    "gate_w8_min": "gate_w8_min",
    "gate_w8_max": "gate_w8_max",
    "gate_w10_min": "gate_w10_min",
    "gate_w10_max": "gate_w10_max",
    "dir_w1_min": "dir_w1_min",
    "dir_w1_max": "dir_w1_max",
    "dir_w3_min": "dir_w3_min",
    "dir_w3_max": "dir_w3_max",
    "dir_w5_min": "dir_w5_min",
    "dir_w5_max": "dir_w5_max",
    "dir_w8_min": "dir_w8_min",
    "dir_w8_max": "dir_w8_max",
    "dir_w10_min": "dir_w10_min",
    "dir_w10_max": "dir_w10_max",
    # protection aliases
    "min_hold_tp_min": "min_hold_min",
    "min_hold_tp_max": "min_hold_max",
    "early_softsl_min_hold_min": "early_softsl_min_hold_min",
    "early_softsl_min_hold_max": "early_softsl_min_hold_max",
    "early_softsl_progress_frac_min": "early_softsl_progress_frac_min",
    "early_softsl_progress_frac_max": "early_softsl_progress_frac_max",
    "early_trail_min_hold_min": "early_trail_min_hold_min",
    "early_trail_min_hold_max": "early_trail_min_hold_max",
    "early_trail_progress_frac_min": "early_trail_progress_frac_min",
    "early_trail_progress_frac_max": "early_trail_progress_frac_max",
    "early_trail_ref_updates_min_min": "early_trail_ref_updates_min_min",
    "early_trail_ref_updates_min_max": "early_trail_ref_updates_min_max",
    # some optional gate aliases
    "gw_top_self_min": "gate_self_min",
    "gw_top_self_max": "gate_self_max",
}


def apply_ranges_overrides(raw_ranges: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (raw_ranges or {}).items():
        canon = _RANGE_KEY_ALIASES.get(k, k)
        out[canon] = v
    # convenience defaults
    if "tune_dynamic" not in out and "tune_dynamic_sl" in out:
        out["tune_dynamic"] = 1
    if "tune_gate_weights" not in out:
        out["tune_gate_weights"] = 1 if ("gate_self_min" in out and "gate_self_max" in out) else 0
    if "tune_dir_weights" not in out:
        out["tune_dir_weights"] = 1 if ("dir_self_min" in out and "dir_self_max" in out) else 0
    return out


# ---------------------------------------------------------------------------
# score helpers
# ---------------------------------------------------------------------------

def trade_penalty(trades: int, target: float, band: float, mode: str, barrier_k: float, shortage_pen: float, excess_pen: float) -> float:
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return 0.0
    lo, hi = float(target) - float(band), float(target) + float(band)
    if mode == "hard":
        return 1e6 if (trades < lo or trades > hi) else 0.0
    shortage = max(0.0, lo - float(trades))
    excess = max(0.0, float(trades) - hi)
    if mode == "soft":
        return float(shortage_pen) * shortage + float(excess_pen) * excess
    if float(target) <= 0.0:
        return float(barrier_k) * (shortage ** 2 + excess ** 2)
    return float(barrier_k) * ((shortage / float(target)) ** 2 + (excess / float(target)) ** 2)


def agg_worst(scores: Sequence[float], mode: str, worst_k: int = 2, worst_q: float = 0.2) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = int(s.size)
    if n == 0:
        return float("-inf")
    mode = str(mode or "min").strip().lower()
    if mode == "min":
        return float(s[0])
    if mode == "quantile":
        q = float(np.clip(worst_q, 0.0, 1.0))
        return float(np.quantile(s, q))
    k = int(max(1, min(n, int(worst_k))))
    return float(np.mean(s[:k]))


def side_balance_penalty_component(long_trades: int, short_trades: int, min_short_trades: int, min_short_share: float, penalty_k: float) -> float:
    if float(penalty_k) <= 0.0:
        return 0.0
    total = int(long_trades) + int(short_trades)
    pen = 0.0
    mst = int(max(0, min_short_trades))
    mss = float(max(0.0, min_short_share))
    if mst > 0 and int(short_trades) < mst:
        gap = float(mst - int(short_trades)) / float(max(mst, 1))
        pen += float(penalty_k) * (gap ** 2)
    if mss > 0.0:
        share = (float(short_trades) / float(total)) if total > 0 else 0.0
        if share < mss:
            gap = (mss - share) / max(mss, 1e-9)
            pen += float(penalty_k) * (gap ** 2)
    return float(pen)


def segment_score(
    net_ret: float,
    mdd: float,
    tail_hit: int,
    trades: int,
    maxh_cnt: int,
    long_trades: int,
    short_trades: int,
    alpha_dd: float,
    beta_tail: float,
    trade_mode: str,
    trade_target: float,
    trade_band: float,
    barrier_k: float,
    shortage_penalty: float,
    excess_penalty: float,
    maxhold_ratio_free: float,
    maxhold_penalty_k: float,
    maxhold_penalty_power: float,
    side_balance_penalty_k: float,
    min_short_trades: int,
    min_short_share: float,
) -> float:
    log_ret = float(np.log1p(float(net_ret)))
    pen_dd = float(alpha_dd) * float(mdd)
    pen_tail = float(beta_tail) * int(tail_hit)
    pen_trade = trade_penalty(int(trades), float(trade_target), float(trade_band), str(trade_mode), float(barrier_k), float(shortage_penalty), float(excess_penalty))
    maxh_ratio = (float(maxh_cnt) / float(trades)) if int(trades) > 0 else 0.0
    if float(maxhold_penalty_k) > 0.0 and float(maxhold_ratio_free) < 1.0:
        excess = max(0.0, maxh_ratio - float(maxhold_ratio_free))
        pen_maxh = float(maxhold_penalty_k) * (excess ** float(maxhold_penalty_power))
    else:
        pen_maxh = 0.0
    pen_side = side_balance_penalty_component(int(long_trades), int(short_trades), int(min_short_trades), float(min_short_share), float(side_balance_penalty_k))
    return float(log_ret - pen_dd - pen_tail - pen_trade - pen_maxh - pen_side)


@dataclass
class SegmentMetrics:
    net_ret: float
    mdd: float
    tail: int
    trades: int
    winrate: float
    score: float
    long_trades: int = 0
    short_trades: int = 0
    short_share: float = 0.0
    side_penalty: float = 0.0
    maxh_cnt: int = 0
    maxh_ratio: float = 0.0


# ---------------------------------------------------------------------------
# quantile helpers
# ---------------------------------------------------------------------------

def quantile_from_history(values: np.ndarray, ready: np.ndarray, start: int, end: int, q: float, min_ready: int, floor: float) -> float:
    if int(end) <= int(start):
        return float(floor)
    hist = values[int(start):int(end)]
    vals = hist[ready[int(start):int(end)]]
    vals = vals[np.isfinite(vals)]
    if vals.size < int(min_ready):
        return float(floor)
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(vals, q))


def quantile_from_values(values: np.ndarray, ready_mask: np.ndarray, q: float, min_ready: int, floor: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return float(floor)
    mask = np.asarray(ready_mask, dtype=bool)
    vals = vals[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < int(min_ready):
        return float(floor)
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(vals, q))


# ---------------------------------------------------------------------------
# dynamic policy helpers
# ---------------------------------------------------------------------------

def resolve_local_soft_sl_hold(base_soft_hold: int, trail_hold: int, relax: int, allow_before_trail: int, hold_floor: int) -> int:
    v = int(base_soft_hold) - int(relax)
    floor_eff = int(hold_floor)
    if int(allow_before_trail) == 0:
        floor_eff = max(floor_eff, int(trail_hold))
    if v < floor_eff:
        v = floor_eff
    if v < 0:
        v = 0
    upper_eff = int(base_soft_hold)
    if int(allow_before_trail) == 0 and floor_eff > upper_eff:
        upper_eff = int(floor_eff)
    if v > upper_eff:
        v = upper_eff
    return int(v)


def _normalize_dynamic_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = {
        "enabled": 1 if safe_int(raw.get("enabled", 0), 0) != 0 else 0,
        "mode": str(raw.get("mode", "entry_latched")).strip().lower(),
        "use_dyn_lev": 1 if safe_int(raw.get("use_dyn_lev", 1), 1) != 0 else 0,
        "use_dyn_gate": 1 if safe_int(raw.get("use_dyn_gate", 1), 1) != 0 else 0,
        "use_dyn_bep": 1 if safe_int(raw.get("use_dyn_bep", 1), 1) != 0 else 0,
        "use_dyn_trail": 1 if safe_int(raw.get("use_dyn_trail", 1), 1) != 0 else 0,
        "use_dyn_sl": 1 if safe_int(raw.get("use_dyn_sl", 0), 0) != 0 else 0,
        "use_dyn_soft_sl": 1 if safe_int(raw.get("use_dyn_soft_sl", 0), 0) != 0 else 0,
        "allow_soft_sl_before_trail": 1 if safe_int(raw.get("allow_soft_sl_before_trail", 0), 0) != 0 else 0,
        "softsl_hold_floor": max(0, safe_int(raw.get("softsl_hold_floor", 0), 0)),
        "post_bep_shield_ignore_softsl_hold": 1 if safe_int(raw.get("post_bep_shield_ignore_softsl_hold", 0), 0) != 0 else 0,
        "funding_soft_min": max(0.0, safe_float(raw.get("funding_soft_min", 0.0), 0.0)),
        "margin_cap": max(1e-6, safe_float(raw.get("margin_cap", 0.50), 0.50)),
        "lev_scale_min": safe_float(raw.get("lev_scale_min", 0.70), 0.70),
        "lev_scale_max": safe_float(raw.get("lev_scale_max", 1.05), 1.05),
        "gate_mult_min": safe_float(raw.get("gate_mult_min", 0.95), 0.95),
        "gate_mult_max": safe_float(raw.get("gate_mult_max", 1.15), 1.15),
        "bep_scale_min": safe_float(raw.get("bep_scale_min", 0.75), 0.75),
        "bep_scale_max": safe_float(raw.get("bep_scale_max", 1.05), 1.05),
        "trail_scale_min": safe_float(raw.get("trail_scale_min", 0.90), 0.90),
        "trail_scale_max": safe_float(raw.get("trail_scale_max", 1.12), 1.12),
        "sl_scale_min": safe_float(raw.get("sl_scale_min", 0.85), 0.85),
        "sl_scale_max": safe_float(raw.get("sl_scale_max", 1.05), 1.05),
        "lev_trend_k": safe_float(raw.get("lev_trend_k", 0.18), 0.18),
        "lev_stress_k": safe_float(raw.get("lev_stress_k", 0.30), 0.30),
        "gate_stress_k": safe_float(raw.get("gate_stress_k", 0.12), 0.12),
        "gate_trend_k": safe_float(raw.get("gate_trend_k", 0.08), 0.08),
        # signed ranges allowed
        "bep_stress_k": safe_float(raw.get("bep_stress_k", 0.0), 0.0),
        "bep_trend_k": safe_float(raw.get("bep_trend_k", 0.05), 0.05),
        "trail_trend_k": safe_float(raw.get("trail_trend_k", 0.15), 0.15),
        "trail_stress_k": safe_float(raw.get("trail_stress_k", 0.0), 0.0),
        "sl_trend_k": safe_float(raw.get("sl_trend_k", 0.0), 0.0),
        "sl_stress_k": safe_float(raw.get("sl_stress_k", 0.0), 0.0),
        "softsl_stress_mid": safe_float(raw.get("softsl_stress_mid", 0.35), 0.35),
        "softsl_stress_hi": safe_float(raw.get("softsl_stress_hi", 0.65), 0.65),
        "softsl_relax_mid": max(0, safe_int(raw.get("softsl_relax_mid", 1), 1)),
        "softsl_relax_hi": max(0, safe_int(raw.get("softsl_relax_hi", 2), 2)),
        "use_margin_gate": 1 if safe_int(raw.get("use_margin_gate", 0), 0) != 0 else 0,
        "margin_req_base": max(0.0, safe_float(raw.get("margin_req_base", 0.0), 0.0)),
        "margin_req_stress_k": safe_float(raw.get("margin_req_stress_k", 0.0), 0.0),
        "margin_req_trend_k": safe_float(raw.get("margin_req_trend_k", 0.0), 0.0),
        "margin_req_max": max(0.0, safe_float(raw.get("margin_req_max", 0.0), 0.0)),
        "use_margin_lev_degrade": 1 if safe_int(raw.get("use_margin_lev_degrade", 0), 0) != 0 else 0,
        "margin_lev_floor": safe_float(raw.get("margin_lev_floor", 0.70), 0.70),
        "margin_lev_band": max(1e-6, safe_float(raw.get("margin_lev_band", 0.05), 0.05)),
        "use_pre_bep_timeout": 1 if safe_int(raw.get("use_pre_bep_timeout", 0), 0) != 0 else 0,
        "pre_bep_timeout_bars": max(0, safe_int(raw.get("pre_bep_timeout_bars", 3), 3)),
        "pre_bep_stress_th": safe_float(raw.get("pre_bep_stress_th", 0.55), 0.55),
        "pre_bep_progress_frac": safe_float(raw.get("pre_bep_progress_frac", 0.55), 0.55),
        "pre_bep_degrade_sl_scale": safe_float(raw.get("pre_bep_degrade_sl_scale", 0.85), 0.85),
        # delta (new canonical). old absolute field is migrated in normalize_single_config_from_any.
        "pre_bep_softsl_delta": max(0, safe_int(raw.get("pre_bep_softsl_delta", 0), 0)),
        "pre_bep_force_close_bars": max(0, safe_int(raw.get("pre_bep_force_close_bars", 0), 0)),
        "pre_bep_force_close_red_only": 1 if safe_int(raw.get("pre_bep_force_close_red_only", 1), 1) != 0 else 0,
        "w_atr": max(0.0, safe_float(raw.get("w_atr", raw.get("w_atr_raw", 0.35)), 0.35)),
        "w_rng": max(0.0, safe_float(raw.get("w_rng", raw.get("w_rng_raw", 0.20)), 0.20)),
        "w_vol": max(0.0, safe_float(raw.get("w_vol", raw.get("w_vol_raw", 0.30)), 0.30)),
        "w_fund": max(0.0, safe_float(raw.get("w_fund", raw.get("w_fund_raw", 0.15)), 0.15)),
    }
    if out["mode"] not in ("entry_latched", "exit_path_adaptive"):
        out["mode"] = "entry_latched"
    for lo_key, hi_key, lo_def, hi_def in [
        ("lev_scale_min", "lev_scale_max", 0.70, 1.05),
        ("gate_mult_min", "gate_mult_max", 0.95, 1.15),
        ("bep_scale_min", "bep_scale_max", 0.75, 1.05),
        ("trail_scale_min", "trail_scale_max", 0.90, 1.12),
        ("sl_scale_min", "sl_scale_max", 0.85, 1.05),
    ]:
        lo = safe_float(out.get(lo_key, lo_def), lo_def)
        hi = safe_float(out.get(hi_key, hi_def), hi_def)
        if hi < lo:
            lo, hi = hi, lo
        out[lo_key], out[hi_key] = float(lo), float(hi)
    mid = float(np.clip(out["softsl_stress_mid"], 0.0, 1.0))
    hi = float(np.clip(out["softsl_stress_hi"], 0.0, 1.0))
    if hi < mid:
        hi = mid
    out["softsl_stress_mid"] = mid
    out["softsl_stress_hi"] = hi
    if out["softsl_relax_hi"] < out["softsl_relax_mid"]:
        out["softsl_relax_hi"] = int(out["softsl_relax_mid"])
    ws = np.array([out["w_atr"], out["w_rng"], out["w_vol"], out["w_fund"]], dtype=np.float64)
    s = float(ws.sum())
    if s <= 0.0:
        ws = np.array([0.35, 0.20, 0.30, 0.15], dtype=np.float64)
        s = float(ws.sum())
    ws /= s
    out["w_atr"], out["w_rng"], out["w_vol"], out["w_fund"] = [float(x) for x in ws]
    return out


DEFAULT_SINGLE_CFG: Dict[str, Any] = {
    "schema": "single_v70",
    "entry_th": 0.0,
    "entry_th_floor": 0.0,
    "q_entry": 0.85,
    "leverage": 10.0,
    "lev_mult": 1.0,
    "gate_weights": {"w1": 0.0, "w3": 0.0, "w5": 1.0, "w8": 0.0, "w10": 0.0},
    "dir_weights": {"w1": 0.0, "w3": 0.0, "w5": 1.0, "w8": 0.0, "w10": 0.0},
    "TP": 0.01,
    "SL": 0.005,
    "BEP_ARM": 0.001,
    "trailing": 0.001,
    "fee_tp_mult": 1.0,
    "bep_arm_fee_mult": 0.2,
    "bep_stop_fee_mult": 1.0,
    "bep_stop_mode": "maker_be",
    "atr_entry_mult": 1.0,
    "range_entry_mult": 1.0,
    "low_vol_filter": 0,
    "trail_after_bep": 1,
    "risk_entry_mode": 0,
    "use_atr_scaling": 1,
    "min_hold_tp_bars": 6,
    "min_hold_bars": 6,
    "min_hold_trail_bars": 8,
    "min_hold_soft_sl_bars": 6,
    "max_hold_bars": 32,
    "integer_leverage": 0,
    "hard_sl_mult_pre_unlock": 1.0,
    "trail_grace_after_bep": 0,
    "trail_grace_after_unlock": 0,
    "runtime_feature_cfg": {
        "vol_feature": "vol_z_60",
        "atr_feature": "atr10_rel",
    },
    "progress_protect_cfg": {
        "early_softsl_enabled": 0,
        "early_softsl_min_hold": 2,
        "early_softsl_progress_frac": 0.50,
        "early_trail_enabled": 0,
        "early_trail_min_hold": 3,
        "early_trail_progress_frac": 0.85,
        "early_trail_ref_updates_min": 1,
    },
    "risk_cfg": {
        "atr_high_th": float("nan"),
        "atr_percentile": 75.0,
        "funding_near_min": 0.0,
        "risk_lev_cap": 12.0,
        "vol_low_th": -1e9,
    },
    "dynamic_cfg": {
        "enabled": 0,
        "mode": "entry_latched",
        "use_dyn_lev": 1,
        "use_dyn_gate": 1,
        "use_dyn_bep": 1,
        "use_dyn_trail": 1,
        "use_dyn_sl": 0,
        "use_dyn_soft_sl": 0,
        "allow_soft_sl_before_trail": 0,
        "softsl_hold_floor": 0,
        "post_bep_shield_ignore_softsl_hold": 0,
        "funding_soft_min": 0.0,
        "margin_cap": 0.50,
        "lev_scale_min": 0.70,
        "lev_scale_max": 1.05,
        "gate_mult_min": 0.95,
        "gate_mult_max": 1.15,
        "bep_scale_min": 0.75,
        "bep_scale_max": 1.05,
        "trail_scale_min": 0.90,
        "trail_scale_max": 1.12,
        "sl_scale_min": 0.85,
        "sl_scale_max": 1.05,
        "lev_trend_k": 0.18,
        "lev_stress_k": 0.30,
        "gate_stress_k": 0.12,
        "gate_trend_k": 0.08,
        "bep_stress_k": 0.0,
        "bep_trend_k": 0.05,
        "trail_trend_k": 0.15,
        "trail_stress_k": 0.0,
        "sl_trend_k": 0.0,
        "sl_stress_k": 0.0,
        "softsl_stress_mid": 0.35,
        "softsl_stress_hi": 0.65,
        "softsl_relax_mid": 1,
        "softsl_relax_hi": 2,
        "use_margin_gate": 0,
        "margin_req_base": 0.0,
        "margin_req_stress_k": 0.0,
        "margin_req_trend_k": 0.0,
        "margin_req_max": 0.0,
        "use_margin_lev_degrade": 0,
        "margin_lev_floor": 0.70,
        "margin_lev_band": 0.05,
        "use_pre_bep_timeout": 0,
        "pre_bep_timeout_bars": 3,
        "pre_bep_stress_th": 0.55,
        "pre_bep_progress_frac": 0.55,
        "pre_bep_degrade_sl_scale": 0.85,
        "pre_bep_softsl_delta": 0,
        "pre_bep_force_close_bars": 0,
        "pre_bep_force_close_red_only": 1,
        "w_atr": 0.35,
        "w_rng": 0.20,
        "w_vol": 0.30,
        "w_fund": 0.15,
    },
    "tail_cfg": {
        "stop_equity": 0.40,
        "stop_dd": 0.35,
        "warmup_steps": 0,
    },
    "tuned_meta": {},
}


def _legacy_top_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    tier_params = raw.get("tier_params", {}) if isinstance(raw.get("tier_params"), dict) else {}
    top = tier_params.get("top") or tier_params.get("TOP") or {}
    if not isinstance(top, dict):
        top = {}
    tier_q = raw.get("tier_q", {}) if isinstance(raw.get("tier_q"), dict) else {}
    risk_cfg = raw.get("risk_cfg", {}) if isinstance(raw.get("risk_cfg"), dict) else {}
    dyn_tier_cfg = raw.get("dynamic_tier_cfg", {}) if isinstance(raw.get("dynamic_tier_cfg"), dict) else {}
    dyn_top_cfg = raw.get("dynamic_top_cfg", {}) if isinstance(raw.get("dynamic_top_cfg"), dict) else {}
    dyn_raw = dyn_tier_cfg.get("top", dyn_top_cfg) if isinstance(dyn_tier_cfg, dict) else dyn_top_cfg
    dir_weights_raw = raw.get("dir_weights", {}) if isinstance(raw.get("dir_weights"), dict) else {}
    top_dir_raw = dir_weights_raw.get("top", dir_weights_raw)
    atr_entry = raw.get("atr_entry_mult_tier", raw.get("atr_entry_mult", None))
    range_entry = raw.get("range_entry_mult_tier", raw.get("range_entry_mult", None))
    if isinstance(atr_entry, dict):
        atr_entry = atr_entry.get("top", atr_entry.get("TOP", 1.0))
    if isinstance(range_entry, dict):
        range_entry = range_entry.get("top", range_entry.get("TOP", 1.0))
    vol_low = risk_cfg.get("vol_low_th_tier", risk_cfg.get("vol_low_th", None))
    if isinstance(vol_low, dict):
        vol_low = vol_low.get("top", vol_low.get("TOP", -1e9))
    return {
        "q_entry": raw.get("entry_q", tier_q.get("top", tier_q.get("TOP", 0.85))),
        "entry_th": raw.get("entry_th", 0.0),
        "entry_th_floor": raw.get("entry_th_floor", raw.get("entry_th", 0.0)),
        "leverage": raw.get("leverage", 10.0),
        "lev_mult": top.get("lev_mult", 1.0),
        "gate_weights": raw.get("gate_weights", {"w1": 0.0, "w3": 0.0, "w5": 1.0}),
        "dir_weights": top_dir_raw if isinstance(top_dir_raw, dict) else {"w1": raw.get("w1", 0.0), "w3": raw.get("w3", 0.0), "w5": raw.get("w5", 1.0)},
        "TP": top.get("TP", raw.get("TP", 0.01)),
        "SL": top.get("SL", raw.get("SL", 0.005)),
        "BEP_ARM": top.get("BEP", raw.get("BEP", raw.get("BEP_ARM", 0.001))),
        "trailing": top.get("trailing", raw.get("trailing", 0.001)),
        "fee_tp_mult": raw.get("fee_tp_mult", 1.0),
        "bep_arm_fee_mult": raw.get("bep_arm_fee_mult", raw.get("fee_bep_mult", 0.2)),
        "bep_stop_fee_mult": raw.get("bep_stop_fee_mult", 1.0),
        "bep_stop_mode": raw.get("bep_stop_mode", "maker_be"),
        "atr_entry_mult": atr_entry if atr_entry is not None else 1.0,
        "range_entry_mult": range_entry if range_entry is not None else 1.0,
        "low_vol_filter": raw.get("low_vol_filter", 0),
        "trail_after_bep": raw.get("trail_after_bep", 1),
        "risk_entry_mode": raw.get("risk_entry_mode", 0),
        "use_atr_scaling": raw.get("use_atr_scaling", 1),
        "min_hold_bars": top.get("min_hold_bars", raw.get("min_hold_bars", 6)),
        "min_hold_trail_bars": top.get("min_hold_trail_bars", top.get("min_hold_bars", raw.get("min_hold_bars", 6))),
        "min_hold_soft_sl_bars": top.get("min_hold_soft_sl_bars", top.get("min_hold_bars", raw.get("min_hold_bars", 6))),
        "max_hold_bars": top.get("max_hold_bars", raw.get("max_hold_bars", 32)),
        "integer_leverage": raw.get("integer_leverage", 0),
        "hard_sl_mult_pre_unlock": raw.get("hard_sl_mult_pre_unlock", 1.0),
        "trail_grace_after_bep": raw.get("trail_grace_after_bep", 0),
        "trail_grace_after_unlock": raw.get("trail_grace_after_unlock", 0),
        "risk_cfg": {
            "atr_high_th": risk_cfg.get("atr_high_th", float("nan")),
            "atr_percentile": risk_cfg.get("atr_percentile", 75.0),
            "funding_near_min": risk_cfg.get("funding_near_min", 0.0),
            "risk_lev_cap": risk_cfg.get("risk_lev_cap", raw.get("risk_lev_cap", 12.0)),
            "vol_low_th": vol_low if vol_low is not None else -1e9,
        },
        "dynamic_cfg": dyn_raw,
        "tail_cfg": raw.get("tail_cfg", {}),
        "tuned_meta": dict(raw.get("tuned_meta", {})),
        "cost_per_side": raw.get("cost_per_side", raw.get("taker_fee_per_side", 0.0005)),
        "slip_per_side": raw.get("slip_per_side", 0.0),
        "maker_fee_per_side": raw.get("maker_fee_per_side", raw.get("cost_per_side", raw.get("taker_fee_per_side", 0.0005))),
    }


def normalize_single_config_from_any(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = copy.deepcopy(raw_cfg or {})
    cfg = copy.deepcopy(DEFAULT_SINGLE_CFG)
    is_legacy_tiered = isinstance(raw_cfg.get("tier_params"), dict)
    if is_legacy_tiered:
        merged = _legacy_top_block(raw_cfg)
    else:
        merged = dict(raw_cfg)

    cfg = _deep_merge(cfg, merged)
    # flat aliases
    if "entry_q" in merged and "q_entry" not in merged:
        cfg["q_entry"] = safe_float(merged.get("entry_q", cfg["q_entry"]), cfg["q_entry"])
    if "fee_bep_mult" in merged and "bep_arm_fee_mult" not in merged:
        cfg["bep_arm_fee_mult"] = safe_float(merged.get("fee_bep_mult", cfg["bep_arm_fee_mult"]), cfg["bep_arm_fee_mult"])
    if "BEP" in merged and "BEP_ARM" not in merged:
        cfg["BEP_ARM"] = safe_float(merged.get("BEP", cfg["BEP_ARM"]), cfg["BEP_ARM"])
    if "min_hold_bars" in merged and "min_hold_tp_bars" not in merged:
        cfg["min_hold_tp_bars"] = safe_int(merged.get("min_hold_bars", cfg.get("min_hold_tp_bars", 6)), cfg.get("min_hold_tp_bars", 6))

    cfg["gate_weights"] = normalize_horizon_weights(cfg.get("gate_weights", None), fallback={1: 0.0, 3: 0.0, 5: 1.0, 8: 0.0, 10: 0.0})
    dir_fallback = cfg.get("dir_weights", {"w1": 0.0, "w3": 0.0, "w5": 1.0, "w8": 0.0, "w10": 0.0})
    if not isinstance(dir_fallback, dict) or not any(k in dir_fallback for k in ("w1", "w3", "w5", "w8", "w10")):
        dir_fallback = {
            "w1": safe_float(raw_cfg.get("w1", 0.0), 0.0),
            "w3": safe_float(raw_cfg.get("w3", 0.0), 0.0),
            "w5": safe_float(raw_cfg.get("w5", 1.0), 1.0),
            "w8": safe_float(raw_cfg.get("w8", 0.0), 0.0),
            "w10": safe_float(raw_cfg.get("w10", 0.0), 0.0),
        }
    cfg["dir_weights"] = normalize_horizon_weights(dir_fallback, fallback={1: 0.0, 3: 0.0, 5: 1.0, 8: 0.0, 10: 0.0})

    cfg["q_entry"] = float(np.clip(safe_float(cfg.get("q_entry", 0.85), 0.85), 0.0, 1.0))
    cfg["entry_th"] = safe_float(cfg.get("entry_th", 0.0), 0.0)
    cfg["entry_th_floor"] = safe_float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)), 0.0)
    cfg["leverage"] = max(1.0, safe_float(cfg.get("leverage", 10.0), 10.0))
    cfg["lev_mult"] = max(0.01, safe_float(cfg.get("lev_mult", 1.0), 1.0))
    cfg["TP"] = max(0.0, safe_float(cfg.get("TP", 0.01), 0.01))
    cfg["SL"] = max(0.0, safe_float(cfg.get("SL", 0.005), 0.005))
    cfg["BEP_ARM"] = max(0.0, safe_float(cfg.get("BEP_ARM", 0.001), 0.001))
    cfg["trailing"] = max(0.0, safe_float(cfg.get("trailing", 0.001), 0.001))
    cfg["fee_tp_mult"] = max(0.0, safe_float(cfg.get("fee_tp_mult", 1.0), 1.0))
    cfg["bep_arm_fee_mult"] = max(0.0, safe_float(cfg.get("bep_arm_fee_mult", 0.2), 0.2))
    cfg["bep_stop_fee_mult"] = max(0.0, safe_float(cfg.get("bep_stop_fee_mult", 1.0), 1.0))
    cfg["bep_stop_mode"] = str(cfg.get("bep_stop_mode", "maker_be") or "maker_be").strip().lower()
    if cfg["bep_stop_mode"] not in ("maker_be", "taker_be", "scaled"):
        cfg["bep_stop_mode"] = "maker_be"
    cfg["atr_entry_mult"] = max(0.0, safe_float(cfg.get("atr_entry_mult", 1.0), 1.0))
    cfg["range_entry_mult"] = max(0.0, safe_float(cfg.get("range_entry_mult", 1.0), 1.0))
    cfg["low_vol_filter"] = 1 if safe_int(cfg.get("low_vol_filter", 0), 0) != 0 else 0
    cfg["trail_after_bep"] = 1 if safe_int(cfg.get("trail_after_bep", 1), 1) != 0 else 0
    cfg["risk_entry_mode"] = safe_int(cfg.get("risk_entry_mode", 0), 0)
    cfg["use_atr_scaling"] = 1 if safe_int(cfg.get("use_atr_scaling", 1), 1) != 0 else 0
    cfg["min_hold_tp_bars"] = max(0, safe_int(cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 6)), 6))
    cfg["min_hold_bars"] = int(cfg["min_hold_tp_bars"])  # legacy alias
    cfg["min_hold_trail_bars"] = max(0, safe_int(cfg.get("min_hold_trail_bars", cfg["min_hold_tp_bars"]), cfg["min_hold_tp_bars"]))
    cfg["min_hold_soft_sl_bars"] = max(0, safe_int(cfg.get("min_hold_soft_sl_bars", cfg["min_hold_tp_bars"]), cfg["min_hold_tp_bars"]))
    cfg["max_hold_bars"] = max(0, safe_int(cfg.get("max_hold_bars", 32), 32))
    for _k in ("min_hold_tp_bars", "min_hold_trail_bars", "min_hold_soft_sl_bars"):
        if cfg["max_hold_bars"] < cfg[_k]:
            cfg["max_hold_bars"] = int(cfg[_k])
    cfg["integer_leverage"] = 1 if safe_int(cfg.get("integer_leverage", 0), 0) != 0 else 0
    cfg["hard_sl_mult_pre_unlock"] = max(0.0, safe_float(cfg.get("hard_sl_mult_pre_unlock", 1.0), 1.0))
    cfg["trail_grace_after_bep"] = max(0, safe_int(cfg.get("trail_grace_after_bep", 0), 0))
    cfg["trail_grace_after_unlock"] = max(0, safe_int(cfg.get("trail_grace_after_unlock", 0), 0))

    risk_cfg = dict(cfg.get("risk_cfg", {}))
    risk_cfg["atr_high_th"] = risk_cfg.get("atr_high_th", float("nan"))
    risk_cfg["atr_percentile"] = safe_float(risk_cfg.get("atr_percentile", 75.0), 75.0)
    risk_cfg["funding_near_min"] = max(0.0, safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0))
    risk_cfg["risk_lev_cap"] = max(1.0, safe_float(risk_cfg.get("risk_lev_cap", 12.0), 12.0))
    risk_cfg["vol_low_th"] = safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9)
    cfg["risk_cfg"] = risk_cfg

    runtime_cfg = dict(cfg.get("runtime_feature_cfg", {}))
    runtime_cfg["vol_feature"] = str(runtime_cfg.get("vol_feature", "vol_z_60") or "vol_z_60")
    runtime_cfg["atr_feature"] = str(runtime_cfg.get("atr_feature", "atr10_rel") or "atr10_rel")
    cfg["runtime_feature_cfg"] = runtime_cfg

    regime_cfg = _normalize_regime_weight_cfg(
        cfg.get("regime_weight_cfg", {}),
        base_gate_weights=cfg.get("gate_weights", {"w5": 1.0}),
        base_dir_weights=cfg.get("dir_weights", {"w5": 1.0}),
    )
    cfg["regime_weight_cfg"] = regime_cfg

    dyn_cfg = dict(cfg.get("dynamic_cfg", {}))
    if "pre_bep_softsl_delta" not in dyn_cfg and "pre_bep_degrade_softsl_hold" in dyn_cfg:
        abs_hold = max(0, safe_int(dyn_cfg.get("pre_bep_degrade_softsl_hold", cfg["min_hold_soft_sl_bars"]), cfg["min_hold_soft_sl_bars"]))
        dyn_cfg["pre_bep_softsl_delta"] = max(0, int(cfg["min_hold_soft_sl_bars"]) - abs_hold)
    cfg["dynamic_cfg"] = _normalize_dynamic_cfg(dyn_cfg)

    prog_cfg = dict(cfg.get("progress_protect_cfg", {}))
    for k, default in {
        "early_softsl_enabled": 0,
        "early_softsl_min_hold": 2,
        "early_softsl_progress_frac": 0.50,
        "early_trail_enabled": 0,
        "early_trail_min_hold": 3,
        "early_trail_progress_frac": 0.85,
        "early_trail_ref_updates_min": 1,
    }.items():
        if k not in prog_cfg and k in cfg:
            prog_cfg[k] = cfg[k]
        prog_cfg[k] = prog_cfg.get(k, default)
    prog_cfg["early_softsl_enabled"] = 1 if safe_int(prog_cfg.get("early_softsl_enabled", 0), 0) != 0 else 0
    prog_cfg["early_softsl_min_hold"] = max(0, safe_int(prog_cfg.get("early_softsl_min_hold", 2), 2))
    prog_cfg["early_softsl_progress_frac"] = max(0.0, safe_float(prog_cfg.get("early_softsl_progress_frac", 0.50), 0.50))
    prog_cfg["early_trail_enabled"] = 1 if safe_int(prog_cfg.get("early_trail_enabled", 0), 0) != 0 else 0
    prog_cfg["early_trail_min_hold"] = max(0, safe_int(prog_cfg.get("early_trail_min_hold", 3), 3))
    prog_cfg["early_trail_progress_frac"] = max(0.0, safe_float(prog_cfg.get("early_trail_progress_frac", 0.85), 0.85))
    prog_cfg["early_trail_ref_updates_min"] = max(0, safe_int(prog_cfg.get("early_trail_ref_updates_min", 1), 1))
    cfg["progress_protect_cfg"] = prog_cfg

    tail_cfg = dict(cfg.get("tail_cfg", {}))
    tail_cfg["stop_equity"] = safe_float(tail_cfg.get("stop_equity", 0.40), 0.40)
    tail_cfg["stop_dd"] = safe_float(tail_cfg.get("stop_dd", 0.35), 0.35)
    tail_cfg["warmup_steps"] = max(0, safe_int(tail_cfg.get("warmup_steps", 0), 0))
    cfg["tail_cfg"] = tail_cfg

    # keep a few flat aliases for readability/backward inspection
    cfg["entry_q"] = float(cfg["q_entry"])
    cfg["BEP"] = float(cfg["BEP_ARM"])
    return cfg



def _file_stamp(path: str) -> Dict[str, Any]:
    name = os.path.basename(path)
    try:
        st = os.stat(path)
        return {"name": name, "mtime": int(st.st_mtime), "size": int(st.st_size)}
    except FileNotFoundError:
        return {"name": name, "mtime": None, "size": None}


def _make_model_paths(models_dir: str) -> Dict[str, str]:
    names = {
        "scaler": "scaler_ethusdt.json",
        "model": "model_hybrid_mr_tcnlstm_multihead_ethusdt.pt",
    }
    paths = {k: os.path.join(models_dir, v) for k, v in names.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {k}: {p}")
    return paths


def precompute_hybrids(
    df_window: pd.DataFrame,
    seq_len: int,
    models_dir: Optional[str] = None,
    cache_npz: str = "",
    batch_size: int = 2048,
    use_amp: int = 1,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    paths = _make_model_paths(str(models_dir))
    infer = HybridScalpInference(models_dir=models_dir, device=None)
    infer.load()
    infer.reset()

    try:
        y_scale = float(getattr(infer, "y_scale", 1.0) or 1.0)
    except Exception:
        y_scale = 1.0
    reg_target_mode = str(getattr(infer, "reg_target_mode", "signed_legacy") or "signed_legacy").strip().lower()
    infer_horizons = [int(h) for h in sorted(getattr(infer, "horizons", [1, 3, 5]))]

    def _decode_reg_tensor(reg_raw_t: torch.Tensor) -> torch.Tensor:
        if hasattr(infer, "_decode_reg_output"):
            return infer._decode_reg_output(reg_raw_t)
        reg_eff_t = reg_raw_t
        if reg_target_mode == "magnitude_v2":
            reg_eff_t = torch.nn.functional.softplus(reg_eff_t)
        if y_scale != 1.0:
            reg_eff_t = reg_eff_t / float(y_scale)
        return reg_eff_t

    def _hybrid_from_tensors(reg_raw_t: torch.Tensor, logit_t: torch.Tensor) -> torch.Tensor:
        if hasattr(infer, "_hybrid_from_logits"):
            return infer._hybrid_from_logits(reg_raw_t, logit_t)
        return _decode_reg_tensor(reg_raw_t) * torch.tanh(logit_t / 2.0)

    feat_cols = list(getattr(infer, "features", None) or getattr(infer, "FEATURES", []))
    miss = [c for c in feat_cols if c not in df_window.columns]
    if miss:
        raise KeyError(f"missing feature cols: {miss[:8]} ... ({len(miss)} total)")

    time_col = "timestamp" if "timestamp" in df_window.columns else "time"
    feat_hash = hashlib.sha1(",".join(feat_cols).encode("utf-8")).hexdigest()[:16]
    data_hash = hashlib.sha1(pd.util.hash_pandas_object(df_window[[time_col]], index=False).values.tobytes()).hexdigest()[:16]
    stamp = {
        "meta_version": 7,
        "seq_len": int(seq_len),
        "n_rows": int(len(df_window)),
        "feat_hash": feat_hash,
        "data_hash": data_hash,
        "y_scale": float(y_scale),
        "reg_target_mode": str(reg_target_mode),
        "horizons": infer_horizons,
        "files": {k: _file_stamp(p) for k, p in paths.items()},
    }

    if cache_npz and os.path.exists(cache_npz):
        try:
            npz = np.load(cache_npz, allow_pickle=True)
            meta = json.loads(npz["meta"].tobytes().decode("utf-8"))
            cached_h = [int(h) for h in meta.get("horizons", [])]
            ok = (
                int(meta.get("meta_version", 0)) >= int(stamp["meta_version"])
                and int(meta.get("seq_len", -1)) == int(stamp["seq_len"])
                and int(meta.get("n_rows", -1)) == int(stamp["n_rows"])
                and meta.get("feat_hash", "") == stamp["feat_hash"]
                and meta.get("data_hash", "") == stamp["data_hash"]
                and float(meta.get("y_scale", 1.0) or 1.0) == float(stamp["y_scale"])
                and str(meta.get("reg_target_mode", "signed_legacy")) == str(stamp["reg_target_mode"])
                and meta.get("files", {}) == stamp["files"]
                and "ready" in npz.files and "meta" in npz.files
            )
            if ok:
                signals = {h: np.zeros(int(stamp["n_rows"]), dtype=np.float32) for h in HORIZONS_ALL}
                for h in cached_h:
                    k = f"h{int(h)}"
                    if k in npz.files:
                        signals[int(h)] = npz[k].astype(np.float32, copy=False)
                return signals, npz["ready"].astype(bool, copy=False)
        except Exception as e:
            print("[CACHE] load failed -> recompute:", e)

    n = len(df_window)
    signals = {h: np.zeros(n, dtype=np.float32) for h in HORIZONS_ALL}
    ready = np.zeros(n, dtype=bool)
    if n < int(seq_len):
        return signals, ready

    X = df_window[feat_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = (X - infer.scaler_mean) / infer.scaler_std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    device = infer.device
    x = torch.from_numpy(Xs).to(device)
    win = x.unfold(0, int(seq_len), 1).transpose(1, 2)
    n_seq = int(win.shape[0])
    bs = int(batch_size) if batch_size else 2048
    amp_enabled = bool(use_amp) and (device.type == "cuda")

    ready[int(seq_len) - 1 :] = True
    if time_col:
        ts = pd.to_datetime(df_window[time_col], utc=True).astype("int64").to_numpy() // 10**9
        dt_mins = np.diff(ts) / 60.0
        gap_indices = np.where(dt_mins > 1.1)[0]
        shut_cnt = 0
        for gi in gap_indices:
            start_wipe = gi + 1
            end_wipe = min(n, gi + 1 + int(seq_len))
            ready[start_wipe:end_wipe] = False
            shut_cnt += 1
        if shut_cnt > 0:
            print(f"[SHUTDOWN GUARD] Detected {shut_cnt} time gaps. Applied {seq_len}-min cooldown.")

    horizon_to_pair = {1: (0, 1), 3: (2, 3), 5: (4, 5), 8: (6, 7), 10: (8, 9)}
    raw_cache = {h: {"reg": np.full(n, np.nan, dtype=np.float32), "logit": np.full(n, np.nan, dtype=np.float32)} for h in infer_horizons}

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            for s in range(0, n_seq, bs):
                xb = win[s : s + bs].contiguous()
                pred = infer.model(xb)
                i0 = (int(seq_len) - 1) + s
                # all heads decoded generically
                for h in infer_horizons:
                    pr, pl = horizon_to_pair[int(h)]
                    reg_t = _decode_reg_tensor(pred[pr])
                    logit_t = pred[pl]
                    hy_t = _hybrid_from_tensors(pred[pr], pred[pl])
                    reg_np = reg_t.detach().float().cpu().numpy()
                    logit_np = logit_t.detach().float().cpu().numpy()
                    hy_np = hy_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
                    i1 = i0 + reg_np.shape[0]
                    raw_cache[int(h)]["reg"][i0:i1] = reg_np
                    raw_cache[int(h)]["logit"][i0:i1] = logit_np
                    signals[int(h)][i0:i1] = hy_np

    try:
        m = ready.astype(bool)
        if np.any(m):
            dbg_parts = []
            for h in infer_horizons:
                logit_np = raw_cache[int(h)]["logit"]
                reg_np = raw_cache[int(h)]["reg"]
                s_all = np.tanh(logit_np / 2.0)
                tiny = 1e-6
                def _safe_mean(x: np.ndarray) -> float:
                    x = x[np.isfinite(x)]
                    return float(np.mean(x)) if x.size else float("nan")
                def _safe_std(x: np.ndarray) -> float:
                    x = x[np.isfinite(x)]
                    return float(np.std(x)) if x.size else float("nan")
                dbg_parts.append(f"h{h}:std={_safe_std(signals[int(h)][m]):.3e},tanh_small={100*_safe_mean((np.abs(s_all[m]) < 0.05).astype(np.float32)):.1f}%,reg_tiny={100*_safe_mean((np.abs(reg_np[m]) < tiny).astype(np.float32)):.1f}%")
            print("[HYBRID DEBUG] y_scale=%g %s" % (y_scale, " | ".join(dbg_parts)))
    except Exception as e:
        print("[HYBRID DEBUG] failed:", e)

    if cache_npz:
        meta = json.dumps(stamp).encode("utf-8")
        payload = {"ready": ready, "meta": meta}
        for h in infer_horizons:
            payload[f"h{int(h)}"] = signals[int(h)]
        np.savez_compressed(cache_npz, **payload)
    return signals, ready

# ---------------------------------------------------------------------------
# shared single-position simulation core
# ---------------------------------------------------------------------------

EXIT_TP = 0
EXIT_SL = 1
EXIT_TRAIL = 2
EXIT_MAXH = 3
EXIT_FORCE = 4
EXIT_RISK = 5
EXIT_NAMES = ["TP", "SL", "TRAIL", "MAX_HOLD", "FORCE_CLOSE", "RISK_CLOSE"]


def _bep_econ_fee(taker_fee_side: float, maker_fee_per_side: float, mode: str) -> float:
    mode = str(mode or "maker_be").strip().lower()
    if mode == "taker_be":
        return float(taker_fee_side) + float(taker_fee_side)
    # "scaled" currently uses the same reference and lets bep_stop_fee_mult do scaling.
    maker_exit = float(maker_fee_per_side) if float(maker_fee_per_side) > 0.0 else float(taker_fee_side)
    return float(taker_fee_side) + maker_exit


def simulate_trading_core_rl_single(
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    gate_strength: np.ndarray,
    dir_signal: np.ndarray,
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    atr_high_th: float,
    vol_low_th: float,
    funding_near_min: float,
    risk_lev_cap: float,
    base_leverage: float,
    cost_per_side: float,
    maker_fee_per_side: float,
    slip_per_side: float,
    fee_tp_mult: float,
    bep_arm_fee_mult: float,
    bep_stop_fee_mult: float,
    bep_stop_mode: str,
    atr_entry_mult: float,
    range_entry_mult: float,
    low_vol_filter: int,
    trail_after_bep: int,
    risk_entry_mode: int,
    use_atr_scaling: int,
    lev_mult: float,
    TP: float,
    SL: float,
    bep_arm_base: float,
    trailing: float,
    min_hold_bars: int,
    min_hold_trail_bars: int,
    min_hold_soft_sl_bars: int,
    max_hold_bars: int,
    dyn_lev_scale_arr: np.ndarray,
    dyn_bep_scale_arr: np.ndarray,
    dyn_trail_scale_arr: np.ndarray,
    dyn_sl_scale_arr: np.ndarray,
    dyn_softsl_relax_arr: np.ndarray,
    dyn_gate_mult_arr: np.ndarray,
    dyn_stress_arr: np.ndarray,
    use_pre_bep_timeout: int,
    pre_bep_timeout_bars: int,
    pre_bep_stress_th: float,
    pre_bep_progress_frac: float,
    pre_bep_degrade_sl_scale: float,
    pre_bep_softsl_delta: int,
    pre_bep_force_close_bars: int,
    pre_bep_force_close_red_only: int,
    dyn_mode_code: int,
    allow_soft_sl_before_trail: int,
    softsl_hold_floor: int,
    post_bep_shield_ignore_softsl_hold: int,
    hard_sl_mult_pre_unlock: float,
    trail_grace_after_bep: int,
    trail_grace_after_unlock: int,
    early_softsl_enabled: int = 0,
    early_softsl_min_hold: int = 2,
    early_softsl_progress_frac: float = 0.5,
    early_trail_enabled: int = 0,
    early_trail_min_hold: int = 3,
    early_trail_progress_frac: float = 0.85,
    early_trail_ref_updates_min: int = 1,
    stop_equity: float = 0.4,
    stop_dd: float = 0.35,
    warmup_steps: int = 0,
    integer_leverage: int = 0,
    seg_start: int = 0,
    intrabar_mode: int = 1,
    regime_alpha_arr: Optional[np.ndarray] = None,
    regime_bucket_arr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n = int(len(close))
    exit_cnt = np.zeros(6, dtype=np.int64)
    exit_gross_sum = np.zeros(6, dtype=np.float64)
    exit_fee_sum = np.zeros(6, dtype=np.float64)
    exit_net_sum = np.zeros(6, dtype=np.float64)
    trade_logs: List[Dict[str, Any]] = []

    if support_strength_ratio_arr is None:
        support_strength_ratio_arr = np.zeros(n, dtype=np.float64)
    else:
        support_strength_ratio_arr = np.asarray(support_strength_ratio_arr, dtype=np.float64)
    if support_weak_eligible_mask is None:
        support_weak_eligible_mask = np.zeros(n, dtype=np.bool_)
    else:
        support_weak_eligible_mask = np.asarray(support_weak_eligible_mask, dtype=np.bool_)
    if support_pass_mask is None:
        support_pass_mask = np.zeros(n, dtype=np.bool_)
    else:
        support_pass_mask = np.asarray(support_pass_mask, dtype=np.bool_)

    if n < 2:
        return {
            "net_ret": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wins": 0,
            "tail_hit": 0,
            "exit_cnt": exit_cnt,
            "exit_gross_sum": exit_gross_sum,
            "exit_fee_sum": exit_fee_sum,
            "exit_net_sum": exit_net_sum,
            "trail_before_bep": 0,
            "trail_after_bep": 0,
            "bep_armed_trades": 0,
            "ref_updates": 0,
            "trade_logs": trade_logs,
            "long_trades": 0,
            "short_trades": 0,
            "maxh_cnt": 0,
        }

    range_rel = (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) / np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    atr_med = float(np.median(atr_rel)) if len(atr_rel) else 1.0
    range_med = float(np.median(range_rel)) if len(range_rel) else 1.0
    if atr_med <= 0.0:
        atr_med = 1.0
    if range_med <= 0.0:
        range_med = 1.0

    taker_fee_side = float(cost_per_side) + float(slip_per_side)
    fee_roundtrip = 2.0 * taker_fee_side
    econ_be_fee = _bep_econ_fee(taker_fee_side, maker_fee_per_side, bep_stop_mode)

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trade_cnt = 0
    win_cnt = 0
    tail_hit = 0
    long_trades = 0
    short_trades = 0
    maxh_cnt = 0
    trail_before_bep_cnt = 0
    trail_after_bep_cnt = 0
    bep_armed_trades = 0
    ref_updates = 0
    pos_ref_updates_local = 0

    pos_side = 0
    entry_price = 0.0
    ref_price = 0.0
    entry_i = -1
    entry_decision_i = -1
    entry_lev = 0.0
    bep_armed = 0
    bep_armed_at = -1

    pos_TP = 0.0
    pos_SL = 0.0
    pos_BEP_ARM = 0.0
    pos_TR = 0.0
    pos_min_hold = 0
    pos_min_hold_trail = 0
    pos_min_hold_soft_sl = 0
    pos_max_hold = 0
    pos_bep_arm_fee = 0.0
    pos_bep_stop_fee = 0.0

    pos_SL_base = 0.0
    pos_BEP_ARM_base = 0.0
    pos_TR_base = 0.0
    pos_bep_arm_fee_base = 0.0
    pos_min_hold_soft_sl_base = 0

    pend_side = 0
    pend_i = -1
    pend_lev = 0.0
    pend_TP = 0.0
    pend_SL = 0.0
    pend_BEP_ARM = 0.0
    pend_TR = 0.0
    pend_min_hold = 0
    pend_min_hold_trail = 0
    pend_min_hold_soft_sl = 0
    pend_max_hold = 0
    pend_bep_arm_fee = 0.0
    pend_bep_stop_fee = 0.0
    pend_SL_base = 0.0
    pend_BEP_ARM_base = 0.0
    pend_TR_base = 0.0
    pend_bep_arm_fee_base = 0.0
    pend_min_hold_soft_sl_base = 0

    entry_gate_strength = 0.0
    entry_dir_signal = 0.0
    entry_atr_rel = 0.0
    entry_vol_z = 0.0
    entry_bep_arm_val = 0.0
    entry_dyn_lev_scale = 1.0
    entry_dyn_bep_scale = 1.0
    entry_dyn_trail_scale = 1.0
    entry_dyn_sl_scale = 1.0
    entry_dyn_gate_mult = 1.0
    entry_dyn_stress = 0.0
    entry_min_hold_soft_sl_init = 0
    entry_regime_alpha = float("nan")
    entry_regime_bucket = -1

    pend_gate_strength = 0.0
    pend_dir_signal = 0.0
    pend_atr_rel = 0.0
    pend_vol_z = 0.0
    pend_dyn_lev_scale = 1.0
    pend_dyn_bep_scale = 1.0
    pend_dyn_trail_scale = 1.0
    pend_dyn_sl_scale = 1.0
    pend_dyn_gate_mult = 1.0
    pend_dyn_stress = 0.0
    pend_decision_i = -1

    mfe_rel = 0.0
    mae_rel = 0.0
    mfe_prev_rel = 0.0
    pre_bep_degraded = 0
    pre_bep_force_closed = 0
    pre_bep_last_progress_prev = 0.0

    for i in range(n - 1):
        if pend_side != 0 and pend_i == i:
            pos_side = int(pend_side)
            pend_side = 0
            entry_price = float(open_[i])
            ref_price = float(open_[i])
            entry_i = int(i)
            entry_decision_i = int(pend_decision_i)
            entry_lev = float(pend_lev)
            bep_armed = 0
            bep_armed_at = -1

            pos_TP = float(pend_TP)
            pos_SL = float(pend_SL)
            pos_BEP_ARM = float(pend_BEP_ARM)
            pos_TR = float(pend_TR)
            pos_min_hold = int(pend_min_hold)
            pos_min_hold_trail = int(pend_min_hold_trail)
            pos_min_hold_soft_sl = int(pend_min_hold_soft_sl)
            pos_max_hold = int(pend_max_hold)
            pos_bep_arm_fee = float(pend_bep_arm_fee)
            pos_bep_stop_fee = float(pend_bep_stop_fee)

            pos_SL_base = float(pend_SL_base)
            pos_BEP_ARM_base = float(pend_BEP_ARM_base)
            pos_TR_base = float(pend_TR_base)
            pos_bep_arm_fee_base = float(pend_bep_arm_fee_base)
            pos_min_hold_soft_sl_base = int(pend_min_hold_soft_sl_base)

            entry_gate_strength = float(pend_gate_strength)
            entry_dir_signal = float(pend_dir_signal)
            entry_atr_rel = float(pend_atr_rel)
            entry_vol_z = float(pend_vol_z)
            entry_dyn_lev_scale = float(pend_dyn_lev_scale)
            entry_dyn_bep_scale = float(pend_dyn_bep_scale)
            entry_dyn_trail_scale = float(pend_dyn_trail_scale)
            entry_dyn_sl_scale = float(pend_dyn_sl_scale)
            entry_dyn_gate_mult = float(pend_dyn_gate_mult)
            entry_dyn_stress = float(pend_dyn_stress)
            entry_min_hold_soft_sl_init = int(pend_min_hold_soft_sl)
            if regime_alpha_arr is not None and 0 <= int(entry_decision_i) < int(len(regime_alpha_arr)):
                entry_regime_alpha = float(regime_alpha_arr[int(entry_decision_i)])
            else:
                entry_regime_alpha = float("nan")
            if regime_bucket_arr is not None and 0 <= int(entry_decision_i) < int(len(regime_bucket_arr)):
                entry_regime_bucket = int(regime_bucket_arr[int(entry_decision_i)])
            else:
                entry_regime_bucket = -1

            arm_v = float(pos_BEP_ARM)
            if arm_v < float(pos_bep_arm_fee):
                arm_v = float(pos_bep_arm_fee)
            entry_bep_arm_val = float(arm_v)
            pos_ref_updates_local = 0
            mfe_rel = 0.0
            mae_rel = 0.0
            mfe_prev_rel = 0.0
            pre_bep_degraded = 0
            pre_bep_force_closed = 0
            pre_bep_last_progress_prev = 0.0

        hi = float(high[i])
        lo = float(low[i])
        c = float(close[i])

        if pos_side != 0:
            side = int(pos_side)
            ep = float(entry_price)
            if ep > 0.0:
                if side == 1:
                    fav = (hi - ep) / ep
                    adv = (lo - ep) / ep
                else:
                    fav = (ep - lo) / ep
                    adv = (ep - hi) / ep
                if fav > mfe_rel:
                    mfe_rel = fav
                if adv < mae_rel:
                    mae_rel = adv
            lev_now = float(entry_lev)
            hold = int(i - entry_i)
            if hold < 0:
                hold = 0

            if int(dyn_mode_code) == 1 and i > entry_i:
                j = i - 1
                cand_sl = float(pos_SL_base) * float(dyn_sl_scale_arr[j])
                min_sl_live = 0.6 * fee_roundtrip
                if cand_sl < min_sl_live:
                    cand_sl = min_sl_live
                if cand_sl < pos_SL:
                    pos_SL = float(cand_sl)

                cand_bep_arm_fee = float(pos_bep_arm_fee_base) * float(dyn_bep_scale_arr[j])
                if cand_bep_arm_fee < 0.0:
                    cand_bep_arm_fee = 0.0
                if cand_bep_arm_fee < pos_bep_arm_fee:
                    pos_bep_arm_fee = float(cand_bep_arm_fee)

                cand_bep = float(pos_BEP_ARM_base) * float(dyn_bep_scale_arr[j])
                if cand_bep < pos_bep_arm_fee:
                    cand_bep = pos_bep_arm_fee
                if cand_bep < pos_BEP_ARM:
                    pos_BEP_ARM = float(cand_bep)

                cand_tr = float(pos_TR_base) * float(dyn_trail_scale_arr[j])
                if int(trail_after_bep) == 0:
                    min_tr_live = max(float(econ_be_fee), fee_roundtrip * float(fee_tp_mult))
                    if cand_tr < min_tr_live:
                        cand_tr = min_tr_live
                if cand_tr < pos_TR:
                    pos_TR = float(cand_tr)

                cand_soft = resolve_local_soft_sl_hold(
                    int(pos_min_hold_soft_sl_base),
                    int(pos_min_hold_trail),
                    int(dyn_softsl_relax_arr[j]),
                    int(allow_soft_sl_before_trail),
                    int(softsl_hold_floor),
                )
                if cand_soft < pos_min_hold_soft_sl:
                    pos_min_hold_soft_sl = int(cand_soft)

            force_close_now = False
            if int(use_pre_bep_timeout) == 1 and bep_armed == 0 and i > entry_i:
                stress_prev = float(dyn_stress_arr[i - 1])
                progress_prev = 0.0
                if entry_bep_arm_val > 1e-12:
                    progress_prev = float(mfe_prev_rel / entry_bep_arm_val)
                pre_bep_last_progress_prev = float(progress_prev)
                timeout_hit = (
                    hold >= int(pre_bep_timeout_bars)
                    and stress_prev >= float(pre_bep_stress_th)
                    and progress_prev < float(pre_bep_progress_frac)
                )
                if timeout_hit:
                    pre_bep_degraded = 1
                    cand_sl = float(pos_SL_base) * float(pre_bep_degrade_sl_scale)
                    min_sl_live = 0.6 * fee_roundtrip
                    if cand_sl < min_sl_live:
                        cand_sl = min_sl_live
                    if cand_sl < pos_SL:
                        pos_SL = float(cand_sl)
                    if int(pre_bep_softsl_delta) > 0:
                        cand_soft = resolve_local_soft_sl_hold(
                            int(pos_min_hold_soft_sl_base),
                            int(pos_min_hold_trail),
                            int(pre_bep_softsl_delta),
                            int(allow_soft_sl_before_trail),
                            int(softsl_hold_floor),
                        )
                        if cand_soft < pos_min_hold_soft_sl:
                            pos_min_hold_soft_sl = int(cand_soft)
                    if int(pre_bep_force_close_bars) > 0 and hold >= int(pre_bep_force_close_bars):
                        unreal = side * (c - ep) / np.maximum(ep, 1e-12)
                        if (int(pre_bep_force_close_red_only) == 0) or (unreal <= 0.0):
                            force_close_now = True
                            pre_bep_force_closed = 1

            TP_v = float(pos_TP)
            SL_v = float(pos_SL)
            BEP_ARM_v = float(pos_BEP_ARM)
            TR_v = float(pos_TR)
            min_hold_soft = int(pos_min_hold_soft_sl)

            tp_price = ep * (1.0 + side * TP_v)
            sl_price = ep * (1.0 - side * SL_v)
            bep_stop_price = ep * (1.0 + side * float(pos_bep_stop_fee))

            if bep_armed == 1:
                if side == 1:
                    if sl_price < bep_stop_price:
                        sl_price = bep_stop_price
                else:
                    if sl_price > bep_stop_price:
                        sl_price = bep_stop_price

            hard_sl_dist = max(float(SL_v) * float(max(hard_sl_mult_pre_unlock, 0.0)), 0.6 * fee_roundtrip)
            hard_sl = ep * (1.0 - side * hard_sl_dist)

            do_exit = False
            exit_price = 0.0
            reason = EXIT_MAXH
            progress_soft = 0.0
            if entry_bep_arm_val > 1e-12:
                progress_soft = float(mfe_rel / entry_bep_arm_val)
            progress_trail_den = max(float(TR_v), float(entry_bep_arm_val), 1e-12)
            progress_trail = float(mfe_rel / progress_trail_den)
            early_softsl_now = (
                int(early_softsl_enabled) == 1
                and hold >= int(early_softsl_min_hold)
                and progress_soft >= float(early_softsl_progress_frac)
            )
            early_trail_now = (
                int(early_trail_enabled) == 1
                and hold >= int(early_trail_min_hold)
                and progress_trail >= float(early_trail_progress_frac)
                and pos_ref_updates_local >= int(early_trail_ref_updates_min)
            )
            allow_tp = hold >= int(pos_min_hold)
            allow_trail = (hold >= int(pos_min_hold_trail)) or early_trail_now
            allow_soft_sl = (hold >= int(min_hold_soft)) or early_softsl_now
            maxh_hit = (int(pos_max_hold) > 0 and hold >= int(pos_max_hold))

            first_is_high = True
            if int(intrabar_mode) == 2:
                first_is_high = False

            for step in range(2):
                use_high = first_is_high if step == 0 else (not first_is_high)
                is_fav = (side == 1 and use_high) or (side == -1 and (not use_high))

                if is_fav:
                    if BEP_ARM_v > 0.0 and bep_armed == 0:
                        bep_arm_price = ep * (1.0 + side * BEP_ARM_v)
                        if (side == 1 and hi >= bep_arm_price) or (side == -1 and lo <= bep_arm_price):
                            bep_armed = 1
                            bep_armed_at = int(i)
                            ref_price = hi if side == 1 else lo

                    if bep_armed == 1:
                        if side == 1:
                            if sl_price < bep_stop_price:
                                sl_price = bep_stop_price
                        else:
                            if sl_price > bep_stop_price:
                                sl_price = bep_stop_price

                    if TR_v > 0.0 and (int(trail_after_bep) == 0 or bep_armed == 1):
                        if side == 1:
                            if hi > ref_price:
                                ref_price = hi
                                ref_updates += 1
                                pos_ref_updates_local += 1
                        else:
                            if lo < ref_price:
                                ref_price = lo
                                ref_updates += 1
                                pos_ref_updates_local += 1

                    if allow_tp:
                        if (side == 1 and hi >= tp_price) or (side == -1 and lo <= tp_price):
                            do_exit = True
                            exit_price = tp_price
                            reason = EXIT_TP
                            break
                else:
                    trail_hit = False
                    sl_hit = False
                    trail_price = 0.0
                    trail_active = False
                    if allow_trail and TR_v > 0.0 and (int(trail_after_bep) == 0 or bep_armed == 1):
                        if not (bep_armed == 1 and bep_armed_at == i):
                            trail_active = True
                    if trail_active:
                        unlock_i = entry_i + int(pos_min_hold_trail)
                        if int(trail_grace_after_unlock) > 0 and (i - unlock_i) < int(trail_grace_after_unlock):
                            trail_active = False
                    if trail_active and int(trail_after_bep) == 1 and bep_armed == 1 and int(trail_grace_after_bep) > 0:
                        if (i - bep_armed_at) < int(trail_grace_after_bep):
                            trail_active = False
                    if trail_active:
                        trail_price = ref_price * (1.0 - side * TR_v)
                        if bep_armed == 1:
                            if side == 1:
                                if trail_price < bep_stop_price:
                                    trail_price = bep_stop_price
                            else:
                                if trail_price > bep_stop_price:
                                    trail_price = bep_stop_price
                        if (side == 1 and lo <= trail_price) or (side == -1 and hi >= trail_price):
                            trail_hit = True

                    post_bep_sl_live = (
                        bep_armed == 1
                        and int(post_bep_shield_ignore_softsl_hold) == 1
                        and bep_armed_at >= 0
                        and bep_armed_at < i
                    )
                    if allow_soft_sl or post_bep_sl_live:
                        if (side == 1 and lo <= sl_price) or (side == -1 and hi >= sl_price):
                            sl_hit = True
                    if trail_active or allow_soft_sl or post_bep_sl_live:
                        if bep_armed == 1:
                            if trail_hit:
                                do_exit = True
                                exit_price = trail_price
                                reason = EXIT_TRAIL
                                break
                            elif sl_hit:
                                do_exit = True
                                exit_price = sl_price
                                reason = EXIT_SL
                                break
                        else:
                            if sl_hit:
                                do_exit = True
                                exit_price = sl_price
                                reason = EXIT_SL
                                break
                            elif trail_hit:
                                do_exit = True
                                exit_price = trail_price
                                reason = EXIT_TRAIL
                                break
                    else:
                        if (side == 1 and lo <= hard_sl) or (side == -1 and hi >= hard_sl):
                            do_exit = True
                            exit_price = hard_sl
                            reason = EXIT_SL
                            break

            if (not do_exit) and force_close_now:
                do_exit = True
                exit_price = c
                reason = EXIT_RISK
            if (not do_exit) and maxh_hit:
                do_exit = True
                exit_price = c
                reason = EXIT_MAXH

            if do_exit:
                exit_fee_side = taker_fee_side
                if float(maker_fee_per_side) > 0.0 and reason in (EXIT_TP, EXIT_TRAIL):
                    exit_fee_side = float(maker_fee_per_side)
                fee_total = (taker_fee_side + exit_fee_side) * lev_now
                gross_pnl = (side * (exit_price - ep) / np.maximum(ep, 1e-12)) * lev_now
                net_pnl = gross_pnl - fee_total
                scaled = max(net_pnl, -0.999)
                equity = equity * (1.0 + scaled)

                trade_cnt += 1
                if side == 1:
                    long_trades += 1
                else:
                    short_trades += 1
                if scaled > 0.0:
                    win_cnt += 1
                if reason == EXIT_MAXH:
                    maxh_cnt += 1

                exit_cnt[reason] += 1
                exit_gross_sum[reason] += gross_pnl
                exit_fee_sum[reason] += fee_total
                exit_net_sum[reason] += net_pnl
                if reason == EXIT_TRAIL:
                    if bep_armed == 1:
                        trail_after_bep_cnt += 1
                    else:
                        trail_before_bep_cnt += 1
                if bep_armed == 1:
                    bep_armed_trades += 1

                trade_logs.append({
                    "entry_idx": int(seg_start + int(entry_i)),
                    "exit_idx": int(seg_start + int(i)),
                    "decision_idx": int(seg_start + int(entry_decision_i)),
                    "tier": 0,
                    "tier_name": "SINGLE",
                    "side": int(side),
                    "side_name": "LONG" if side == 1 else "SHORT",
                    "entry_th_used": np.nan,
                    "gate_strength_entry": float(entry_gate_strength),
                    "dir_signal_entry": float(entry_dir_signal),
                    "atr_rel_entry": float(entry_atr_rel),
                    "vol_z_60_entry": float(entry_vol_z),
                    "bep_arm_value": float(entry_bep_arm_val),
                    "entry_bep_arm_fee": float(pos_bep_arm_fee),
                    "entry_bep_stop_fee": float(pos_bep_stop_fee),
                    "entry_dyn_gate_mult": float(entry_dyn_gate_mult),
                    "entry_dyn_lev_scale": float(entry_dyn_lev_scale),
                    "entry_dyn_bep_scale": float(entry_dyn_bep_scale),
                    "entry_dyn_trail_scale": float(entry_dyn_trail_scale),
                    "entry_dyn_sl_scale": float(entry_dyn_sl_scale),
                    "entry_dyn_stress": float(entry_dyn_stress),
                    "entry_dyn_mode": int(dyn_mode_code),
                    "entry_allow_soft_sl_before_trail": int(allow_soft_sl_before_trail),
                    "entry_softsl_hold_floor": int(softsl_hold_floor),
                    "entry_post_bep_shield_ignore_softsl_hold": int(post_bep_shield_ignore_softsl_hold),
                    "entry_min_hold_soft_sl_local": int(entry_min_hold_soft_sl_init),
                    "entry_regime_alpha": float(entry_regime_alpha),
                    "entry_regime_bucket": int(entry_regime_bucket),
                    "entry_regime_name": ("active" if int(entry_regime_bucket) == 2 else ("mid" if int(entry_regime_bucket) == 1 else ("calm" if int(entry_regime_bucket) == 0 else "unknown"))),
                    "final_min_hold_soft_sl_local": int(pos_min_hold_soft_sl),
                    "pre_bep_degraded": int(pre_bep_degraded),
                    "pre_bep_force_closed": int(pre_bep_force_closed),
                    "pre_bep_last_progress_prev": float(pre_bep_last_progress_prev),
                    "final_live_sl": float(pos_SL),
                    "final_live_bep_arm": float(pos_BEP_ARM),
                    "final_live_bep_stop_fee": float(pos_bep_stop_fee),
                    "final_live_trail": float(pos_TR),
                    "entry_early_softsl_enabled": int(early_softsl_enabled),
                    "entry_early_softsl_min_hold": int(early_softsl_min_hold),
                    "entry_early_softsl_progress_frac": float(early_softsl_progress_frac),
                    "entry_early_trail_enabled": int(early_trail_enabled),
                    "entry_early_trail_min_hold": int(early_trail_min_hold),
                    "entry_early_trail_progress_frac": float(early_trail_progress_frac),
                    "entry_early_trail_ref_updates_min": int(early_trail_ref_updates_min),
                    "ref_updates_local": int(pos_ref_updates_local),
                    "mfe": float(mfe_rel),
                    "mae": float(mae_rel),
                    "entry_price": float(ep),
                    "exit_price": float(exit_price),
                    "lev": float(lev_now),
                    "gross_pnl": float(gross_pnl),
                    "fee_total": float(fee_total),
                    "net_pnl": float(net_pnl),
                    "net_pnl_alloc": float(scaled),
                    "exit_reason_id": int(reason),
                    "exit_reason": EXIT_NAMES[int(reason)],
                    "hold_bars": int(hold),
                })

                pos_side = 0
                bep_armed = 0
                bep_armed_at = -1

            if pos_side != 0:
                fav_cur = 0.0
                if side == 1:
                    fav_cur = (hi - ep) / np.maximum(ep, 1e-12)
                else:
                    fav_cur = (ep - lo) / np.maximum(ep, 1e-12)
                if fav_cur > mfe_prev_rel:
                    mfe_prev_rel = fav_cur

        if equity > peak:
            peak = equity
        dd = 1.0 - equity / peak
        if dd > mdd:
            mdd = dd
        if equity < float(stop_equity) or dd > float(stop_dd):
            tail_hit = 1
            break

        if i < int(warmup_steps):
            continue

        if pos_side == 0 and pend_side == 0:
            if not bool(ready[i]):
                continue
            gsig = float(gate_strength[i])
            dsig = float(dir_signal[i])
            if gsig <= 0.0 or dsig == 0.0:
                continue
            if float(minutes_to_next_funding[i]) < float(funding_near_min):
                continue
            if int(low_vol_filter) == 1:
                if float(vol_z[i]) <= float(vol_low_th):
                    continue
                if float(atr_rel[i]) < float(atr_entry_mult) * fee_roundtrip:
                    continue
                if float(range_rel[i]) < float(range_entry_mult) * fee_roundtrip:
                    continue
            if int(use_atr_scaling) == 1 and np.isfinite(float(atr_high_th)) and float(atr_rel[i]) > float(atr_high_th):
                continue
            if int(risk_entry_mode) == 3:
                if float(atr_rel[i]) > atr_med * float(atr_entry_mult):
                    continue
                if float(range_rel[i]) > range_med * float(range_entry_mult):
                    continue

            side = 1 if dsig > 0.0 else -1
            lev_now = float(base_leverage) * float(lev_mult) * float(dyn_lev_scale_arr[i])
            if int(integer_leverage) == 1:
                lev_now = float(int(lev_now + 0.5))
            if lev_now < 1.0:
                lev_now = 1.0
            if lev_now > float(risk_lev_cap):
                lev_now = float(risk_lev_cap)

            atr_e = float(atr_rel[i]) if int(use_atr_scaling) == 1 else 1.0
            sl_base_v = float(SL) * atr_e
            tp_v = float(TP) * atr_e
            bep_arm_base_v = float(bep_arm_base) * atr_e
            tr_base_v = float(trailing) * atr_e

            sl_v = sl_base_v * float(dyn_sl_scale_arr[i])
            bep_arm_v = bep_arm_base_v * float(dyn_bep_scale_arr[i])
            tr_v = tr_base_v * float(dyn_trail_scale_arr[i])

            min_tp = fee_roundtrip * float(fee_tp_mult)
            if tp_v < min_tp:
                tp_v = min_tp

            bep_arm_fee_local = econ_be_fee * float(bep_arm_fee_mult) * float(dyn_bep_scale_arr[i])
            if bep_arm_fee_local < 0.0:
                bep_arm_fee_local = 0.0
            if bep_arm_v < bep_arm_fee_local:
                bep_arm_v = bep_arm_fee_local

            min_sl = 0.6 * fee_roundtrip
            if sl_v < min_sl:
                sl_v = min_sl

            if int(trail_after_bep) == 0:
                min_tr = max(econ_be_fee, fee_roundtrip * float(fee_tp_mult))
                if tr_v < min_tr:
                    tr_v = min_tr

            local_soft_sl_hold = resolve_local_soft_sl_hold(
                int(min_hold_soft_sl_bars),
                int(min_hold_trail_bars),
                int(dyn_softsl_relax_arr[i]),
                int(allow_soft_sl_before_trail),
                int(softsl_hold_floor),
            )

            pend_decision_i = int(i)
            pend_gate_strength = float(gsig)
            pend_dir_signal = float(dsig)
            pend_atr_rel = float(atr_rel[i])
            pend_vol_z = float(vol_z[i])
            pend_dyn_lev_scale = float(dyn_lev_scale_arr[i])
            pend_dyn_bep_scale = float(dyn_bep_scale_arr[i])
            pend_dyn_trail_scale = float(dyn_trail_scale_arr[i])
            pend_dyn_sl_scale = float(dyn_sl_scale_arr[i])
            pend_dyn_gate_mult = float(dyn_gate_mult_arr[i])
            pend_dyn_stress = float(dyn_stress_arr[i])

            pend_side = int(side)
            pend_i = int(i + 1)
            pend_lev = float(lev_now)
            pend_TP = float(tp_v)
            pend_SL = float(sl_v)
            pend_BEP_ARM = float(bep_arm_v)
            pend_TR = float(tr_v)
            pend_min_hold = int(min_hold_bars)
            pend_min_hold_trail = int(min_hold_trail_bars)
            pend_min_hold_soft_sl = int(local_soft_sl_hold)
            pend_max_hold = int(max_hold_bars)
            pend_bep_arm_fee = float(bep_arm_fee_local)
            pend_bep_stop_fee = float(econ_be_fee * float(bep_stop_fee_mult))

            pend_SL_base = float(sl_base_v)
            pend_BEP_ARM_base = float(bep_arm_base_v)
            pend_TR_base = float(tr_base_v)
            pend_bep_arm_fee_base = float(econ_be_fee * float(bep_arm_fee_mult))
            pend_min_hold_soft_sl_base = int(min_hold_soft_sl_bars)

    if tail_hit == 0 and pos_side != 0:
        last_c = float(close[-1])
        side = int(pos_side)
        ep = float(entry_price)
        lev_now = float(entry_lev)
        fee_total = (taker_fee_side + taker_fee_side) * lev_now
        gross_pnl = (side * (last_c - ep) / np.maximum(ep, 1e-12)) * lev_now
        net_pnl = gross_pnl - fee_total
        scaled = max(net_pnl, -0.999)
        equity = equity * (1.0 + scaled)
        trade_cnt += 1
        if side == 1:
            long_trades += 1
        else:
            short_trades += 1
        if scaled > 0.0:
            win_cnt += 1
        exit_cnt[EXIT_FORCE] += 1
        exit_gross_sum[EXIT_FORCE] += gross_pnl
        exit_fee_sum[EXIT_FORCE] += fee_total
        exit_net_sum[EXIT_FORCE] += net_pnl
        if equity > peak:
            peak = equity
        dd = 1.0 - equity / peak
        if dd > mdd:
            mdd = dd

        hi_last = float(high[-1])
        lo_last = float(low[-1])
        if ep > 0.0:
            if side == 1:
                fav = (hi_last - ep) / ep
                adv = (lo_last - ep) / ep
            else:
                fav = (ep - lo_last) / ep
                adv = (ep - hi_last) / ep
            if fav > mfe_rel:
                mfe_rel = fav
            if adv < mae_rel:
                mae_rel = adv

        hold_bars = int((n - 1) - int(entry_i))
        trade_logs.append({
            "entry_idx": int(seg_start + int(entry_i)),
            "exit_idx": int(seg_start + int(n - 1)),
            "decision_idx": int(seg_start + int(entry_decision_i)),
            "tier": 0,
            "tier_name": "SINGLE",
            "side": int(side),
            "side_name": "LONG" if side == 1 else "SHORT",
            "entry_th_used": np.nan,
            "gate_strength_entry": float(entry_gate_strength),
            "dir_signal_entry": float(entry_dir_signal),
            "atr_rel_entry": float(entry_atr_rel),
            "vol_z_60_entry": float(entry_vol_z),
            "bep_arm_value": float(entry_bep_arm_val),
            "entry_bep_arm_fee": float(pos_bep_arm_fee),
            "entry_bep_stop_fee": float(pos_bep_stop_fee),
            "entry_dyn_gate_mult": float(entry_dyn_gate_mult),
            "entry_dyn_lev_scale": float(entry_dyn_lev_scale),
            "entry_dyn_bep_scale": float(entry_dyn_bep_scale),
            "entry_dyn_trail_scale": float(entry_dyn_trail_scale),
            "entry_dyn_sl_scale": float(entry_dyn_sl_scale),
            "entry_dyn_stress": float(entry_dyn_stress),
            "entry_dyn_mode": int(dyn_mode_code),
            "entry_allow_soft_sl_before_trail": int(allow_soft_sl_before_trail),
            "entry_softsl_hold_floor": int(softsl_hold_floor),
            "entry_post_bep_shield_ignore_softsl_hold": int(post_bep_shield_ignore_softsl_hold),
            "entry_min_hold_soft_sl_local": int(entry_min_hold_soft_sl_init),
            "entry_regime_alpha": float(entry_regime_alpha),
            "entry_regime_bucket": int(entry_regime_bucket),
            "entry_regime_name": ("active" if int(entry_regime_bucket) == 2 else ("mid" if int(entry_regime_bucket) == 1 else ("calm" if int(entry_regime_bucket) == 0 else "unknown"))),
            "final_min_hold_soft_sl_local": int(pos_min_hold_soft_sl),
            "pre_bep_degraded": int(pre_bep_degraded),
            "pre_bep_force_closed": int(pre_bep_force_closed),
            "pre_bep_last_progress_prev": float(pre_bep_last_progress_prev),
            "final_live_sl": float(pos_SL),
            "final_live_bep_arm": float(pos_BEP_ARM),
            "final_live_bep_stop_fee": float(pos_bep_stop_fee),
            "final_live_trail": float(pos_TR),
            "mfe": float(mfe_rel),
            "mae": float(mae_rel),
            "entry_price": float(ep),
            "exit_price": float(last_c),
            "lev": float(lev_now),
            "gross_pnl": float(gross_pnl),
            "fee_total": float(fee_total),
            "net_pnl": float(net_pnl),
            "net_pnl_alloc": float(scaled),
            "exit_reason_id": int(EXIT_FORCE),
            "exit_reason": EXIT_NAMES[int(EXIT_FORCE)],
            "hold_bars": int(hold_bars),
        })
        pos_side = 0
        bep_armed = 0
        bep_armed_at = -1

    return {
        "net_ret": float(equity - 1.0),
        "mdd": float(mdd),
        "trades": int(trade_cnt),
        "wins": int(win_cnt),
        "tail_hit": int(tail_hit),
        "exit_cnt": exit_cnt,
        "exit_gross_sum": exit_gross_sum,
        "exit_fee_sum": exit_fee_sum,
        "exit_net_sum": exit_net_sum,
        "trail_before_bep": int(trail_before_bep_cnt),
        "trail_after_bep": int(trail_after_bep_cnt),
        "bep_armed_trades": int(bep_armed_trades),
        "ref_updates": int(ref_updates),
        "trade_logs": trade_logs,
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "maxh_cnt": int(maxh_cnt),
    }


def evaluate_single_segment(
    seg_start: int,
    seg_end: int,
    open_px: np.ndarray,
    close_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    h1_sig: np.ndarray,
    h3_sig: np.ndarray,
    h5_sig: np.ndarray,
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    cfg: Dict[str, Any],
    score_cfg: Dict[str, Any],
    cost_per_side: float,
    slip_per_side: float,
    maker_fee_per_side: float,
    entry_q_lookback: int,
    entry_q_min_ready: int,
) -> Dict[str, Any]:
    cfg = normalize_single_config_from_any(cfg)
    score_cfg = dict(score_cfg or {})

    seg = slice(int(seg_start), int(seg_end))
    hist_end = int(seg_start)
    hist_start = max(0, hist_end - int(entry_q_lookback))

    gate_w = normalize_horizon_weights(cfg["gate_weights"], fallback={1: 0.0, 3: 0.0, 5: 1.0, 8: 0.0, 10: 0.0})
    dir_w = normalize_horizon_weights(cfg["dir_weights"], fallback={1: 0.0, 3: 0.0, 5: 1.0, 8: 0.0, 10: 0.0})

    gate_signal_all = np.abs(
        float(gate_w["w1"]) * h1_sig.astype(np.float64)
        + float(gate_w["w3"]) * h3_sig.astype(np.float64)
        + float(gate_w["w5"]) * h5_sig.astype(np.float64)
    )
    dir_signal_all = (
        float(dir_w["w1"]) * h1_sig.astype(np.float64)
        + float(dir_w["w3"]) * h3_sig.astype(np.float64)
        + float(dir_w["w5"]) * h5_sig.astype(np.float64)
    )

    thr_entry = quantile_from_history(
        gate_signal_all,
        ready,
        hist_start,
        hist_end,
        safe_float(cfg.get("q_entry", 0.85), 0.85),
        int(entry_q_min_ready),
        safe_float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)), 0.0),
    )

    risk_cfg = dict(cfg.get("risk_cfg", {}))
    atr_high_th = risk_cfg.get("atr_high_th", np.nan)
    try:
        atr_high_th_ = float(atr_high_th) if atr_high_th is not None else float("nan")
    except Exception:
        atr_high_th_ = float("nan")
    if np.isfinite(float(atr_high_th_)) and float(atr_high_th_) <= 0.0:
        atr_hist = atr_rel[hist_start:hist_end]
        atr_high_th_ = float(np.percentile(atr_hist, safe_float(risk_cfg.get("atr_percentile", 75.0), 75.0))) if atr_hist.size > 10 else 1e9

    range_hist = (high_px[hist_start:hist_end] - low_px[hist_start:hist_end]) / np.maximum(close_px[hist_start:hist_end], 1e-12)
    range_med = float(np.median(range_hist)) if range_hist.size > 0 else 1.0
    if range_med <= 0.0:
        range_med = 1.0
    range_seg = (high_px[seg] - low_px[seg]) / np.maximum(close_px[seg], 1e-12)
    range_cut = safe_float(cfg.get("range_entry_mult", 1.0), 1.0) * float(range_med)

    gate_strength_seg = gate_signal_all[seg].astype(np.float64)
    dir_signal_seg = dir_signal_all[seg].astype(np.float64)
    ready_seg = ready[seg].astype(bool)
    atr_seg = atr_rel[seg].astype(np.float64)
    vol_seg = vol_z[seg].astype(np.float64)
    funding_seg = minutes_to_next_funding[seg].astype(np.float64)

    dyn_gate_mult_arr, dyn_lev_scale_arr, dyn_bep_scale_arr, dyn_trail_scale_arr, dyn_sl_scale_arr, dyn_softsl_relax_arr, dyn_stress_arr = build_dynamic_arrays(
        dynamic_cfg=cfg["dynamic_cfg"],
        gate_strength_seg=gate_strength_seg,
        thr_entry=float(thr_entry),
        atr_seg=atr_seg,
        atr_high_th=float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        range_seg=range_seg.astype(np.float64),
        range_cut=float(range_cut),
        vol_seg=vol_seg,
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_seg=funding_seg,
    )
    gate_strength_used = gate_strength_seg * (ready_seg.astype(bool) & (gate_strength_seg >= (float(thr_entry) * dyn_gate_mult_arr)))

    dyn = cfg["dynamic_cfg"]
    sim_kwargs = dict(
        open_=open_px[seg].astype(np.float64),
        close=close_px[seg].astype(np.float64),
        high=high_px[seg].astype(np.float64),
        low=low_px[seg].astype(np.float64),
        gate_strength=gate_strength_used.astype(np.float64),
        dir_signal=dir_signal_seg.astype(np.float64),
        ready=ready_seg,
        vol_z=vol_seg,
        atr_rel=atr_seg,
        minutes_to_next_funding=funding_seg,
        atr_high_th=float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_near_min=safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0),
        risk_lev_cap=safe_float(risk_cfg.get("risk_lev_cap", 12.0), 12.0),
        base_leverage=safe_float(cfg.get("leverage", 10.0), 10.0),
        cost_per_side=float(cost_per_side),
        maker_fee_per_side=float(maker_fee_per_side),
        slip_per_side=float(slip_per_side),
        fee_tp_mult=safe_float(cfg.get("fee_tp_mult", 1.0), 1.0),
        bep_arm_fee_mult=safe_float(cfg.get("bep_arm_fee_mult", cfg.get("fee_bep_mult", 0.2)), 0.2),
        bep_stop_fee_mult=safe_float(cfg.get("bep_stop_fee_mult", 1.0), 1.0),
        bep_stop_mode=str(cfg.get("bep_stop_mode", "maker_be")),
        atr_entry_mult=safe_float(cfg.get("atr_entry_mult", 1.0), 1.0),
        range_entry_mult=safe_float(cfg.get("range_entry_mult", 1.0), 1.0),
        low_vol_filter=safe_int(cfg.get("low_vol_filter", 0), 0),
        trail_after_bep=safe_int(cfg.get("trail_after_bep", 1), 1),
        risk_entry_mode=safe_int(cfg.get("risk_entry_mode", 0), 0),
        use_atr_scaling=safe_int(cfg.get("use_atr_scaling", 1), 1),
        lev_mult=safe_float(cfg.get("lev_mult", 1.0), 1.0),
        TP=safe_float(cfg.get("TP", 0.0), 0.0),
        SL=safe_float(cfg.get("SL", 0.0), 0.0),
        bep_arm_base=safe_float(cfg.get("BEP_ARM", cfg.get("BEP", 0.0)), 0.0),
        trailing=safe_float(cfg.get("trailing", 0.0), 0.0),
        min_hold_bars=safe_int(cfg.get("min_hold_bars", 0), 0),
        min_hold_trail_bars=safe_int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_bars", 0)), 0),
        min_hold_soft_sl_bars=safe_int(cfg.get("min_hold_soft_sl_bars", cfg.get("min_hold_bars", 0)), 0),
        max_hold_bars=safe_int(cfg.get("max_hold_bars", 0), 0),
        dyn_lev_scale_arr=dyn_lev_scale_arr.astype(np.float64),
        dyn_bep_scale_arr=dyn_bep_scale_arr.astype(np.float64),
        dyn_trail_scale_arr=dyn_trail_scale_arr.astype(np.float64),
        dyn_sl_scale_arr=dyn_sl_scale_arr.astype(np.float64),
        dyn_softsl_relax_arr=dyn_softsl_relax_arr.astype(np.int64),
        dyn_gate_mult_arr=dyn_gate_mult_arr.astype(np.float64),
        dyn_stress_arr=dyn_stress_arr.astype(np.float64),
        use_pre_bep_timeout=safe_int(dyn.get("use_pre_bep_timeout", 0), 0),
        pre_bep_timeout_bars=safe_int(dyn.get("pre_bep_timeout_bars", 3), 3),
        pre_bep_stress_th=safe_float(dyn.get("pre_bep_stress_th", 0.55), 0.55),
        pre_bep_progress_frac=safe_float(dyn.get("pre_bep_progress_frac", 0.55), 0.55),
        pre_bep_degrade_sl_scale=safe_float(dyn.get("pre_bep_degrade_sl_scale", 0.75), 0.75),
        pre_bep_softsl_delta=safe_int(dyn.get("pre_bep_softsl_delta", 0), 0),
        pre_bep_force_close_bars=safe_int(dyn.get("pre_bep_force_close_bars", 0), 0),
        pre_bep_force_close_red_only=safe_int(dyn.get("pre_bep_force_close_red_only", 1), 1),
        dyn_mode_code=(1 if str(dyn.get("mode", "entry_latched")).strip().lower() == "exit_path_adaptive" else 0),
        allow_soft_sl_before_trail=safe_int(dyn.get("allow_soft_sl_before_trail", 0), 0),
        softsl_hold_floor=safe_int(dyn.get("softsl_hold_floor", 0), 0),
        post_bep_shield_ignore_softsl_hold=safe_int(dyn.get("post_bep_shield_ignore_softsl_hold", 0), 0),
        hard_sl_mult_pre_unlock=safe_float(cfg.get("hard_sl_mult_pre_unlock", 1.0), 1.0),
        trail_grace_after_bep=safe_int(cfg.get("trail_grace_after_bep", 0), 0),
        trail_grace_after_unlock=safe_int(cfg.get("trail_grace_after_unlock", 0), 0),
        early_softsl_enabled=safe_int(cfg.get("progress_protect_cfg", {}).get("early_softsl_enabled", 0), 0),
        early_softsl_min_hold=safe_int(cfg.get("progress_protect_cfg", {}).get("early_softsl_min_hold", 2), 2),
        early_softsl_progress_frac=safe_float(cfg.get("progress_protect_cfg", {}).get("early_softsl_progress_frac", 0.5), 0.5),
        early_trail_enabled=safe_int(cfg.get("progress_protect_cfg", {}).get("early_trail_enabled", 0), 0),
        early_trail_min_hold=safe_int(cfg.get("progress_protect_cfg", {}).get("early_trail_min_hold", 3), 3),
        early_trail_progress_frac=safe_float(cfg.get("progress_protect_cfg", {}).get("early_trail_progress_frac", 0.85), 0.85),
        early_trail_ref_updates_min=safe_int(cfg.get("progress_protect_cfg", {}).get("early_trail_ref_updates_min", 1), 1),
        stop_equity=safe_float(cfg.get("tail_cfg", {}).get("stop_equity", 0.0), 0.0),
        stop_dd=safe_float(cfg.get("tail_cfg", {}).get("stop_dd", 1.0), 1.0),
        warmup_steps=safe_int(cfg.get("tail_cfg", {}).get("warmup_steps", 0), 0),
        integer_leverage=safe_int(cfg.get("integer_leverage", 0), 0),
        seg_start=int(seg_start),
    )

    res_ohlc = simulate_trading_core_rl_single(**sim_kwargs, intrabar_mode=1)
    res_olhc = simulate_trading_core_rl_single(**sim_kwargs, intrabar_mode=2)

    def _score(res: Dict[str, Any]) -> float:
        return segment_score(
            net_ret=float(res["net_ret"]),
            mdd=float(res["mdd"]),
            tail_hit=int(res["tail_hit"]),
            trades=int(res["trades"]),
            maxh_cnt=int(res["maxh_cnt"]),
            long_trades=int(res["long_trades"]),
            short_trades=int(res["short_trades"]),
            alpha_dd=float(score_cfg.get("alpha_dd", 0.9)),
            beta_tail=float(score_cfg.get("beta_tail", 2.0)),
            trade_mode=str(score_cfg.get("trade_mode", "none")),
            trade_target=float(score_cfg.get("trade_target", 0.0)),
            trade_band=float(score_cfg.get("trade_band", 0.0)),
            barrier_k=float(score_cfg.get("barrier_k", 2.0)),
            shortage_penalty=float(score_cfg.get("trade_shortage_penalty", 0.05)),
            excess_penalty=float(score_cfg.get("trade_excess_penalty", 0.01)),
            maxhold_ratio_free=float(score_cfg.get("maxhold_ratio_free", 1.0)),
            maxhold_penalty_k=float(score_cfg.get("maxhold_penalty_k", 0.0)),
            maxhold_penalty_power=float(score_cfg.get("maxhold_penalty_power", 2.0)),
            side_balance_penalty_k=float(score_cfg.get("side_balance_penalty_k", 0.0)),
            min_short_trades=int(score_cfg.get("min_short_trades_global", 0)),
            min_short_share=float(score_cfg.get("min_short_share_global", 0.0)),
        )

    score_ohlc = _score(res_ohlc)
    score_olhc = _score(res_olhc)
    chosen = res_ohlc
    chosen_score = score_ohlc
    if (score_olhc < score_ohlc) or (
        score_olhc == score_ohlc and (
            float(res_olhc["net_ret"]) < float(res_ohlc["net_ret"])
            or (float(res_olhc["net_ret"]) == float(res_ohlc["net_ret"]) and float(res_olhc["mdd"]) > float(res_ohlc["mdd"]))
        )
    ):
        chosen = res_olhc
        chosen_score = score_olhc

    trades = int(chosen["trades"])
    wins = int(chosen["wins"])
    long_trades = int(chosen["long_trades"])
    short_trades = int(chosen["short_trades"])
    short_share = float(short_trades / trades) if trades > 0 else 0.0
    side_penalty = side_balance_penalty_component(
        long_trades,
        short_trades,
        int(score_cfg.get("min_short_trades_global", 0)),
        float(score_cfg.get("min_short_share_global", 0.0)),
        float(score_cfg.get("side_balance_penalty_k", 0.0)),
    )
    maxh_ratio = (float(chosen["maxh_cnt"]) / float(trades)) if trades > 0 else 0.0
    return {
        "net_ret": float(chosen["net_ret"]),
        "mdd_net": float(chosen["mdd"]),
        "winrate_net": float(wins / trades) if trades > 0 else 0.0,
        "trades": trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "short_share": float(short_share),
        "trade_logs": list(chosen["trade_logs"]),
        "tail": int(chosen["tail_hit"]),
        "thr_entry": float(thr_entry),
        "atr_high_th": float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        "exit_cnt": chosen["exit_cnt"],
        "exit_gross_sum": chosen["exit_gross_sum"],
        "exit_fee_sum": chosen["exit_fee_sum"],
        "exit_net_sum": chosen["exit_net_sum"],
        "trail_before_bep": int(chosen["trail_before_bep"]),
        "trail_after_bep": int(chosen["trail_after_bep"]),
        "bep_armed_trades": int(chosen["bep_armed_trades"]),
        "ref_updates": int(chosen["ref_updates"]),
        "maxh_cnt": int(chosen["maxh_cnt"]),
        "score": float(chosen_score),
        "side_penalty": float(side_penalty),
        "maxh_ratio": float(maxh_ratio),
    }


# ---------------------------------------------------------------------------
# v2bc appended helpers: separate detect / weight / threshold / filter cfgs
# ---------------------------------------------------------------------------

REGIME_BUCKET_NAMES: Dict[int, str] = {0: "calm", 1: "mid", 2: "active"}
REGIME_DETECT_KEYS = (
    "enabled",
    "source",
    "stress_lo",
    "stress_hi",
    "alpha_ema",
    "alpha_hysteresis",
    "w_atr",
    "w_rng",
    "w_vol",
    "w_fund",
)

DEFAULT_REGIME_DETECT_CFG: Dict[str, Any] = {
    "enabled": 0,
    "source": "exo_stress",
    "stress_lo": 0.25,
    "stress_hi": 0.65,
    "alpha_ema": 0.15,
    "alpha_hysteresis": 0.03,
    "w_atr": 0.35,
    "w_rng": 0.20,
    "w_vol": 0.30,
    "w_fund": 0.15,
}
DEFAULT_REGIME_WEIGHT_CFG = {
    "enabled": 0,
    "gate_calm_mix": 0.60,
    "gate_active_mix": 0.55,
    "dir_calm_mix": 0.35,
    "dir_active_mix": 0.50,
    "gate_calm_anchor": {"w1": 0.45, "w3": 0.40, "w5": 0.12, "w8": 0.03, "w10": 0.00},
    "gate_active_anchor": {"w1": 0.08, "w3": 0.28, "w5": 0.32, "w8": 0.22, "w10": 0.10},
    "dir_calm_anchor": {"w1": 0.30, "w3": 0.24, "w5": 0.24, "w8": 0.16, "w10": 0.06},
    "dir_active_anchor": {"w1": 0.08, "w3": 0.12, "w5": 0.32, "w8": 0.32, "w10": 0.16},
}
DEFAULT_REGIME_THRESHOLD_CFG: Dict[str, Any] = {
    "enabled": 0,
    "bucket_min_ready": 0,
    "bucket_fallback_global": 1,
    "q_entry_calm": None,
    "q_entry_mid": None,
    "q_entry_active": None,
    "entry_th_floor_calm": None,
    "entry_th_floor_mid": None,
    "entry_th_floor_active": None,
}
DEFAULT_REGIME_FILTER_CFG: Dict[str, Any] = {
    "enabled": 0,
    "use_vol_split": 1,
    "use_entry_mult_split": 1,
    "mid_interp_mode": "linear",
    "vol_low_th_calm": None,
    "vol_low_th_mid": None,
    "vol_low_th_active": None,
    "atr_entry_mult_calm": None,
    "atr_entry_mult_active": None,
    "range_entry_mult_calm": None,
    "range_entry_mult_active": None,
}


def regime_bucket_name(bucket: int) -> str:
    return REGIME_BUCKET_NAMES.get(int(bucket), "unknown")


def _normalize_regime_detect_cfg(raw: Optional[Dict[str, Any]], *, default_enabled: Optional[int] = None) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_REGIME_DETECT_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in DEFAULT_REGIME_DETECT_CFG})
    if default_enabled is not None and "enabled" not in raw:
        out["enabled"] = int(default_enabled)
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["source"] = str(out.get("source", "exo_stress") or "exo_stress").strip().lower()
    if out["source"] != "exo_stress":
        out["source"] = "exo_stress"
    out["stress_lo"] = float(np.clip(safe_float(out.get("stress_lo", 0.25), 0.25), 0.0, 1.0))
    out["stress_hi"] = float(np.clip(safe_float(out.get("stress_hi", 0.65), 0.65), 0.0, 1.0))
    if out["stress_hi"] <= out["stress_lo"]:
        out["stress_hi"] = min(1.0, out["stress_lo"] + 0.05)
        if out["stress_hi"] <= out["stress_lo"]:
            out["stress_lo"] = max(0.0, out["stress_hi"] - 0.05)
    out["alpha_ema"] = float(np.clip(safe_float(out.get("alpha_ema", 0.15), 0.15), 0.0, 1.0))
    out["alpha_hysteresis"] = float(max(0.0, safe_float(out.get("alpha_hysteresis", 0.03), 0.03)))
    ws = np.asarray([
        max(0.0, safe_float(out.get("w_atr", 0.35), 0.35)),
        max(0.0, safe_float(out.get("w_rng", 0.20), 0.20)),
        max(0.0, safe_float(out.get("w_vol", 0.30), 0.30)),
        max(0.0, safe_float(out.get("w_fund", 0.15), 0.15)),
    ], dtype=np.float64)
    s = float(ws.sum())
    if s <= 0.0:
        ws[:] = np.asarray([0.35, 0.20, 0.30, 0.15], dtype=np.float64)
        s = float(ws.sum())
    ws /= max(s, 1e-12)
    out["w_atr"], out["w_rng"], out["w_vol"], out["w_fund"] = [float(x) for x in ws]
    return out


def _normalize_regime_weight_cfg(
    raw: Optional[Dict[str, Any]],
    base_gate_weights: Optional[Dict[str, float]] = None,
    base_dir_weights: Optional[Dict[str, float]] = None,
    available_horizons: Optional[Sequence[int]] = None,
    *,
    default_enabled: Optional[int] = None,
) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_REGIME_WEIGHT_CFG)
    allowed = set(DEFAULT_REGIME_WEIGHT_CFG.keys())
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in allowed and k not in ("gate_calm_anchor", "gate_active_anchor", "dir_calm_anchor", "dir_active_anchor")})
    if default_enabled is not None and "enabled" not in raw:
        out["enabled"] = int(default_enabled)
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    for k, dflt in {"gate_calm_mix": 0.60, "gate_active_mix": 0.55, "dir_calm_mix": 0.35, "dir_active_mix": 0.50}.items():
        out[k] = float(np.clip(safe_float(out.get(k, dflt), dflt), 0.0, 1.0))
    base_gate = normalize_horizon_weights(base_gate_weights or {"w5": 1.0}, fallback={5: 1.0}, available_horizons=available_horizons)
    base_dir = normalize_horizon_weights(base_dir_weights or {"w5": 1.0}, fallback={5: 1.0}, available_horizons=available_horizons)
    out["gate_calm_anchor"] = normalize_horizon_weights(raw.get("gate_calm_anchor", DEFAULT_REGIME_WEIGHT_CFG["gate_calm_anchor"]), fallback={1:0.45,3:0.40,5:0.12,8:0.03,10:0.00}, available_horizons=available_horizons)
    out["gate_active_anchor"] = normalize_horizon_weights(raw.get("gate_active_anchor", DEFAULT_REGIME_WEIGHT_CFG["gate_active_anchor"]), fallback={1:0.08,3:0.28,5:0.32,8:0.22,10:0.10}, available_horizons=available_horizons)
    out["dir_calm_anchor"] = normalize_horizon_weights(raw.get("dir_calm_anchor", DEFAULT_REGIME_WEIGHT_CFG["dir_calm_anchor"]), fallback={1:0.30,3:0.24,5:0.24,8:0.16,10:0.06}, available_horizons=available_horizons)
    out["dir_active_anchor"] = normalize_horizon_weights(raw.get("dir_active_anchor", DEFAULT_REGIME_WEIGHT_CFG["dir_active_anchor"]), fallback={1:0.08,3:0.12,5:0.32,8:0.32,10:0.16}, available_horizons=available_horizons)
    out["gate_base"] = base_gate
    out["dir_base"] = base_dir
    return out


def _normalize_regime_threshold_cfg(raw: Optional[Dict[str, Any]], *, base_q_entry: float, base_entry_th_floor: float, default_enabled: Optional[int] = None) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_REGIME_THRESHOLD_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in DEFAULT_REGIME_THRESHOLD_CFG})
    if default_enabled is not None and "enabled" not in raw:
        out["enabled"] = int(default_enabled)
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["bucket_min_ready"] = max(0, safe_int(out.get("bucket_min_ready", 0), 0))
    out["bucket_fallback_global"] = 1 if safe_int(out.get("bucket_fallback_global", 1), 1) != 0 else 0
    for name in ("calm", "mid", "active"):
        q_key = f"q_entry_{name}"
        floor_key = f"entry_th_floor_{name}"
        q_val = out.get(q_key, None)
        floor_val = out.get(floor_key, None)
        out[q_key] = float(np.clip(safe_float(base_q_entry if q_val is None else q_val, base_q_entry), 0.0, 1.0))
        out[floor_key] = float(safe_float(base_entry_th_floor if floor_val is None else floor_val, base_entry_th_floor))
    return out


def _interp_mode_value(lo: float, hi: float, mode: str) -> float:
    mode = str(mode or "linear").strip().lower()
    if mode in ("linear", "mid", "avg", "average"):
        return 0.5 * (float(lo) + float(hi))
    if mode in ("lo", "calm"):
        return float(lo)
    if mode in ("hi", "active"):
        return float(hi)
    return 0.5 * (float(lo) + float(hi))


def _normalize_regime_filter_cfg(
    raw: Optional[Dict[str, Any]],
    *,
    base_vol_low_th: float,
    base_atr_entry_mult: float,
    base_range_entry_mult: float,
    default_enabled: Optional[int] = None,
) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_REGIME_FILTER_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in DEFAULT_REGIME_FILTER_CFG})
    if default_enabled is not None and "enabled" not in raw:
        out["enabled"] = int(default_enabled)
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["use_vol_split"] = 1 if safe_int(out.get("use_vol_split", 1), 1) != 0 else 0
    out["use_entry_mult_split"] = 1 if safe_int(out.get("use_entry_mult_split", 1), 1) != 0 else 0
    out["mid_interp_mode"] = str(out.get("mid_interp_mode", "linear") or "linear").strip().lower()

    base_vol = safe_float(base_vol_low_th, -1e9)
    base_atr = max(0.0, safe_float(base_atr_entry_mult, 1.0))
    base_range = max(0.0, safe_float(base_range_entry_mult, 1.0))

    vol_calm = safe_float(out.get("vol_low_th_calm", base_vol), base_vol)
    vol_mid = safe_float(out.get("vol_low_th_mid", base_vol), base_vol)
    vol_active = safe_float(out.get("vol_low_th_active", base_vol), base_vol)
    if vol_mid < min(vol_calm, vol_active):
        vol_mid = min(vol_calm, vol_active)
    if vol_mid > max(vol_calm, vol_active):
        vol_mid = max(vol_calm, vol_active)
    ordered = sorted([float(vol_calm), float(vol_mid), float(vol_active)])
    vol_calm, vol_mid, vol_active = ordered[0], ordered[1], ordered[2]
    out["vol_low_th_calm"] = float(vol_calm)
    out["vol_low_th_mid"] = float(vol_mid)
    out["vol_low_th_active"] = float(vol_active)

    atr_calm = max(0.0, safe_float(out.get("atr_entry_mult_calm", base_atr), base_atr))
    atr_active = max(0.0, safe_float(out.get("atr_entry_mult_active", base_atr), base_atr))
    if atr_active < atr_calm:
        atr_calm, atr_active = atr_active, atr_calm
    atr_mid = _interp_mode_value(atr_calm, atr_active, out["mid_interp_mode"])
    out["atr_entry_mult_calm"] = float(atr_calm)
    out["atr_entry_mult_mid"] = float(atr_mid)
    out["atr_entry_mult_active"] = float(atr_active)

    rng_calm = max(0.0, safe_float(out.get("range_entry_mult_calm", base_range), base_range))
    rng_active = max(0.0, safe_float(out.get("range_entry_mult_active", base_range), base_range))
    if rng_active < rng_calm:
        rng_calm, rng_active = rng_active, rng_calm
    rng_mid = _interp_mode_value(rng_calm, rng_active, out["mid_interp_mode"])
    out["range_entry_mult_calm"] = float(rng_calm)
    out["range_entry_mult_mid"] = float(rng_mid)
    out["range_entry_mult_active"] = float(rng_active)
    return out


def build_regime_adaptive_signal_bundle(
    signals_by_h: Dict[int, np.ndarray],
    available_horizons: Sequence[int],
    base_gate_weights: Dict[str, float],
    base_dir_weights: Dict[str, float],
    regime_detect_cfg: Optional[Dict[str, Any]] = None,
    regime_weight_cfg: Optional[Dict[str, Any]] = None,
    *,
    atr_arr: np.ndarray,
    range_arr: np.ndarray,
    vol_arr: np.ndarray,
    funding_arr: np.ndarray,
    atr_high_th: float,
    range_cut: float,
    vol_low_th: float,
    funding_soft_min: float,
    stress_weights: Optional[Dict[str, float]] = None,
    detect_required: bool = False,
) -> Dict[str, Any]:
    avail = tuple(int(h) for h in available_horizons)
    base_gate = normalize_horizon_weights(base_gate_weights, fallback={5: 1.0}, available_horizons=avail)
    base_dir = normalize_horizon_weights(base_dir_weights, fallback={5: 1.0}, available_horizons=avail)
    detect_cfg = _normalize_regime_detect_cfg(regime_detect_cfg or {}, default_enabled=int(detect_required))
    if stress_weights:
        sw = dict(stress_weights)
        detect_cfg["w_atr"] = max(0.0, safe_float(sw.get("w_atr", detect_cfg["w_atr"]), detect_cfg["w_atr"]))
        detect_cfg["w_rng"] = max(0.0, safe_float(sw.get("w_rng", detect_cfg["w_rng"]), detect_cfg["w_rng"]))
        detect_cfg["w_vol"] = max(0.0, safe_float(sw.get("w_vol", detect_cfg["w_vol"]), detect_cfg["w_vol"]))
        detect_cfg["w_fund"] = max(0.0, safe_float(sw.get("w_fund", detect_cfg["w_fund"]), detect_cfg["w_fund"]))
        detect_cfg = _normalize_regime_detect_cfg(detect_cfg, default_enabled=detect_cfg.get("enabled", 0))
    weight_cfg = _normalize_regime_weight_cfg(regime_weight_cfg or {}, base_gate, base_dir, available_horizons=avail)
    n = 0
    for h in avail:
        arr = signals_by_h.get(int(h))
        if arr is not None:
            n = int(len(arr))
            break
    if n <= 0:
        zf = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int8)
        return {"enabled": 0, "detect_enabled": 0, "weight_enabled": 0, "gate_signal_all": zf, "dir_signal_all": zf, "stress_arr": zf, "alpha_arr": zf, "bucket_arr": zi, "gate_calm_profile": base_gate, "gate_active_profile": base_gate, "dir_calm_profile": base_dir, "dir_active_profile": base_dir}

    gate_base_signal = np.zeros(n, dtype=np.float64)
    dir_base_signal = np.zeros(n, dtype=np.float64)
    for h in avail:
        sig = np.asarray(signals_by_h[int(h)], dtype=np.float64)
        gate_base_signal += float(base_gate.get(_weight_key(h), 0.0)) * sig
        dir_base_signal += float(base_dir.get(_weight_key(h), 0.0)) * sig

    need_detect = bool(int(detect_cfg.get("enabled", 0)) != 0 or int(weight_cfg.get("enabled", 0)) != 0 or bool(detect_required))
    if not need_detect:
        z = np.zeros(n, dtype=np.float64)
        return {"enabled": 0, "detect_enabled": 0, "weight_enabled": 0, "gate_signal_all": np.abs(gate_base_signal), "dir_signal_all": dir_base_signal, "stress_arr": z, "alpha_arr": z, "bucket_arr": np.zeros(n, dtype=np.int8), "gate_calm_profile": base_gate, "gate_active_profile": base_gate, "dir_calm_profile": base_dir, "dir_active_profile": base_dir}

    stress_arr, alpha_arr, bucket_arr = build_regime_alpha_exogenous(
        atr_arr=np.asarray(atr_arr, dtype=np.float64),
        range_arr=np.asarray(range_arr, dtype=np.float64),
        vol_arr=np.asarray(vol_arr, dtype=np.float64),
        funding_arr=np.asarray(funding_arr, dtype=np.float64),
        atr_high_th=float(atr_high_th),
        range_cut=float(range_cut),
        vol_low_th=float(vol_low_th),
        funding_soft_min=float(funding_soft_min),
        stress_lo=float(detect_cfg.get("stress_lo", 0.25)),
        stress_hi=float(detect_cfg.get("stress_hi", 0.65)),
        alpha_ema=float(detect_cfg.get("alpha_ema", 0.15)),
        alpha_hysteresis=float(detect_cfg.get("alpha_hysteresis", 0.03)),
        w_atr=float(detect_cfg.get("w_atr", 0.35)),
        w_rng=float(detect_cfg.get("w_rng", 0.20)),
        w_vol=float(detect_cfg.get("w_vol", 0.30)),
        w_fund=float(detect_cfg.get("w_fund", 0.15)),
    )

    if int(weight_cfg.get("enabled", 0)) == 0:
        return {"enabled": 0, "detect_enabled": int(detect_cfg.get("enabled", 0) != 0 or detect_required), "weight_enabled": 0, "gate_signal_all": np.abs(gate_base_signal), "dir_signal_all": dir_base_signal, "stress_arr": stress_arr, "alpha_arr": alpha_arr, "bucket_arr": bucket_arr, "gate_calm_profile": base_gate, "gate_active_profile": base_gate, "dir_calm_profile": base_dir, "dir_active_profile": base_dir}

    gate_calm_profile = _blend_base_to_anchor(base_gate, weight_cfg["gate_calm_anchor"], weight_cfg.get("gate_calm_mix", 0.60), available_horizons=avail)
    gate_active_profile = _blend_base_to_anchor(base_gate, weight_cfg["gate_active_anchor"], weight_cfg.get("gate_active_mix", 0.55), available_horizons=avail)
    dir_calm_profile = _blend_base_to_anchor(base_dir, weight_cfg["dir_calm_anchor"], weight_cfg.get("dir_calm_mix", 0.35), available_horizons=avail)
    dir_active_profile = _blend_base_to_anchor(base_dir, weight_cfg["dir_active_anchor"], weight_cfg.get("dir_active_mix", 0.50), available_horizons=avail)
    gate_sum = np.zeros(n, dtype=np.float64)
    dir_sum = np.zeros(n, dtype=np.float64)
    one_minus_alpha = 1.0 - alpha_arr
    for h in avail:
        sig = np.asarray(signals_by_h[int(h)], dtype=np.float64)
        k = _weight_key(int(h))
        gate_w_series = one_minus_alpha * float(gate_calm_profile.get(k, 0.0)) + alpha_arr * float(gate_active_profile.get(k, 0.0))
        dir_w_series = one_minus_alpha * float(dir_calm_profile.get(k, 0.0)) + alpha_arr * float(dir_active_profile.get(k, 0.0))
        gate_sum += gate_w_series * sig
        dir_sum += dir_w_series * sig
    return {"enabled": 1, "detect_enabled": 1, "weight_enabled": 1, "gate_signal_all": np.abs(gate_sum), "dir_signal_all": dir_sum, "stress_arr": stress_arr, "alpha_arr": alpha_arr, "bucket_arr": bucket_arr, "gate_calm_profile": gate_calm_profile, "gate_active_profile": gate_active_profile, "dir_calm_profile": dir_calm_profile, "dir_active_profile": dir_active_profile}


def build_bucketed_entry_threshold_pack(
    gate_signal_all: np.ndarray,
    ready: np.ndarray,
    bucket_arr: Optional[np.ndarray],
    *,
    hist_start: int,
    hist_end: int,
    seg_start: int,
    seg_end: int,
    q_entry: float,
    entry_th_floor: float,
    entry_q_min_ready: int,
    regime_threshold_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_signal_all = np.asarray(gate_signal_all, dtype=np.float64)
    ready = np.asarray(ready, dtype=bool)
    seg_start = int(seg_start)
    seg_end = int(seg_end)
    seg_len = max(0, seg_end - seg_start)
    thr_entry_global = quantile_from_history(
        gate_signal_all,
        ready,
        int(hist_start),
        int(hist_end),
        float(q_entry),
        int(entry_q_min_ready),
        float(entry_th_floor),
    )
    threshold_cfg = _normalize_regime_threshold_cfg(
        regime_threshold_cfg,
        base_q_entry=float(q_entry),
        base_entry_th_floor=float(entry_th_floor),
    )
    entry_threshold_base_seg = np.full(seg_len, float(thr_entry_global), dtype=np.float64)
    entry_q_used_seg = np.full(seg_len, float(q_entry), dtype=np.float64)
    entry_floor_used_seg = np.full(seg_len, float(entry_th_floor), dtype=np.float64)
    bucket_seg = np.full(seg_len, -1, dtype=np.int8)
    hist_bucket = None
    if bucket_arr is not None:
        bucket_all = np.asarray(bucket_arr, dtype=np.int8)
        if seg_len > 0:
            bucket_seg = np.asarray(bucket_all[seg_start:seg_end], dtype=np.int8)
        hist_bucket = np.asarray(bucket_all[int(hist_start):int(hist_end)], dtype=np.int8)
    bucket_min_ready = int(threshold_cfg.get("bucket_min_ready", 0))
    if bucket_min_ready <= 0:
        bucket_min_ready = int(entry_q_min_ready)
    bucket_meta: Dict[str, Any] = {}
    if int(threshold_cfg.get("enabled", 0)) != 0 and hist_bucket is not None and seg_len > 0:
        hist_vals = np.asarray(gate_signal_all[int(hist_start):int(hist_end)], dtype=np.float64)
        hist_ready = np.asarray(ready[int(hist_start):int(hist_end)], dtype=bool)
        for bucket_id, bucket_name in REGIME_BUCKET_NAMES.items():
            q_key = f"q_entry_{bucket_name}"
            floor_key = f"entry_th_floor_{bucket_name}"
            q_bucket = float(threshold_cfg.get(q_key, q_entry))
            floor_bucket = float(threshold_cfg.get(floor_key, entry_th_floor))
            hist_mask = hist_ready & (hist_bucket == int(bucket_id))
            hist_count = int(np.sum(hist_mask))
            use_global = False
            if hist_count >= int(bucket_min_ready):
                thr_bucket = quantile_from_values(hist_vals, hist_mask, q_bucket, int(bucket_min_ready), floor_bucket)
                q_eff = q_bucket
                floor_eff = floor_bucket
            elif int(threshold_cfg.get("bucket_fallback_global", 1)) != 0:
                thr_bucket = float(thr_entry_global)
                q_eff = float(q_entry)
                floor_eff = float(entry_th_floor)
                use_global = True
            else:
                thr_bucket = float(floor_bucket)
                q_eff = float(q_bucket)
                floor_eff = float(floor_bucket)
            seg_mask = bucket_seg == int(bucket_id)
            if np.any(seg_mask):
                entry_threshold_base_seg[seg_mask] = float(thr_bucket)
                entry_q_used_seg[seg_mask] = float(q_eff)
                entry_floor_used_seg[seg_mask] = float(floor_eff)
            bucket_meta[bucket_name] = {
                "hist_ready_count": int(hist_count),
                "seg_bar_count": int(np.sum(seg_mask)),
                "threshold": float(thr_bucket),
                "q_used": float(q_eff),
                "floor_used": float(floor_eff),
                "used_global_fallback": int(use_global),
            }
    else:
        hist_ready = np.asarray(ready[int(hist_start):int(hist_end)], dtype=bool)
        for bucket_id, bucket_name in REGIME_BUCKET_NAMES.items():
            seg_mask = bucket_seg == int(bucket_id)
            hist_count = 0 if hist_bucket is None else int(np.sum((hist_bucket == int(bucket_id)) & hist_ready))
            bucket_meta[bucket_name] = {
                "hist_ready_count": int(hist_count),
                "seg_bar_count": int(np.sum(seg_mask)),
                "threshold": float(thr_entry_global),
                "q_used": float(q_entry),
                "floor_used": float(entry_th_floor),
                "used_global_fallback": 1,
            }
    return {
        "thr_entry_global": float(thr_entry_global),
        "entry_threshold_base_seg": entry_threshold_base_seg,
        "entry_q_used_seg": entry_q_used_seg,
        "entry_floor_used_seg": entry_floor_used_seg,
        "bucket_seg": bucket_seg,
        "bucket_meta": bucket_meta,
        "bucket_min_ready": int(bucket_min_ready),
        "bucket_fallback_global": int(threshold_cfg.get("bucket_fallback_global", 1)),
        "threshold_enabled": int(threshold_cfg.get("enabled", 0)),
    }


def build_bucketed_filter_pack(
    *,
    bucket_arr: Optional[np.ndarray],
    seg_start: int,
    seg_end: int,
    regime_filter_cfg: Optional[Dict[str, Any]],
    base_vol_low_th: float,
    base_atr_entry_mult: float,
    base_range_entry_mult: float,
) -> Dict[str, Any]:
    seg_start = int(seg_start)
    seg_end = int(seg_end)
    seg_len = max(0, seg_end - seg_start)
    cfg = _normalize_regime_filter_cfg(
        regime_filter_cfg,
        base_vol_low_th=float(base_vol_low_th),
        base_atr_entry_mult=float(base_atr_entry_mult),
        base_range_entry_mult=float(base_range_entry_mult),
    )
    vol_arr = np.full(seg_len, float(base_vol_low_th), dtype=np.float64)
    atr_arr = np.full(seg_len, float(base_atr_entry_mult), dtype=np.float64)
    rng_arr = np.full(seg_len, float(base_range_entry_mult), dtype=np.float64)
    bucket_seg = np.full(seg_len, -1, dtype=np.int8)
    if bucket_arr is not None and seg_len > 0:
        bucket_all = np.asarray(bucket_arr, dtype=np.int8)
        bucket_seg = np.asarray(bucket_all[seg_start:seg_end], dtype=np.int8)
    if int(cfg.get("enabled", 0)) != 0 and seg_len > 0:
        if int(cfg.get("use_vol_split", 1)) != 0:
            vol_arr[bucket_seg == 0] = float(cfg.get("vol_low_th_calm", base_vol_low_th))
            vol_arr[bucket_seg == 1] = float(cfg.get("vol_low_th_mid", base_vol_low_th))
            vol_arr[bucket_seg == 2] = float(cfg.get("vol_low_th_active", base_vol_low_th))
        if int(cfg.get("use_entry_mult_split", 1)) != 0:
            atr_arr[bucket_seg == 0] = float(cfg.get("atr_entry_mult_calm", base_atr_entry_mult))
            atr_arr[bucket_seg == 1] = float(cfg.get("atr_entry_mult_mid", base_atr_entry_mult))
            atr_arr[bucket_seg == 2] = float(cfg.get("atr_entry_mult_active", base_atr_entry_mult))
            rng_arr[bucket_seg == 0] = float(cfg.get("range_entry_mult_calm", base_range_entry_mult))
            rng_arr[bucket_seg == 1] = float(cfg.get("range_entry_mult_mid", base_range_entry_mult))
            rng_arr[bucket_seg == 2] = float(cfg.get("range_entry_mult_active", base_range_entry_mult))
    bucket_meta = {}
    for bid, bname in REGIME_BUCKET_NAMES.items():
        mask = bucket_seg == int(bid)
        bucket_meta[bname] = {
            "seg_bar_count": int(np.sum(mask)),
            "vol_low_th": float(np.mean(vol_arr[mask])) if np.any(mask) else float(base_vol_low_th),
            "atr_entry_mult": float(np.mean(atr_arr[mask])) if np.any(mask) else float(base_atr_entry_mult),
            "range_entry_mult": float(np.mean(rng_arr[mask])) if np.any(mask) else float(base_range_entry_mult),
        }
    return {
        "enabled": int(cfg.get("enabled", 0)),
        "use_vol_split": int(cfg.get("use_vol_split", 1)),
        "use_entry_mult_split": int(cfg.get("use_entry_mult_split", 1)),
        "mid_interp_mode": str(cfg.get("mid_interp_mode", "linear")),
        "vol_low_th_arr": vol_arr,
        "atr_entry_mult_arr": atr_arr,
        "range_entry_mult_arr": rng_arr,
        "filter_bucket_seg": bucket_seg,
        "bucket_meta": bucket_meta,
        "cfg": cfg,
    }


# extend search-bounds aliases for v2a / v2bc
_RANGE_KEY_ALIASES.update({
    "tune_regime_detection": "tune_regime_detection",
    "tune_regime_thresholds": "tune_regime_thresholds",
    "tune_regime_filters": "tune_regime_filters",
    "regime_detect_enabled_min": "regime_detect_enabled_min",
    "regime_detect_enabled_max": "regime_detect_enabled_max",
    "regime_threshold_enabled_min": "regime_threshold_enabled_min",
    "regime_threshold_enabled_max": "regime_threshold_enabled_max",
    "regime_filter_enabled_min": "regime_filter_enabled_min",
    "regime_filter_enabled_max": "regime_filter_enabled_max",
    "bucket_min_ready_min": "bucket_min_ready_min",
    "bucket_min_ready_max": "bucket_min_ready_max",
    "bucket_fallback_global_min": "bucket_fallback_global_min",
    "bucket_fallback_global_max": "bucket_fallback_global_max",
    "q_entry_calm_min": "q_entry_calm_min",
    "q_entry_calm_max": "q_entry_calm_max",
    "q_entry_mid_min": "q_entry_mid_min",
    "q_entry_mid_max": "q_entry_mid_max",
    "q_entry_active_min": "q_entry_active_min",
    "q_entry_active_max": "q_entry_active_max",
    "q_entry_calm_delta_min": "q_entry_calm_delta_min",
    "q_entry_calm_delta_max": "q_entry_calm_delta_max",
    "q_entry_mid_delta_min": "q_entry_mid_delta_min",
    "q_entry_mid_delta_max": "q_entry_mid_delta_max",
    "q_entry_active_delta_min": "q_entry_active_delta_min",
    "q_entry_active_delta_max": "q_entry_active_delta_max",
    "entry_th_calm_min": "entry_th_calm_min",
    "entry_th_calm_max": "entry_th_calm_max",
    "entry_th_mid_min": "entry_th_mid_min",
    "entry_th_mid_max": "entry_th_mid_max",
    "entry_th_active_min": "entry_th_active_min",
    "entry_th_active_max": "entry_th_active_max",
    "entry_th_calm_delta_min": "entry_th_calm_delta_min",
    "entry_th_calm_delta_max": "entry_th_calm_delta_max",
    "entry_th_mid_delta_min": "entry_th_mid_delta_min",
    "entry_th_mid_delta_max": "entry_th_mid_delta_max",
    "entry_th_active_delta_min": "entry_th_active_delta_min",
    "entry_th_active_delta_max": "entry_th_active_delta_max",
    "regime_filter_use_vol_split_min": "regime_filter_use_vol_split_min",
    "regime_filter_use_vol_split_max": "regime_filter_use_vol_split_max",
    "regime_filter_use_entry_mult_split_min": "regime_filter_use_entry_mult_split_min",
    "regime_filter_use_entry_mult_split_max": "regime_filter_use_entry_mult_split_max",
    "vol_low_th_calm_min": "vol_low_th_calm_min",
    "vol_low_th_calm_max": "vol_low_th_calm_max",
    "vol_low_th_mid_min": "vol_low_th_mid_min",
    "vol_low_th_mid_max": "vol_low_th_mid_max",
    "vol_low_th_active_min": "vol_low_th_active_min",
    "vol_low_th_active_max": "vol_low_th_active_max",
    "vol_low_th_calm_delta_min": "vol_low_th_calm_delta_min",
    "vol_low_th_calm_delta_max": "vol_low_th_calm_delta_max",
    "vol_low_th_mid_delta_min": "vol_low_th_mid_delta_min",
    "vol_low_th_mid_delta_max": "vol_low_th_mid_delta_max",
    "vol_low_th_active_delta_min": "vol_low_th_active_delta_min",
    "vol_low_th_active_delta_max": "vol_low_th_active_delta_max",
    "atr_entry_mult_calm_min": "atr_entry_mult_calm_min",
    "atr_entry_mult_calm_max": "atr_entry_mult_calm_max",
    "atr_entry_mult_active_min": "atr_entry_mult_active_min",
    "atr_entry_mult_active_max": "atr_entry_mult_active_max",
    "atr_entry_mult_calm_delta_min": "atr_entry_mult_calm_delta_min",
    "atr_entry_mult_calm_delta_max": "atr_entry_mult_calm_delta_max",
    "atr_entry_mult_active_delta_min": "atr_entry_mult_active_delta_min",
    "atr_entry_mult_active_delta_max": "atr_entry_mult_active_delta_max",
    "range_entry_mult_calm_min": "range_entry_mult_calm_min",
    "range_entry_mult_calm_max": "range_entry_mult_calm_max",
    "range_entry_mult_active_min": "range_entry_mult_active_min",
    "range_entry_mult_active_max": "range_entry_mult_active_max",
    "range_entry_mult_calm_delta_min": "range_entry_mult_calm_delta_min",
    "range_entry_mult_calm_delta_max": "range_entry_mult_calm_delta_max",
    "range_entry_mult_active_delta_min": "range_entry_mult_active_delta_min",
    "range_entry_mult_active_delta_max": "range_entry_mult_active_delta_max",
})


_normalize_single_config_from_any_base = normalize_single_config_from_any


def normalize_single_config_from_any(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = copy.deepcopy(raw_cfg or {})
    cfg = _normalize_single_config_from_any_base(raw_cfg)
    merged = dict(raw_cfg)
    merged_regime_weight_raw = dict(merged.get("regime_weight_cfg", {}) if isinstance(merged.get("regime_weight_cfg", {}), dict) else {})
    merged_regime_detect_raw = dict(merged.get("regime_detect_cfg", {}) if isinstance(merged.get("regime_detect_cfg", {}), dict) else {})
    merged_regime_threshold_raw = dict(merged.get("regime_threshold_cfg", {}) if isinstance(merged.get("regime_threshold_cfg", {}), dict) else {})
    merged_regime_filter_raw = dict(merged.get("regime_filter_cfg", {}) if isinstance(merged.get("regime_filter_cfg", {}), dict) else {})

    if not merged_regime_weight_raw and isinstance(cfg.get("regime_weight_cfg", {}), dict):
        merged_regime_weight_raw = dict(cfg.get("regime_weight_cfg", {}))
    if not merged_regime_detect_raw and isinstance(cfg.get("regime_detect_cfg", {}), dict):
        merged_regime_detect_raw = dict(cfg.get("regime_detect_cfg", {}))
    if not merged_regime_threshold_raw and isinstance(cfg.get("regime_threshold_cfg", {}), dict):
        merged_regime_threshold_raw = dict(cfg.get("regime_threshold_cfg", {}))
    if not merged_regime_filter_raw and isinstance(cfg.get("regime_filter_cfg", {}), dict):
        merged_regime_filter_raw = dict(cfg.get("regime_filter_cfg", {}))

    detect_seed = dict(merged_regime_detect_raw or {})
    if (not detect_seed) and merged_regime_weight_raw:
        detect_seed = {k: copy.deepcopy(v) for k, v in merged_regime_weight_raw.items() if k in REGIME_DETECT_KEYS}

    cfg["regime_detect_cfg"] = _normalize_regime_detect_cfg(
        detect_seed,
        default_enabled=safe_int(merged_regime_weight_raw.get("enabled", cfg.get("regime_detect_cfg", {}).get("enabled", 0)), 0),
    )
    cfg["regime_weight_cfg"] = _normalize_regime_weight_cfg(
        merged_regime_weight_raw if merged_regime_weight_raw else cfg.get("regime_weight_cfg", {}),
        base_gate_weights=cfg.get("gate_weights", {"w5": 1.0}),
        base_dir_weights=cfg.get("dir_weights", {"w5": 1.0}),
        default_enabled=safe_int(merged_regime_weight_raw.get("enabled", cfg.get("regime_weight_cfg", {}).get("enabled", 0)), 0),
    )
    cfg["regime_threshold_cfg"] = _normalize_regime_threshold_cfg(
        merged_regime_threshold_raw if merged_regime_threshold_raw else cfg.get("regime_threshold_cfg", {}),
        base_q_entry=float(cfg.get("q_entry", 0.85)),
        base_entry_th_floor=float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0))),
        default_enabled=safe_int((merged_regime_threshold_raw or {}).get("enabled", cfg.get("regime_threshold_cfg", {}).get("enabled", 0)), 0),
    )
    cfg["regime_filter_cfg"] = _normalize_regime_filter_cfg(
        merged_regime_filter_raw if merged_regime_filter_raw else cfg.get("regime_filter_cfg", {}),
        base_vol_low_th=float(cfg.get("risk_cfg", {}).get("vol_low_th", -1e9)),
        base_atr_entry_mult=float(cfg.get("atr_entry_mult", 1.0)),
        base_range_entry_mult=float(cfg.get("range_entry_mult", 1.0)),
        default_enabled=safe_int((merged_regime_filter_raw or {}).get("enabled", cfg.get("regime_filter_cfg", {}).get("enabled", 0)), 0),
    )
    if any(int(cfg.get(k, {}).get("enabled", 0)) != 0 for k in ("regime_weight_cfg", "regime_threshold_cfg", "regime_filter_cfg")) and int(cfg.get("regime_detect_cfg", {}).get("enabled", 0)) == 0:
        cfg["regime_detect_cfg"]["enabled"] = 1
    cfg["schema"] = "single_v90"
    return cfg


def build_dynamic_arrays(
    dynamic_cfg: Dict[str, Any],
    gate_strength_seg: np.ndarray,
    thr_entry: float,
    atr_seg: np.ndarray,
    atr_high_th: float,
    range_seg: np.ndarray,
    range_cut: float,
    vol_seg: np.ndarray,
    vol_low_th: float,
    funding_seg: np.ndarray,
    thr_entry_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = _normalize_dynamic_cfg(dynamic_cfg)
    n = int(len(gate_strength_seg))
    gate_mult_arr = np.ones(n, dtype=np.float64)
    lev_scale_arr = np.ones(n, dtype=np.float64)
    bep_scale_arr = np.ones(n, dtype=np.float64)
    trail_scale_arr = np.ones(n, dtype=np.float64)
    sl_scale_arr = np.ones(n, dtype=np.float64)
    softsl_relax_arr = np.zeros(n, dtype=np.int64)
    stress_arr = np.zeros(n, dtype=np.float64)
    if cfg.get("enabled", 0) == 0:
        return gate_mult_arr, lev_scale_arr, bep_scale_arr, trail_scale_arr, sl_scale_arr, softsl_relax_arr, stress_arr

    eps = 1e-12
    strength = np.asarray(gate_strength_seg, dtype=np.float64)
    if thr_entry_arr is not None:
        thr_ref_arr = np.maximum(np.asarray(thr_entry_arr, dtype=np.float64), eps)
    else:
        thr_ref_arr = np.full(int(len(strength)), max(float(thr_entry), eps), dtype=np.float64)
    trend_raw = np.maximum((strength / thr_ref_arr) - 1.0, 0.0)
    trend = np.clip(trend_raw / max(float(cfg.get("margin_cap", 0.5)), 1e-6), 0.0, 1.0)

    if np.isfinite(float(atr_high_th)) and float(atr_high_th) > 0.0:
        atr_stress = np.clip(np.maximum((np.asarray(atr_seg, dtype=np.float64) / max(float(atr_high_th), eps)) - 1.0, 0.0), 0.0, 1.0)
    else:
        atr_stress = np.zeros(n, dtype=np.float64)

    if np.isfinite(float(range_cut)) and float(range_cut) > 0.0:
        range_stress = np.clip(np.maximum((np.asarray(range_seg, dtype=np.float64) / max(float(range_cut), eps)) - 1.0, 0.0), 0.0, 1.0)
    else:
        range_stress = np.zeros(n, dtype=np.float64)

    vol_soft = float(vol_low_th) if np.isfinite(float(vol_low_th)) else 0.0
    vol_norm = max(abs(vol_soft), 1.0)
    vol_deficit = np.clip(np.maximum(vol_soft - np.asarray(vol_seg, dtype=np.float64), 0.0) / vol_norm, 0.0, 1.0)

    funding_soft = max(float(cfg.get("funding_soft_min", 0.0)), 0.0)
    if funding_soft > 0.0:
        funding_stress = np.where(
            np.asarray(funding_seg, dtype=np.float64) < funding_soft,
            1.0 - np.clip(np.asarray(funding_seg, dtype=np.float64) / max(funding_soft, eps), 0.0, 1.0),
            0.0,
        )
    else:
        funding_stress = np.zeros(n, dtype=np.float64)

    stress = (
        float(cfg["w_atr"]) * atr_stress
        + float(cfg["w_rng"]) * range_stress
        + float(cfg["w_vol"]) * vol_deficit
        + float(cfg["w_fund"]) * funding_stress
    )
    stress = np.clip(stress, 0.0, 1.0)
    stress_arr[:] = stress

    if int(cfg.get("use_dyn_gate", 1)) == 1:
        gate_mult_arr[:] = np.clip(
            1.0 + float(cfg.get("gate_stress_k", 0.12)) * stress - float(cfg.get("gate_trend_k", 0.08)) * trend,
            float(cfg.get("gate_mult_min", 0.95)),
            float(cfg.get("gate_mult_max", 1.15)),
        )

    if int(cfg.get("use_margin_gate", 0)) == 1:
        margin_req = np.clip(
            float(cfg.get("margin_req_base", 0.0))
            + float(cfg.get("margin_req_stress_k", 0.0)) * stress
            - float(cfg.get("margin_req_trend_k", 0.0)) * trend,
            0.0,
            float(cfg.get("margin_req_max", 0.0)),
        )
        gate_mult_arr[:] = gate_mult_arr * (1.0 + margin_req)

    if int(cfg.get("use_dyn_lev", 1)) == 1:
        lev_scale_arr[:] = np.clip(
            1.0 + float(cfg.get("lev_trend_k", 0.18)) * trend - float(cfg.get("lev_stress_k", 0.30)) * stress,
            float(cfg.get("lev_scale_min", 0.70)),
            float(cfg.get("lev_scale_max", 1.05)),
        )

    if int(cfg.get("use_margin_lev_degrade", 0)) == 1:
        eff_thr = np.maximum(thr_ref_arr * gate_mult_arr, eps)
        surplus = np.maximum((strength / eff_thr) - 1.0, 0.0)
        band = max(float(cfg.get("margin_lev_band", 0.05)), 1e-6)
        lev_floor = float(np.clip(cfg.get("margin_lev_floor", 0.70), 0.0, 1.0))
        lev_margin_scale = lev_floor + (1.0 - lev_floor) * np.clip(surplus / band, 0.0, 1.0)
        lev_scale_arr[:] = np.clip(lev_scale_arr * lev_margin_scale, 0.05, float(cfg.get("lev_scale_max", 1.05)))

    if int(cfg.get("use_dyn_bep", 1)) == 1:
        bep_scale_arr[:] = np.clip(
            1.0 - float(cfg.get("bep_stress_k", 0.0)) * stress + float(cfg.get("bep_trend_k", 0.05)) * trend,
            float(cfg.get("bep_scale_min", 0.75)),
            float(cfg.get("bep_scale_max", 1.05)),
        )

    if int(cfg.get("use_dyn_trail", 1)) == 1:
        trail_scale_arr[:] = np.clip(
            1.0 + float(cfg.get("trail_trend_k", 0.15)) * trend - float(cfg.get("trail_stress_k", 0.0)) * stress,
            float(cfg.get("trail_scale_min", 0.90)),
            float(cfg.get("trail_scale_max", 1.12)),
        )

    if int(cfg.get("use_dyn_sl", 0)) == 1:
        sl_scale_arr[:] = np.clip(
            1.0 + float(cfg.get("sl_trend_k", 0.0)) * trend - float(cfg.get("sl_stress_k", 0.0)) * stress,
            float(cfg.get("sl_scale_min", 0.85)),
            float(cfg.get("sl_scale_max", 1.05)),
        )

    if int(cfg.get("use_dyn_soft_sl", 0)) == 1:
        stress_mid = float(cfg.get("softsl_stress_mid", 0.35))
        stress_hi = float(cfg.get("softsl_stress_hi", 0.65))
        relax_mid = int(max(0, cfg.get("softsl_relax_mid", 1)))
        relax_hi = int(max(relax_mid, cfg.get("softsl_relax_hi", 2)))
        softsl_relax_arr[:] = np.where(stress >= stress_mid, relax_mid, 0)
        softsl_relax_arr[:] = np.where(stress >= stress_hi, relax_hi, softsl_relax_arr)

    return gate_mult_arr, lev_scale_arr, bep_scale_arr, trail_scale_arr, sl_scale_arr, softsl_relax_arr, stress_arr


# ---------------------------------------------------------------------------
# v33 appended objective alignment + tp_window + entry_episode helpers
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveBreakdown:
    score_raw: float
    score_cost_mean: float
    score_cost_worst: float
    penalty_min_seg_trades: float
    penalty_min_short_trades: float
    penalty_min_short_share: float
    penalty_regime_extreme: float
    penalty_bottom2_trades: float
    penalty_seg_trade_floor: float
    penalty_trade_cv: float
    bottom2_mean_trades: float
    seg_trade_mean: float
    seg_trade_std: float
    seg_trade_cv: float
    seg_trade_min: int
    seg_trade_max: int
    objective_final: float
    feasible_min_seg_trades: bool
    feasible_short_trades: bool
    feasible_short_share: bool
    feasible_all: bool


def _safe_trade_cv(seg_trades_arr: np.ndarray) -> Tuple[float, float, float]:
    vals = np.asarray(seg_trades_arr, dtype=np.float64)
    if vals.size <= 0:
        return 0.0, 0.0, 0.0
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))
    if mean_v <= 1e-12:
        return mean_v, std_v, 0.0
    return mean_v, std_v, float(std_v / mean_v)


def assemble_objective(
    raw_score: float,
    *,
    score_cost_mean: Optional[float] = None,
    score_cost_worst: Optional[float] = None,
    min_seg_seen: int = 0,
    total_short: int = 0,
    short_share_all: float = 0.0,
    regime_calm_frac: Optional[float] = None,
    regime_active_frac: Optional[float] = None,
    seg_trades: Optional[Sequence[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ObjectiveBreakdown:
    cfg = dict(cfg or {})
    score_raw = float(raw_score)
    score_cost_mean = float(score_cost_mean if score_cost_mean is not None else raw_score)
    score_cost_worst = float(score_cost_worst if score_cost_worst is not None else raw_score)

    seg_trades_arr = np.asarray(seg_trades if seg_trades is not None else [], dtype=np.float64)
    if seg_trades_arr.size > 0:
        seg_sorted = np.sort(seg_trades_arr)
        seg_trade_min = int(seg_sorted[0])
        seg_trade_max = int(seg_sorted[-1])
        bottom_k = 2 if seg_sorted.size >= 2 else 1
        bottom2_mean_trades = float(np.mean(seg_sorted[:bottom_k]))
        seg_trade_mean, seg_trade_std, seg_trade_cv = _safe_trade_cv(seg_trades_arr)
        min_seg_seen_eff = int(seg_trade_min)
    else:
        min_seg_seen_eff = int(min_seg_seen)
        seg_trade_min = int(min_seg_seen_eff)
        seg_trade_max = int(min_seg_seen_eff)
        bottom2_mean_trades = float(min_seg_seen_eff)
        seg_trade_mean = float(min_seg_seen_eff) if min_seg_seen_eff > 0 else 0.0
        seg_trade_std = 0.0
        seg_trade_cv = 0.0

    min_seg_target = max(0, safe_int(cfg.get("min_seg_trades", 0), 0))
    min_seg_mode = str(cfg.get("min_seg_trades_mode", "hard") or "hard").strip().lower()
    min_seg_penalty_k = float(max(0.0, safe_float(cfg.get("min_seg_trades_penalty_k", 1.0), 1.0)))
    min_seg_penalty_power = float(max(1.0, safe_float(cfg.get("min_seg_trades_penalty_power", 1.0), 1.0)))
    hard_base = float(max(0.0, safe_float(cfg.get("hard_guard_base", 1e6), 1e6)))
    hard_step = float(max(0.0, safe_float(cfg.get("hard_guard_step", 1.0), 1.0)))

    feasible_min_seg = True
    penalty_min_seg = 0.0
    if min_seg_target > 0 and int(min_seg_seen_eff) < int(min_seg_target):
        feasible_min_seg = False
        gap = float(int(min_seg_target) - int(min_seg_seen_eff))
        if min_seg_mode == "soft":
            penalty_min_seg = float(min_seg_penalty_k) * (gap ** float(min_seg_penalty_power))
        else:
            penalty_min_seg = float(hard_base) + float(hard_step) * gap

    min_short_target = max(0, safe_int(cfg.get("min_short_trades_global", 0), 0))
    short_guard_mode = str(cfg.get("short_trades_guard_mode", "hard") or "hard").strip().lower()
    short_trades_penalty_k = float(max(0.0, safe_float(cfg.get("short_trades_penalty_k", 1.0), 1.0)))
    short_trades_penalty_power = float(max(1.0, safe_float(cfg.get("short_trades_penalty_power", 1.0), 1.0)))

    feasible_short_trades = True
    penalty_min_short_trades = 0.0
    if min_short_target > 0 and int(total_short) < int(min_short_target):
        feasible_short_trades = False
        gap = float(int(min_short_target) - int(total_short))
        if short_guard_mode == "soft":
            penalty_min_short_trades = float(short_trades_penalty_k) * (gap ** float(short_trades_penalty_power))
        else:
            penalty_min_short_trades = float(hard_base) + float(hard_step) * gap

    min_short_share = float(max(0.0, safe_float(cfg.get("min_short_share_global", 0.0), 0.0)))
    short_share_guard_mode = str(cfg.get("short_share_guard_mode", "hard") or "hard").strip().lower()
    short_share_penalty_k = float(max(0.0, safe_float(cfg.get("short_share_penalty_k", 1000.0), 1000.0)))
    short_share_penalty_power = float(max(1.0, safe_float(cfg.get("short_share_penalty_power", 1.0), 1.0)))

    feasible_short_share = True
    penalty_min_short_share = 0.0
    if min_short_share > 0.0 and float(short_share_all) < float(min_short_share):
        feasible_short_share = False
        gap = float(min_short_share) - float(short_share_all)
        if short_share_guard_mode == "soft":
            penalty_min_short_share = float(short_share_penalty_k) * (gap ** float(short_share_penalty_power))
        else:
            penalty_min_short_share = float(hard_base) + 1000.0 * gap

    regime_extreme_penalty = 0.0
    regime_extreme_penalty_k = float(max(0.0, safe_float(cfg.get("regime_extreme_penalty_k", 0.0), 0.0)))
    regime_extreme_max_frac = float(np.clip(safe_float(cfg.get("regime_extreme_max_frac", 1.0), 1.0), 0.0, 1.0))
    if regime_extreme_penalty_k > 0.0:
        if regime_calm_frac is not None:
            rc = float(np.clip(safe_float(regime_calm_frac, 0.0), 0.0, 1.0))
            if rc > regime_extreme_max_frac:
                regime_extreme_penalty += regime_extreme_penalty_k * ((rc - regime_extreme_max_frac) / max(1.0 - regime_extreme_max_frac, 1e-9)) ** 2
        if regime_active_frac is not None:
            ra = float(np.clip(safe_float(regime_active_frac, 0.0), 0.0, 1.0))
            if ra > regime_extreme_max_frac:
                regime_extreme_penalty += regime_extreme_penalty_k * ((ra - regime_extreme_max_frac) / max(1.0 - regime_extreme_max_frac, 1e-9)) ** 2

    seg_bottom2_target = float(max(0.0, safe_float(cfg.get("seg_bottom2_target", 0.0), 0.0)))
    seg_bottom2_penalty_k = float(max(0.0, safe_float(cfg.get("seg_bottom2_penalty_k", 0.0), 0.0)))
    penalty_bottom2_trades = 0.0
    if seg_bottom2_target > 0.0 and seg_bottom2_penalty_k > 0.0 and float(bottom2_mean_trades) < seg_bottom2_target:
        penalty_bottom2_trades = seg_bottom2_penalty_k * ((seg_bottom2_target - float(bottom2_mean_trades)) / max(seg_bottom2_target, 1e-12))

    seg_floor_target = float(max(0.0, safe_float(cfg.get("seg_floor_target", 0.0), 0.0)))
    seg_floor_penalty_k = float(max(0.0, safe_float(cfg.get("seg_floor_penalty_k", 0.0), 0.0)))
    penalty_seg_trade_floor = 0.0
    if seg_floor_target > 0.0 and seg_floor_penalty_k > 0.0 and float(min_seg_seen_eff) < seg_floor_target:
        penalty_seg_trade_floor = seg_floor_penalty_k * ((seg_floor_target - float(min_seg_seen_eff)) / max(seg_floor_target, 1e-12))

    trade_cv_cap = float(max(0.0, safe_float(cfg.get("trade_cv_cap", 0.0), 0.0)))
    trade_cv_penalty_k = float(max(0.0, safe_float(cfg.get("trade_cv_penalty_k", 0.0), 0.0)))
    penalty_trade_cv = 0.0
    if trade_cv_cap > 0.0 and trade_cv_penalty_k > 0.0 and float(seg_trade_mean) > 1e-12 and float(seg_trade_cv) > trade_cv_cap:
        penalty_trade_cv = trade_cv_penalty_k * ((float(seg_trade_cv) - trade_cv_cap) / max(trade_cv_cap, 1e-12))

    objective_final = float(
        score_raw
        - penalty_min_seg
        - penalty_min_short_trades
        - penalty_min_short_share
        - regime_extreme_penalty
        - penalty_bottom2_trades
        - penalty_seg_trade_floor
        - penalty_trade_cv
    )
    feasible_all = bool(feasible_min_seg and feasible_short_trades and feasible_short_share)
    return ObjectiveBreakdown(
        score_raw=float(score_raw),
        score_cost_mean=float(score_cost_mean),
        score_cost_worst=float(score_cost_worst),
        penalty_min_seg_trades=float(penalty_min_seg),
        penalty_min_short_trades=float(penalty_min_short_trades),
        penalty_min_short_share=float(penalty_min_short_share),
        penalty_regime_extreme=float(regime_extreme_penalty),
        penalty_bottom2_trades=float(penalty_bottom2_trades),
        penalty_seg_trade_floor=float(penalty_seg_trade_floor),
        penalty_trade_cv=float(penalty_trade_cv),
        bottom2_mean_trades=float(bottom2_mean_trades),
        seg_trade_mean=float(seg_trade_mean),
        seg_trade_std=float(seg_trade_std),
        seg_trade_cv=float(seg_trade_cv),
        seg_trade_min=int(seg_trade_min),
        seg_trade_max=int(seg_trade_max),
        objective_final=float(objective_final),
        feasible_min_seg_trades=bool(feasible_min_seg),
        feasible_short_trades=bool(feasible_short_trades),
        feasible_short_share=bool(feasible_short_share),
        feasible_all=bool(feasible_all),
    )


DEFAULT_TP_WINDOW_CFG: Dict[str, Any] = {
    "enabled": 0,
    "progress_frac_arm": 0.70,
    "extend_bars": 0,
    "block_early_trail": 1,
    "block_early_soft_sl": 1,
    "floor_trail_hold_to_tp": 1,
    "floor_soft_sl_hold_to_tp": 1,
    "suspend_post_bep_shield_before_tp": 1,
    "expire_on_pullback_frac": 0.35,
}

DEFAULT_ENTRY_EPISODE_CFG: Dict[str, Any] = {
    "enabled": 0,
    "rearm_enabled": 0,
    "run_gap_reset_bars": 1,
    "episode_max_entries_per_run": 1,
    "rearm_same_side_only": 1,
    "rearm_cooldown_bars": 1,
    "rearm_max_bars_after_exit": 8,
    "rearm_gate_reset_frac": 0.45,
    "rearm_gate_refresh_frac": 0.70,
    "rearm_price_reset_frac": 0.0004,
    "rearm_after_trail": 1,
    "rearm_after_tp": 1,
    "rearm_after_sl": 0,
}


def _normalize_tp_window_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_TP_WINDOW_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in out})
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["progress_frac_arm"] = float(np.clip(safe_float(out.get("progress_frac_arm", 0.70), 0.70), 0.0, 2.0))
    out["extend_bars"] = max(0, safe_int(out.get("extend_bars", 0), 0))
    out["block_early_trail"] = 1 if safe_int(out.get("block_early_trail", 1), 1) != 0 else 0
    out["block_early_soft_sl"] = 1 if safe_int(out.get("block_early_soft_sl", 1), 1) != 0 else 0
    out["floor_trail_hold_to_tp"] = 1 if safe_int(out.get("floor_trail_hold_to_tp", 1), 1) != 0 else 0
    out["floor_soft_sl_hold_to_tp"] = 1 if safe_int(out.get("floor_soft_sl_hold_to_tp", 1), 1) != 0 else 0
    out["suspend_post_bep_shield_before_tp"] = 1 if safe_int(out.get("suspend_post_bep_shield_before_tp", 1), 1) != 0 else 0
    out["expire_on_pullback_frac"] = float(np.clip(safe_float(out.get("expire_on_pullback_frac", 0.35), 0.35), 0.0, 1.0))
    return out


def _normalize_entry_episode_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_ENTRY_EPISODE_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in out})
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["rearm_enabled"] = 1 if safe_int(out.get("rearm_enabled", 0), 0) != 0 else 0
    out["run_gap_reset_bars"] = max(0, safe_int(out.get("run_gap_reset_bars", 1), 1))
    out["episode_max_entries_per_run"] = max(1, safe_int(out.get("episode_max_entries_per_run", 1), 1))
    out["rearm_same_side_only"] = 1 if safe_int(out.get("rearm_same_side_only", 1), 1) != 0 else 0
    out["rearm_cooldown_bars"] = max(0, safe_int(out.get("rearm_cooldown_bars", 1), 1))
    out["rearm_max_bars_after_exit"] = max(0, safe_int(out.get("rearm_max_bars_after_exit", 8), 8))
    out["rearm_gate_reset_frac"] = float(np.clip(safe_float(out.get("rearm_gate_reset_frac", 0.45), 0.45), 0.0, 1.5))
    out["rearm_gate_refresh_frac"] = float(np.clip(safe_float(out.get("rearm_gate_refresh_frac", 0.70), 0.70), 0.0, 2.0))
    out["rearm_price_reset_frac"] = float(max(0.0, safe_float(out.get("rearm_price_reset_frac", 0.0004), 0.0004)))
    out["rearm_after_trail"] = 1 if safe_int(out.get("rearm_after_trail", 1), 1) != 0 else 0
    out["rearm_after_tp"] = 1 if safe_int(out.get("rearm_after_tp", 1), 1) != 0 else 0
    out["rearm_after_sl"] = 1 if safe_int(out.get("rearm_after_sl", 0), 0) != 0 else 0
    if out["enabled"] == 0:
        out["rearm_enabled"] = 0
    return out


_RANGE_KEY_ALIASES.update({
    "tune_entry_episode": "tune_entry_episode",
    "tp_window_progress_frac_arm_min": "tp_window_progress_frac_arm_min",
    "tp_window_progress_frac_arm_max": "tp_window_progress_frac_arm_max",
    "tp_window_expire_on_pullback_frac_min": "tp_window_expire_on_pullback_frac_min",
    "tp_window_expire_on_pullback_frac_max": "tp_window_expire_on_pullback_frac_max",
    "entry_episode_enabled_min": "entry_episode_enabled_min",
    "entry_episode_enabled_max": "entry_episode_enabled_max",
    "rearm_enabled_min": "rearm_enabled_min",
    "rearm_enabled_max": "rearm_enabled_max",
    "run_gap_reset_bars_min": "run_gap_reset_bars_min",
    "run_gap_reset_bars_max": "run_gap_reset_bars_max",
    "episode_max_entries_per_run_min": "episode_max_entries_per_run_min",
    "episode_max_entries_per_run_max": "episode_max_entries_per_run_max",
    "rearm_same_side_only_min": "rearm_same_side_only_min",
    "rearm_same_side_only_max": "rearm_same_side_only_max",
    "rearm_cooldown_bars_min": "rearm_cooldown_bars_min",
    "rearm_cooldown_bars_max": "rearm_cooldown_bars_max",
    "rearm_max_bars_after_exit_min": "rearm_max_bars_after_exit_min",
    "rearm_max_bars_after_exit_max": "rearm_max_bars_after_exit_max",
    "rearm_gate_reset_frac_min": "rearm_gate_reset_frac_min",
    "rearm_gate_reset_frac_max": "rearm_gate_reset_frac_max",
    "rearm_gate_refresh_frac_min": "rearm_gate_refresh_frac_min",
    "rearm_gate_refresh_frac_max": "rearm_gate_refresh_frac_max",
    "rearm_price_reset_frac_min": "rearm_price_reset_frac_min",
    "rearm_price_reset_frac_max": "rearm_price_reset_frac_max",
    "rearm_after_trail_min": "rearm_after_trail_min",
    "rearm_after_trail_max": "rearm_after_trail_max",
    "rearm_after_tp_min": "rearm_after_tp_min",
    "rearm_after_tp_max": "rearm_after_tp_max",
    "rearm_after_sl_min": "rearm_after_sl_min",
    "rearm_after_sl_max": "rearm_after_sl_max",
})


_normalize_single_config_from_any_v33_base = normalize_single_config_from_any


def normalize_single_config_from_any(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_single_config_from_any_v33_base(raw_cfg)
    raw_cfg = copy.deepcopy(raw_cfg or {})
    cfg["tp_window_cfg"] = _normalize_tp_window_cfg(raw_cfg.get("tp_window_cfg", cfg.get("tp_window_cfg", {})))
    cfg["entry_episode_cfg"] = _normalize_entry_episode_cfg(raw_cfg.get("entry_episode_cfg", cfg.get("entry_episode_cfg", {})))
    cfg["schema"] = "single_v100"
    return cfg



def _rearm_reason_code(reason: int) -> int:
    if int(reason) == int(EXIT_TRAIL):
        return 1
    if int(reason) == int(EXIT_TP):
        return 2
    if int(reason) == int(EXIT_SL):
        return 3
    if int(reason) == int(EXIT_RISK):
        return 4
    if int(reason) == int(EXIT_FORCE):
        return 5
    if int(reason) == int(EXIT_MAXH):
        return 6
    return 0


def _rearm_exit_allowed(reason: int, rearm_after_trail: int, rearm_after_tp: int, rearm_after_sl: int) -> bool:
    if int(reason) == int(EXIT_TRAIL):
        return bool(int(rearm_after_trail) != 0)
    if int(reason) == int(EXIT_TP):
        return bool(int(rearm_after_tp) != 0)
    if int(reason) == int(EXIT_SL):
        return bool(int(rearm_after_sl) != 0)
    return False


# ---------------------------------------------------------------------------
# v33 appended enhanced detailed simulator (tp_window + entry_episode)
# ---------------------------------------------------------------------------

_simulate_trading_core_rl_single_v33_base = simulate_trading_core_rl_single


def simulate_trading_core_rl_single(
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    gate_strength: np.ndarray,
    dir_signal: np.ndarray,
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    atr_high_th: float,
    vol_low_th: float,
    funding_near_min: float,
    risk_lev_cap: float,
    base_leverage: float,
    cost_per_side: float,
    maker_fee_per_side: float,
    slip_per_side: float,
    fee_tp_mult: float,
    bep_arm_fee_mult: float,
    bep_stop_fee_mult: float,
    bep_stop_mode: str,
    atr_entry_mult: float,
    range_entry_mult: float,
    low_vol_filter: int,
    trail_after_bep: int,
    risk_entry_mode: int,
    use_atr_scaling: int,
    lev_mult: float,
    TP: float,
    SL: float,
    bep_arm_base: float,
    trailing: float,
    min_hold_bars: int,
    min_hold_trail_bars: int,
    min_hold_soft_sl_bars: int,
    max_hold_bars: int,
    dyn_lev_scale_arr: np.ndarray,
    dyn_bep_scale_arr: np.ndarray,
    dyn_trail_scale_arr: np.ndarray,
    dyn_sl_scale_arr: np.ndarray,
    dyn_softsl_relax_arr: np.ndarray,
    dyn_gate_mult_arr: np.ndarray,
    dyn_stress_arr: np.ndarray,
    use_pre_bep_timeout: int,
    pre_bep_timeout_bars: int,
    pre_bep_stress_th: float,
    pre_bep_progress_frac: float,
    pre_bep_degrade_sl_scale: float,
    pre_bep_softsl_delta: int,
    pre_bep_force_close_bars: int,
    pre_bep_force_close_red_only: int,
    dyn_mode_code: int,
    allow_soft_sl_before_trail: int,
    softsl_hold_floor: int,
    post_bep_shield_ignore_softsl_hold: int,
    hard_sl_mult_pre_unlock: float,
    trail_grace_after_bep: int,
    trail_grace_after_unlock: int,
    early_softsl_enabled: int = 0,
    early_softsl_min_hold: int = 2,
    early_softsl_progress_frac: float = 0.5,
    early_trail_enabled: int = 0,
    early_trail_min_hold: int = 3,
    early_trail_progress_frac: float = 0.85,
    early_trail_ref_updates_min: int = 1,
    stop_equity: float = 0.4,
    stop_dd: float = 0.35,
    warmup_steps: int = 0,
    integer_leverage: int = 0,
    seg_start: int = 0,
    intrabar_mode: int = 1,
    regime_alpha_arr: Optional[np.ndarray] = None,
    regime_bucket_arr: Optional[np.ndarray] = None,
    tp_window_enabled: int = 0,
    tp_window_progress_frac_arm: float = 0.70,
    tp_window_extend_bars: int = 0,
    tp_window_block_early_trail: int = 1,
    tp_window_block_early_soft_sl: int = 1,
    tp_window_floor_trail_hold_to_tp: int = 1,
    tp_window_floor_soft_sl_hold_to_tp: int = 1,
    tp_window_suspend_post_bep_shield_before_tp: int = 1,
    tp_window_expire_on_pullback_frac: float = 0.35,
    entry_episode_enabled: int = 0,
    rearm_enabled: int = 0,
    run_gap_reset_bars: int = 1,
    episode_max_entries_per_run: int = 1,
    rearm_same_side_only: int = 1,
    rearm_cooldown_bars: int = 1,
    rearm_max_bars_after_exit: int = 8,
    rearm_gate_reset_frac: float = 0.45,
    rearm_gate_refresh_frac: float = 0.70,
    rearm_price_reset_frac: float = 0.0004,
    rearm_after_trail: int = 1,
    rearm_after_tp: int = 1,
    rearm_after_sl: int = 0,
    same_side_hold_enabled: int = 0,
    same_side_hold_weak_enabled: int = 1,
    same_side_hold_strong_ratio: float = 1.0,
    same_side_hold_weak_ratio: float = 0.82,
    same_side_hold_weak_min_progress_frac: float = 0.35,
    same_side_hold_allow_pre_bep_weak: int = 1,
    same_side_hold_pre_bep_max_bonus_bars: int = 1,
    same_side_hold_bonus_bars_strong: int = 2,
    same_side_hold_bonus_bars_weak: int = 1,
    same_side_hold_max_extra_bars: int = 4,
    same_side_hold_grace_after_bep_strong: int = 1,
    same_side_hold_grace_after_bep_weak: int = 0,
    same_side_hold_grace_after_unlock_strong: int = 1,
    same_side_hold_grace_after_unlock_weak: int = 0,
    support_strength_ratio_arr: Optional[np.ndarray] = None,
    support_weak_eligible_mask: Optional[np.ndarray] = None,
    support_pass_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if (
        int(tp_window_enabled) == 0
        and int(entry_episode_enabled) == 0
        and int(rearm_enabled) == 0
        and int(same_side_hold_enabled) == 0
    ):
        return _simulate_trading_core_rl_single_v33_base(
            open_=open_,
            close=close,
            high=high,
            low=low,
            gate_strength=gate_strength,
            dir_signal=dir_signal,
            ready=ready,
            vol_z=vol_z,
            atr_rel=atr_rel,
            minutes_to_next_funding=minutes_to_next_funding,
            atr_high_th=atr_high_th,
            vol_low_th=vol_low_th,
            funding_near_min=funding_near_min,
            risk_lev_cap=risk_lev_cap,
            base_leverage=base_leverage,
            cost_per_side=cost_per_side,
            maker_fee_per_side=maker_fee_per_side,
            slip_per_side=slip_per_side,
            fee_tp_mult=fee_tp_mult,
            bep_arm_fee_mult=bep_arm_fee_mult,
            bep_stop_fee_mult=bep_stop_fee_mult,
            bep_stop_mode=bep_stop_mode,
            atr_entry_mult=atr_entry_mult,
            range_entry_mult=range_entry_mult,
            low_vol_filter=low_vol_filter,
            trail_after_bep=trail_after_bep,
            risk_entry_mode=risk_entry_mode,
            use_atr_scaling=use_atr_scaling,
            lev_mult=lev_mult,
            TP=TP,
            SL=SL,
            bep_arm_base=bep_arm_base,
            trailing=trailing,
            min_hold_bars=min_hold_bars,
            min_hold_trail_bars=min_hold_trail_bars,
            min_hold_soft_sl_bars=min_hold_soft_sl_bars,
            max_hold_bars=max_hold_bars,
            dyn_lev_scale_arr=dyn_lev_scale_arr,
            dyn_bep_scale_arr=dyn_bep_scale_arr,
            dyn_trail_scale_arr=dyn_trail_scale_arr,
            dyn_sl_scale_arr=dyn_sl_scale_arr,
            dyn_softsl_relax_arr=dyn_softsl_relax_arr,
            dyn_gate_mult_arr=dyn_gate_mult_arr,
            dyn_stress_arr=dyn_stress_arr,
            use_pre_bep_timeout=use_pre_bep_timeout,
            pre_bep_timeout_bars=pre_bep_timeout_bars,
            pre_bep_stress_th=pre_bep_stress_th,
            pre_bep_progress_frac=pre_bep_progress_frac,
            pre_bep_degrade_sl_scale=pre_bep_degrade_sl_scale,
            pre_bep_softsl_delta=pre_bep_softsl_delta,
            pre_bep_force_close_bars=pre_bep_force_close_bars,
            pre_bep_force_close_red_only=pre_bep_force_close_red_only,
            dyn_mode_code=dyn_mode_code,
            allow_soft_sl_before_trail=allow_soft_sl_before_trail,
            softsl_hold_floor=softsl_hold_floor,
            post_bep_shield_ignore_softsl_hold=post_bep_shield_ignore_softsl_hold,
            hard_sl_mult_pre_unlock=hard_sl_mult_pre_unlock,
            trail_grace_after_bep=trail_grace_after_bep,
            trail_grace_after_unlock=trail_grace_after_unlock,
            early_softsl_enabled=early_softsl_enabled,
            early_softsl_min_hold=early_softsl_min_hold,
            early_softsl_progress_frac=early_softsl_progress_frac,
            early_trail_enabled=early_trail_enabled,
            early_trail_min_hold=early_trail_min_hold,
            early_trail_progress_frac=early_trail_progress_frac,
            early_trail_ref_updates_min=early_trail_ref_updates_min,
            stop_equity=stop_equity,
            stop_dd=stop_dd,
            warmup_steps=warmup_steps,
            integer_leverage=integer_leverage,
            seg_start=seg_start,
            intrabar_mode=intrabar_mode,
            regime_alpha_arr=regime_alpha_arr,
            regime_bucket_arr=regime_bucket_arr,
        )

    n = int(len(close))
    exit_cnt = np.zeros(6, dtype=np.int64)
    exit_gross_sum = np.zeros(6, dtype=np.float64)
    exit_fee_sum = np.zeros(6, dtype=np.float64)
    exit_net_sum = np.zeros(6, dtype=np.float64)
    trade_logs: List[Dict[str, Any]] = []

    if support_strength_ratio_arr is None:
        support_strength_ratio_arr = np.zeros(n, dtype=np.float64)
    else:
        support_strength_ratio_arr = np.asarray(support_strength_ratio_arr, dtype=np.float64)
    if support_weak_eligible_mask is None:
        support_weak_eligible_mask = np.zeros(n, dtype=np.bool_)
    else:
        support_weak_eligible_mask = np.asarray(support_weak_eligible_mask, dtype=np.bool_)
    if support_pass_mask is None:
        support_pass_mask = np.zeros(n, dtype=np.bool_)
    else:
        support_pass_mask = np.asarray(support_pass_mask, dtype=np.bool_)

    if n < 2:
        return {
            "net_ret": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wins": 0,
            "tail_hit": 0,
            "exit_cnt": exit_cnt,
            "exit_gross_sum": exit_gross_sum,
            "exit_fee_sum": exit_fee_sum,
            "exit_net_sum": exit_net_sum,
            "trail_before_bep": 0,
            "trail_after_bep": 0,
            "bep_armed_trades": 0,
            "ref_updates": 0,
            "trade_logs": trade_logs,
            "long_trades": 0,
            "short_trades": 0,
            "maxh_cnt": 0,
            "tp_window_armed_trades": 0,
            "tp_window_live_bars_total": 0,
            "tp_window_blocked_early_trail": 0,
            "tp_window_blocked_softsl": 0,
            "rearm_entries": 0,
            "rearm_entries_after_trail": 0,
            "rearm_entries_after_tp": 0,
            "rearm_entries_after_sl": 0,
            "same_side_hold_events": 0,
            "same_side_hold_strong_events": 0,
            "same_side_hold_weak_events": 0,
        }

    range_rel = (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) / np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    atr_med = float(np.median(atr_rel)) if len(atr_rel) else 1.0
    range_med = float(np.median(range_rel)) if len(range_rel) else 1.0
    if atr_med <= 0.0:
        atr_med = 1.0
    if range_med <= 0.0:
        range_med = 1.0

    taker_fee_side = float(cost_per_side) + float(slip_per_side)
    fee_roundtrip = 2.0 * taker_fee_side
    econ_be_fee = _bep_econ_fee(taker_fee_side, maker_fee_per_side, bep_stop_mode)

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trade_cnt = 0
    win_cnt = 0
    tail_hit = 0
    long_trades = 0
    short_trades = 0
    maxh_cnt = 0
    trail_before_bep_cnt = 0
    trail_after_bep_cnt = 0
    bep_armed_trades = 0
    ref_updates = 0
    pos_ref_updates_local = 0
    tp_window_armed_trades = 0
    tp_window_live_bars_total = 0
    tp_window_blocked_early_trail_total = 0
    tp_window_blocked_softsl_total = 0
    rearm_entries = 0
    rearm_entries_after_trail = 0
    rearm_entries_after_tp = 0
    rearm_entries_after_sl = 0

    pos_side = 0
    entry_price = 0.0
    ref_price = 0.0
    entry_i = -1
    entry_decision_i = -1
    entry_lev = 0.0
    bep_armed = 0
    bep_armed_at = -1

    pos_TP = 0.0
    pos_SL = 0.0
    pos_BEP_ARM = 0.0
    pos_TR = 0.0
    pos_min_hold = 0
    pos_min_hold_trail = 0
    pos_min_hold_soft_sl = 0
    pos_max_hold = 0
    pos_bep_arm_fee = 0.0
    pos_bep_stop_fee = 0.0

    pos_SL_base = 0.0
    pos_BEP_ARM_base = 0.0
    pos_TR_base = 0.0
    pos_bep_arm_fee_base = 0.0
    pos_min_hold_soft_sl_base = 0

    pend_side = 0
    pend_i = -1
    pend_lev = 0.0
    pend_TP = 0.0
    pend_SL = 0.0
    pend_BEP_ARM = 0.0
    pend_TR = 0.0
    pend_min_hold = 0
    pend_min_hold_trail = 0
    pend_min_hold_soft_sl = 0
    pend_max_hold = 0
    pend_bep_arm_fee = 0.0
    pend_bep_stop_fee = 0.0
    pend_SL_base = 0.0
    pend_BEP_ARM_base = 0.0
    pend_TR_base = 0.0
    pend_bep_arm_fee_base = 0.0
    pend_min_hold_soft_sl_base = 0

    entry_gate_strength = 0.0
    entry_dir_signal = 0.0
    entry_atr_rel = 0.0
    entry_vol_z = 0.0
    entry_bep_arm_val = 0.0
    entry_dyn_lev_scale = 1.0
    entry_dyn_bep_scale = 1.0
    entry_dyn_trail_scale = 1.0
    entry_dyn_sl_scale = 1.0
    entry_dyn_gate_mult = 1.0
    entry_dyn_stress = 0.0
    entry_min_hold_soft_sl_init = 0
    entry_regime_alpha = float("nan")
    entry_regime_bucket = -1
    entry_run_id = -1
    entry_episode_idx_in_run = 0
    entry_is_rearm = 0
    entry_rearm_reason_code = 0
    entry_bars_since_last_exit = -1
    entry_run_peak_gate = 0.0
    entry_gate_reset_seen = 0
    entry_price_reset_frac = 0.0
    entry_last_exit_reason = -1

    pend_gate_strength = 0.0
    pend_dir_signal = 0.0
    pend_atr_rel = 0.0
    pend_vol_z = 0.0
    pend_dyn_lev_scale = 1.0
    pend_dyn_bep_scale = 1.0
    pend_dyn_trail_scale = 1.0
    pend_dyn_sl_scale = 1.0
    pend_dyn_gate_mult = 1.0
    pend_dyn_stress = 0.0
    pend_decision_i = -1
    pend_run_id = -1
    pend_episode_idx_in_run = 0
    pend_is_rearm = 0
    pend_rearm_reason_code = 0
    pend_bars_since_last_exit = -1
    pend_run_peak_gate = 0.0
    pend_gate_reset_seen = 0
    pend_price_reset_frac = 0.0
    pend_last_exit_reason = -1

    mfe_rel = 0.0
    mae_rel = 0.0
    mfe_prev_rel = 0.0
    pre_bep_degraded = 0
    pre_bep_force_closed = 0
    pre_bep_last_progress_prev = 0.0

    tpw_live = 0
    tpw_armed_at = -1
    tpw_progress_at_arm = 0.0
    tpw_peak_progress = 0.0
    tpw_live_bars_local = 0
    tpw_blocked_early_trail_local = 0
    tpw_blocked_softsl_local = 0

    pos_same_side_hold_floor_trail = 0
    pos_same_side_grace_bep_until_i = -1
    pos_same_side_grace_unlock_until_i = -1
    same_side_hold_events = 0
    same_side_hold_strong_events = 0
    same_side_hold_weak_events = 0
    same_side_hold_last_class = 0
    same_side_hold_last_strength_ratio = 0.0
    same_side_hold_events_local = 0
    same_side_hold_strong_events_local = 0
    same_side_hold_weak_events_local = 0

    run_id = 0
    run_active = 0
    run_start_i = -1
    run_last_i = -1
    run_side = 0
    run_peak_gate = 0.0
    run_entries = 0
    run_gap_bars = 0
    run_reset_seen_after_exit = 0

    last_exit_i = -10**9
    last_exit_reason = -1
    last_exit_side = 0
    last_exit_price = 0.0
    last_run_id = -1

    entry_episode_enabled = 1 if int(entry_episode_enabled) != 0 else 0
    rearm_enabled = 1 if int(rearm_enabled) != 0 else 0
    run_gap_reset_bars = max(0, int(run_gap_reset_bars))
    episode_max_entries_per_run = max(1, int(episode_max_entries_per_run))
    rearm_cooldown_bars = max(0, int(rearm_cooldown_bars))
    rearm_max_bars_after_exit = max(0, int(rearm_max_bars_after_exit))
    tp_window_enabled = 1 if int(tp_window_enabled) != 0 else 0
    tp_window_extend_bars = max(0, int(tp_window_extend_bars))

    for i in range(n - 1):
        if pend_side != 0 and pend_i == i:
            pos_side = int(pend_side)
            pend_side = 0
            entry_price = float(open_[i])
            ref_price = float(open_[i])
            entry_i = int(i)
            entry_decision_i = int(pend_decision_i)
            entry_lev = float(pend_lev)
            bep_armed = 0
            bep_armed_at = -1

            pos_TP = float(pend_TP)
            pos_SL = float(pend_SL)
            pos_BEP_ARM = float(pend_BEP_ARM)
            pos_TR = float(pend_TR)
            pos_min_hold = int(pend_min_hold)
            pos_min_hold_trail = int(pend_min_hold_trail)
            pos_min_hold_soft_sl = int(pend_min_hold_soft_sl)
            pos_max_hold = int(pend_max_hold)
            pos_bep_arm_fee = float(pend_bep_arm_fee)
            pos_bep_stop_fee = float(pend_bep_stop_fee)

            pos_SL_base = float(pend_SL_base)
            pos_BEP_ARM_base = float(pend_BEP_ARM_base)
            pos_TR_base = float(pend_TR_base)
            pos_bep_arm_fee_base = float(pend_bep_arm_fee_base)
            pos_min_hold_soft_sl_base = int(pend_min_hold_soft_sl_base)

            entry_gate_strength = float(pend_gate_strength)
            entry_dir_signal = float(pend_dir_signal)
            entry_atr_rel = float(pend_atr_rel)
            entry_vol_z = float(pend_vol_z)
            entry_dyn_lev_scale = float(pend_dyn_lev_scale)
            entry_dyn_bep_scale = float(pend_dyn_bep_scale)
            entry_dyn_trail_scale = float(pend_dyn_trail_scale)
            entry_dyn_sl_scale = float(pend_dyn_sl_scale)
            entry_dyn_gate_mult = float(pend_dyn_gate_mult)
            entry_dyn_stress = float(pend_dyn_stress)
            entry_min_hold_soft_sl_init = int(pend_min_hold_soft_sl)
            if regime_alpha_arr is not None and 0 <= int(entry_decision_i) < int(len(regime_alpha_arr)):
                entry_regime_alpha = float(regime_alpha_arr[int(entry_decision_i)])
            else:
                entry_regime_alpha = float("nan")
            if regime_bucket_arr is not None and 0 <= int(entry_decision_i) < int(len(regime_bucket_arr)):
                entry_regime_bucket = int(regime_bucket_arr[int(entry_decision_i)])
            else:
                entry_regime_bucket = -1
            entry_run_id = int(pend_run_id)
            entry_episode_idx_in_run = int(pend_episode_idx_in_run)
            entry_is_rearm = int(pend_is_rearm)
            entry_rearm_reason_code = int(pend_rearm_reason_code)
            entry_bars_since_last_exit = int(pend_bars_since_last_exit)
            entry_run_peak_gate = float(pend_run_peak_gate)
            entry_gate_reset_seen = int(pend_gate_reset_seen)
            entry_price_reset_frac = float(pend_price_reset_frac)
            entry_last_exit_reason = int(pend_last_exit_reason)
            if int(entry_is_rearm) == 1:
                rearm_entries += 1
                if int(entry_last_exit_reason) == int(EXIT_TRAIL):
                    rearm_entries_after_trail += 1
                elif int(entry_last_exit_reason) == int(EXIT_TP):
                    rearm_entries_after_tp += 1
                elif int(entry_last_exit_reason) == int(EXIT_SL):
                    rearm_entries_after_sl += 1

            arm_v = float(pos_BEP_ARM)
            if arm_v < float(pos_bep_arm_fee):
                arm_v = float(pos_bep_arm_fee)
            entry_bep_arm_val = float(arm_v)
            pos_ref_updates_local = 0
            mfe_rel = 0.0
            mae_rel = 0.0
            mfe_prev_rel = 0.0
            pre_bep_degraded = 0
            pre_bep_force_closed = 0
            pre_bep_last_progress_prev = 0.0
            tpw_live = 0
            tpw_armed_at = -1
            tpw_progress_at_arm = 0.0
            tpw_peak_progress = 0.0
            tpw_live_bars_local = 0
            tpw_blocked_early_trail_local = 0
            tpw_blocked_softsl_local = 0
            pos_same_side_hold_floor_trail = 0
            pos_same_side_grace_bep_until_i = -1
            pos_same_side_grace_unlock_until_i = -1
            same_side_hold_last_class = 0
            same_side_hold_last_strength_ratio = 0.0
            same_side_hold_events_local = 0
            same_side_hold_strong_events_local = 0
            same_side_hold_weak_events_local = 0

        hi = float(high[i])
        lo = float(low[i])
        c = float(close[i])

        if pos_side != 0:
            side = int(pos_side)
            ep = float(entry_price)
            if ep > 0.0:
                den = max(ep, 1e-12)
                if side == 1:
                    fav = (hi - ep) / den
                    adv = (lo - ep) / den
                else:
                    fav = (ep - lo) / den
                    adv = (ep - hi) / den
                if fav > mfe_rel:
                    mfe_rel = fav
                if adv < mae_rel:
                    mae_rel = adv
            lev_now = float(entry_lev)
            hold = int(i - entry_i)
            if hold < 0:
                hold = 0

            if int(dyn_mode_code) == 1 and i > entry_i:
                j = i - 1
                cand_sl = float(pos_SL_base) * float(dyn_sl_scale_arr[j])
                min_sl_live = 0.6 * fee_roundtrip
                if cand_sl < min_sl_live:
                    cand_sl = min_sl_live
                if cand_sl < pos_SL:
                    pos_SL = float(cand_sl)

                cand_bep_arm_fee = float(pos_bep_arm_fee_base) * float(dyn_bep_scale_arr[j])
                if cand_bep_arm_fee < 0.0:
                    cand_bep_arm_fee = 0.0
                if cand_bep_arm_fee < pos_bep_arm_fee:
                    pos_bep_arm_fee = float(cand_bep_arm_fee)

                cand_bep = float(pos_BEP_ARM_base) * float(dyn_bep_scale_arr[j])
                if cand_bep < pos_bep_arm_fee:
                    cand_bep = pos_bep_arm_fee
                if cand_bep < pos_BEP_ARM:
                    pos_BEP_ARM = float(cand_bep)

                cand_tr = float(pos_TR_base) * float(dyn_trail_scale_arr[j])
                if int(trail_after_bep) == 0:
                    min_tr_live = max(float(econ_be_fee), fee_roundtrip * float(fee_tp_mult))
                    if cand_tr < min_tr_live:
                        cand_tr = min_tr_live
                if cand_tr < pos_TR:
                    pos_TR = float(cand_tr)

                cand_soft = resolve_local_soft_sl_hold(
                    int(pos_min_hold_soft_sl_base),
                    int(pos_min_hold_trail),
                    int(dyn_softsl_relax_arr[j]),
                    int(allow_soft_sl_before_trail),
                    int(softsl_hold_floor),
                )
                if cand_soft < pos_min_hold_soft_sl:
                    pos_min_hold_soft_sl = int(cand_soft)

            force_close_now = False
            if int(use_pre_bep_timeout) == 1 and bep_armed == 0 and i > entry_i:
                stress_prev = float(dyn_stress_arr[i - 1])
                progress_prev = 0.0
                if entry_bep_arm_val > 1e-12:
                    progress_prev = float(mfe_prev_rel / entry_bep_arm_val)
                pre_bep_last_progress_prev = float(progress_prev)
                timeout_hit = (
                    hold >= int(pre_bep_timeout_bars)
                    and stress_prev >= float(pre_bep_stress_th)
                    and progress_prev < float(pre_bep_progress_frac)
                )
                if timeout_hit:
                    pre_bep_degraded = 1
                    cand_sl = float(pos_SL_base) * float(pre_bep_degrade_sl_scale)
                    min_sl_live = 0.6 * fee_roundtrip
                    if cand_sl < min_sl_live:
                        cand_sl = min_sl_live
                    if cand_sl < pos_SL:
                        pos_SL = float(cand_sl)
                    if int(pre_bep_softsl_delta) > 0:
                        cand_soft = resolve_local_soft_sl_hold(
                            int(pos_min_hold_soft_sl_base),
                            int(pos_min_hold_trail),
                            int(pre_bep_softsl_delta),
                            int(allow_soft_sl_before_trail),
                            int(softsl_hold_floor),
                        )
                        if cand_soft < pos_min_hold_soft_sl:
                            pos_min_hold_soft_sl = int(cand_soft)
                    if int(pre_bep_force_close_bars) > 0 and hold >= int(pre_bep_force_close_bars):
                        unreal = side * (c - ep) / max(ep, 1e-12)
                        if (int(pre_bep_force_close_red_only) == 0) or (unreal <= 0.0):
                            force_close_now = True
                            pre_bep_force_closed = 1

            TP_v = float(pos_TP)
            SL_v = float(pos_SL)
            BEP_ARM_v = float(pos_BEP_ARM)
            TR_v = float(pos_TR)
            min_hold_soft = int(pos_min_hold_soft_sl)

            tp_price = ep * (1.0 + side * TP_v)
            sl_price = ep * (1.0 - side * SL_v)
            bep_stop_price = ep * (1.0 + side * float(pos_bep_stop_fee))

            if bep_armed == 1:
                if side == 1:
                    if sl_price < bep_stop_price:
                        sl_price = bep_stop_price
                else:
                    if sl_price > bep_stop_price:
                        sl_price = bep_stop_price

            hard_sl_dist = max(float(SL_v) * float(max(hard_sl_mult_pre_unlock, 0.0)), 0.6 * fee_roundtrip)
            hard_sl = ep * (1.0 - side * hard_sl_dist)

            progress_soft = 0.0
            if entry_bep_arm_val > 1e-12:
                progress_soft = float(mfe_rel / entry_bep_arm_val)
            progress_trail_den = max(float(TR_v), float(entry_bep_arm_val), 1e-12)
            progress_trail = float(mfe_rel / progress_trail_den)
            raw_early_softsl_now = (
                int(early_softsl_enabled) == 1
                and hold >= int(early_softsl_min_hold)
                and progress_soft >= float(early_softsl_progress_frac)
            )
            raw_early_trail_now = (
                int(early_trail_enabled) == 1
                and hold >= int(early_trail_min_hold)
                and progress_trail >= float(early_trail_progress_frac)
                and pos_ref_updates_local >= int(early_trail_ref_updates_min)
            )

            if int(tp_window_enabled) == 1 and float(TP_v) > 1e-12:
                tp_progress = float(mfe_rel / max(float(TP_v), 1e-12))
                cur_progress = 0.0
                if ep > 0.0:
                    cur_progress = max(0.0, side * (c - ep) / max(ep, 1e-12) / max(float(TP_v), 1e-12))
                if tp_progress > tpw_peak_progress:
                    tpw_peak_progress = float(tp_progress)
                if tpw_live == 0 and tp_progress >= float(tp_window_progress_frac_arm):
                    tpw_live = 1
                    tpw_armed_at = int(i)
                    tpw_progress_at_arm = float(tp_progress)
                    tpw_peak_progress = float(tp_progress)
                if tpw_live == 1:
                    tpw_live_bars_local = int(i - tpw_armed_at + 1)
                    expire_allowed = tpw_live_bars_local > int(tp_window_extend_bars)
                    if expire_allowed and float(tp_window_expire_on_pullback_frac) > 0.0 and tpw_peak_progress > 0.0:
                        pullback_ok = cur_progress < (tpw_peak_progress * (1.0 - float(tp_window_expire_on_pullback_frac)))
                        if pullback_ok:
                            tpw_live = 0
                else:
                    tpw_live_bars_local = 0
            else:
                tpw_live = 0
                tpw_live_bars_local = 0

            early_softsl_now = raw_early_softsl_now
            early_trail_now = raw_early_trail_now
            if tpw_live == 1 and int(tp_window_block_early_soft_sl) == 1 and raw_early_softsl_now:
                early_softsl_now = False
                tpw_blocked_softsl_local = 1
                tp_window_blocked_softsl_total += 1
            if tpw_live == 1 and int(tp_window_block_early_trail) == 1 and raw_early_trail_now:
                early_trail_now = False
                tpw_blocked_early_trail_local = 1
                tp_window_blocked_early_trail_total += 1
            if tpw_live == 1 and int(tp_window_floor_trail_hold_to_tp) == 1:
                pos_min_hold_trail = max(int(pos_min_hold_trail), int(pos_min_hold))
            if tpw_live == 1 and int(tp_window_floor_soft_sl_hold_to_tp) == 1:
                min_hold_soft = max(int(min_hold_soft), int(pos_min_hold))

            eff_pos_min_hold_trail = int(pos_min_hold_trail)
            if int(same_side_hold_enabled) == 1 and int(pos_same_side_hold_floor_trail) > int(eff_pos_min_hold_trail):
                eff_pos_min_hold_trail = int(pos_same_side_hold_floor_trail)

            allow_tp = hold >= int(pos_min_hold)
            allow_trail = (hold >= int(eff_pos_min_hold_trail)) or early_trail_now
            allow_soft_sl = (hold >= int(min_hold_soft)) or early_softsl_now
            maxh_hit = (int(pos_max_hold) > 0 and hold >= int(pos_max_hold))

            do_exit = False
            exit_price = 0.0
            reason = EXIT_MAXH
            first_is_high = True
            if int(intrabar_mode) == 2:
                first_is_high = False

            for step in range(2):
                use_high = first_is_high if step == 0 else (not first_is_high)
                is_fav = (side == 1 and use_high) or (side == -1 and (not use_high))

                if is_fav:
                    if BEP_ARM_v > 0.0 and bep_armed == 0:
                        bep_arm_price = ep * (1.0 + side * BEP_ARM_v)
                        if (side == 1 and hi >= bep_arm_price) or (side == -1 and lo <= bep_arm_price):
                            bep_armed = 1
                            bep_armed_at = int(i)
                            ref_price = hi if side == 1 else lo

                    if bep_armed == 1:
                        if side == 1:
                            if sl_price < bep_stop_price:
                                sl_price = bep_stop_price
                        else:
                            if sl_price > bep_stop_price:
                                sl_price = bep_stop_price

                    if TR_v > 0.0 and (int(trail_after_bep) == 0 or bep_armed == 1):
                        if side == 1:
                            if hi > ref_price:
                                ref_price = hi
                                ref_updates += 1
                                pos_ref_updates_local += 1
                        else:
                            if lo < ref_price:
                                ref_price = lo
                                ref_updates += 1
                                pos_ref_updates_local += 1

                    if allow_tp:
                        if (side == 1 and hi >= tp_price) or (side == -1 and lo <= tp_price):
                            do_exit = True
                            exit_price = tp_price
                            reason = EXIT_TP
                            break
                else:
                    trail_hit = False
                    sl_hit = False
                    trail_price = 0.0
                    trail_active = False
                    if allow_trail and TR_v > 0.0 and (int(trail_after_bep) == 0 or bep_armed == 1):
                        if not (bep_armed == 1 and bep_armed_at == i):
                            trail_active = True
                    if trail_active:
                        unlock_i = entry_i + int(eff_pos_min_hold_trail)
                        unlock_grace_until_i = int(unlock_i + int(trail_grace_after_unlock))
                        if int(same_side_hold_enabled) == 1 and int(pos_same_side_grace_unlock_until_i) > int(unlock_grace_until_i):
                            unlock_grace_until_i = int(pos_same_side_grace_unlock_until_i)
                        if i < int(unlock_grace_until_i):
                            trail_active = False
                    if trail_active and int(trail_after_bep) == 1 and bep_armed == 1:
                        bep_grace_until_i = int(bep_armed_at + int(trail_grace_after_bep))
                        if int(same_side_hold_enabled) == 1 and int(pos_same_side_grace_bep_until_i) > int(bep_grace_until_i):
                            bep_grace_until_i = int(pos_same_side_grace_bep_until_i)
                        if i < int(bep_grace_until_i):
                            trail_active = False
                    if trail_active:
                        trail_price = ref_price * (1.0 - side * TR_v)
                        if bep_armed == 1:
                            if side == 1:
                                if trail_price < bep_stop_price:
                                    trail_price = bep_stop_price
                            else:
                                if trail_price > bep_stop_price:
                                    trail_price = bep_stop_price
                        if (side == 1 and lo <= trail_price) or (side == -1 and hi >= trail_price):
                            trail_hit = True

                    post_bep_sl_live = (
                        bep_armed == 1
                        and int(post_bep_shield_ignore_softsl_hold) == 1
                        and bep_armed_at >= 0
                        and bep_armed_at < i
                    )
                    if tpw_live == 1 and int(tp_window_suspend_post_bep_shield_before_tp) == 1:
                        post_bep_sl_live = False
                    if allow_soft_sl or post_bep_sl_live:
                        if (side == 1 and lo <= sl_price) or (side == -1 and hi >= sl_price):
                            sl_hit = True
                    if trail_active or allow_soft_sl or post_bep_sl_live:
                        if bep_armed == 1:
                            if trail_hit:
                                do_exit = True
                                exit_price = trail_price
                                reason = EXIT_TRAIL
                                break
                            elif sl_hit:
                                do_exit = True
                                exit_price = sl_price
                                reason = EXIT_SL
                                break
                        else:
                            if sl_hit:
                                do_exit = True
                                exit_price = sl_price
                                reason = EXIT_SL
                                break
                            elif trail_hit:
                                do_exit = True
                                exit_price = trail_price
                                reason = EXIT_TRAIL
                                break
                    else:
                        if (side == 1 and lo <= hard_sl) or (side == -1 and hi >= hard_sl):
                            do_exit = True
                            exit_price = hard_sl
                            reason = EXIT_SL
                            break

            if (not do_exit) and force_close_now:
                do_exit = True
                exit_price = c
                reason = EXIT_RISK
            if (not do_exit) and maxh_hit:
                do_exit = True
                exit_price = c
                reason = EXIT_MAXH

            if do_exit:
                exit_fee_side = taker_fee_side
                if float(maker_fee_per_side) > 0.0 and reason in (EXIT_TP, EXIT_TRAIL):
                    exit_fee_side = float(maker_fee_per_side)
                fee_total = (taker_fee_side + exit_fee_side) * lev_now
                gross_pnl = (side * (exit_price - ep) / max(ep, 1e-12)) * lev_now
                net_pnl = gross_pnl - fee_total
                scaled = max(net_pnl, -0.999)
                equity = equity * (1.0 + scaled)

                trade_cnt += 1
                if side == 1:
                    long_trades += 1
                else:
                    short_trades += 1
                if scaled > 0.0:
                    win_cnt += 1
                if reason == EXIT_MAXH:
                    maxh_cnt += 1

                exit_cnt[reason] += 1
                exit_gross_sum[reason] += gross_pnl
                exit_fee_sum[reason] += fee_total
                exit_net_sum[reason] += net_pnl
                if reason == EXIT_TRAIL:
                    if bep_armed == 1:
                        trail_after_bep_cnt += 1
                    else:
                        trail_before_bep_cnt += 1
                if bep_armed == 1:
                    bep_armed_trades += 1
                if tpw_armed_at >= 0:
                    tp_window_armed_trades += 1
                    tp_window_live_bars_total += int(max(0, tpw_live_bars_local))

                trade_logs.append({
                    "entry_idx": int(seg_start + int(entry_i)),
                    "exit_idx": int(seg_start + int(i)),
                    "decision_idx": int(seg_start + int(entry_decision_i)),
                    "tier": 0,
                    "tier_name": "SINGLE",
                    "side": int(side),
                    "side_name": "LONG" if side == 1 else "SHORT",
                    "entry_th_used": np.nan,
                    "gate_strength_entry": float(entry_gate_strength),
                    "dir_signal_entry": float(entry_dir_signal),
                    "atr_rel_entry": float(entry_atr_rel),
                    "vol_z_60_entry": float(entry_vol_z),
                    "bep_arm_value": float(entry_bep_arm_val),
                    "entry_bep_arm_fee": float(pos_bep_arm_fee),
                    "entry_bep_stop_fee": float(pos_bep_stop_fee),
                    "entry_dyn_gate_mult": float(entry_dyn_gate_mult),
                    "entry_dyn_lev_scale": float(entry_dyn_lev_scale),
                    "entry_dyn_bep_scale": float(entry_dyn_bep_scale),
                    "entry_dyn_trail_scale": float(entry_dyn_trail_scale),
                    "entry_dyn_sl_scale": float(entry_dyn_sl_scale),
                    "entry_dyn_stress": float(entry_dyn_stress),
                    "entry_dyn_mode": int(dyn_mode_code),
                    "entry_allow_soft_sl_before_trail": int(allow_soft_sl_before_trail),
                    "entry_softsl_hold_floor": int(softsl_hold_floor),
                    "entry_post_bep_shield_ignore_softsl_hold": int(post_bep_shield_ignore_softsl_hold),
                    "entry_min_hold_soft_sl_local": int(entry_min_hold_soft_sl_init),
                    "entry_regime_alpha": float(entry_regime_alpha),
                    "entry_regime_bucket": int(entry_regime_bucket),
                    "entry_regime_name": ("active" if int(entry_regime_bucket) == 2 else ("mid" if int(entry_regime_bucket) == 1 else ("calm" if int(entry_regime_bucket) == 0 else "unknown"))),
                    "final_min_hold_soft_sl_local": int(pos_min_hold_soft_sl),
                    "pre_bep_degraded": int(pre_bep_degraded),
                    "pre_bep_force_closed": int(pre_bep_force_closed),
                    "pre_bep_last_progress_prev": float(pre_bep_last_progress_prev),
                    "final_live_sl": float(pos_SL),
                    "final_live_bep_arm": float(pos_BEP_ARM),
                    "final_live_bep_stop_fee": float(pos_bep_stop_fee),
                    "final_live_trail": float(pos_TR),
                    "entry_early_softsl_enabled": int(early_softsl_enabled),
                    "entry_early_softsl_min_hold": int(early_softsl_min_hold),
                    "entry_early_softsl_progress_frac": float(early_softsl_progress_frac),
                    "entry_early_trail_enabled": int(early_trail_enabled),
                    "entry_early_trail_min_hold": int(early_trail_min_hold),
                    "entry_early_trail_progress_frac": float(early_trail_progress_frac),
                    "entry_early_trail_ref_updates_min": int(early_trail_ref_updates_min),
                    "entry_run_id": int(entry_run_id),
                    "entry_episode_idx_in_run": int(entry_episode_idx_in_run),
                    "entry_is_rearm": int(entry_is_rearm),
                    "rearm_reason_code": int(entry_rearm_reason_code),
                    "bars_since_last_exit": int(entry_bars_since_last_exit),
                    "run_peak_gate": float(entry_run_peak_gate),
                    "gate_reset_seen": int(entry_gate_reset_seen),
                    "price_reset_frac": float(entry_price_reset_frac),
                    "last_exit_reason": int(entry_last_exit_reason),
                    "tp_window_armed": int(tpw_armed_at >= 0),
                    "tp_window_armed_bar": int(seg_start + int(tpw_armed_at)) if tpw_armed_at >= 0 else -1,
                    "tp_progress_at_arm": float(tpw_progress_at_arm),
                    "tp_window_live_bars": int(max(0, tpw_live_bars_local)),
                    "tp_window_exit_blocked_early_trail": int(tpw_blocked_early_trail_local),
                    "tp_window_exit_blocked_softsl": int(tpw_blocked_softsl_local),
                    "same_side_hold_events_local": int(same_side_hold_events_local),
                    "same_side_hold_strong_events_local": int(same_side_hold_strong_events_local),
                    "same_side_hold_weak_events_local": int(same_side_hold_weak_events_local),
                    "same_side_hold_last_class": int(same_side_hold_last_class),
                    "same_side_hold_last_strength_ratio": float(same_side_hold_last_strength_ratio),
                    "same_side_hold_floor_trail_local": int(pos_same_side_hold_floor_trail),
                    "ref_updates_local": int(pos_ref_updates_local),
                    "mfe": float(mfe_rel),
                    "mae": float(mae_rel),
                    "entry_price": float(ep),
                    "exit_price": float(exit_price),
                    "lev": float(lev_now),
                    "gross_pnl": float(gross_pnl),
                    "fee_total": float(fee_total),
                    "net_pnl": float(net_pnl),
                    "net_pnl_alloc": float(scaled),
                    "exit_reason_id": int(reason),
                    "exit_reason": EXIT_NAMES[int(reason)],
                    "hold_bars": int(hold),
                })

                last_exit_i = int(i)
                last_exit_reason = int(reason)
                last_exit_side = int(side)
                last_exit_price = float(exit_price)
                last_run_id = int(entry_run_id if entry_run_id >= 0 else run_id)
                if run_active == 1 and last_run_id == run_id:
                    run_reset_seen_after_exit = 0

                pos_side = 0
                bep_armed = 0
                bep_armed_at = -1
                tpw_live = 0
                tpw_armed_at = -1
                tpw_progress_at_arm = 0.0
                tpw_peak_progress = 0.0
                tpw_live_bars_local = 0
                tpw_blocked_early_trail_local = 0
                tpw_blocked_softsl_local = 0
                tpw_live_bars_local = 0
                pos_same_side_hold_floor_trail = 0
                pos_same_side_grace_bep_until_i = -1
                pos_same_side_grace_unlock_until_i = -1
                same_side_hold_last_class = 0
                same_side_hold_last_strength_ratio = 0.0
                same_side_hold_events_local = 0
                same_side_hold_strong_events_local = 0
                same_side_hold_weak_events_local = 0

            if pos_side != 0:
                fav_cur = 0.0
                if side == 1:
                    fav_cur = (hi - ep) / max(ep, 1e-12)
                else:
                    fav_cur = (ep - lo) / max(ep, 1e-12)
                if fav_cur > mfe_prev_rel:
                    mfe_prev_rel = fav_cur

                if int(same_side_hold_enabled) == 1:
                    ratio_now = float(support_strength_ratio_arr[i]) if i < len(support_strength_ratio_arr) else 0.0
                    signal_side_now = 1 if float(dir_signal[i]) > 0.0 else (-1 if float(dir_signal[i]) < 0.0 else 0)
                    same_side_now = int(bool(ready[i]) and signal_side_now != 0 and signal_side_now == int(pos_side))
                    pass_now = bool(support_pass_mask[i]) if i < len(support_pass_mask) else False
                    weak_ok_now = bool(support_weak_eligible_mask[i]) if i < len(support_weak_eligible_mask) else False

                    support_class = 0
                    if same_side_now == 1:
                        if pass_now and ratio_now >= float(same_side_hold_strong_ratio):
                            support_class = 2
                        elif int(same_side_hold_weak_enabled) == 1 and weak_ok_now and ratio_now >= float(same_side_hold_weak_ratio):
                            progress_ok = (bep_armed == 1)
                            if (not progress_ok) and int(same_side_hold_allow_pre_bep_weak) == 1 and entry_bep_arm_val > 1e-12:
                                progress_ok = (float(mfe_prev_rel) / max(float(entry_bep_arm_val), 1e-12)) >= float(same_side_hold_weak_min_progress_frac)
                            if progress_ok:
                                support_class = 1

                    if support_class > 0:
                        next_hold = int(i + 1 - entry_i)
                        bonus = int(same_side_hold_bonus_bars_strong) if support_class == 2 else int(same_side_hold_bonus_bars_weak)
                        if support_class == 1 and bep_armed == 0:
                            bonus = min(int(bonus), int(same_side_hold_pre_bep_max_bonus_bars))

                        if bonus > 0:
                            cap_floor = int(pos_min_hold_trail + max(0, int(same_side_hold_max_extra_bars)))
                            target_floor = int(next_hold + bonus)
                            if target_floor > pos_same_side_hold_floor_trail:
                                pos_same_side_hold_floor_trail = int(target_floor)
                            if pos_same_side_hold_floor_trail > cap_floor:
                                pos_same_side_hold_floor_trail = int(cap_floor)

                        if support_class == 2:
                            if int(same_side_hold_grace_after_bep_strong) > 0:
                                pos_same_side_grace_bep_until_i = max(int(pos_same_side_grace_bep_until_i), int(i + int(same_side_hold_grace_after_bep_strong)))
                            if int(same_side_hold_grace_after_unlock_strong) > 0:
                                pos_same_side_grace_unlock_until_i = max(int(pos_same_side_grace_unlock_until_i), int(i + int(same_side_hold_grace_after_unlock_strong)))
                            same_side_hold_strong_events += 1
                            same_side_hold_strong_events_local += 1
                        else:
                            if int(same_side_hold_grace_after_bep_weak) > 0:
                                pos_same_side_grace_bep_until_i = max(int(pos_same_side_grace_bep_until_i), int(i + int(same_side_hold_grace_after_bep_weak)))
                            if int(same_side_hold_grace_after_unlock_weak) > 0:
                                pos_same_side_grace_unlock_until_i = max(int(pos_same_side_grace_unlock_until_i), int(i + int(same_side_hold_grace_after_unlock_weak)))
                            same_side_hold_weak_events += 1
                            same_side_hold_weak_events_local += 1

                        same_side_hold_events += 1
                        same_side_hold_events_local += 1
                        same_side_hold_last_class = int(support_class)
                        same_side_hold_last_strength_ratio = float(ratio_now)
                    else:
                        same_side_hold_last_class = 0
                        same_side_hold_last_strength_ratio = float(ratio_now)

        if equity > peak:
            peak = equity
        dd = 1.0 - equity / peak
        if dd > mdd:
            mdd = dd
        if equity < float(stop_equity) or dd > float(stop_dd):
            tail_hit = 1
            break

        if i < int(warmup_steps):
            continue

        gsig = float(gate_strength[i]) if bool(ready[i]) else 0.0
        dsig = float(dir_signal[i]) if bool(ready[i]) else 0.0
        cand_on = bool(ready[i]) and gsig > 0.0 and dsig != 0.0
        cand_side = 1 if dsig > 0.0 else (-1 if dsig < 0.0 else 0)

        if int(entry_episode_enabled) == 1:
            if run_active == 0:
                if cand_on:
                    run_id += 1
                    run_active = 1
                    run_start_i = int(i)
                    run_last_i = int(i)
                    run_side = int(cand_side)
                    run_peak_gate = float(gsig)
                    run_entries = 0
                    run_gap_bars = 0
                    run_reset_seen_after_exit = 0
            else:
                if cand_on and int(cand_side) == int(run_side):
                    run_last_i = int(i)
                    run_gap_bars = 0
                    if float(gsig) > float(run_peak_gate):
                        run_peak_gate = float(gsig)
                elif cand_on and int(cand_side) != int(run_side):
                    run_id += 1
                    run_active = 1
                    run_start_i = int(i)
                    run_last_i = int(i)
                    run_side = int(cand_side)
                    run_peak_gate = float(gsig)
                    run_entries = 0
                    run_gap_bars = 0
                    run_reset_seen_after_exit = 0
                else:
                    run_gap_bars += 1
                    if run_gap_bars > int(run_gap_reset_bars):
                        run_active = 0
                        run_start_i = -1
                        run_last_i = -1
                        run_side = 0
                        run_peak_gate = 0.0
                        run_entries = 0
                        run_gap_bars = 0
                        run_reset_seen_after_exit = 0
            if run_active == 1 and int(last_run_id) == int(run_id) and int(last_exit_i) >= 0 and i > int(last_exit_i) and pos_side == 0 and pend_side == 0:
                if (not cand_on) or (int(cand_side) != int(run_side)):
                    run_reset_seen_after_exit = 1
                elif float(gsig) <= float(rearm_gate_reset_frac) * max(float(run_peak_gate), 1e-12):
                    run_reset_seen_after_exit = 1

        if pos_side == 0 and pend_side == 0:
            if not cand_on:
                continue
            if float(minutes_to_next_funding[i]) < float(funding_near_min):
                continue
            if int(low_vol_filter) == 1:
                if float(vol_z[i]) <= float(vol_low_th):
                    continue
                if float(atr_rel[i]) < float(atr_entry_mult) * fee_roundtrip:
                    continue
                if float(range_rel[i]) < float(range_entry_mult) * fee_roundtrip:
                    continue
            if int(use_atr_scaling) == 1 and np.isfinite(float(atr_high_th)) and float(atr_rel[i]) > float(atr_high_th):
                continue
            if int(risk_entry_mode) == 3:
                if float(atr_rel[i]) > atr_med * float(atr_entry_mult):
                    continue
                if float(range_rel[i]) > range_med * float(range_entry_mult):
                    continue

            allow_entry = True
            is_rearm_now = 0
            rearm_reason_code_now = 0
            bars_since_last_exit_now = -1
            gate_reset_seen_now = 0
            price_reset_frac_now = 0.0
            last_exit_reason_now = -1
            run_peak_gate_now = float(gsig)
            run_id_now = int(run_id if run_active == 1 else -1)
            episode_idx_now = 1

            if int(entry_episode_enabled) == 1 and run_active == 1:
                run_peak_gate_now = float(run_peak_gate)
                if run_entries == 0 or int(last_run_id) != int(run_id):
                    episode_idx_now = int(run_entries) + 1
                else:
                    episode_idx_now = int(run_entries) + 1
                    if int(rearm_enabled) == 0:
                        allow_entry = False
                    else:
                        bars_since_last_exit_now = int(i - last_exit_i) if int(last_exit_i) > -10**8 else -1
                        last_exit_reason_now = int(last_exit_reason)
                        gate_reset_seen_now = int(run_reset_seen_after_exit)
                        if int(last_exit_i) > -10**8 and float(last_exit_price) > 0.0:
                            price_reset_frac_now = abs(float(c) - float(last_exit_price)) / max(abs(float(last_exit_price)), 1e-12)
                        reason_ok = _rearm_exit_allowed(int(last_exit_reason), int(rearm_after_trail), int(rearm_after_tp), int(rearm_after_sl))
                        same_side_ok = (int(rearm_same_side_only) == 0) or (int(cand_side) == int(last_exit_side))
                        cooldown_ok = bars_since_last_exit_now >= int(rearm_cooldown_bars)
                        max_bars_ok = True if int(rearm_max_bars_after_exit) <= 0 else (bars_since_last_exit_now <= int(rearm_max_bars_after_exit))
                        episode_cap_ok = int(run_entries) < int(episode_max_entries_per_run)
                        gate_refresh_ok = float(gsig) >= float(rearm_gate_refresh_frac) * max(float(run_peak_gate), 1e-12)
                        reset_ok = bool(int(gate_reset_seen_now) == 1 or float(price_reset_frac_now) >= float(rearm_price_reset_frac))
                        allow_entry = bool(reason_ok and same_side_ok and cooldown_ok and max_bars_ok and episode_cap_ok and gate_refresh_ok and reset_ok)
                        is_rearm_now = 1 if allow_entry else 0
                        rearm_reason_code_now = _rearm_reason_code(int(last_exit_reason)) if allow_entry else 0
                if not allow_entry:
                    continue
                if int(run_entries) >= int(episode_max_entries_per_run):
                    continue
                run_entries = int(run_entries) + 1
                episode_idx_now = int(run_entries)
            elif int(entry_episode_enabled) == 1 and run_active == 0:
                run_id += 1
                run_active = 1
                run_start_i = int(i)
                run_last_i = int(i)
                run_side = int(cand_side)
                run_peak_gate = float(gsig)
                run_entries = 1
                run_gap_bars = 0
                run_reset_seen_after_exit = 0
                run_id_now = int(run_id)
                episode_idx_now = 1
                run_peak_gate_now = float(run_peak_gate)

            side = 1 if dsig > 0.0 else -1
            lev_now = float(base_leverage) * float(lev_mult) * float(dyn_lev_scale_arr[i])
            if int(integer_leverage) == 1:
                lev_now = float(int(lev_now + 0.5))
            if lev_now < 1.0:
                lev_now = 1.0
            if lev_now > float(risk_lev_cap):
                lev_now = float(risk_lev_cap)

            atr_e = float(atr_rel[i]) if int(use_atr_scaling) == 1 else 1.0
            sl_base_v = float(SL) * atr_e
            tp_v = float(TP) * atr_e
            bep_arm_base_v = float(bep_arm_base) * atr_e
            tr_base_v = float(trailing) * atr_e

            sl_v = sl_base_v * float(dyn_sl_scale_arr[i])
            bep_arm_v = bep_arm_base_v * float(dyn_bep_scale_arr[i])
            tr_v = tr_base_v * float(dyn_trail_scale_arr[i])

            min_tp = fee_roundtrip * float(fee_tp_mult)
            if tp_v < min_tp:
                tp_v = min_tp

            bep_arm_fee_local = econ_be_fee * float(bep_arm_fee_mult) * float(dyn_bep_scale_arr[i])
            if bep_arm_fee_local < 0.0:
                bep_arm_fee_local = 0.0
            if bep_arm_v < bep_arm_fee_local:
                bep_arm_v = bep_arm_fee_local

            min_sl = 0.6 * fee_roundtrip
            if sl_v < min_sl:
                sl_v = min_sl

            if int(trail_after_bep) == 0:
                min_tr = max(econ_be_fee, fee_roundtrip * float(fee_tp_mult))
                if tr_v < min_tr:
                    tr_v = min_tr

            local_soft_sl_hold = resolve_local_soft_sl_hold(
                int(min_hold_soft_sl_bars),
                int(min_hold_trail_bars),
                int(dyn_softsl_relax_arr[i]),
                int(allow_soft_sl_before_trail),
                int(softsl_hold_floor),
            )

            pend_decision_i = int(i)
            pend_gate_strength = float(gsig)
            pend_dir_signal = float(dsig)
            pend_atr_rel = float(atr_rel[i])
            pend_vol_z = float(vol_z[i])
            pend_dyn_lev_scale = float(dyn_lev_scale_arr[i])
            pend_dyn_bep_scale = float(dyn_bep_scale_arr[i])
            pend_dyn_trail_scale = float(dyn_trail_scale_arr[i])
            pend_dyn_sl_scale = float(dyn_sl_scale_arr[i])
            pend_dyn_gate_mult = float(dyn_gate_mult_arr[i])
            pend_dyn_stress = float(dyn_stress_arr[i])
            pend_run_id = int(run_id_now)
            pend_episode_idx_in_run = int(episode_idx_now)
            pend_is_rearm = int(is_rearm_now)
            pend_rearm_reason_code = int(rearm_reason_code_now)
            pend_bars_since_last_exit = int(bars_since_last_exit_now)
            pend_run_peak_gate = float(run_peak_gate_now)
            pend_gate_reset_seen = int(gate_reset_seen_now)
            pend_price_reset_frac = float(price_reset_frac_now)
            pend_last_exit_reason = int(last_exit_reason_now)

            pend_side = int(side)
            pend_i = int(i + 1)
            pend_lev = float(lev_now)
            pend_TP = float(tp_v)
            pend_SL = float(sl_v)
            pend_BEP_ARM = float(bep_arm_v)
            pend_TR = float(tr_v)
            pend_min_hold = int(min_hold_bars)
            pend_min_hold_trail = int(min_hold_trail_bars)
            pend_min_hold_soft_sl = int(local_soft_sl_hold)
            pend_max_hold = int(max_hold_bars)
            pend_bep_arm_fee = float(bep_arm_fee_local)
            pend_bep_stop_fee = float(econ_be_fee * float(bep_stop_fee_mult))

            pend_SL_base = float(sl_base_v)
            pend_BEP_ARM_base = float(bep_arm_base_v)
            pend_TR_base = float(tr_base_v)
            pend_bep_arm_fee_base = float(econ_be_fee * float(bep_arm_fee_mult))
            pend_min_hold_soft_sl_base = int(min_hold_soft_sl_bars)

    if tail_hit == 0 and pos_side != 0:
        last_c = float(close[-1])
        side = int(pos_side)
        ep = float(entry_price)
        lev_now = float(entry_lev)
        fee_total = (taker_fee_side + taker_fee_side) * lev_now
        gross_pnl = (side * (last_c - ep) / max(ep, 1e-12)) * lev_now
        net_pnl = gross_pnl - fee_total
        scaled = max(net_pnl, -0.999)
        equity = equity * (1.0 + scaled)
        trade_cnt += 1
        if side == 1:
            long_trades += 1
        else:
            short_trades += 1
        if scaled > 0.0:
            win_cnt += 1
        exit_cnt[EXIT_FORCE] += 1
        exit_gross_sum[EXIT_FORCE] += gross_pnl
        exit_fee_sum[EXIT_FORCE] += fee_total
        exit_net_sum[EXIT_FORCE] += net_pnl
        if tpw_armed_at >= 0:
            tp_window_armed_trades += 1
            tp_window_live_bars_total += int(max(0, tpw_live_bars_local))
        if equity > peak:
            peak = equity
        dd = 1.0 - equity / peak
        if dd > mdd:
            mdd = dd

        hi_last = float(high[-1])
        lo_last = float(low[-1])
        if ep > 0.0:
            if side == 1:
                fav = (hi_last - ep) / max(ep, 1e-12)
                adv = (lo_last - ep) / max(ep, 1e-12)
            else:
                fav = (ep - lo_last) / max(ep, 1e-12)
                adv = (ep - hi_last) / max(ep, 1e-12)
            if fav > mfe_rel:
                mfe_rel = fav
            if adv < mae_rel:
                mae_rel = adv

        hold_bars = int((n - 1) - int(entry_i))
        trade_logs.append({
            "entry_idx": int(seg_start + int(entry_i)),
            "exit_idx": int(seg_start + int(n - 1)),
            "decision_idx": int(seg_start + int(entry_decision_i)),
            "tier": 0,
            "tier_name": "SINGLE",
            "side": int(side),
            "side_name": "LONG" if side == 1 else "SHORT",
            "entry_th_used": np.nan,
            "gate_strength_entry": float(entry_gate_strength),
            "dir_signal_entry": float(entry_dir_signal),
            "atr_rel_entry": float(entry_atr_rel),
            "vol_z_60_entry": float(entry_vol_z),
            "bep_arm_value": float(entry_bep_arm_val),
            "entry_bep_arm_fee": float(pos_bep_arm_fee),
            "entry_bep_stop_fee": float(pos_bep_stop_fee),
            "entry_dyn_gate_mult": float(entry_dyn_gate_mult),
            "entry_dyn_lev_scale": float(entry_dyn_lev_scale),
            "entry_dyn_bep_scale": float(entry_dyn_bep_scale),
            "entry_dyn_trail_scale": float(entry_dyn_trail_scale),
            "entry_dyn_sl_scale": float(entry_dyn_sl_scale),
            "entry_dyn_stress": float(entry_dyn_stress),
            "entry_dyn_mode": int(dyn_mode_code),
            "entry_allow_soft_sl_before_trail": int(allow_soft_sl_before_trail),
            "entry_softsl_hold_floor": int(softsl_hold_floor),
            "entry_post_bep_shield_ignore_softsl_hold": int(post_bep_shield_ignore_softsl_hold),
            "entry_min_hold_soft_sl_local": int(entry_min_hold_soft_sl_init),
            "entry_regime_alpha": float(entry_regime_alpha),
            "entry_regime_bucket": int(entry_regime_bucket),
            "entry_regime_name": ("active" if int(entry_regime_bucket) == 2 else ("mid" if int(entry_regime_bucket) == 1 else ("calm" if int(entry_regime_bucket) == 0 else "unknown"))),
            "final_min_hold_soft_sl_local": int(pos_min_hold_soft_sl),
            "pre_bep_degraded": int(pre_bep_degraded),
            "pre_bep_force_closed": int(pre_bep_force_closed),
            "pre_bep_last_progress_prev": float(pre_bep_last_progress_prev),
            "final_live_sl": float(pos_SL),
            "final_live_bep_arm": float(pos_BEP_ARM),
            "final_live_bep_stop_fee": float(pos_bep_stop_fee),
            "final_live_trail": float(pos_TR),
            "entry_run_id": int(entry_run_id),
            "entry_episode_idx_in_run": int(entry_episode_idx_in_run),
            "entry_is_rearm": int(entry_is_rearm),
            "rearm_reason_code": int(entry_rearm_reason_code),
            "bars_since_last_exit": int(entry_bars_since_last_exit),
            "run_peak_gate": float(entry_run_peak_gate),
            "gate_reset_seen": int(entry_gate_reset_seen),
            "price_reset_frac": float(entry_price_reset_frac),
            "last_exit_reason": int(entry_last_exit_reason),
            "tp_window_armed": int(tpw_armed_at >= 0),
            "tp_window_armed_bar": int(seg_start + int(tpw_armed_at)) if tpw_armed_at >= 0 else -1,
            "tp_progress_at_arm": float(tpw_progress_at_arm),
            "tp_window_live_bars": int(max(0, tpw_live_bars_local)),
            "tp_window_exit_blocked_early_trail": int(tpw_blocked_early_trail_local),
            "tp_window_exit_blocked_softsl": int(tpw_blocked_softsl_local),
            "mfe": float(mfe_rel),
            "mae": float(mae_rel),
            "entry_price": float(ep),
            "exit_price": float(last_c),
            "lev": float(lev_now),
            "gross_pnl": float(gross_pnl),
            "fee_total": float(fee_total),
            "net_pnl": float(net_pnl),
            "net_pnl_alloc": float(scaled),
            "exit_reason_id": int(EXIT_FORCE),
            "exit_reason": EXIT_NAMES[int(EXIT_FORCE)],
            "hold_bars": int(hold_bars),
        })
        last_exit_i = int(n - 1)
        last_exit_reason = int(EXIT_FORCE)
        last_exit_side = int(side)
        last_exit_price = float(last_c)
        last_run_id = int(entry_run_id if entry_run_id >= 0 else run_id)
        pos_side = 0
        bep_armed = 0
        bep_armed_at = -1

    return {
        "net_ret": float(equity - 1.0),
        "mdd": float(mdd),
        "trades": int(trade_cnt),
        "wins": int(win_cnt),
        "tail_hit": int(tail_hit),
        "exit_cnt": exit_cnt,
        "exit_gross_sum": exit_gross_sum,
        "exit_fee_sum": exit_fee_sum,
        "exit_net_sum": exit_net_sum,
        "trail_before_bep": int(trail_before_bep_cnt),
        "trail_after_bep": int(trail_after_bep_cnt),
        "bep_armed_trades": int(bep_armed_trades),
        "ref_updates": int(ref_updates),
        "trade_logs": trade_logs,
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "maxh_cnt": int(maxh_cnt),
        "tp_window_armed_trades": int(tp_window_armed_trades),
        "tp_window_live_bars_total": int(tp_window_live_bars_total),
        "tp_window_blocked_early_trail": int(tp_window_blocked_early_trail_total),
        "tp_window_blocked_softsl": int(tp_window_blocked_softsl_total),
        "rearm_entries": int(rearm_entries),
        "rearm_entries_after_trail": int(rearm_entries_after_trail),
        "rearm_entries_after_tp": int(rearm_entries_after_tp),
        "rearm_entries_after_sl": int(rearm_entries_after_sl),
        "same_side_hold_events": int(same_side_hold_events),
        "same_side_hold_strong_events": int(same_side_hold_strong_events),
        "same_side_hold_weak_events": int(same_side_hold_weak_events),
    }


# --- v110 regime lane active dual-lane overlay ---
DEFAULT_REGIME_LANE_CFG: Dict[str, Any] = {
    "enabled": 0,
    "active_sparse_enabled": 0,
    "active_sparse_min_ready": 160,
    "sparse_gate_q": 0.55,
    "sparse_gate_floor_q": 0.00,
    "sparse_atr_q": 0.65,
    "sparse_range_q": 0.65,
    "sparse_vol_q": 0.00,
    "sparse_require_high_vol": 0,
    "sparse_high_logic": "or",
}
REGIME_PROFILE_NAMES: Dict[int, str] = {
    0: "calm",
    1: "mid",
    2: "active_dense",
    3: "active_sparse",
}
REGIME_PROFILE_IDS: Dict[str, int] = {v: k for k, v in REGIME_PROFILE_NAMES.items()}


def regime_profile_name(profile: int) -> str:
    return REGIME_PROFILE_NAMES.get(int(profile), "unknown")


def _bucket_to_profile_arr(bucket_arr: Optional[np.ndarray]) -> np.ndarray:
    if bucket_arr is None:
        return np.zeros(0, dtype=np.int8)
    bucket = np.asarray(bucket_arr, dtype=np.int8)
    out = np.full(bucket.shape, -1, dtype=np.int8)
    out[bucket == 0] = 0
    out[bucket == 1] = 1
    out[bucket == 2] = 2
    return out


def _normalize_regime_lane_cfg(raw: Optional[Dict[str, Any]], *, default_enabled: Optional[int] = None) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = copy.deepcopy(DEFAULT_REGIME_LANE_CFG)
    out.update({k: copy.deepcopy(v) for k, v in raw.items() if k in DEFAULT_REGIME_LANE_CFG})
    if default_enabled is not None and "enabled" not in raw:
        out["enabled"] = int(default_enabled)
    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["active_sparse_enabled"] = 1 if safe_int(out.get("active_sparse_enabled", 0), 0) != 0 else 0
    out["active_sparse_min_ready"] = max(0, safe_int(out.get("active_sparse_min_ready", 160), 160))
    out["sparse_gate_q"] = float(np.clip(safe_float(out.get("sparse_gate_q", 0.55), 0.55), 0.0, 1.0))
    out["sparse_gate_floor_q"] = float(np.clip(safe_float(out.get("sparse_gate_floor_q", 0.0), 0.0), 0.0, 1.0))
    if out["sparse_gate_floor_q"] > out["sparse_gate_q"]:
        out["sparse_gate_floor_q"] = float(out["sparse_gate_q"])
    out["sparse_atr_q"] = float(np.clip(safe_float(out.get("sparse_atr_q", 0.65), 0.65), 0.0, 1.0))
    out["sparse_range_q"] = float(np.clip(safe_float(out.get("sparse_range_q", 0.65), 0.65), 0.0, 1.0))
    out["sparse_vol_q"] = float(np.clip(safe_float(out.get("sparse_vol_q", 0.0), 0.0), 0.0, 1.0))
    out["sparse_require_high_vol"] = 1 if safe_int(out.get("sparse_require_high_vol", 0), 0) != 0 else 0
    out["sparse_high_logic"] = str(out.get("sparse_high_logic", "or") or "or").strip().lower()
    if out["sparse_high_logic"] not in {"or", "and", "atr_only", "range_only"}:
        out["sparse_high_logic"] = "or"
    if int(out.get("enabled", 0)) == 0:
        out["active_sparse_enabled"] = 0
    return out


_normalize_regime_threshold_cfg_v110_base = _normalize_regime_threshold_cfg

def _normalize_regime_threshold_cfg(raw: Optional[Dict[str, Any]], *, base_q_entry: float, base_entry_th_floor: float, default_enabled: Optional[int] = None) -> Dict[str, Any]:
    out = dict(_normalize_regime_threshold_cfg_v110_base(raw, base_q_entry=float(base_q_entry), base_entry_th_floor=float(base_entry_th_floor), default_enabled=default_enabled))
    raw = dict(raw or {})

    q_dense_seed = raw.get("q_entry_active_dense", raw.get("q_entry_active", out.get("q_entry_active", base_q_entry)))
    q_dense = float(np.clip(safe_float(q_dense_seed, out.get("q_entry_active", base_q_entry)), 0.0, 1.0))
    q_sparse_direct = raw.get("q_entry_active_sparse", None)
    q_sparse_delta = safe_float(raw.get("q_entry_active_sparse_delta", 0.0), 0.0)
    if q_sparse_direct is None:
        q_sparse = float(np.clip(q_dense + q_sparse_delta, 0.0, 1.0))
    else:
        q_sparse = float(np.clip(safe_float(q_sparse_direct, q_dense), 0.0, 1.0))
    if q_sparse > q_dense:
        q_sparse = q_dense
    out["q_entry_active_dense"] = float(q_dense)
    out["q_entry_active_sparse"] = float(q_sparse)
    out["q_entry_active_sparse_delta"] = float(q_sparse - q_dense)
    out["q_entry_active"] = float(q_dense)

    floor_dense_seed = raw.get("entry_th_floor_active_dense", raw.get("entry_th_active_dense", raw.get("entry_th_floor_active", raw.get("entry_th_active", out.get("entry_th_floor_active", base_entry_th_floor)))))
    floor_dense = float(safe_float(floor_dense_seed, out.get("entry_th_floor_active", base_entry_th_floor)))
    floor_sparse_direct = raw.get("entry_th_floor_active_sparse", raw.get("entry_th_active_sparse", None))
    floor_sparse_delta = safe_float(raw.get("entry_th_floor_active_sparse_delta", raw.get("entry_th_active_sparse_delta", 0.0)), 0.0)
    if floor_sparse_direct is None:
        floor_sparse = float(floor_dense + floor_sparse_delta)
    else:
        floor_sparse = float(safe_float(floor_sparse_direct, floor_dense))
    if floor_sparse > floor_dense:
        floor_sparse = floor_dense
    out["entry_th_floor_active_dense"] = float(floor_dense)
    out["entry_th_floor_active_sparse"] = float(floor_sparse)
    out["entry_th_floor_active_sparse_delta"] = float(floor_sparse - floor_dense)
    out["entry_th_floor_active"] = float(floor_dense)
    return out


_normalize_regime_filter_cfg_v110_base = _normalize_regime_filter_cfg

def _normalize_regime_filter_cfg(
    raw: Optional[Dict[str, Any]],
    *,
    base_vol_low_th: float,
    base_atr_entry_mult: float,
    base_range_entry_mult: float,
    default_enabled: Optional[int] = None,
) -> Dict[str, Any]:
    out = dict(_normalize_regime_filter_cfg_v110_base(
        raw,
        base_vol_low_th=float(base_vol_low_th),
        base_atr_entry_mult=float(base_atr_entry_mult),
        base_range_entry_mult=float(base_range_entry_mult),
        default_enabled=default_enabled,
    ))
    raw = dict(raw or {})

    vol_dense_seed = raw.get("vol_low_th_active_dense", raw.get("vol_low_th_active", out.get("vol_low_th_active", base_vol_low_th)))
    vol_dense = float(safe_float(vol_dense_seed, out.get("vol_low_th_active", base_vol_low_th)))
    vol_sparse_direct = raw.get("vol_low_th_active_sparse", None)
    vol_sparse_delta = safe_float(raw.get("vol_low_th_active_sparse_delta", 0.0), 0.0)
    if vol_sparse_direct is None:
        vol_sparse = float(vol_dense + vol_sparse_delta)
    else:
        vol_sparse = float(safe_float(vol_sparse_direct, vol_dense))
    if vol_sparse > vol_dense:
        vol_sparse = vol_dense
    out["vol_low_th_active_dense"] = float(vol_dense)
    out["vol_low_th_active_sparse"] = float(vol_sparse)
    out["vol_low_th_active_sparse_delta"] = float(vol_sparse - vol_dense)
    out["vol_low_th_active"] = float(vol_dense)

    atr_dense_seed = raw.get("atr_entry_mult_active_dense", raw.get("atr_entry_mult_active", out.get("atr_entry_mult_active", base_atr_entry_mult)))
    atr_dense = max(0.0, float(safe_float(atr_dense_seed, out.get("atr_entry_mult_active", base_atr_entry_mult))))
    atr_sparse_direct = raw.get("atr_entry_mult_active_sparse", None)
    atr_sparse_delta = safe_float(raw.get("atr_entry_mult_active_sparse_delta", 0.0), 0.0)
    if atr_sparse_direct is None:
        atr_sparse = max(0.0, float(atr_dense + atr_sparse_delta))
    else:
        atr_sparse = max(0.0, float(safe_float(atr_sparse_direct, atr_dense)))
    if atr_sparse > atr_dense:
        atr_sparse = atr_dense
    out["atr_entry_mult_active_dense"] = float(atr_dense)
    out["atr_entry_mult_active_sparse"] = float(atr_sparse)
    out["atr_entry_mult_active_sparse_delta"] = float(atr_sparse - atr_dense)
    out["atr_entry_mult_active"] = float(atr_dense)

    rng_dense_seed = raw.get("range_entry_mult_active_dense", raw.get("range_entry_mult_active", out.get("range_entry_mult_active", base_range_entry_mult)))
    rng_dense = max(0.0, float(safe_float(rng_dense_seed, out.get("range_entry_mult_active", base_range_entry_mult))))
    rng_sparse_direct = raw.get("range_entry_mult_active_sparse", None)
    rng_sparse_delta = safe_float(raw.get("range_entry_mult_active_sparse_delta", 0.0), 0.0)
    if rng_sparse_direct is None:
        rng_sparse = max(0.0, float(rng_dense + rng_sparse_delta))
    else:
        rng_sparse = max(0.0, float(safe_float(rng_sparse_direct, rng_dense)))
    if rng_sparse > rng_dense:
        rng_sparse = rng_dense
    out["range_entry_mult_active_dense"] = float(rng_dense)
    out["range_entry_mult_active_sparse"] = float(rng_sparse)
    out["range_entry_mult_active_sparse_delta"] = float(rng_sparse - rng_dense)
    out["range_entry_mult_active"] = float(rng_dense)
    return out


def build_active_sparse_lane_pack(
    gate_signal_all: np.ndarray,
    ready: np.ndarray,
    atr_rel_all: np.ndarray,
    range_rel_all: np.ndarray,
    vol_z_all: np.ndarray,
    bucket_arr: Optional[np.ndarray],
    *,
    hist_start: int,
    hist_end: int,
    seg_start: int,
    seg_end: int,
    regime_lane_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_signal_all = np.asarray(gate_signal_all, dtype=np.float64)
    ready = np.asarray(ready, dtype=bool)
    atr_rel_all = np.asarray(atr_rel_all, dtype=np.float64)
    range_rel_all = np.asarray(range_rel_all, dtype=np.float64)
    vol_z_all = np.asarray(vol_z_all, dtype=np.float64)
    if bucket_arr is None:
        bucket_all = np.full(gate_signal_all.shape, -1, dtype=np.int8)
    else:
        bucket_all = np.asarray(bucket_arr, dtype=np.int8)
    hist_start = int(hist_start)
    hist_end = int(hist_end)
    seg_start = int(seg_start)
    seg_end = int(seg_end)
    hist_len = max(0, hist_end - hist_start)
    seg_len = max(0, seg_end - seg_start)
    bucket_hist = np.asarray(bucket_all[hist_start:hist_end], dtype=np.int8) if hist_len > 0 else np.zeros(0, dtype=np.int8)
    bucket_seg = np.asarray(bucket_all[seg_start:seg_end], dtype=np.int8) if seg_len > 0 else np.zeros(0, dtype=np.int8)
    profile_hist = _bucket_to_profile_arr(bucket_hist)
    profile_seg = _bucket_to_profile_arr(bucket_seg)
    active_sparse_flag_hist = np.zeros(hist_len, dtype=np.bool_)
    active_sparse_flag_seg = np.zeros(seg_len, dtype=np.bool_)
    lane_cfg = _normalize_regime_lane_cfg(regime_lane_cfg, default_enabled=0)
    meta: Dict[str, Any] = {
        "lane_enabled": int(lane_cfg.get("enabled", 0)),
        "active_sparse_requested": int(lane_cfg.get("active_sparse_enabled", 0)),
        "active_sparse_enabled": 0,
        "active_hist_ready_count": 0,
        "active_sparse_hist_count": 0,
        "active_sparse_seg_count": 0,
        "active_sparse_fallback_dense_cnt": 0,
        "sparse_gate_cut": None,
        "sparse_gate_floor_cut": None,
        "sparse_atr_cut": None,
        "sparse_range_cut": None,
        "sparse_vol_cut": None,
        "sparse_high_logic": str(lane_cfg.get("sparse_high_logic", "or") or "or"),
        "sparse_require_high_vol": int(lane_cfg.get("sparse_require_high_vol", 0)),
        "active_sparse_after_gate_band_hist_count": 0,
        "active_sparse_after_high_hist_count": 0,
        "active_sparse_after_vol_hist_count": 0,
        "active_sparse_after_gate_band_seg_count": 0,
        "active_sparse_after_high_seg_count": 0,
        "active_sparse_after_vol_seg_count": 0,
    }
    if int(lane_cfg.get("enabled", 0)) == 0 or int(lane_cfg.get("active_sparse_enabled", 0)) == 0 or hist_len <= 0 or seg_len <= 0:
        return {
            "enabled": int(lane_cfg.get("enabled", 0)),
            "cfg": lane_cfg,
            "bucket_hist": bucket_hist,
            "bucket_seg": bucket_seg,
            "profile_hist": profile_hist,
            "profile_seg": profile_seg,
            "active_sparse_flag_hist": active_sparse_flag_hist,
            "active_sparse_flag_seg": active_sparse_flag_seg,
            "profile_meta": meta,
        }

    hist_gate = np.asarray(gate_signal_all[hist_start:hist_end], dtype=np.float64)
    hist_atr = np.asarray(atr_rel_all[hist_start:hist_end], dtype=np.float64)
    hist_rng = np.asarray(range_rel_all[hist_start:hist_end], dtype=np.float64)
    hist_vol = np.asarray(vol_z_all[hist_start:hist_end], dtype=np.float64)
    hist_ready = np.asarray(ready[hist_start:hist_end], dtype=bool)
    active_hist_mask = hist_ready & (bucket_hist == 2) & np.isfinite(hist_gate) & np.isfinite(hist_atr) & np.isfinite(hist_rng)
    active_hist_count = int(np.sum(active_hist_mask))
    meta["active_hist_ready_count"] = int(active_hist_count)
    min_ready = int(lane_cfg.get("active_sparse_min_ready", 160))
    if active_hist_count < max(1, min_ready):
        meta["active_sparse_fallback_dense_cnt"] = int(np.sum(bucket_seg == 2))
        return {
            "enabled": int(lane_cfg.get("enabled", 0)),
            "cfg": lane_cfg,
            "bucket_hist": bucket_hist,
            "bucket_seg": bucket_seg,
            "profile_hist": profile_hist,
            "profile_seg": profile_seg,
            "active_sparse_flag_hist": active_sparse_flag_hist,
            "active_sparse_flag_seg": active_sparse_flag_seg,
            "profile_meta": meta,
        }

    gate_q = float(np.clip(lane_cfg.get("sparse_gate_q", 0.55), 0.0, 1.0))
    gate_floor_q = float(np.clip(lane_cfg.get("sparse_gate_floor_q", 0.0), 0.0, gate_q))
    atr_q = float(np.clip(lane_cfg.get("sparse_atr_q", 0.65), 0.0, 1.0))
    range_q = float(np.clip(lane_cfg.get("sparse_range_q", 0.65), 0.0, 1.0))
    vol_q = float(np.clip(lane_cfg.get("sparse_vol_q", 0.0), 0.0, 1.0))
    require_high_vol = int(lane_cfg.get("sparse_require_high_vol", 0)) != 0

    gate_cut = float(np.quantile(hist_gate[active_hist_mask], gate_q))
    if gate_floor_q > 0.0:
        gate_floor_cut = float(np.quantile(hist_gate[active_hist_mask], gate_floor_q))
    else:
        gate_floor_cut = float(-np.inf)
    atr_cut = float(np.quantile(hist_atr[active_hist_mask], atr_q))
    range_cut = float(np.quantile(hist_rng[active_hist_mask], range_q))
    hist_vol_mask = active_hist_mask & np.isfinite(hist_vol)
    if require_high_vol and vol_q > 0.0 and int(np.sum(hist_vol_mask)) > 0:
        vol_cut = float(np.quantile(hist_vol[hist_vol_mask], vol_q))
    else:
        vol_cut = float(-np.inf)
    meta["sparse_gate_cut"] = float(gate_cut)
    meta["sparse_gate_floor_cut"] = float(gate_floor_cut) if np.isfinite(gate_floor_cut) else None
    meta["sparse_atr_cut"] = float(atr_cut)
    meta["sparse_range_cut"] = float(range_cut)
    meta["sparse_vol_cut"] = float(vol_cut) if np.isfinite(vol_cut) else None

    def _sparse_mask(
        bucket_local: np.ndarray,
        gate_local: np.ndarray,
        atr_local: np.ndarray,
        range_local: np.ndarray,
        vol_local: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        finite_base = np.isfinite(gate_local) & np.isfinite(atr_local) & np.isfinite(range_local)
        active_local = (bucket_local == 2) & finite_base
        weak_gate_hi = gate_local <= float(gate_cut)
        weak_gate_lo = gate_local >= float(gate_floor_cut)
        gate_band_mask = active_local & weak_gate_hi & weak_gate_lo

        high_atr = atr_local >= float(atr_cut)
        high_rng = range_local >= float(range_cut)
        logic = str(lane_cfg.get("sparse_high_logic", "or") or "or").strip().lower()
        if logic == "and":
            high_ok = high_atr & high_rng
        elif logic == "atr_only":
            high_ok = high_atr
        elif logic == "range_only":
            high_ok = high_rng
        else:
            high_ok = high_atr | high_rng
        after_high_mask = gate_band_mask & high_ok

        if require_high_vol and np.isfinite(float(vol_cut)):
            vol_ok = np.isfinite(vol_local) & (vol_local >= float(vol_cut))
        else:
            vol_ok = np.ones_like(after_high_mask, dtype=np.bool_)
        after_vol_mask = after_high_mask & vol_ok
        return after_vol_mask, gate_band_mask, after_high_mask, after_vol_mask

    active_sparse_flag_hist, gate_band_hist, after_high_hist, after_vol_hist = _sparse_mask(bucket_hist, hist_gate, hist_atr, hist_rng, hist_vol)
    seg_gate = np.asarray(gate_signal_all[seg_start:seg_end], dtype=np.float64)
    seg_atr = np.asarray(atr_rel_all[seg_start:seg_end], dtype=np.float64)
    seg_rng = np.asarray(range_rel_all[seg_start:seg_end], dtype=np.float64)
    seg_vol = np.asarray(vol_z_all[seg_start:seg_end], dtype=np.float64)
    active_sparse_flag_seg, gate_band_seg, after_high_seg, after_vol_seg = _sparse_mask(bucket_seg, seg_gate, seg_atr, seg_rng, seg_vol)
    profile_hist[active_sparse_flag_hist] = 3
    profile_seg[active_sparse_flag_seg] = 3
    meta["active_sparse_enabled"] = 1
    meta["active_sparse_hist_count"] = int(np.sum(active_sparse_flag_hist))
    meta["active_sparse_seg_count"] = int(np.sum(active_sparse_flag_seg))
    meta["active_sparse_after_gate_band_hist_count"] = int(np.sum(gate_band_hist))
    meta["active_sparse_after_high_hist_count"] = int(np.sum(after_high_hist))
    meta["active_sparse_after_vol_hist_count"] = int(np.sum(after_vol_hist))
    meta["active_sparse_after_gate_band_seg_count"] = int(np.sum(gate_band_seg))
    meta["active_sparse_after_high_seg_count"] = int(np.sum(after_high_seg))
    meta["active_sparse_after_vol_seg_count"] = int(np.sum(after_vol_seg))
    return {
        "enabled": int(lane_cfg.get("enabled", 0)),
        "cfg": lane_cfg,
        "bucket_hist": bucket_hist,
        "bucket_seg": bucket_seg,
        "profile_hist": profile_hist,
        "profile_seg": profile_seg,
        "active_sparse_flag_hist": active_sparse_flag_hist,
        "active_sparse_flag_seg": active_sparse_flag_seg,
        "profile_meta": meta,
    }


def build_profiled_entry_threshold_pack(
    gate_signal_all: np.ndarray,
    ready: np.ndarray,
    bucket_arr: Optional[np.ndarray],
    profile_hist: Optional[np.ndarray],
    profile_seg: Optional[np.ndarray],
    *,
    hist_start: int,
    hist_end: int,
    seg_start: int,
    seg_end: int,
    q_entry: float,
    entry_th_floor: float,
    entry_q_min_ready: int,
    regime_threshold_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_signal_all = np.asarray(gate_signal_all, dtype=np.float64)
    ready = np.asarray(ready, dtype=bool)
    seg_start = int(seg_start)
    seg_end = int(seg_end)
    seg_len = max(0, seg_end - seg_start)
    thr_entry_global = quantile_from_history(
        gate_signal_all,
        ready,
        int(hist_start),
        int(hist_end),
        float(q_entry),
        int(entry_q_min_ready),
        float(entry_th_floor),
    )
    threshold_cfg = _normalize_regime_threshold_cfg(
        regime_threshold_cfg,
        base_q_entry=float(q_entry),
        base_entry_th_floor=float(entry_th_floor),
    )
    entry_threshold_base_seg = np.full(seg_len, float(thr_entry_global), dtype=np.float64)
    entry_q_used_seg = np.full(seg_len, float(q_entry), dtype=np.float64)
    entry_floor_used_seg = np.full(seg_len, float(entry_th_floor), dtype=np.float64)

    bucket_seg = np.full(seg_len, -1, dtype=np.int8)
    if bucket_arr is not None and seg_len > 0:
        bucket_all = np.asarray(bucket_arr, dtype=np.int8)
        bucket_seg = np.asarray(bucket_all[seg_start:seg_end], dtype=np.int8)
    if profile_seg is None:
        profile_seg_arr = _bucket_to_profile_arr(bucket_seg)
    else:
        profile_seg_arr = np.asarray(profile_seg, dtype=np.int8)
    if profile_hist is None:
        hist_len = max(0, int(hist_end) - int(hist_start))
        if bucket_arr is not None and hist_len > 0:
            bucket_all = np.asarray(bucket_arr, dtype=np.int8)
            profile_hist_arr = _bucket_to_profile_arr(np.asarray(bucket_all[int(hist_start):int(hist_end)], dtype=np.int8))
        else:
            profile_hist_arr = np.full(hist_len, -1, dtype=np.int8)
    else:
        profile_hist_arr = np.asarray(profile_hist, dtype=np.int8)

    bucket_min_ready = int(threshold_cfg.get("bucket_min_ready", 0))
    if bucket_min_ready <= 0:
        bucket_min_ready = int(entry_q_min_ready)
    hist_vals = np.asarray(gate_signal_all[int(hist_start):int(hist_end)], dtype=np.float64)
    hist_ready = np.asarray(ready[int(hist_start):int(hist_end)], dtype=bool)

    profile_specs = [
        (0, "calm", "q_entry_calm", "entry_th_floor_calm"),
        (1, "mid", "q_entry_mid", "entry_th_floor_mid"),
        (2, "active_dense", "q_entry_active_dense", "entry_th_floor_active_dense"),
        (3, "active_sparse", "q_entry_active_sparse", "entry_th_floor_active_sparse"),
    ]
    profile_meta: Dict[str, Any] = {}
    for pid, pname, q_key, floor_key in profile_specs:
        q_prof = float(threshold_cfg.get(q_key, q_entry))
        floor_prof = float(threshold_cfg.get(floor_key, entry_th_floor))
        hist_mask = hist_ready & (profile_hist_arr == int(pid))
        hist_count = int(np.sum(hist_mask))
        use_global = False
        if hist_count >= int(bucket_min_ready):
            thr_prof = quantile_from_values(hist_vals, hist_mask, q_prof, int(bucket_min_ready), floor_prof)
            q_eff = q_prof
            floor_eff = floor_prof
        elif int(threshold_cfg.get("bucket_fallback_global", 1)) != 0:
            thr_prof = float(thr_entry_global)
            q_eff = float(q_entry)
            floor_eff = float(entry_th_floor)
            use_global = True
        else:
            thr_prof = float(floor_prof)
            q_eff = float(q_prof)
            floor_eff = float(floor_prof)
        seg_mask = profile_seg_arr == int(pid)
        if np.any(seg_mask):
            entry_threshold_base_seg[seg_mask] = float(thr_prof)
            entry_q_used_seg[seg_mask] = float(q_eff)
            entry_floor_used_seg[seg_mask] = float(floor_eff)
        profile_meta[pname] = {
            "hist_ready_count": int(hist_count),
            "seg_bar_count": int(np.sum(seg_mask)),
            "threshold": float(thr_prof),
            "q_used": float(q_eff),
            "floor_used": float(floor_eff),
            "used_global_fallback": int(use_global),
        }
    return {
        "thr_entry_global": float(thr_entry_global),
        "entry_threshold_base_seg": entry_threshold_base_seg,
        "entry_q_used_seg": entry_q_used_seg,
        "entry_floor_used_seg": entry_floor_used_seg,
        "bucket_seg": bucket_seg,
        "profile_seg": profile_seg_arr,
        "profile_meta": profile_meta,
        "bucket_min_ready": int(bucket_min_ready),
        "bucket_fallback_global": int(threshold_cfg.get("bucket_fallback_global", 1)),
        "threshold_enabled": int(threshold_cfg.get("enabled", 0)),
    }


def build_profiled_filter_pack(
    *,
    profile_seg: Optional[np.ndarray],
    regime_filter_cfg: Optional[Dict[str, Any]],
    base_vol_low_th: float,
    base_atr_entry_mult: float,
    base_range_entry_mult: float,
) -> Dict[str, Any]:
    if profile_seg is None:
        profile_seg_arr = np.zeros(0, dtype=np.int8)
    else:
        profile_seg_arr = np.asarray(profile_seg, dtype=np.int8)
    seg_len = int(profile_seg_arr.size)
    cfg = _normalize_regime_filter_cfg(
        regime_filter_cfg,
        base_vol_low_th=float(base_vol_low_th),
        base_atr_entry_mult=float(base_atr_entry_mult),
        base_range_entry_mult=float(base_range_entry_mult),
    )
    vol_arr = np.full(seg_len, float(base_vol_low_th), dtype=np.float64)
    atr_arr = np.full(seg_len, float(base_atr_entry_mult), dtype=np.float64)
    rng_arr = np.full(seg_len, float(base_range_entry_mult), dtype=np.float64)
    if int(cfg.get("enabled", 0)) != 0 and seg_len > 0:
        if int(cfg.get("use_vol_split", 1)) != 0:
            vol_arr[profile_seg_arr == 0] = float(cfg.get("vol_low_th_calm", base_vol_low_th))
            vol_arr[profile_seg_arr == 1] = float(cfg.get("vol_low_th_mid", base_vol_low_th))
            vol_arr[profile_seg_arr == 2] = float(cfg.get("vol_low_th_active_dense", cfg.get("vol_low_th_active", base_vol_low_th)))
            vol_arr[profile_seg_arr == 3] = float(cfg.get("vol_low_th_active_sparse", cfg.get("vol_low_th_active_dense", cfg.get("vol_low_th_active", base_vol_low_th))))
        if int(cfg.get("use_entry_mult_split", 1)) != 0:
            atr_arr[profile_seg_arr == 0] = float(cfg.get("atr_entry_mult_calm", base_atr_entry_mult))
            atr_arr[profile_seg_arr == 1] = float(cfg.get("atr_entry_mult_mid", base_atr_entry_mult))
            atr_arr[profile_seg_arr == 2] = float(cfg.get("atr_entry_mult_active_dense", cfg.get("atr_entry_mult_active", base_atr_entry_mult)))
            atr_arr[profile_seg_arr == 3] = float(cfg.get("atr_entry_mult_active_sparse", cfg.get("atr_entry_mult_active_dense", cfg.get("atr_entry_mult_active", base_atr_entry_mult))))
            rng_arr[profile_seg_arr == 0] = float(cfg.get("range_entry_mult_calm", base_range_entry_mult))
            rng_arr[profile_seg_arr == 1] = float(cfg.get("range_entry_mult_mid", base_range_entry_mult))
            rng_arr[profile_seg_arr == 2] = float(cfg.get("range_entry_mult_active_dense", cfg.get("range_entry_mult_active", base_range_entry_mult)))
            rng_arr[profile_seg_arr == 3] = float(cfg.get("range_entry_mult_active_sparse", cfg.get("range_entry_mult_active_dense", cfg.get("range_entry_mult_active", base_range_entry_mult))))
    profile_meta: Dict[str, Any] = {}
    for pid, pname in REGIME_PROFILE_NAMES.items():
        mask = profile_seg_arr == int(pid)
        profile_meta[pname] = {
            "seg_bar_count": int(np.sum(mask)),
            "vol_low_th": float(np.mean(vol_arr[mask])) if np.any(mask) else float(base_vol_low_th),
            "atr_entry_mult": float(np.mean(atr_arr[mask])) if np.any(mask) else float(base_atr_entry_mult),
            "range_entry_mult": float(np.mean(rng_arr[mask])) if np.any(mask) else float(base_range_entry_mult),
        }
    return {
        "enabled": int(cfg.get("enabled", 0)),
        "use_vol_split": int(cfg.get("use_vol_split", 1)),
        "use_entry_mult_split": int(cfg.get("use_entry_mult_split", 1)),
        "mid_interp_mode": str(cfg.get("mid_interp_mode", "linear")),
        "vol_low_th_arr": vol_arr,
        "atr_entry_mult_arr": atr_arr,
        "range_entry_mult_arr": rng_arr,
        "profile_seg": profile_seg_arr,
        "profile_meta": profile_meta,
        "cfg": cfg,
    }


_RANGE_KEY_ALIASES.update({
    "tune_regime_lanes": "tune_regime_lanes",
    "regime_lane_enabled_min": "regime_lane_enabled_min",
    "regime_lane_enabled_max": "regime_lane_enabled_max",
    "active_sparse_enabled_min": "active_sparse_enabled_min",
    "active_sparse_enabled_max": "active_sparse_enabled_max",
    "active_sparse_min_ready_min": "active_sparse_min_ready_min",
    "active_sparse_min_ready_max": "active_sparse_min_ready_max",
    "sparse_gate_q_min": "sparse_gate_q_min",
    "sparse_gate_q_max": "sparse_gate_q_max",
    "sparse_gate_floor_q_min": "sparse_gate_floor_q_min",
    "sparse_gate_floor_q_max": "sparse_gate_floor_q_max",
    "sparse_atr_q_min": "sparse_atr_q_min",
    "sparse_atr_q_max": "sparse_atr_q_max",
    "sparse_range_q_min": "sparse_range_q_min",
    "sparse_range_q_max": "sparse_range_q_max",
    "sparse_vol_q_min": "sparse_vol_q_min",
    "sparse_vol_q_max": "sparse_vol_q_max",
    "sparse_require_high_vol_min": "sparse_require_high_vol_min",
    "sparse_require_high_vol_max": "sparse_require_high_vol_max",
    "sparse_high_logic_choices": "sparse_high_logic_choices",
    "q_entry_active_sparse_min": "q_entry_active_sparse_min",
    "q_entry_active_sparse_max": "q_entry_active_sparse_max",
    "q_entry_active_sparse_delta_min": "q_entry_active_sparse_delta_min",
    "q_entry_active_sparse_delta_max": "q_entry_active_sparse_delta_max",
    "entry_th_active_sparse_min": "entry_th_active_sparse_min",
    "entry_th_active_sparse_max": "entry_th_active_sparse_max",
    "entry_th_active_sparse_delta_min": "entry_th_active_sparse_delta_min",
    "entry_th_active_sparse_delta_max": "entry_th_active_sparse_delta_max",
    "vol_low_th_active_sparse_min": "vol_low_th_active_sparse_min",
    "vol_low_th_active_sparse_max": "vol_low_th_active_sparse_max",
    "vol_low_th_active_sparse_delta_min": "vol_low_th_active_sparse_delta_min",
    "vol_low_th_active_sparse_delta_max": "vol_low_th_active_sparse_delta_max",
    "atr_entry_mult_active_sparse_min": "atr_entry_mult_active_sparse_min",
    "atr_entry_mult_active_sparse_max": "atr_entry_mult_active_sparse_max",
    "atr_entry_mult_active_sparse_delta_min": "atr_entry_mult_active_sparse_delta_min",
    "atr_entry_mult_active_sparse_delta_max": "atr_entry_mult_active_sparse_delta_max",
    "range_entry_mult_active_sparse_min": "range_entry_mult_active_sparse_min",
    "range_entry_mult_active_sparse_max": "range_entry_mult_active_sparse_max",
    "range_entry_mult_active_sparse_delta_min": "range_entry_mult_active_sparse_delta_min",
    "range_entry_mult_active_sparse_delta_max": "range_entry_mult_active_sparse_delta_max",
})


def _enforce_softsl_trail_guard_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg or {})
    dyn = cfg.get("dynamic_cfg", {})
    if not isinstance(dyn, dict):
        dyn = {}
        cfg["dynamic_cfg"] = dyn
    allow_before = 1 if safe_int(dyn.get("allow_soft_sl_before_trail", 0), 0) != 0 else 0
    trail_hold = max(0, safe_int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))), cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))))
    soft_hold = max(0, safe_int(cfg.get("min_hold_soft_sl_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))), cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))))
    if allow_before == 0 and soft_hold < trail_hold:
        cfg["min_hold_soft_sl_bars"] = int(trail_hold)
    return cfg


_normalize_single_config_from_any_v110_base = normalize_single_config_from_any

# --- v111 optional runner-alignment cfg (audit-only) ---
DEFAULT_RUNNER_ALIGNMENT_CFG: Dict[str, Any] = {
    "enabled": 0,
    "profit_floor_enabled": 0,
    "thesis_monitor_enabled": 0,
}

DEFAULT_SAME_SIDE_HOLD_CFG: Dict[str, Any] = {
    "enabled": 0,
    "weak_enabled": 1,
    "strong_ratio": 1.00,
    "weak_ratio": 0.82,
    "weak_min_progress_frac": 0.35,
    "allow_pre_bep_weak": 1,
    "pre_bep_max_bonus_bars": 1,
    "strong_bonus_bars": 2,
    "weak_bonus_bars": 1,
    "max_extra_bars": 4,
    "strong_grace_after_bep_bars": 1,
    "weak_grace_after_bep_bars": 0,
    "strong_grace_after_unlock_bars": 1,
    "weak_grace_after_unlock_bars": 0,
    "debug_log": 0,
}

def _normalize_same_side_hold_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = dict(DEFAULT_SAME_SIDE_HOLD_CFG)
    for k, v in raw.items():
        out[k] = copy.deepcopy(v)

    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["weak_enabled"] = 1 if safe_int(out.get("weak_enabled", 1), 1) != 0 else 0

    out["strong_ratio"] = max(0.0, safe_float(out.get("strong_ratio", 1.0), 1.0))
    out["weak_ratio"] = max(0.0, safe_float(out.get("weak_ratio", 0.82), 0.82))
    out["weak_min_progress_frac"] = float(np.clip(safe_float(out.get("weak_min_progress_frac", 0.35), 0.35), 0.0, 2.0))
    out["allow_pre_bep_weak"] = 1 if safe_int(out.get("allow_pre_bep_weak", 1), 1) != 0 else 0
    out["pre_bep_max_bonus_bars"] = max(0, safe_int(out.get("pre_bep_max_bonus_bars", 1), 1))

    out["strong_bonus_bars"] = max(0, safe_int(out.get("strong_bonus_bars", 2), 2))
    out["weak_bonus_bars"] = max(0, safe_int(out.get("weak_bonus_bars", 1), 1))
    out["max_extra_bars"] = max(0, safe_int(out.get("max_extra_bars", 4), 4))

    out["strong_grace_after_bep_bars"] = max(0, safe_int(out.get("strong_grace_after_bep_bars", 1), 1))
    out["weak_grace_after_bep_bars"] = max(0, safe_int(out.get("weak_grace_after_bep_bars", 0), 0))
    out["strong_grace_after_unlock_bars"] = max(0, safe_int(out.get("strong_grace_after_unlock_bars", 1), 1))
    out["weak_grace_after_unlock_bars"] = max(0, safe_int(out.get("weak_grace_after_unlock_bars", 0), 0))
    out["debug_log"] = 1 if safe_int(out.get("debug_log", 0), 0) != 0 else 0
    return out

def _normalize_runner_alignment_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(raw or {})
    out = dict(DEFAULT_RUNNER_ALIGNMENT_CFG)
    for k, v in raw.items():
        out[k] = copy.deepcopy(v)

    out["enabled"] = 1 if safe_int(out.get("enabled", 0), 0) != 0 else 0
    out["profit_floor_enabled"] = 1 if safe_int(out.get("profit_floor_enabled", 0), 0) != 0 else 0
    out["thesis_monitor_enabled"] = 1 if safe_int(out.get("thesis_monitor_enabled", 0), 0) != 0 else 0
    return out

def normalize_single_config_from_any(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = copy.deepcopy(raw_cfg or {})
    cfg = _normalize_single_config_from_any_v110_base(raw_cfg)
    merged = dict(raw_cfg)
    merged_regime_threshold_raw = dict(merged.get("regime_threshold_cfg", {}) if isinstance(merged.get("regime_threshold_cfg", {}), dict) else {})
    merged_regime_filter_raw = dict(merged.get("regime_filter_cfg", {}) if isinstance(merged.get("regime_filter_cfg", {}), dict) else {})
    merged_regime_lane_raw = dict(merged.get("regime_lane_cfg", {}) if isinstance(merged.get("regime_lane_cfg", {}), dict) else {})

    if not merged_regime_threshold_raw and isinstance(cfg.get("regime_threshold_cfg", {}), dict):
        merged_regime_threshold_raw = dict(cfg.get("regime_threshold_cfg", {}))
    if not merged_regime_filter_raw and isinstance(cfg.get("regime_filter_cfg", {}), dict):
        merged_regime_filter_raw = dict(cfg.get("regime_filter_cfg", {}))
    if not merged_regime_lane_raw and isinstance(cfg.get("regime_lane_cfg", {}), dict):
        merged_regime_lane_raw = dict(cfg.get("regime_lane_cfg", {}))

    cfg["regime_threshold_cfg"] = _normalize_regime_threshold_cfg(
        merged_regime_threshold_raw if merged_regime_threshold_raw else cfg.get("regime_threshold_cfg", {}),
        base_q_entry=float(cfg.get("q_entry", 0.85)),
        base_entry_th_floor=float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0))),
        default_enabled=safe_int((merged_regime_threshold_raw or {}).get("enabled", cfg.get("regime_threshold_cfg", {}).get("enabled", 0)), 0),
    )
    cfg["regime_filter_cfg"] = _normalize_regime_filter_cfg(
        merged_regime_filter_raw if merged_regime_filter_raw else cfg.get("regime_filter_cfg", {}),
        base_vol_low_th=float(cfg.get("risk_cfg", {}).get("vol_low_th", -1e9)),
        base_atr_entry_mult=float(cfg.get("atr_entry_mult", 1.0)),
        base_range_entry_mult=float(cfg.get("range_entry_mult", 1.0)),
        default_enabled=safe_int((merged_regime_filter_raw or {}).get("enabled", cfg.get("regime_filter_cfg", {}).get("enabled", 0)), 0),
    )
    cfg["regime_lane_cfg"] = _normalize_regime_lane_cfg(
        merged_regime_lane_raw if merged_regime_lane_raw else cfg.get("regime_lane_cfg", {}),
        default_enabled=safe_int((merged_regime_lane_raw or {}).get("enabled", cfg.get("regime_lane_cfg", {}).get("enabled", 0)), 0),
    )
    if any(int(cfg.get(k, {}).get("enabled", 0)) != 0 for k in ("regime_weight_cfg", "regime_threshold_cfg", "regime_filter_cfg", "regime_lane_cfg")) and int(cfg.get("regime_detect_cfg", {}).get("enabled", 0)) == 0:
        cfg["regime_detect_cfg"]["enabled"] = 1
    cfg["runner_alignment_cfg"] = _normalize_runner_alignment_cfg(
        merged.get("runner_alignment_cfg", cfg.get("runner_alignment_cfg", {}))
    )
    cfg["same_side_hold_cfg"] = _normalize_same_side_hold_cfg(
        merged.get("same_side_hold_cfg", cfg.get("same_side_hold_cfg", {}))
    )
    cfg = _enforce_softsl_trail_guard_cfg(cfg)
    cfg["schema"] = "single_v110"
    return cfg


def build_single_best_config(base_cfg: Dict[str, Any], tuned_updates: Dict[str, Any], tuned_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = normalize_single_config_from_any(_deep_merge(copy.deepcopy(base_cfg), tuned_updates or {}))
    meta = dict(cfg.get("tuned_meta", {}))
    if tuned_meta:
        meta.update(copy.deepcopy(tuned_meta))
    cfg["tuned_meta"] = meta
    cfg["schema"] = "single_v110"
    return cfg
