# -*- coding: utf-8 -*-
"""
Single-tier RL-mode autotune (v58).

Highlights:
- flat config schema (no low/mid/top, no multi_pos, no pos_frac)
- split-hold parameters are independent
- gate and direction weights are both tunable weighted h1/h3/h5 mixtures
- BEP arm and BEP stop are separated
- same segment score as backtest is used for intrabar worst-case selection
- coverage-aware objective supports bottom-2 / floor / trade-dispersion penalties
- coverage-aware objective supports bottom-2 / floor / trade-dispersion penalties
- regime threshold/filter architecture supports active_dense / active_sparse dual-lane tuning with band-pass sparse lane v2
- supports legacy top-only tiered base_json and old *_top search-bounds aliases
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hybrid_inference_scalp_v7 import HybridScalpInference
from hybrid_core_modified_v8 import (
    HORIZONS_ALL,
    SegmentMetrics,
    agg_worst,
    apply_ranges_overrides,
    assemble_objective,
    prepare_trial_context,
    prepare_single_segment_fast_inputs_from_context,
    evaluate_prepared_single_segment_fast,
    normalize_single_config_from_any,
    parse_float_list,
    precompute_hybrids,
    safe_float,
    safe_int,
    weights_from_self_mix,
    weights_from_raw_vector,
    warmup_single_fast_core,
)


def _suggest_float(trial: optuna.Trial, name: str, lo: float, hi: float, *, log: bool = False) -> float:
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) <= 1e-15:
        return float(lo)
    return float(trial.suggest_float(name, lo, hi, log=log))


def _suggest_int(trial: optuna.Trial, name: str, lo: int, hi: int, *, step: int = 1) -> int:
    lo = int(lo)
    hi = int(hi)
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return int(lo)
    return int(trial.suggest_int(name, lo, hi, step=max(1, int(step))))


def _bounds_float(ranges: Dict[str, Any], key: str, default: float) -> Tuple[float, float]:
    lo = ranges.get(f"{key}_min", default)
    hi = ranges.get(f"{key}_max", default)
    return float(lo), float(hi)


def _bounds_int(ranges: Dict[str, Any], key: str, default: int) -> Tuple[int, int]:
    lo = ranges.get(f"{key}_min", default)
    hi = ranges.get(f"{key}_max", default)
    return int(lo), int(hi)

def _has_range_bounds(ranges: Dict[str, Any], key: str) -> bool:
    return (f"{key}_min" in ranges) or (f"{key}_max" in ranges)

def _has_any_weight_bounds(ranges: Dict[str, Any], prefix: str) -> bool:
    return any((f"{prefix}_w{h}_min" in ranges) or (f"{prefix}_w{h}_max" in ranges) for h in HORIZONS_ALL)


def _suggest_weight_vector(trial: optuna.Trial, ranges: Dict[str, Any], prefix: str, current: Dict[str, Any]) -> Dict[str, float]:
    raw: Dict[int, float] = {}
    for h in HORIZONS_ALL:
        key = f"{prefix}_w{int(h)}"
        cur = safe_float(current.get(f"w{int(h)}", 0.0), 0.0)
        lo, hi = _bounds_float(ranges, key, cur)
        raw[int(h)] = max(0.0, _suggest_float(trial, key, lo, hi))
    return weights_from_raw_vector(raw)


def _flag_from_any(*vals: Any) -> bool:
    for v in vals:
        if v is None:
            continue
        try:
            return bool(int(v) != 0)
        except Exception:
            pass
        if isinstance(v, bool):
            return bool(v)
    return False


def _normalize_cost_scenarios(cost_list: Sequence[float], slip_list: Sequence[float], maker_fee_per_side: float, default_cost: float, default_slip: float) -> List[Dict[str, float]]:
    costs = list(cost_list) if cost_list else [float(default_cost)]
    slips = list(slip_list) if slip_list else [float(default_slip)]
    if len(slips) == 1 and len(costs) > 1:
        slips = slips * len(costs)
    if len(costs) == 1 and len(slips) > 1:
        costs = costs * len(slips)
    if len(costs) != len(slips):
        raise ValueError(f"cost_list len({len(costs)}) != slip_list len({len(slips)})")
    out: List[Dict[str, float]] = []
    for c, s in zip(costs, slips):
        out.append({"taker": float(c), "maker": float(maker_fee_per_side), "slip": float(s)})
    return out


def _materialize_candidate(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    ranges: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = copy.deepcopy(base_cfg)
    meta: Dict[str, Any] = {}

    # canonical tune flags: range json OR CLI aliases
    tune_fee_bep_mult = _flag_from_any(ranges.get("tune_fee_bep_mult"), getattr(args, "tune_fee_bep_mult", 0))
    tune_dynamic = _flag_from_any(ranges.get("tune_dynamic"), ranges.get("tune_dynamic_top"), getattr(args, "tune_dynamic", 0), getattr(args, "tune_dynamic_top", 0))
    tune_dynamic_sl = _flag_from_any(ranges.get("tune_dynamic_sl"), getattr(args, "tune_dynamic_sl", 0))
    tune_dynamic_soft_sl = _flag_from_any(ranges.get("tune_dynamic_soft_sl"), getattr(args, "tune_dynamic_soft_sl", 0))
    tune_margin_gate = _flag_from_any(ranges.get("tune_margin_gate"), ranges.get("tune_margin_gate_top"), getattr(args, "tune_margin_gate", 0), getattr(args, "tune_margin_gate_top", 0))
    tune_entry_filters = _flag_from_any(ranges.get("tune_entry_filters"), getattr(args, "tune_entry_filters", 0))
    tune_pre_bep = _flag_from_any(ranges.get("tune_pre_bep"), ranges.get("tune_pre_bep_top"), getattr(args, "tune_pre_bep", 0), getattr(args, "tune_pre_bep_top", 0))
    tune_pre_tp_window = _flag_from_any(ranges.get("tune_pre_tp_window"), getattr(args, "tune_pre_tp_window", 0))
    tune_entry_episode = _flag_from_any(ranges.get("tune_entry_episode"), getattr(args, "tune_entry_episode", 0))
    tune_dir_weights = _flag_from_any(ranges.get("tune_dir_weights"), getattr(args, "tune_dir_weights", 0))
    tune_gate_weights = _flag_from_any(ranges.get("tune_gate_weights"), getattr(args, "tune_gate_weights", 0))
    tune_regime_detection = _flag_from_any(ranges.get("tune_regime_detection"), getattr(args, "tune_regime_detection", 0), ranges.get("tune_regime_weights"), getattr(args, "tune_regime_weights", 0))
    tune_regime_weights = _flag_from_any(ranges.get("tune_regime_weights"), getattr(args, "tune_regime_weights", 0))
    tune_regime_thresholds = _flag_from_any(ranges.get("tune_regime_thresholds"), getattr(args, "tune_regime_thresholds", 0))
    tune_regime_filters = _flag_from_any(ranges.get("tune_regime_filters"), getattr(args, "tune_regime_filters", 0))
    tune_regime_lanes = _flag_from_any(ranges.get("tune_regime_lanes"), getattr(args, "tune_regime_lanes", 0))
    if not tune_regime_lanes:
        _lane_keys = (
            "regime_lane_enabled", "active_sparse_enabled", "active_sparse_min_ready",
            "sparse_gate_q", "sparse_gate_floor_q", "sparse_atr_q", "sparse_range_q",
            "sparse_vol_q", "sparse_require_high_vol",
        )
        tune_regime_lanes = any(_has_range_bounds(ranges, _k) for _k in _lane_keys)

    q_lo, q_hi = _bounds_float(ranges, "q_entry", cfg["q_entry"])
    cfg["q_entry"] = _suggest_float(trial, "q_entry", q_lo, q_hi)
    meta["q_entry"] = float(cfg["q_entry"])

    th_lo, th_hi = _bounds_float(ranges, "entry_th", cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)))
    cfg["entry_th_floor"] = _suggest_float(trial, "entry_th_floor", th_lo, th_hi)
    cfg["entry_th"] = float(cfg["entry_th_floor"])
    meta["entry_th_floor"] = float(cfg["entry_th_floor"])

    lev_lo, lev_hi = _bounds_float(ranges, "leverage", cfg["leverage"])
    cfg["leverage"] = _suggest_float(trial, "leverage", lev_lo, lev_hi)
    meta["leverage"] = float(cfg["leverage"])

    levm_lo, levm_hi = _bounds_float(ranges, "lev_mult", cfg["lev_mult"])
    cfg["lev_mult"] = _suggest_float(trial, "lev_mult", levm_lo, levm_hi)
    meta["lev_mult"] = float(cfg["lev_mult"])

    # independent split holds
    if ("min_hold_tp_min" in ranges) or ("min_hold_tp_max" in ranges):
        mhtp_lo, mhtp_hi = _bounds_int(ranges, "min_hold_tp", cfg["min_hold_tp_bars"])
    else:
        # backward compatible: old bounds used generic min_hold for TP hold
        mhtp_lo, mhtp_hi = _bounds_int(ranges, "min_hold", cfg["min_hold_tp_bars"])
    mht_lo, mht_hi = _bounds_int(ranges, "min_hold_trail", cfg["min_hold_trail_bars"])
    mhs_lo, mhs_hi = _bounds_int(ranges, "min_hold_soft_sl", cfg["min_hold_soft_sl_bars"])
    mxh_lo, mxh_hi = _bounds_int(ranges, "max_hold", cfg["max_hold_bars"])
    cfg["min_hold_tp_bars"] = _suggest_int(trial, "min_hold_tp_bars", mhtp_lo, mhtp_hi)
    cfg["min_hold_bars"] = int(cfg["min_hold_tp_bars"])
    cfg["min_hold_trail_bars"] = _suggest_int(trial, "min_hold_trail_bars", mht_lo, mht_hi)
    cfg["min_hold_soft_sl_bars"] = _suggest_int(trial, "min_hold_soft_sl_bars", mhs_lo, mhs_hi)
    mxh_min_eff = max(int(cfg["min_hold_tp_bars"]), int(cfg["min_hold_trail_bars"]), int(cfg["min_hold_soft_sl_bars"]), int(mxh_lo))
    cfg["max_hold_bars"] = _suggest_int(trial, "max_hold_bars", mxh_min_eff, max(int(mxh_hi), mxh_min_eff))
    meta["min_hold_tp_bars"] = int(cfg["min_hold_tp_bars"])
    meta["min_hold_bars"] = int(cfg["min_hold_bars"])
    meta["min_hold_trail_bars"] = int(cfg["min_hold_trail_bars"])
    meta["min_hold_soft_sl_bars"] = int(cfg["min_hold_soft_sl_bars"])
    meta["max_hold_bars"] = int(cfg["max_hold_bars"])

    # geometry via ratios (legacy-friendly search space)
    slb_lo, slb_hi = _bounds_float(ranges, "sl_base", max(cfg["SL"], 1e-6))
    slm_lo, slm_hi = _bounds_float(ranges, "sl_mult", 1.0)
    tpr_lo, tpr_hi = _bounds_float(ranges, "tp_sl_ratio", max(cfg["TP"] / max(cfg["SL"], 1e-12), 1e-6))
    tpm_lo, tpm_hi = _bounds_float(ranges, "tp_mult", 1.0)
    bepr_lo, bepr_hi = _bounds_float(ranges, "bep_ratio", max(cfg["BEP_ARM"] / max(cfg["TP"], 1e-12), 1e-6))
    trr_lo, trr_hi = _bounds_float(ranges, "trail_ratio", max(cfg["trailing"] / max(cfg["TP"], 1e-12), 1e-6))
    sl_base = _suggest_float(trial, "sl_base", slb_lo, slb_hi)
    sl_mult = _suggest_float(trial, "sl_mult", slm_lo, slm_hi)
    tp_sl_ratio = _suggest_float(trial, "tp_sl_ratio", tpr_lo, tpr_hi)
    tp_mult = _suggest_float(trial, "tp_mult", tpm_lo, tpm_hi)
    bep_ratio = _suggest_float(trial, "bep_ratio", bepr_lo, bepr_hi)
    trail_ratio = _suggest_float(trial, "trail_ratio", trr_lo, trr_hi)
    cfg["SL"] = float(max(1e-8, sl_base * sl_mult))
    cfg["TP"] = float(max(1e-8, cfg["SL"] * tp_sl_ratio * tp_mult))
    cfg["BEP_ARM"] = float(max(0.0, cfg["TP"] * bep_ratio))
    cfg["trailing"] = float(max(1e-8, cfg["TP"] * trail_ratio))
    meta.update({
        "sl_base": float(sl_base),
        "sl_mult": float(sl_mult),
        "tp_sl_ratio": float(tp_sl_ratio),
        "tp_mult": float(tp_mult),
        "bep_ratio": float(bep_ratio),
        "trail_ratio": float(trail_ratio),
    })

    # BEP arm / stop separation
    if tune_fee_bep_mult:
        arm_lo, arm_hi = _bounds_float(ranges, "bep_arm_fee_mult", cfg["bep_arm_fee_mult"])
        cfg["bep_arm_fee_mult"] = _suggest_float(trial, "bep_arm_fee_mult", arm_lo, arm_hi)
    stop_lo, stop_hi = _bounds_float(ranges, "bep_stop_fee_mult", cfg.get("bep_stop_fee_mult", 1.0))
    cfg["bep_stop_fee_mult"] = _suggest_float(trial, "bep_stop_fee_mult", stop_lo, stop_hi)
    meta["bep_arm_fee_mult"] = float(cfg["bep_arm_fee_mult"])
    meta["bep_stop_fee_mult"] = float(cfg["bep_stop_fee_mult"])
    if "bep_stop_mode_choices" in ranges:
        choices = ranges.get("bep_stop_mode_choices")
        if isinstance(choices, list) and choices:
            cfg["bep_stop_mode"] = str(trial.suggest_categorical("bep_stop_mode", choices))

    # entry filters / risk
    if tune_entry_filters:
        ae_lo, ae_hi = _bounds_float(ranges, "atr_entry_mult", cfg["atr_entry_mult"])
        re_lo, re_hi = _bounds_float(ranges, "range_entry_mult", cfg["range_entry_mult"])
        vv_lo, vv_hi = _bounds_float(ranges, "vol_low_th", cfg["risk_cfg"]["vol_low_th"])
        cfg["atr_entry_mult"] = _suggest_float(trial, "atr_entry_mult", ae_lo, ae_hi)
        cfg["range_entry_mult"] = _suggest_float(trial, "range_entry_mult", re_lo, re_hi)
        cfg["risk_cfg"]["vol_low_th"] = _suggest_float(trial, "vol_low_th", vv_lo, vv_hi)
        if "funding_near_min_min" in ranges or "funding_near_min_max" in ranges:
            fn_lo, fn_hi = _bounds_float(ranges, "funding_near_min", cfg["risk_cfg"]["funding_near_min"])
            cfg["risk_cfg"]["funding_near_min"] = _suggest_float(trial, "funding_near_min", fn_lo, fn_hi)
        if "atr_high_th_min" in ranges or "atr_high_th_max" in ranges:
            ah_lo, ah_hi = _bounds_float(ranges, "atr_high_th", safe_float(cfg["risk_cfg"].get("atr_high_th", np.nan), np.nan))
            cfg["risk_cfg"]["atr_high_th"] = _suggest_float(trial, "atr_high_th", ah_lo, ah_hi)
    meta["atr_entry_mult"] = float(cfg["atr_entry_mult"])
    meta["range_entry_mult"] = float(cfg["range_entry_mult"])
    meta["vol_low_th"] = float(cfg["risk_cfg"]["vol_low_th"])
    meta["funding_near_min"] = float(cfg["risk_cfg"]["funding_near_min"])
    meta["atr_high_th"] = float(cfg["risk_cfg"].get("atr_high_th", np.nan)) if np.isfinite(safe_float(cfg["risk_cfg"].get("atr_high_th", np.nan), np.nan)) else None

    # gate / dir weights (separate)
    if tune_gate_weights:
        if _has_any_weight_bounds(ranges, "gate"):
            cfg["gate_weights"] = _suggest_weight_vector(trial, ranges, "gate", cfg["gate_weights"])
            for h in HORIZONS_ALL:
                meta[f"gate_w{int(h)}"] = float(cfg["gate_weights"].get(f"w{int(h)}", 0.0))
        else:
            gs_lo, gs_hi = _bounds_float(ranges, "gate_self", cfg["gate_weights"].get("w5", 1.0))
            gm_lo, gm_hi = _bounds_float(ranges, "gate_mix", safe_float(cfg["gate_weights"].get("w1", 0.0) / max(cfg["gate_weights"].get("w1", 0.0) + cfg["gate_weights"].get("w3", 0.0), 1e-12), 0.5))
            gate_self = _suggest_float(trial, "gate_self", gs_lo, gs_hi)
            gate_mix = _suggest_float(trial, "gate_mix", gm_lo, gm_hi)
            cfg["gate_weights"] = weights_from_self_mix(gate_self, gate_mix)
            meta["gate_self"] = float(gate_self)
            meta["gate_mix"] = float(gate_mix)
    else:
        meta["gate_self"] = float(cfg["gate_weights"].get("w5", 1.0))
    if tune_dir_weights:
        if _has_any_weight_bounds(ranges, "dir"):
            cfg["dir_weights"] = _suggest_weight_vector(trial, ranges, "dir", cfg["dir_weights"])
            for h in HORIZONS_ALL:
                meta[f"dir_w{int(h)}"] = float(cfg["dir_weights"].get(f"w{int(h)}", 0.0))
        else:
            ds_lo, ds_hi = _bounds_float(ranges, "dir_self", cfg["dir_weights"].get("w5", 1.0))
            dm_lo, dm_hi = _bounds_float(ranges, "dir_mix", safe_float(cfg["dir_weights"].get("w1", 0.0) / max(cfg["dir_weights"].get("w1", 0.0) + cfg["dir_weights"].get("w3", 0.0), 1e-12), 0.5))
            dir_self = _suggest_float(trial, "dir_self", ds_lo, ds_hi)
            dir_mix = _suggest_float(trial, "dir_mix", dm_lo, dm_hi)
            cfg["dir_weights"] = weights_from_self_mix(dir_self, dir_mix)
            meta["dir_self"] = float(dir_self)
            meta["dir_mix"] = float(dir_mix)
    else:
        meta["dir_self"] = float(cfg["dir_weights"].get("w5", 1.0))

    regime_detect_cfg = copy.deepcopy(cfg.get("regime_detect_cfg", {}))
    regime_weight_cfg = copy.deepcopy(cfg.get("regime_weight_cfg", {}))
    regime_threshold_cfg = copy.deepcopy(cfg.get("regime_threshold_cfg", {}))
    regime_filter_cfg = copy.deepcopy(cfg.get("regime_filter_cfg", {}))
    regime_lane_cfg = copy.deepcopy(cfg.get("regime_lane_cfg", {}))

    if tune_regime_detection:
        if _has_range_bounds(ranges, "regime_detect_enabled"):
            lo, hi = _bounds_int(ranges, "regime_detect_enabled", regime_detect_cfg.get("enabled", 1))
            regime_detect_cfg["enabled"] = _suggest_int(trial, "regime_detect_enabled", lo, hi)
        elif tune_regime_thresholds or tune_regime_weights or tune_regime_filters:
            regime_detect_cfg["enabled"] = 1
        if int(regime_detect_cfg.get("enabled", 0)) != 0:
            lo, hi = _bounds_float(ranges, "regime_stress_lo", regime_detect_cfg.get("stress_lo", 0.25))
            regime_detect_cfg["stress_lo"] = _suggest_float(trial, "regime_stress_lo", lo, hi)
            lo, hi = _bounds_float(ranges, "regime_stress_hi", regime_detect_cfg.get("stress_hi", 0.65))
            regime_detect_cfg["stress_hi"] = _suggest_float(trial, "regime_stress_hi", lo, hi)
            if float(regime_detect_cfg["stress_hi"]) <= float(regime_detect_cfg["stress_lo"]):
                regime_detect_cfg["stress_hi"] = min(1.0, float(regime_detect_cfg["stress_lo"]) + 0.05)
            lo, hi = _bounds_float(ranges, "regime_alpha_ema", regime_detect_cfg.get("alpha_ema", 0.15))
            regime_detect_cfg["alpha_ema"] = _suggest_float(trial, "regime_alpha_ema", lo, hi)
            lo, hi = _bounds_float(ranges, "regime_alpha_hysteresis", regime_detect_cfg.get("alpha_hysteresis", 0.03))
            regime_detect_cfg["alpha_hysteresis"] = _suggest_float(trial, "regime_alpha_hysteresis", lo, hi)
            if any(k in ranges for k in ("regime_w_atr_raw_min", "regime_w_atr_raw_max", "regime_w_rng_raw_min", "regime_w_rng_raw_max", "regime_w_vol_raw_min", "regime_w_vol_raw_max", "regime_w_fund_raw_min", "regime_w_fund_raw_max")):
                raw_w = []
                defaults = {
                    "regime_w_atr_raw": safe_float(regime_detect_cfg.get("w_atr", cfg.get("dynamic_cfg", {}).get("w_atr", 0.35)), 0.35),
                    "regime_w_rng_raw": safe_float(regime_detect_cfg.get("w_rng", cfg.get("dynamic_cfg", {}).get("w_rng", 0.20)), 0.20),
                    "regime_w_vol_raw": safe_float(regime_detect_cfg.get("w_vol", cfg.get("dynamic_cfg", {}).get("w_vol", 0.30)), 0.30),
                    "regime_w_fund_raw": safe_float(regime_detect_cfg.get("w_fund", cfg.get("dynamic_cfg", {}).get("w_fund", 0.15)), 0.15),
                }
                for key in ("regime_w_atr_raw", "regime_w_rng_raw", "regime_w_vol_raw", "regime_w_fund_raw"):
                    lo, hi = _bounds_float(ranges, key, defaults[key])
                    raw_w.append(max(0.0, _suggest_float(trial, key, lo, hi)))
                ws = np.asarray(raw_w, dtype=np.float64)
                if float(ws.sum()) <= 0.0:
                    ws[:] = np.array([0.35, 0.20, 0.30, 0.15], dtype=np.float64)
                ws /= float(ws.sum())
                regime_detect_cfg["w_atr"], regime_detect_cfg["w_rng"], regime_detect_cfg["w_vol"], regime_detect_cfg["w_fund"] = [float(x) for x in ws]

    if tune_regime_weights:
        if _has_range_bounds(ranges, "regime_weight_enabled"):
            lo, hi = _bounds_int(ranges, "regime_weight_enabled", regime_weight_cfg.get("enabled", 1))
            regime_weight_cfg["enabled"] = _suggest_int(trial, "regime_weight_enabled", lo, hi)
        else:
            regime_weight_cfg["enabled"] = 1
        if int(regime_weight_cfg.get("enabled", 0)) != 0:
            for key, dflt in (("gate_calm_mix", regime_weight_cfg.get("gate_calm_mix", 0.60)), ("gate_active_mix", regime_weight_cfg.get("gate_active_mix", 0.55)), ("dir_calm_mix", regime_weight_cfg.get("dir_calm_mix", 0.35)), ("dir_active_mix", regime_weight_cfg.get("dir_active_mix", 0.50))):
                lo, hi = _bounds_float(ranges, key, dflt)
                regime_weight_cfg[key] = _suggest_float(trial, key, lo, hi)

    if tune_regime_thresholds:
        if _has_range_bounds(ranges, "regime_threshold_enabled"):
            lo, hi = _bounds_int(ranges, "regime_threshold_enabled", regime_threshold_cfg.get("enabled", 1))
            regime_threshold_cfg["enabled"] = _suggest_int(trial, "regime_threshold_enabled", lo, hi)
        else:
            regime_threshold_cfg["enabled"] = 1
        if _has_range_bounds(ranges, "bucket_min_ready"):
            lo, hi = _bounds_int(ranges, "bucket_min_ready", regime_threshold_cfg.get("bucket_min_ready", 0))
            regime_threshold_cfg["bucket_min_ready"] = _suggest_int(trial, "bucket_min_ready", lo, hi)
        if _has_range_bounds(ranges, "bucket_fallback_global"):
            lo, hi = _bounds_int(ranges, "bucket_fallback_global", regime_threshold_cfg.get("bucket_fallback_global", 1))
            regime_threshold_cfg["bucket_fallback_global"] = _suggest_int(trial, "bucket_fallback_global", lo, hi)
        if int(regime_threshold_cfg.get("enabled", 0)) != 0:
            q_base = float(cfg["q_entry"])
            q_calm = float(regime_threshold_cfg.get("q_entry_calm", q_base))
            q_mid = float(regime_threshold_cfg.get("q_entry_mid", q_base))
            q_active = float(regime_threshold_cfg.get("q_entry_active", q_base))
            if _has_range_bounds(ranges, "q_entry_calm_delta"):
                lo, hi = _bounds_float(ranges, "q_entry_calm_delta", q_calm - q_base)
                q_calm = float(np.clip(q_base + _suggest_float(trial, "q_entry_calm_delta", lo, hi), 0.0, 1.0))
            elif _has_range_bounds(ranges, "q_entry_calm"):
                lo, hi = _bounds_float(ranges, "q_entry_calm", q_calm)
                q_calm = float(np.clip(_suggest_float(trial, "q_entry_calm", lo, hi), 0.0, 1.0))
            if _has_range_bounds(ranges, "q_entry_mid_delta"):
                lo, hi = _bounds_float(ranges, "q_entry_mid_delta", q_mid - q_base)
                q_mid = float(np.clip(q_base + _suggest_float(trial, "q_entry_mid_delta", lo, hi), 0.0, 1.0))
            elif _has_range_bounds(ranges, "q_entry_mid"):
                lo, hi = _bounds_float(ranges, "q_entry_mid", q_mid)
                q_mid = float(np.clip(_suggest_float(trial, "q_entry_mid", lo, hi), 0.0, 1.0))
            if _has_range_bounds(ranges, "q_entry_active_delta"):
                lo, hi = _bounds_float(ranges, "q_entry_active_delta", q_active - q_base)
                q_active = float(np.clip(q_base + _suggest_float(trial, "q_entry_active_delta", lo, hi), 0.0, 1.0))
            elif _has_range_bounds(ranges, "q_entry_active"):
                lo, hi = _bounds_float(ranges, "q_entry_active", q_active)
                q_active = float(np.clip(_suggest_float(trial, "q_entry_active", lo, hi), 0.0, 1.0))
            q_calm = min(q_calm, q_mid, q_active)
            q_active = max(q_calm, q_mid, q_active)
            q_mid = min(max(q_mid, q_calm), q_active)
            regime_threshold_cfg["q_entry_calm"] = float(q_calm)
            regime_threshold_cfg["q_entry_mid"] = float(q_mid)
            regime_threshold_cfg["q_entry_active"] = float(q_active)

            th_base = float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)))
            th_calm = float(regime_threshold_cfg.get("entry_th_floor_calm", th_base))
            th_mid = float(regime_threshold_cfg.get("entry_th_floor_mid", th_base))
            th_active = float(regime_threshold_cfg.get("entry_th_floor_active", th_base))
            if _has_range_bounds(ranges, "entry_th_calm_delta"):
                lo, hi = _bounds_float(ranges, "entry_th_calm_delta", th_calm - th_base)
                th_calm = float(th_base + _suggest_float(trial, "entry_th_calm_delta", lo, hi))
            elif _has_range_bounds(ranges, "entry_th_calm"):
                lo, hi = _bounds_float(ranges, "entry_th_calm", th_calm)
                th_calm = float(_suggest_float(trial, "entry_th_calm", lo, hi))
            if _has_range_bounds(ranges, "entry_th_mid_delta"):
                lo, hi = _bounds_float(ranges, "entry_th_mid_delta", th_mid - th_base)
                th_mid = float(th_base + _suggest_float(trial, "entry_th_mid_delta", lo, hi))
            elif _has_range_bounds(ranges, "entry_th_mid"):
                lo, hi = _bounds_float(ranges, "entry_th_mid", th_mid)
                th_mid = float(_suggest_float(trial, "entry_th_mid", lo, hi))
            if _has_range_bounds(ranges, "entry_th_active_delta"):
                lo, hi = _bounds_float(ranges, "entry_th_active_delta", th_active - th_base)
                th_active = float(th_base + _suggest_float(trial, "entry_th_active_delta", lo, hi))
            elif _has_range_bounds(ranges, "entry_th_active"):
                lo, hi = _bounds_float(ranges, "entry_th_active", th_active)
                th_active = float(_suggest_float(trial, "entry_th_active", lo, hi))
            regime_threshold_cfg["entry_th_floor_calm"] = float(th_calm)
            regime_threshold_cfg["entry_th_floor_mid"] = float(th_mid)
            regime_threshold_cfg["entry_th_floor_active"] = float(th_active)
            regime_threshold_cfg["q_entry_active_dense"] = float(q_active)
            q_sparse = float(regime_threshold_cfg.get("q_entry_active_sparse", q_active))
            if _has_range_bounds(ranges, "q_entry_active_sparse_delta"):
                lo, hi = _bounds_float(ranges, "q_entry_active_sparse_delta", q_sparse - q_active)
                q_sparse = float(np.clip(q_active + _suggest_float(trial, "q_entry_active_sparse_delta", lo, hi), 0.0, 1.0))
            elif _has_range_bounds(ranges, "q_entry_active_sparse"):
                lo, hi = _bounds_float(ranges, "q_entry_active_sparse", q_sparse)
                q_sparse = float(np.clip(_suggest_float(trial, "q_entry_active_sparse", lo, hi), 0.0, 1.0))
            q_sparse = min(float(q_sparse), float(q_active))
            regime_threshold_cfg["q_entry_active_sparse"] = float(q_sparse)
            regime_threshold_cfg["q_entry_active_sparse_delta"] = float(q_sparse - q_active)
            regime_threshold_cfg["entry_th_floor_active_dense"] = float(th_active)
            th_sparse = float(regime_threshold_cfg.get("entry_th_floor_active_sparse", th_active))
            if _has_range_bounds(ranges, "entry_th_active_sparse_delta"):
                lo, hi = _bounds_float(ranges, "entry_th_active_sparse_delta", th_sparse - th_active)
                th_sparse = float(th_active + _suggest_float(trial, "entry_th_active_sparse_delta", lo, hi))
            elif _has_range_bounds(ranges, "entry_th_active_sparse"):
                lo, hi = _bounds_float(ranges, "entry_th_active_sparse", th_sparse)
                th_sparse = float(_suggest_float(trial, "entry_th_active_sparse", lo, hi))
            if th_sparse > th_active:
                th_sparse = th_active
            regime_threshold_cfg["entry_th_floor_active_sparse"] = float(th_sparse)
            regime_threshold_cfg["entry_th_floor_active_sparse_delta"] = float(th_sparse - th_active)

    if tune_regime_filters:
        if _has_range_bounds(ranges, "regime_filter_enabled"):
            lo, hi = _bounds_int(ranges, "regime_filter_enabled", regime_filter_cfg.get("enabled", 1))
            regime_filter_cfg["enabled"] = _suggest_int(trial, "regime_filter_enabled", lo, hi)
        else:
            regime_filter_cfg["enabled"] = 1
        if _has_range_bounds(ranges, "regime_filter_use_vol_split"):
            lo, hi = _bounds_int(ranges, "regime_filter_use_vol_split", regime_filter_cfg.get("use_vol_split", 1))
            regime_filter_cfg["use_vol_split"] = _suggest_int(trial, "regime_filter_use_vol_split", lo, hi)
        if _has_range_bounds(ranges, "regime_filter_use_entry_mult_split"):
            lo, hi = _bounds_int(ranges, "regime_filter_use_entry_mult_split", regime_filter_cfg.get("use_entry_mult_split", 1))
            regime_filter_cfg["use_entry_mult_split"] = _suggest_int(trial, "regime_filter_use_entry_mult_split", lo, hi)
        if int(regime_filter_cfg.get("enabled", 0)) != 0:
            base_vol = float(cfg.get("risk_cfg", {}).get("vol_low_th", -1e9))
            base_atr = float(cfg.get("atr_entry_mult", 1.0))
            base_rng = float(cfg.get("range_entry_mult", 1.0))
            vol_calm = float(regime_filter_cfg.get("vol_low_th_calm", base_vol))
            vol_mid = float(regime_filter_cfg.get("vol_low_th_mid", base_vol))
            vol_active = float(regime_filter_cfg.get("vol_low_th_active", base_vol))
            if _has_range_bounds(ranges, "vol_low_th_calm_delta"):
                lo, hi = _bounds_float(ranges, "vol_low_th_calm_delta", vol_calm - base_vol)
                vol_calm = float(base_vol + _suggest_float(trial, "vol_low_th_calm_delta", lo, hi))
            elif _has_range_bounds(ranges, "vol_low_th_calm"):
                lo, hi = _bounds_float(ranges, "vol_low_th_calm", vol_calm)
                vol_calm = float(_suggest_float(trial, "vol_low_th_calm", lo, hi))
            if _has_range_bounds(ranges, "vol_low_th_mid_delta"):
                lo, hi = _bounds_float(ranges, "vol_low_th_mid_delta", vol_mid - base_vol)
                vol_mid = float(base_vol + _suggest_float(trial, "vol_low_th_mid_delta", lo, hi))
            elif _has_range_bounds(ranges, "vol_low_th_mid"):
                lo, hi = _bounds_float(ranges, "vol_low_th_mid", vol_mid)
                vol_mid = float(_suggest_float(trial, "vol_low_th_mid", lo, hi))
            if _has_range_bounds(ranges, "vol_low_th_active_delta"):
                lo, hi = _bounds_float(ranges, "vol_low_th_active_delta", vol_active - base_vol)
                vol_active = float(base_vol + _suggest_float(trial, "vol_low_th_active_delta", lo, hi))
            elif _has_range_bounds(ranges, "vol_low_th_active"):
                lo, hi = _bounds_float(ranges, "vol_low_th_active", vol_active)
                vol_active = float(_suggest_float(trial, "vol_low_th_active", lo, hi))
            ordered = sorted([float(vol_calm), float(vol_mid), float(vol_active)])
            regime_filter_cfg["vol_low_th_calm"], regime_filter_cfg["vol_low_th_mid"], regime_filter_cfg["vol_low_th_active"] = ordered

            atr_calm = float(regime_filter_cfg.get("atr_entry_mult_calm", base_atr))
            atr_active = float(regime_filter_cfg.get("atr_entry_mult_active", base_atr))
            if _has_range_bounds(ranges, "atr_entry_mult_calm_delta"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_calm_delta", atr_calm - base_atr)
                atr_calm = float(base_atr + _suggest_float(trial, "atr_entry_mult_calm_delta", lo, hi))
            elif _has_range_bounds(ranges, "atr_entry_mult_calm"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_calm", atr_calm)
                atr_calm = float(_suggest_float(trial, "atr_entry_mult_calm", lo, hi))
            if _has_range_bounds(ranges, "atr_entry_mult_active_delta"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_active_delta", atr_active - base_atr)
                atr_active = float(base_atr + _suggest_float(trial, "atr_entry_mult_active_delta", lo, hi))
            elif _has_range_bounds(ranges, "atr_entry_mult_active"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_active", atr_active)
                atr_active = float(_suggest_float(trial, "atr_entry_mult_active", lo, hi))
            if atr_active < atr_calm:
                atr_calm, atr_active = atr_active, atr_calm
            regime_filter_cfg["atr_entry_mult_calm"] = float(atr_calm)
            regime_filter_cfg["atr_entry_mult_active"] = float(atr_active)

            rng_calm = float(regime_filter_cfg.get("range_entry_mult_calm", base_rng))
            rng_active = float(regime_filter_cfg.get("range_entry_mult_active", base_rng))
            if _has_range_bounds(ranges, "range_entry_mult_calm_delta"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_calm_delta", rng_calm - base_rng)
                rng_calm = float(base_rng + _suggest_float(trial, "range_entry_mult_calm_delta", lo, hi))
            elif _has_range_bounds(ranges, "range_entry_mult_calm"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_calm", rng_calm)
                rng_calm = float(_suggest_float(trial, "range_entry_mult_calm", lo, hi))
            if _has_range_bounds(ranges, "range_entry_mult_active_delta"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_active_delta", rng_active - base_rng)
                rng_active = float(base_rng + _suggest_float(trial, "range_entry_mult_active_delta", lo, hi))
            elif _has_range_bounds(ranges, "range_entry_mult_active"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_active", rng_active)
                rng_active = float(_suggest_float(trial, "range_entry_mult_active", lo, hi))
            if rng_active < rng_calm:
                rng_calm, rng_active = rng_active, rng_calm
            regime_filter_cfg["range_entry_mult_calm"] = float(rng_calm)
            regime_filter_cfg["range_entry_mult_active"] = float(rng_active)
            regime_filter_cfg["vol_low_th_active_dense"] = float(regime_filter_cfg.get("vol_low_th_active", base_vol))
            vol_sparse = float(regime_filter_cfg.get("vol_low_th_active_sparse", regime_filter_cfg["vol_low_th_active_dense"]))
            if _has_range_bounds(ranges, "vol_low_th_active_sparse_delta"):
                lo, hi = _bounds_float(ranges, "vol_low_th_active_sparse_delta", vol_sparse - regime_filter_cfg["vol_low_th_active_dense"])
                vol_sparse = float(regime_filter_cfg["vol_low_th_active_dense"] + _suggest_float(trial, "vol_low_th_active_sparse_delta", lo, hi))
            elif _has_range_bounds(ranges, "vol_low_th_active_sparse"):
                lo, hi = _bounds_float(ranges, "vol_low_th_active_sparse", vol_sparse)
                vol_sparse = float(_suggest_float(trial, "vol_low_th_active_sparse", lo, hi))
            if vol_sparse > regime_filter_cfg["vol_low_th_active_dense"]:
                vol_sparse = float(regime_filter_cfg["vol_low_th_active_dense"])
            regime_filter_cfg["vol_low_th_active_sparse"] = float(vol_sparse)
            regime_filter_cfg["vol_low_th_active_sparse_delta"] = float(vol_sparse - regime_filter_cfg["vol_low_th_active_dense"])
            regime_filter_cfg["atr_entry_mult_active_dense"] = float(regime_filter_cfg.get("atr_entry_mult_active", base_atr))
            atr_sparse = float(regime_filter_cfg.get("atr_entry_mult_active_sparse", regime_filter_cfg["atr_entry_mult_active_dense"]))
            if _has_range_bounds(ranges, "atr_entry_mult_active_sparse_delta"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_active_sparse_delta", atr_sparse - regime_filter_cfg["atr_entry_mult_active_dense"])
                atr_sparse = float(regime_filter_cfg["atr_entry_mult_active_dense"] + _suggest_float(trial, "atr_entry_mult_active_sparse_delta", lo, hi))
            elif _has_range_bounds(ranges, "atr_entry_mult_active_sparse"):
                lo, hi = _bounds_float(ranges, "atr_entry_mult_active_sparse", atr_sparse)
                atr_sparse = float(_suggest_float(trial, "atr_entry_mult_active_sparse", lo, hi))
            if atr_sparse > regime_filter_cfg["atr_entry_mult_active_dense"]:
                atr_sparse = float(regime_filter_cfg["atr_entry_mult_active_dense"])
            regime_filter_cfg["atr_entry_mult_active_sparse"] = float(atr_sparse)
            regime_filter_cfg["atr_entry_mult_active_sparse_delta"] = float(atr_sparse - regime_filter_cfg["atr_entry_mult_active_dense"])
            regime_filter_cfg["range_entry_mult_active_dense"] = float(regime_filter_cfg.get("range_entry_mult_active", base_rng))
            rng_sparse = float(regime_filter_cfg.get("range_entry_mult_active_sparse", regime_filter_cfg["range_entry_mult_active_dense"]))
            if _has_range_bounds(ranges, "range_entry_mult_active_sparse_delta"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_active_sparse_delta", rng_sparse - regime_filter_cfg["range_entry_mult_active_dense"])
                rng_sparse = float(regime_filter_cfg["range_entry_mult_active_dense"] + _suggest_float(trial, "range_entry_mult_active_sparse_delta", lo, hi))
            elif _has_range_bounds(ranges, "range_entry_mult_active_sparse"):
                lo, hi = _bounds_float(ranges, "range_entry_mult_active_sparse", rng_sparse)
                rng_sparse = float(_suggest_float(trial, "range_entry_mult_active_sparse", lo, hi))
            if rng_sparse > regime_filter_cfg["range_entry_mult_active_dense"]:
                rng_sparse = float(regime_filter_cfg["range_entry_mult_active_dense"])
            regime_filter_cfg["range_entry_mult_active_sparse"] = float(rng_sparse)
            regime_filter_cfg["range_entry_mult_active_sparse_delta"] = float(rng_sparse - regime_filter_cfg["range_entry_mult_active_dense"])

    if tune_regime_lanes:
        if _has_range_bounds(ranges, "regime_lane_enabled"):
            lo, hi = _bounds_int(ranges, "regime_lane_enabled", regime_lane_cfg.get("enabled", 1))
            regime_lane_cfg["enabled"] = _suggest_int(trial, "regime_lane_enabled", lo, hi)
        else:
            regime_lane_cfg["enabled"] = int(regime_lane_cfg.get("enabled", 1))
        if int(regime_lane_cfg.get("enabled", 0)) != 0:
            if _has_range_bounds(ranges, "active_sparse_enabled"):
                lo, hi = _bounds_int(ranges, "active_sparse_enabled", regime_lane_cfg.get("active_sparse_enabled", 1))
                regime_lane_cfg["active_sparse_enabled"] = _suggest_int(trial, "active_sparse_enabled", lo, hi)
            else:
                regime_lane_cfg["active_sparse_enabled"] = int(regime_lane_cfg.get("active_sparse_enabled", 1))
            if int(regime_lane_cfg.get("active_sparse_enabled", 0)) != 0:
                if _has_range_bounds(ranges, "active_sparse_min_ready"):
                    lo, hi = _bounds_int(ranges, "active_sparse_min_ready", regime_lane_cfg.get("active_sparse_min_ready", 160))
                    regime_lane_cfg["active_sparse_min_ready"] = _suggest_int(trial, "active_sparse_min_ready", lo, hi)
                if _has_range_bounds(ranges, "sparse_gate_q"):
                    lo, hi = _bounds_float(ranges, "sparse_gate_q", regime_lane_cfg.get("sparse_gate_q", 0.55))
                    regime_lane_cfg["sparse_gate_q"] = _suggest_float(trial, "sparse_gate_q", lo, hi)
                if _has_range_bounds(ranges, "sparse_gate_floor_q"):
                    lo, hi = _bounds_float(ranges, "sparse_gate_floor_q", regime_lane_cfg.get("sparse_gate_floor_q", 0.0))
                    regime_lane_cfg["sparse_gate_floor_q"] = _suggest_float(trial, "sparse_gate_floor_q", lo, hi)
                if float(regime_lane_cfg.get("sparse_gate_floor_q", 0.0)) > float(regime_lane_cfg.get("sparse_gate_q", 0.55)):
                    regime_lane_cfg["sparse_gate_floor_q"] = float(regime_lane_cfg.get("sparse_gate_q", 0.55))
                if _has_range_bounds(ranges, "sparse_atr_q"):
                    lo, hi = _bounds_float(ranges, "sparse_atr_q", regime_lane_cfg.get("sparse_atr_q", 0.65))
                    regime_lane_cfg["sparse_atr_q"] = _suggest_float(trial, "sparse_atr_q", lo, hi)
                if _has_range_bounds(ranges, "sparse_range_q"):
                    lo, hi = _bounds_float(ranges, "sparse_range_q", regime_lane_cfg.get("sparse_range_q", 0.65))
                    regime_lane_cfg["sparse_range_q"] = _suggest_float(trial, "sparse_range_q", lo, hi)
                if _has_range_bounds(ranges, "sparse_vol_q"):
                    lo, hi = _bounds_float(ranges, "sparse_vol_q", regime_lane_cfg.get("sparse_vol_q", 0.0))
                    regime_lane_cfg["sparse_vol_q"] = _suggest_float(trial, "sparse_vol_q", lo, hi)
                if _has_range_bounds(ranges, "sparse_require_high_vol"):
                    lo, hi = _bounds_int(ranges, "sparse_require_high_vol", regime_lane_cfg.get("sparse_require_high_vol", 0))
                    regime_lane_cfg["sparse_require_high_vol"] = _suggest_int(trial, "sparse_require_high_vol", lo, hi)
                elif "sparse_require_high_vol" in regime_lane_cfg:
                    regime_lane_cfg["sparse_require_high_vol"] = int(regime_lane_cfg.get("sparse_require_high_vol", 0))
                choices = ranges.get("sparse_high_logic_choices")
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    choices_norm = [str(x).strip().lower() for x in choices if str(x).strip()]
                    if choices_norm:
                        regime_lane_cfg["sparse_high_logic"] = str(trial.suggest_categorical("sparse_high_logic", choices_norm)).strip().lower()

    if (int(regime_weight_cfg.get("enabled", 0)) != 0 or int(regime_threshold_cfg.get("enabled", 0)) != 0 or int(regime_filter_cfg.get("enabled", 0)) != 0 or int(regime_lane_cfg.get("enabled", 0)) != 0) and int(regime_detect_cfg.get("enabled", 0)) == 0:
        regime_detect_cfg["enabled"] = 1

    cfg["regime_detect_cfg"] = regime_detect_cfg
    cfg["regime_weight_cfg"] = regime_weight_cfg
    cfg["regime_threshold_cfg"] = regime_threshold_cfg
    cfg["regime_filter_cfg"] = regime_filter_cfg
    cfg["regime_lane_cfg"] = regime_lane_cfg

    meta["regime_detect_enabled"] = int(cfg.get("regime_detect_cfg", {}).get("enabled", 0))
    meta["regime_weight_enabled"] = int(cfg.get("regime_weight_cfg", {}).get("enabled", 0))
    meta["regime_threshold_enabled"] = int(cfg.get("regime_threshold_cfg", {}).get("enabled", 0))
    meta["regime_filter_enabled"] = int(cfg.get("regime_filter_cfg", {}).get("enabled", 0))
    meta["regime_stress_lo"] = float(cfg.get("regime_detect_cfg", {}).get("stress_lo", 0.25))
    meta["regime_stress_hi"] = float(cfg.get("regime_detect_cfg", {}).get("stress_hi", 0.65))
    meta["regime_alpha_ema"] = float(cfg.get("regime_detect_cfg", {}).get("alpha_ema", 0.15))
    meta["regime_alpha_hysteresis"] = float(cfg.get("regime_detect_cfg", {}).get("alpha_hysteresis", 0.03))
    meta["regime_w_atr"] = float(cfg.get("regime_detect_cfg", {}).get("w_atr", cfg.get("dynamic_cfg", {}).get("w_atr", 0.35)))
    meta["regime_w_rng"] = float(cfg.get("regime_detect_cfg", {}).get("w_rng", cfg.get("dynamic_cfg", {}).get("w_rng", 0.20)))
    meta["regime_w_vol"] = float(cfg.get("regime_detect_cfg", {}).get("w_vol", cfg.get("dynamic_cfg", {}).get("w_vol", 0.30)))
    meta["regime_w_fund"] = float(cfg.get("regime_detect_cfg", {}).get("w_fund", cfg.get("dynamic_cfg", {}).get("w_fund", 0.15)))
    meta["gate_calm_mix"] = float(cfg.get("regime_weight_cfg", {}).get("gate_calm_mix", 0.60))
    meta["gate_active_mix"] = float(cfg.get("regime_weight_cfg", {}).get("gate_active_mix", 0.55))
    meta["dir_calm_mix"] = float(cfg.get("regime_weight_cfg", {}).get("dir_calm_mix", 0.35))
    meta["dir_active_mix"] = float(cfg.get("regime_weight_cfg", {}).get("dir_active_mix", 0.50))
    meta["bucket_min_ready"] = int(cfg.get("regime_threshold_cfg", {}).get("bucket_min_ready", 0))
    meta["bucket_fallback_global"] = int(cfg.get("regime_threshold_cfg", {}).get("bucket_fallback_global", 1))
    meta["q_entry_calm"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_calm", cfg["q_entry"]))
    meta["q_entry_mid"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_mid", cfg["q_entry"]))
    meta["q_entry_active"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_active", cfg["q_entry"]))
    meta["entry_th_calm"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_calm", cfg.get("entry_th_floor", cfg.get("entry_th", 0.0))))
    meta["entry_th_mid"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_mid", cfg.get("entry_th_floor", cfg.get("entry_th", 0.0))))
    meta["entry_th_active"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_active", cfg.get("entry_th_floor", cfg.get("entry_th", 0.0))))
    meta["regime_filter_use_vol_split"] = int(cfg.get("regime_filter_cfg", {}).get("use_vol_split", 1))
    meta["regime_filter_use_entry_mult_split"] = int(cfg.get("regime_filter_cfg", {}).get("use_entry_mult_split", 1))
    meta["vol_low_th_calm"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_calm", cfg.get("risk_cfg", {}).get("vol_low_th", -1e9)))
    meta["vol_low_th_mid"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_mid", cfg.get("risk_cfg", {}).get("vol_low_th", -1e9)))
    meta["vol_low_th_active"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_active", cfg.get("risk_cfg", {}).get("vol_low_th", -1e9)))
    meta["atr_entry_mult_calm"] = float(cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_calm", cfg.get("atr_entry_mult", 1.0)))
    meta["atr_entry_mult_active"] = float(cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_active", cfg.get("atr_entry_mult", 1.0)))
    meta["range_entry_mult_calm"] = float(cfg.get("regime_filter_cfg", {}).get("range_entry_mult_calm", cfg.get("range_entry_mult", 1.0)))
    meta["range_entry_mult_active"] = float(cfg.get("regime_filter_cfg", {}).get("range_entry_mult_active", cfg.get("range_entry_mult", 1.0)))
    meta["regime_lane_enabled"] = int(cfg.get("regime_lane_cfg", {}).get("enabled", 0))
    meta["active_sparse_enabled"] = int(cfg.get("regime_lane_cfg", {}).get("active_sparse_enabled", 0))
    meta["active_sparse_min_ready"] = int(cfg.get("regime_lane_cfg", {}).get("active_sparse_min_ready", 160))
    meta["sparse_gate_q"] = float(cfg.get("regime_lane_cfg", {}).get("sparse_gate_q", 0.55))
    meta["sparse_gate_floor_q"] = float(cfg.get("regime_lane_cfg", {}).get("sparse_gate_floor_q", 0.0))
    meta["sparse_atr_q"] = float(cfg.get("regime_lane_cfg", {}).get("sparse_atr_q", 0.65))
    meta["sparse_range_q"] = float(cfg.get("regime_lane_cfg", {}).get("sparse_range_q", 0.65))
    meta["sparse_vol_q"] = float(cfg.get("regime_lane_cfg", {}).get("sparse_vol_q", 0.0))
    meta["sparse_require_high_vol"] = int(cfg.get("regime_lane_cfg", {}).get("sparse_require_high_vol", 0))
    meta["sparse_high_logic"] = str(cfg.get("regime_lane_cfg", {}).get("sparse_high_logic", "or"))
    meta["tune_regime_lanes"] = int(tune_regime_lanes)
    meta["q_entry_active_dense"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_active_dense", cfg.get("regime_threshold_cfg", {}).get("q_entry_active", cfg["q_entry"])))
    meta["q_entry_active_sparse"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_active_sparse", cfg.get("regime_threshold_cfg", {}).get("q_entry_active_dense", cfg.get("regime_threshold_cfg", {}).get("q_entry_active", cfg["q_entry"]))))
    meta["q_entry_active_sparse_delta"] = float(cfg.get("regime_threshold_cfg", {}).get("q_entry_active_sparse_delta", meta["q_entry_active_sparse"] - meta["q_entry_active_dense"]))
    meta["entry_th_active_dense"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_active_dense", cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_active", cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)))))
    meta["entry_th_active_sparse"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_active_sparse", meta["entry_th_active_dense"]))
    meta["entry_th_active_sparse_delta"] = float(cfg.get("regime_threshold_cfg", {}).get("entry_th_floor_active_sparse_delta", meta["entry_th_active_sparse"] - meta["entry_th_active_dense"]))
    meta["vol_low_th_active_dense"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_active_dense", cfg.get("regime_filter_cfg", {}).get("vol_low_th_active", cfg.get("risk_cfg", {}).get("vol_low_th", -1e9))))
    meta["vol_low_th_active_sparse"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_active_sparse", meta["vol_low_th_active_dense"]))
    meta["vol_low_th_active_sparse_delta"] = float(cfg.get("regime_filter_cfg", {}).get("vol_low_th_active_sparse_delta", meta["vol_low_th_active_sparse"] - meta["vol_low_th_active_dense"]))
    meta["atr_entry_mult_active_dense"] = float(cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_active_dense", cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_active", cfg.get("atr_entry_mult", 1.0))))
    meta["atr_entry_mult_active_sparse"] = float(cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_active_sparse", meta["atr_entry_mult_active_dense"]))
    meta["atr_entry_mult_active_sparse_delta"] = float(cfg.get("regime_filter_cfg", {}).get("atr_entry_mult_active_sparse_delta", meta["atr_entry_mult_active_sparse"] - meta["atr_entry_mult_active_dense"]))
    meta["range_entry_mult_active_dense"] = float(cfg.get("regime_filter_cfg", {}).get("range_entry_mult_active_dense", cfg.get("regime_filter_cfg", {}).get("range_entry_mult_active", cfg.get("range_entry_mult", 1.0))))
    meta["range_entry_mult_active_sparse"] = float(cfg.get("regime_filter_cfg", {}).get("range_entry_mult_active_sparse", meta["range_entry_mult_active_dense"]))
    meta["range_entry_mult_active_sparse_delta"] = float(cfg.get("regime_filter_cfg", {}).get("range_entry_mult_active_sparse_delta", meta["range_entry_mult_active_sparse"] - meta["range_entry_mult_active_dense"]))

    # new structural knobs
    hs_lo, hs_hi = _bounds_float(ranges, "hard_sl_mult_pre_unlock", cfg.get("hard_sl_mult_pre_unlock", 1.0))
    cfg["hard_sl_mult_pre_unlock"] = _suggest_float(trial, "hard_sl_mult_pre_unlock", hs_lo, hs_hi)
    tgb_lo, tgb_hi = _bounds_int(ranges, "trail_grace_after_bep", cfg.get("trail_grace_after_bep", 0))
    tgu_lo, tgu_hi = _bounds_int(ranges, "trail_grace_after_unlock", cfg.get("trail_grace_after_unlock", 0))
    cfg["trail_grace_after_bep"] = _suggest_int(trial, "trail_grace_after_bep", tgb_lo, tgb_hi)
    cfg["trail_grace_after_unlock"] = _suggest_int(trial, "trail_grace_after_unlock", tgu_lo, tgu_hi)
    meta["hard_sl_mult_pre_unlock"] = float(cfg["hard_sl_mult_pre_unlock"])
    meta["trail_grace_after_bep"] = int(cfg["trail_grace_after_bep"])
    meta["trail_grace_after_unlock"] = int(cfg["trail_grace_after_unlock"])

    # progress-aware protection
    prog = copy.deepcopy(cfg.get("progress_protect_cfg", {}))
    if ("early_softsl_min_hold_min" in ranges) or ("early_softsl_min_hold_max" in ranges):
        lo, hi = _bounds_int(ranges, "early_softsl_min_hold", prog.get("early_softsl_min_hold", 2))
        prog["early_softsl_min_hold"] = _suggest_int(trial, "early_softsl_min_hold", lo, hi)
    if ("early_softsl_progress_frac_min" in ranges) or ("early_softsl_progress_frac_max" in ranges):
        lo, hi = _bounds_float(ranges, "early_softsl_progress_frac", prog.get("early_softsl_progress_frac", 0.5))
        prog["early_softsl_progress_frac"] = _suggest_float(trial, "early_softsl_progress_frac", lo, hi)
        prog["early_softsl_enabled"] = 1
    if ("early_trail_min_hold_min" in ranges) or ("early_trail_min_hold_max" in ranges):
        lo, hi = _bounds_int(ranges, "early_trail_min_hold", prog.get("early_trail_min_hold", 3))
        prog["early_trail_min_hold"] = _suggest_int(trial, "early_trail_min_hold", lo, hi)
    if ("early_trail_progress_frac_min" in ranges) or ("early_trail_progress_frac_max" in ranges):
        lo, hi = _bounds_float(ranges, "early_trail_progress_frac", prog.get("early_trail_progress_frac", 0.85))
        prog["early_trail_progress_frac"] = _suggest_float(trial, "early_trail_progress_frac", lo, hi)
        prog["early_trail_enabled"] = 1
    if ("early_trail_ref_updates_min_min" in ranges) or ("early_trail_ref_updates_min_max" in ranges):
        lo, hi = _bounds_int(ranges, "early_trail_ref_updates_min", prog.get("early_trail_ref_updates_min", 1))
        prog["early_trail_ref_updates_min"] = _suggest_int(trial, "early_trail_ref_updates_min", lo, hi)
        prog["early_trail_enabled"] = 1
    cfg["progress_protect_cfg"] = prog
    meta["early_softsl_enabled"] = int(prog.get("early_softsl_enabled", 0))
    meta["early_softsl_min_hold"] = int(prog.get("early_softsl_min_hold", 2))
    meta["early_softsl_progress_frac"] = float(prog.get("early_softsl_progress_frac", 0.5))
    meta["early_trail_enabled"] = int(prog.get("early_trail_enabled", 0))
    meta["early_trail_min_hold"] = int(prog.get("early_trail_min_hold", 3))
    meta["early_trail_progress_frac"] = float(prog.get("early_trail_progress_frac", 0.85))
    meta["early_trail_ref_updates_min"] = int(prog.get("early_trail_ref_updates_min", 1))

    # dynamic block
    dyn = copy.deepcopy(cfg["dynamic_cfg"])
    if tune_dynamic:
        dyn["enabled"] = 1
        mm_lo, mm_hi = _bounds_float(ranges, "dyn_margin_cap", dyn.get("margin_cap", 0.50))
        fs_lo, fs_hi = _bounds_float(ranges, "dyn_funding_soft_min", dyn.get("funding_soft_min", 0.0))
        dyn["margin_cap"] = _suggest_float(trial, "dyn_margin_cap", mm_lo, mm_hi)
        dyn["funding_soft_min"] = _suggest_float(trial, "dyn_funding_soft_min", fs_lo, fs_hi)
        raw_w = []
        for key, dflt in [("dyn_w_atr_raw", dyn.get("w_atr", 0.35)), ("dyn_w_rng_raw", dyn.get("w_rng", 0.20)), ("dyn_w_vol_raw", dyn.get("w_vol", 0.30)), ("dyn_w_fund_raw", dyn.get("w_fund", 0.15))]:
            lo, hi = _bounds_float(ranges, key, dflt)
            raw_w.append(max(0.0, _suggest_float(trial, key, lo, hi)))
        ws = np.asarray(raw_w, dtype=np.float64)
        if float(ws.sum()) <= 0.0:
            ws[:] = np.array([0.35, 0.20, 0.30, 0.15], dtype=np.float64)
        ws /= float(ws.sum())
        dyn["w_atr"], dyn["w_rng"], dyn["w_vol"], dyn["w_fund"] = [float(x) for x in ws]
        for key, dflt in [
            ("dyn_lev_trend_k", dyn.get("lev_trend_k", 0.18)),
            ("dyn_lev_stress_k", dyn.get("lev_stress_k", 0.30)),
            ("dyn_gate_stress_k", dyn.get("gate_stress_k", 0.12)),
            ("dyn_gate_trend_k", dyn.get("gate_trend_k", 0.08)),
            ("dyn_bep_stress_k", dyn.get("bep_stress_k", 0.0)),
            ("dyn_bep_trend_k", dyn.get("bep_trend_k", 0.05)),
            ("dyn_trail_trend_k", dyn.get("trail_trend_k", 0.15)),
            ("dyn_trail_stress_k", dyn.get("trail_stress_k", 0.0)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            val = _suggest_float(trial, key, lo, hi)
            dyn_map = {
                "dyn_lev_trend_k": "lev_trend_k",
                "dyn_lev_stress_k": "lev_stress_k",
                "dyn_gate_stress_k": "gate_stress_k",
                "dyn_gate_trend_k": "gate_trend_k",
                "dyn_bep_stress_k": "bep_stress_k",
                "dyn_bep_trend_k": "bep_trend_k",
                "dyn_trail_trend_k": "trail_trend_k",
                "dyn_trail_stress_k": "trail_stress_k",
            }
            dyn[dyn_map[key]] = float(val)
        for key, dflt in [
            ("dyn_lev_scale_min", dyn.get("lev_scale_min", 0.70)),
            ("dyn_lev_scale_max", dyn.get("lev_scale_max", 1.05)),
            ("dyn_gate_mult_min", dyn.get("gate_mult_min", 0.95)),
            ("dyn_gate_mult_max", dyn.get("gate_mult_max", 1.15)),
            ("dyn_bep_scale_min", dyn.get("bep_scale_min", 0.75)),
            ("dyn_bep_scale_max", dyn.get("bep_scale_max", 1.05)),
            ("dyn_trail_scale_min", dyn.get("trail_scale_min", 0.90)),
            ("dyn_trail_scale_max", dyn.get("trail_scale_max", 1.12)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            val = _suggest_float(trial, key, lo, hi)
            dyn_map = {
                "dyn_lev_scale_min": "lev_scale_min",
                "dyn_lev_scale_max": "lev_scale_max",
                "dyn_gate_mult_min": "gate_mult_min",
                "dyn_gate_mult_max": "gate_mult_max",
                "dyn_bep_scale_min": "bep_scale_min",
                "dyn_bep_scale_max": "bep_scale_max",
                "dyn_trail_scale_min": "trail_scale_min",
                "dyn_trail_scale_max": "trail_scale_max",
            }
            dyn[dyn_map[key]] = float(val)

    if tune_dynamic_sl:
        dyn["enabled"] = 1
        dyn["use_dyn_sl"] = 1
        for key, dflt in [
            ("dyn_sl_scale_min", dyn.get("sl_scale_min", 0.85)),
            ("dyn_sl_scale_max", dyn.get("sl_scale_max", 1.05)),
            ("dyn_sl_trend_k", dyn.get("sl_trend_k", 0.0)),
            ("dyn_sl_stress_k", dyn.get("sl_stress_k", 0.0)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            val = _suggest_float(trial, key, lo, hi)
            dyn_map = {
                "dyn_sl_scale_min": "sl_scale_min",
                "dyn_sl_scale_max": "sl_scale_max",
                "dyn_sl_trend_k": "sl_trend_k",
                "dyn_sl_stress_k": "sl_stress_k",
            }
            dyn[dyn_map[key]] = float(val)

    if tune_dynamic_soft_sl:
        dyn["enabled"] = 1
        dyn["use_dyn_soft_sl"] = 1
        for key, dflt in [
            ("dyn_softsl_stress_mid", dyn.get("softsl_stress_mid", 0.35)),
            ("dyn_softsl_stress_hi", dyn.get("softsl_stress_hi", 0.65)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            val = _suggest_float(trial, key, lo, hi)
            dyn_map = {
                "dyn_softsl_stress_mid": "softsl_stress_mid",
                "dyn_softsl_stress_hi": "softsl_stress_hi",
            }
            dyn[dyn_map[key]] = float(val)
        for key, dflt in [
            ("dyn_softsl_relax_mid", dyn.get("softsl_relax_mid", 1)),
            ("dyn_softsl_relax_hi", dyn.get("softsl_relax_hi", 2)),
        ]:
            lo, hi = _bounds_int(ranges, key, dflt)
            val = _suggest_int(trial, key, lo, hi)
            dyn_map = {"dyn_softsl_relax_mid": "softsl_relax_mid", "dyn_softsl_relax_hi": "softsl_relax_hi"}
            dyn[dyn_map[key]] = int(val)
        if ("allow_soft_sl_before_trail_min" in ranges) or ("allow_soft_sl_before_trail_max" in ranges):
            lo, hi = _bounds_int(ranges, "allow_soft_sl_before_trail", dyn.get("allow_soft_sl_before_trail", 0))
            dyn["allow_soft_sl_before_trail"] = _suggest_int(trial, "allow_soft_sl_before_trail", lo, hi)
        if ("softsl_hold_floor_min" in ranges) or ("softsl_hold_floor_max" in ranges):
            lo, hi = _bounds_int(ranges, "softsl_hold_floor", dyn.get("softsl_hold_floor", 0))
            dyn["softsl_hold_floor"] = _suggest_int(trial, "softsl_hold_floor", lo, hi)

    tpw = copy.deepcopy(cfg.get("tp_window_cfg", {}))
    if tune_pre_tp_window:
        for key, dflt in [
            ("tp_window_enabled", tpw.get("enabled", 0)),
            ("tp_window_extend_bars", tpw.get("extend_bars", 0)),
            ("tp_window_block_early_trail", tpw.get("block_early_trail", 1)),
            ("tp_window_block_early_soft_sl", tpw.get("block_early_soft_sl", 1)),
            ("tp_window_floor_trail_hold_to_tp", tpw.get("floor_trail_hold_to_tp", 1)),
            ("tp_window_floor_soft_sl_hold_to_tp", tpw.get("floor_soft_sl_hold_to_tp", 1)),
            ("tp_window_suspend_post_bep_shield_before_tp", tpw.get("suspend_post_bep_shield_before_tp", 1)),
        ]:
            lo, hi = _bounds_int(ranges, key, dflt)
            tpw_map = {
                "tp_window_enabled": "enabled",
                "tp_window_extend_bars": "extend_bars",
                "tp_window_block_early_trail": "block_early_trail",
                "tp_window_block_early_soft_sl": "block_early_soft_sl",
                "tp_window_floor_trail_hold_to_tp": "floor_trail_hold_to_tp",
                "tp_window_floor_soft_sl_hold_to_tp": "floor_soft_sl_hold_to_tp",
                "tp_window_suspend_post_bep_shield_before_tp": "suspend_post_bep_shield_before_tp",
            }
            tpw[tpw_map[key]] = int(_suggest_int(trial, key, lo, hi))
        if ("tp_window_progress_frac_arm_min" in ranges) or ("tp_window_progress_frac_arm_max" in ranges):
            lo, hi = _bounds_float(ranges, "tp_window_progress_frac_arm", tpw.get("progress_frac_arm", 0.70))
            tpw["progress_frac_arm"] = _suggest_float(trial, "tp_window_progress_frac_arm", lo, hi)
        if ("tp_window_expire_on_pullback_frac_min" in ranges) or ("tp_window_expire_on_pullback_frac_max" in ranges):
            lo, hi = _bounds_float(ranges, "tp_window_expire_on_pullback_frac", tpw.get("expire_on_pullback_frac", 0.35))
            tpw["expire_on_pullback_frac"] = _suggest_float(trial, "tp_window_expire_on_pullback_frac", lo, hi)
        if ("post_bep_shield_ignore_softsl_hold_min" in ranges) or ("post_bep_shield_ignore_softsl_hold_max" in ranges):
            lo, hi = _bounds_int(ranges, "post_bep_shield_ignore_softsl_hold", dyn.get("post_bep_shield_ignore_softsl_hold", 0))
            dyn["post_bep_shield_ignore_softsl_hold"] = _suggest_int(trial, "post_bep_shield_ignore_softsl_hold", lo, hi)
    cfg["tp_window_cfg"] = tpw

    epcfg = copy.deepcopy(cfg.get("entry_episode_cfg", {}))
    if tune_entry_episode:
        for key, dflt in [
            ("entry_episode_enabled", epcfg.get("enabled", 0)),
            ("rearm_enabled", epcfg.get("rearm_enabled", 0)),
            ("run_gap_reset_bars", epcfg.get("run_gap_reset_bars", 1)),
            ("episode_max_entries_per_run", epcfg.get("episode_max_entries_per_run", 1)),
            ("rearm_same_side_only", epcfg.get("rearm_same_side_only", 1)),
            ("rearm_cooldown_bars", epcfg.get("rearm_cooldown_bars", 1)),
            ("rearm_max_bars_after_exit", epcfg.get("rearm_max_bars_after_exit", 8)),
            ("rearm_after_trail", epcfg.get("rearm_after_trail", 1)),
            ("rearm_after_tp", epcfg.get("rearm_after_tp", 1)),
            ("rearm_after_sl", epcfg.get("rearm_after_sl", 0)),
        ]:
            lo, hi = _bounds_int(ranges, key, dflt)
            ep_map = {
                "entry_episode_enabled": "enabled",
                "rearm_enabled": "rearm_enabled",
                "run_gap_reset_bars": "run_gap_reset_bars",
                "episode_max_entries_per_run": "episode_max_entries_per_run",
                "rearm_same_side_only": "rearm_same_side_only",
                "rearm_cooldown_bars": "rearm_cooldown_bars",
                "rearm_max_bars_after_exit": "rearm_max_bars_after_exit",
                "rearm_after_trail": "rearm_after_trail",
                "rearm_after_tp": "rearm_after_tp",
                "rearm_after_sl": "rearm_after_sl",
            }
            epcfg[ep_map[key]] = int(_suggest_int(trial, key, lo, hi))
        for key, dflt in [
            ("rearm_gate_reset_frac", epcfg.get("rearm_gate_reset_frac", 0.45)),
            ("rearm_gate_refresh_frac", epcfg.get("rearm_gate_refresh_frac", 0.70)),
            ("rearm_price_reset_frac", epcfg.get("rearm_price_reset_frac", 0.0004)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            epcfg[key] = _suggest_float(trial, key, lo, hi)
    cfg["entry_episode_cfg"] = epcfg

    if tune_margin_gate:
        dyn["enabled"] = 1
        dyn["use_margin_gate"] = 1
        dyn["use_margin_lev_degrade"] = 1 if _flag_from_any(ranges.get("dyn_use_margin_lev_degrade", 0)) else dyn.get("use_margin_lev_degrade", 0)
        for key, dflt in [
            ("margin_req_base", dyn.get("margin_req_base", 0.0)),
            ("margin_req_stress_k", dyn.get("margin_req_stress_k", 0.0)),
            ("margin_req_trend_k", dyn.get("margin_req_trend_k", 0.0)),
            ("margin_req_max", dyn.get("margin_req_max", 0.0)),
            ("margin_lev_floor", dyn.get("margin_lev_floor", 0.70)),
            ("margin_lev_band", dyn.get("margin_lev_band", 0.05)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            dyn[key] = float(_suggest_float(trial, key, lo, hi))

    if tune_pre_bep:
        dyn["enabled"] = 1
        dyn["use_pre_bep_timeout"] = 1
        for key, dflt in [
            ("pre_bep_timeout_bars", dyn.get("pre_bep_timeout_bars", 3)),
            ("pre_bep_force_close_bars", dyn.get("pre_bep_force_close_bars", 0)),
            ("pre_bep_softsl_delta", dyn.get("pre_bep_softsl_delta", 0)),
        ]:
            lo, hi = _bounds_int(ranges, key, dflt)
            dyn[key] = int(_suggest_int(trial, key, lo, hi))
        if ("pre_bep_force_close_red_only_min" in ranges) or ("pre_bep_force_close_red_only_max" in ranges):
            lo, hi = _bounds_int(ranges, "pre_bep_force_close_red_only", dyn.get("pre_bep_force_close_red_only", 1))
            dyn["pre_bep_force_close_red_only"] = int(_suggest_int(trial, "pre_bep_force_close_red_only", lo, hi))
        for key, dflt in [
            ("pre_bep_stress_th", dyn.get("pre_bep_stress_th", 0.55)),
            ("pre_bep_progress_frac", dyn.get("pre_bep_progress_frac", 0.55)),
            ("pre_bep_degrade_sl_scale", dyn.get("pre_bep_degrade_sl_scale", 0.85)),
        ]:
            lo, hi = _bounds_float(ranges, key, dflt)
            dyn[key] = float(_suggest_float(trial, key, lo, hi))

    cfg["dynamic_cfg"] = dyn

    meta["min_hold_soft_sl_bars_raw"] = int(meta.get("min_hold_soft_sl_bars", cfg.get("min_hold_soft_sl_bars", 0)))
    if int(dyn.get("allow_soft_sl_before_trail", 0)) == 0:
        _trail_hold_eff = max(0, int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0)))))
        if int(cfg.get("min_hold_soft_sl_bars", 0)) < _trail_hold_eff:
            cfg["min_hold_soft_sl_bars"] = int(_trail_hold_eff)
            meta["softsl_guard_clamped"] = 1
        else:
            meta["softsl_guard_clamped"] = 0
    else:
        meta["softsl_guard_clamped"] = 0
    meta["min_hold_soft_sl_bars"] = int(cfg.get("min_hold_soft_sl_bars", 0))
    meta["min_hold_soft_sl_bars_effective"] = int(cfg.get("min_hold_soft_sl_bars", 0))

    meta["allow_soft_sl_before_trail"] = int(dyn.get("allow_soft_sl_before_trail", 0))
    meta["softsl_hold_floor"] = int(dyn.get("softsl_hold_floor", 0))
    meta["post_bep_shield_ignore_softsl_hold"] = int(dyn.get("post_bep_shield_ignore_softsl_hold", 0))
    meta["pre_bep_force_close_red_only"] = int(dyn.get("pre_bep_force_close_red_only", 1))
    meta["tp_window_enabled"] = int(cfg.get("tp_window_cfg", {}).get("enabled", 0))
    meta["tp_window_progress_frac_arm"] = float(cfg.get("tp_window_cfg", {}).get("progress_frac_arm", 0.70))
    meta["tp_window_extend_bars"] = int(cfg.get("tp_window_cfg", {}).get("extend_bars", 0))
    meta["tp_window_block_early_trail"] = int(cfg.get("tp_window_cfg", {}).get("block_early_trail", 1))
    meta["tp_window_block_early_soft_sl"] = int(cfg.get("tp_window_cfg", {}).get("block_early_soft_sl", 1))
    meta["tp_window_floor_trail_hold_to_tp"] = int(cfg.get("tp_window_cfg", {}).get("floor_trail_hold_to_tp", 1))
    meta["tp_window_floor_soft_sl_hold_to_tp"] = int(cfg.get("tp_window_cfg", {}).get("floor_soft_sl_hold_to_tp", 1))
    meta["tp_window_suspend_post_bep_shield_before_tp"] = int(cfg.get("tp_window_cfg", {}).get("suspend_post_bep_shield_before_tp", 1))
    meta["tp_window_expire_on_pullback_frac"] = float(cfg.get("tp_window_cfg", {}).get("expire_on_pullback_frac", 0.35))
    meta["entry_episode_enabled"] = int(cfg.get("entry_episode_cfg", {}).get("enabled", 0))
    meta["rearm_enabled"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_enabled", 0))
    meta["run_gap_reset_bars"] = int(cfg.get("entry_episode_cfg", {}).get("run_gap_reset_bars", 1))
    meta["episode_max_entries_per_run"] = int(cfg.get("entry_episode_cfg", {}).get("episode_max_entries_per_run", 1))
    meta["rearm_same_side_only"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_same_side_only", 1))
    meta["rearm_cooldown_bars"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_cooldown_bars", 1))
    meta["rearm_max_bars_after_exit"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_max_bars_after_exit", 8))
    meta["rearm_gate_reset_frac"] = float(cfg.get("entry_episode_cfg", {}).get("rearm_gate_reset_frac", 0.45))
    meta["rearm_gate_refresh_frac"] = float(cfg.get("entry_episode_cfg", {}).get("rearm_gate_refresh_frac", 0.70))
    meta["rearm_price_reset_frac"] = float(cfg.get("entry_episode_cfg", {}).get("rearm_price_reset_frac", 0.0004))
    meta["rearm_after_trail"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_after_trail", 1))
    meta["rearm_after_tp"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_after_tp", 1))
    meta["rearm_after_sl"] = int(cfg.get("entry_episode_cfg", {}).get("rearm_after_sl", 0))

    meta.update({
        "tune_fee_bep_mult": int(tune_fee_bep_mult),
        "tune_dynamic": int(tune_dynamic),
        "tune_dynamic_sl": int(tune_dynamic_sl),
        "tune_dynamic_soft_sl": int(tune_dynamic_soft_sl),
        "tune_margin_gate": int(tune_margin_gate),
        "tune_entry_filters": int(tune_entry_filters),
        "tune_pre_bep": int(tune_pre_bep),
        "tune_pre_tp_window": int(tune_pre_tp_window),
        "tune_entry_episode": int(tune_entry_episode),
        "tune_gate_weights": int(tune_gate_weights),
        "tune_dir_weights": int(tune_dir_weights),
        "tune_regime_detection": int(tune_regime_detection),
        "tune_regime_weights": int(tune_regime_weights),
        "tune_regime_thresholds": int(tune_regime_thresholds),
        "tune_regime_filters": int(tune_regime_filters),
    })
    # IMPORTANT:
    # base_cfg is already normalized once in main().
    # cfg is a deepcopy(base_cfg) that we mutate in-place for this trial.
    # Running build_single_best_config(base_cfg, cfg, ...) here does an extra
    # deep-merge + normalize, and prepare_trial_context() used to normalize again.
    #
    # Keep exactly ONE final normalize here, then mark the candidate as already normalized
    # so prepare_trial_context() can skip redundant work.
    cfg = normalize_single_config_from_any(cfg)
    cfg["__normalized_single_v110__"] = 1

    return cfg, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--base_json", type=str, required=True)
    ap.add_argument("--ranges_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--log_csv", type=str, required=True)
    ap.add_argument("--log_flush_every", type=int, default=0, help="0=end-only, N=rewrite csv every N appended trials")

    ap.add_argument("--models_dir", type=str, default="")
    ap.add_argument("--cache_npz", type=str, default="")
    ap.add_argument("--seq_len", type=int, default=300)
    ap.add_argument("--window", type=int, default=60000)
    ap.add_argument("--window-includes-hist-extra", dest="window_includes_hist_extra", type=int, default=0)
    ap.add_argument("--oos_len", type=int, default=30000)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--entry_q_lookback", type=int, default=6000)
    ap.add_argument("--entry_q_min_ready", type=int, default=300)
    ap.add_argument("--vol-feature", dest="vol_feature", type=str, default="")
    ap.add_argument("--atr-feature", dest="atr_feature", type=str, default="")

    ap.add_argument("--cost_list", type=str, default="")
    ap.add_argument("--slip_list", type=str, default="")
    ap.add_argument("--maker_fee_per_side", type=float, default=0.0)
    ap.add_argument("--risk_lev_cap", type=float, default=12.0)
    ap.add_argument("--fee_tp_mult", type=float, default=None)
    ap.add_argument("--low_vol_filter", type=int, default=None)
    ap.add_argument("--trail_after_bep", type=int, default=None)
    ap.add_argument("--risk_entry_mode", type=int, default=None)
    ap.add_argument("--use_atr_scaling", type=int, default=None)

    ap.add_argument("--w_cost_mean", type=float, default=0.0)
    ap.add_argument("--w_cost_worst", type=float, default=1.0)
    ap.add_argument("--cost_worst_agg", type=str, default="min")
    ap.add_argument("--cost_worst_k", type=int, default=1)
    ap.add_argument("--cost_worst_q", type=float, default=0.2)
    ap.add_argument("--w_mean", type=float, default=0.18)
    ap.add_argument("--w_worst", type=float, default=0.82)
    ap.add_argument("--worst_agg", type=str, default="cvar")
    ap.add_argument("--worst_k", type=int, default=4)
    ap.add_argument("--worst_q", type=float, default=0.2)

    ap.add_argument("--alpha_dd", type=float, default=0.9)
    ap.add_argument("--beta_tail", type=float, default=2.0)
    ap.add_argument("--stop_equity", type=float, default=0.40)
    ap.add_argument("--stop_dd", type=float, default=0.35)
    ap.add_argument("--warmup_steps", type=int, default=0)

    ap.add_argument("--trade_mode", default="soft")
    ap.add_argument("--trade_target", type=float, default=300.0)
    ap.add_argument("--trade_band", type=float, default=150.0)
    ap.add_argument("--barrier_k", type=float, default=2.0)
    ap.add_argument("--trade_shortage_penalty", type=float, default=0.05)
    ap.add_argument("--trade_excess_penalty", type=float, default=0.01)
    ap.add_argument("--side_balance_penalty_k", type=float, default=0.0)
    ap.add_argument("--min_short_trades_global", type=int, default=0)
    ap.add_argument("--min_short_share_global", type=float, default=0.0)
    ap.add_argument("--min_seg_trades", type=int, default=0)
    ap.add_argument("--min_seg_trades_mode", type=str, default="soft", choices=["hard", "soft"])
    ap.add_argument("--min_seg_trades_penalty_k", type=float, default=1.0)
    ap.add_argument("--min_seg_trades_penalty_power", type=float, default=1.0)
    ap.add_argument("--short_trades_guard_mode", type=str, default="hard", choices=["hard", "soft"])
    ap.add_argument("--short_trades_penalty_k", type=float, default=1.0)
    ap.add_argument("--short_trades_penalty_power", type=float, default=1.0)
    ap.add_argument("--short_share_guard_mode", type=str, default="hard", choices=["hard", "soft"])
    ap.add_argument("--short_share_penalty_k", type=float, default=1000.0)
    ap.add_argument("--short_share_penalty_power", type=float, default=1.0)
    ap.add_argument("--hard_guard_base", type=float, default=1000000.0)
    ap.add_argument("--hard_guard_step", type=float, default=1.0)
    ap.add_argument("--seg_bottom2_target", "--seg-bottom2-target", dest="seg_bottom2_target", type=float, default=90.0)
    ap.add_argument("--seg_bottom2_penalty_k", "--seg-bottom2-penalty-k", dest="seg_bottom2_penalty_k", type=float, default=0.55)
    ap.add_argument("--seg_floor_target", "--seg-floor-target", dest="seg_floor_target", type=float, default=85.0)
    ap.add_argument("--seg_floor_penalty_k", "--seg-floor-penalty-k", dest="seg_floor_penalty_k", type=float, default=0.30)
    ap.add_argument("--trade_cv_cap", "--trade-cv-cap", dest="trade_cv_cap", type=float, default=0.36)
    ap.add_argument("--trade_cv_penalty_k", "--trade-cv-penalty-k", dest="trade_cv_penalty_k", type=float, default=0.10)
    ap.add_argument("--min_seg_trades_tier", type=str, default="")  # compatibility; last value wins
    ap.add_argument("--maxhold_ratio_free", type=float, default=1.0)
    ap.add_argument("--maxhold_penalty_k", type=float, default=0.0)
    ap.add_argument("--maxhold_penalty_power", type=float, default=2.0)

    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--pruner", type=str, default="median", choices=["none", "median"])
    ap.add_argument("--prune_startup_trials", type=int, default=100)
    ap.add_argument("--prune_warmup_steps", type=int, default=2)

    # tuning flags (single-tier canonical)
    ap.add_argument("--tune_fee_bep_mult", action="store_true")
    ap.add_argument("--tune_dynamic", action="store_true")
    ap.add_argument("--tune_dynamic_sl", action="store_true")
    ap.add_argument("--tune_dynamic_soft_sl", action="store_true")
    ap.add_argument("--tune_margin_gate", action="store_true")
    ap.add_argument("--tune_entry_filters", action="store_true")
    ap.add_argument("--tune_pre_bep", action="store_true")
    ap.add_argument("--tune_pre_tp_window", action="store_true")
    ap.add_argument("--tune_entry_episode", action="store_true")
    ap.add_argument("--tune_gate_weights", action="store_true")
    ap.add_argument("--tune_dir_weights", action="store_true")
    ap.add_argument("--tune_regime_detection", action="store_true")
    ap.add_argument("--tune_regime_weights", action="store_true")
    ap.add_argument("--tune_regime_thresholds", action="store_true")
    ap.add_argument("--tune_regime_filters", action="store_true")
    ap.add_argument("--tune_regime_lanes", action="store_true")
    ap.add_argument("--regime-extreme-max-frac", dest="regime_extreme_max_frac", type=float, default=1.0)
    ap.add_argument("--regime-extreme-penalty-k", dest="regime_extreme_penalty_k", type=float, default=0.0)

    # compatibility aliases from tiered CLI
    ap.add_argument("--tier_logic", type=str, default="")
    ap.add_argument("--multi_pos", type=int, default=0)
    ap.add_argument("--pos_frac", type=str, default="")
    ap.add_argument("--entry_q_lookback_top", type=int, default=0)
    ap.add_argument("--tune_dynamic_top", action="store_true")
    ap.add_argument("--tune_pre_bep_top", action="store_true")
    ap.add_argument("--tune_margin_gate_top", action="store_true")

    ap.add_argument("--hybrid-batch-size", dest="hybrid_batch_size", type=int, default=2048)
    ap.add_argument("--hybrid-amp", type=int, default=1)

    args = ap.parse_args()

    with open(args.base_json, "r", encoding="utf-8") as f:
        base_raw = json.load(f)
    base_cfg = normalize_single_config_from_any(base_raw)
    with open(args.ranges_json, "r", encoding="utf-8") as f:
        ranges_raw = json.load(f)
    ranges = apply_ranges_overrides(ranges_raw)

    # CLI hard overrides into base config
    base_cfg["risk_cfg"]["risk_lev_cap"] = float(args.risk_lev_cap)
    base_cfg["tail_cfg"]["stop_equity"] = float(args.stop_equity)
    base_cfg["tail_cfg"]["stop_dd"] = float(args.stop_dd)
    base_cfg["tail_cfg"]["warmup_steps"] = int(args.warmup_steps)
    if args.fee_tp_mult is not None:
        base_cfg["fee_tp_mult"] = float(args.fee_tp_mult)
    if args.low_vol_filter is not None:
        base_cfg["low_vol_filter"] = int(args.low_vol_filter)
    if args.trail_after_bep is not None:
        base_cfg["trail_after_bep"] = int(args.trail_after_bep)
    if args.risk_entry_mode is not None:
        base_cfg["risk_entry_mode"] = int(args.risk_entry_mode)
    if args.use_atr_scaling is not None:
        base_cfg["use_atr_scaling"] = 1 if int(args.use_atr_scaling) != 0 else 0
    if str(args.vol_feature).strip():
        base_cfg.setdefault("runtime_feature_cfg", {})["vol_feature"] = str(args.vol_feature).strip()
    if str(args.atr_feature).strip():
        base_cfg.setdefault("runtime_feature_cfg", {})["atr_feature"] = str(args.atr_feature).strip()
    runtime_feature_cfg = dict(base_cfg.get("runtime_feature_cfg", {}))
    vol_feature = str(runtime_feature_cfg.get("vol_feature", "vol_z_60") or "vol_z_60")
    atr_feature = str(runtime_feature_cfg.get("atr_feature", "atr10_rel") or "atr10_rel")

    if args.entry_q_lookback_top and args.entry_q_lookback <= 0:
        args.entry_q_lookback = int(args.entry_q_lookback_top)

    csv_path = args.csv
    cols0 = pd.read_csv(csv_path, nrows=0).columns
    time_col = "timestamp" if "timestamp" in cols0 else ("time" if "time" in cols0 else None)
    if time_col is None:
        raise ValueError("CSV must contain 'time' or 'timestamp' column.")
    required_cols = [time_col, "open", "high", "low", "close", "minutes_to_next_funding", vol_feature, atr_feature] + list(HybridScalpInference.FEATURES)
    required_cols = list(dict.fromkeys(required_cols))
    missing_cols = [c for c in required_cols if c not in cols0]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    df = pd.read_csv(csv_path, usecols=required_cols)

    hist_extra = int(max(args.entry_q_lookback, args.seq_len, args.entry_q_min_ready, args.entry_q_lookback_top))
    if int(args.window_includes_hist_extra) != 0:
        need = int(args.window)
        window_oos = int(max(1, need - hist_extra))
    else:
        window_oos = int(args.window)
        need = int(window_oos + hist_extra)

    if int(args.oos_len) > int(window_oos):
        raise ValueError(f"oos_len({args.oos_len}) must be <= window({window_oos})")
    if len(df) < need:
        raise ValueError(f"rows={len(df)} < need={need} (window={window_oos} + hist_extra={hist_extra})")
    df_window = df.iloc[-need:].reset_index(drop=True)

    print(f"[RUNTIME FEATURES] vol_feature={vol_feature} atr_feature={atr_feature}")
    signals_by_h, ready = precompute_hybrids(
        df_window=df_window,
        seq_len=int(args.seq_len),
        models_dir=args.models_dir.strip() or None,
        cache_npz=args.cache_npz.strip() or "",
        batch_size=int(args.hybrid_batch_size),
        use_amp=int(args.hybrid_amp),
    )

    open_ = df_window["open"].to_numpy(dtype=np.float64)
    close = df_window["close"].to_numpy(dtype=np.float64)
    high = df_window["high"].to_numpy(dtype=np.float64)
    low = df_window["low"].to_numpy(dtype=np.float64)
    vol_z = df_window[vol_feature].to_numpy(dtype=np.float64)
    atr_rel_arr = df_window[atr_feature].to_numpy(dtype=np.float64)
    minutes_to_next_funding = df_window["minutes_to_next_funding"].to_numpy(dtype=np.float64)

    window_start = hist_extra
    oos_end = int(window_start + window_oos)
    oos_start = max(int(window_start), int(oos_end - int(args.oos_len)))
    seg_len = int(args.oos_len // args.splits)
    if seg_len * int(args.splits) != int(args.oos_len):
        raise ValueError("oos_len must be divisible by splits")

    seg_bounds = [(int(oos_start + k * seg_len), int(oos_start + (k + 1) * seg_len)) for k in range(int(args.splits))]

    if args.min_seg_trades <= 0 and args.min_seg_trades_tier:
        vals = [safe_int(x, 0) for x in parse_float_list(args.min_seg_trades_tier)]
        if vals:
            args.min_seg_trades = int(vals[-1])

    # Warm numba fast-core once so first trial does not pay compilation cost.
    warmup_single_fast_core()

    cost_scenarios = _normalize_cost_scenarios(
        parse_float_list(args.cost_list, [safe_float(base_raw.get("cost_per_side", base_raw.get("taker_fee_per_side", 0.0005)), 0.0005)]),
        parse_float_list(args.slip_list, [safe_float(base_raw.get("slip_per_side", 0.0), 0.0)]),
        maker_fee_per_side=float(args.maker_fee_per_side),
        default_cost=safe_float(base_raw.get("cost_per_side", base_raw.get("taker_fee_per_side", 0.0005)), 0.0005),
        default_slip=safe_float(base_raw.get("slip_per_side", 0.0), 0.0),
    )

    if args.pruner == "median":
        pruner = MedianPruner(n_startup_trials=int(args.prune_startup_trials), n_warmup_steps=int(args.prune_warmup_steps))
    else:
        pruner = NopPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=int(args.seed)),
        pruner=pruner,
    )

    trial_rows: List[Dict[str, Any]] = []

    score_cfg: Dict[str, Any] = {
        "alpha_dd": float(args.alpha_dd),
        "beta_tail": float(args.beta_tail),
        "trade_mode": str(args.trade_mode),
        "trade_target": float(args.trade_target),
        "trade_band": float(args.trade_band),
        "barrier_k": float(args.barrier_k),
        "trade_shortage_penalty": float(args.trade_shortage_penalty),
        "trade_excess_penalty": float(args.trade_excess_penalty),
        "side_balance_penalty_k": float(args.side_balance_penalty_k),
        "min_short_trades_global": int(args.min_short_trades_global),
        "short_trades_guard_mode": str(args.short_trades_guard_mode),
        "short_trades_penalty_k": float(args.short_trades_penalty_k),
        "short_trades_penalty_power": float(args.short_trades_penalty_power),
        "min_short_share_global": float(args.min_short_share_global),
        "short_share_guard_mode": str(args.short_share_guard_mode),
        "short_share_penalty_k": float(args.short_share_penalty_k),
        "short_share_penalty_power": float(args.short_share_penalty_power),
        "maxhold_ratio_free": float(args.maxhold_ratio_free),
        "maxhold_penalty_k": float(args.maxhold_penalty_k),
        "maxhold_penalty_power": float(args.maxhold_penalty_power),
    }
    objective_cfg: Dict[str, Any] = {
        "min_seg_trades": int(args.min_seg_trades),
        "min_seg_trades_mode": str(args.min_seg_trades_mode),
        "min_seg_trades_penalty_k": float(args.min_seg_trades_penalty_k),
        "min_seg_trades_penalty_power": float(args.min_seg_trades_penalty_power),
        "min_short_trades_global": int(args.min_short_trades_global),
        "short_trades_guard_mode": str(args.short_trades_guard_mode),
        "short_trades_penalty_k": float(args.short_trades_penalty_k),
        "short_trades_penalty_power": float(args.short_trades_penalty_power),
        "min_short_share_global": float(args.min_short_share_global),
        "short_share_guard_mode": str(args.short_share_guard_mode),
        "short_share_penalty_k": float(args.short_share_penalty_k),
        "short_share_penalty_power": float(args.short_share_penalty_power),
        "regime_extreme_max_frac": float(args.regime_extreme_max_frac),
        "regime_extreme_penalty_k": float(args.regime_extreme_penalty_k),
        "hard_guard_base": float(args.hard_guard_base),
        "hard_guard_step": float(args.hard_guard_step),
        "seg_bottom2_target": float(args.seg_bottom2_target),
        "seg_bottom2_penalty_k": float(args.seg_bottom2_penalty_k),
        "seg_floor_target": float(args.seg_floor_target),
        "seg_floor_penalty_k": float(args.seg_floor_penalty_k),
        "trade_cv_cap": float(args.trade_cv_cap),
        "trade_cv_penalty_k": float(args.trade_cv_penalty_k),
    }

    last_flushed_rows = 0

    def _flush_log(force: bool = False) -> None:
        nonlocal last_flushed_rows
        if not trial_rows:
            return
        if not force:
            if int(args.log_flush_every) <= 0:
                return
            if (len(trial_rows) - int(last_flushed_rows)) < int(args.log_flush_every):
                return
        pd.DataFrame(trial_rows).to_csv(args.log_csv, index=False, encoding="utf-8")
        last_flushed_rows = int(len(trial_rows))

    def objective(trial: optuna.Trial) -> float:
        cfg_candidate, sampled_meta = _materialize_candidate(trial, base_cfg, ranges, args)
        row: Dict[str, Any] = {"trial": int(trial.number), "status": "running"}
        row.update(sampled_meta)
        row["schema"] = cfg_candidate.get("schema", "single_v110")
        row["objective_version"] = "coverage_v40+softsl_guard_fix"
        row["regime_lane_version"] = "active_bandpass_lane_v57"
        row["softsl_guard_fix"] = 1
        row["seg_bottom2_target"] = float(args.seg_bottom2_target)
        row["seg_bottom2_penalty_k"] = float(args.seg_bottom2_penalty_k)
        row["seg_floor_target"] = float(args.seg_floor_target)
        row["seg_floor_penalty_k"] = float(args.seg_floor_penalty_k)
        row["trade_cv_cap"] = float(args.trade_cv_cap)
        row["trade_cv_penalty_k"] = float(args.trade_cv_penalty_k)

        scenario_scores: List[float] = []
        base_seg_scores: List[float] = []
        base_seg_trades: List[int] = []
        base_seg_net: List[float] = []
        total_long = 0
        total_short = 0
        total_trades_all = 0
        total_tails = 0

        try:
            trial_ctx = prepare_trial_context(
                open_px=open_,
                close_px=close,
                high_px=high,
                low_px=low,
                signals_by_h=signals_by_h,
                ready=ready,
                vol_z=vol_z,
                atr_rel=atr_rel_arr,
                minutes_to_next_funding=minutes_to_next_funding,
                cfg=cfg_candidate,
            )
            prepared_segments = []
            for seg_start, seg_end in seg_bounds:
                prepared_segments.append(
                    prepare_single_segment_fast_inputs_from_context(
                        ctx=trial_ctx,
                        seg_start=seg_start,
                        seg_end=seg_end,
                        entry_q_lookback=int(args.entry_q_lookback),
                        entry_q_min_ready=int(args.entry_q_min_ready),
                    )
                )
            regime_calm_proxy = float(np.mean([float(p.get("regime_calm_frac", 0.0)) for p in prepared_segments])) if prepared_segments else 0.0
            regime_active_proxy = float(np.mean([float(p.get("regime_active_frac", 0.0)) for p in prepared_segments])) if prepared_segments else 0.0
            row["regime_calm_frac_proxy"] = float(regime_calm_proxy)
            row["regime_active_frac_proxy"] = float(regime_active_proxy)

            for si, sc in enumerate(cost_scenarios):
                seg_scores: List[float] = []
                seg_trades: List[int] = []
                seg_net: List[float] = []
                seg_tails = 0
                scen_long = 0
                scen_short = 0
                for step, prepared_seg in enumerate(prepared_segments):
                    res = evaluate_prepared_single_segment_fast(
                        prepared=prepared_seg,
                        score_cfg=score_cfg,
                        cost_per_side=float(sc["taker"]),
                        slip_per_side=float(sc["slip"]),
                        maker_fee_per_side=float(sc["maker"]),
                        need_diag=False,
                    )
                    seg_scores.append(float(res["score"]))
                    seg_trades.append(int(res["trades"]))
                    seg_net.append(float(res["net_ret"]))
                    seg_tails += int(res["tail"])
                    scen_long += int(res["long_trades"])
                    scen_short += int(res["short_trades"])

                    if si == 0:
                        base_seg_scores.append(float(res["score"]))
                        base_seg_trades.append(int(res["trades"]))
                        base_seg_net.append(float(res["net_ret"]))
                        total_long += int(res["long_trades"])
                        total_short += int(res["short_trades"])
                        total_trades_all += int(res["trades"])
                        total_tails += int(res["tail"])
                        running = float(np.mean(base_seg_scores))
                        trial.report(running, step)
                        if trial.should_prune():
                            row.update({
                                "status": "pruned",
                                "base_seg_done": int(step + 1),
                                "base_score_running": float(running),
                            })
                            trial_rows.append(row)
                            _flush_log()
                            raise optuna.TrialPruned()

                mean_score = float(np.mean(seg_scores)) if seg_scores else float("-inf")
                worst_score = agg_worst(seg_scores, args.worst_agg, int(args.worst_k), float(args.worst_q))
                scen_score = float(args.w_mean) * mean_score + float(args.w_worst) * worst_score
                scenario_scores.append(float(scen_score))
                row[f"score_s{si}"] = float(scen_score)
                row[f"score_s{si}_mean"] = float(mean_score)
                row[f"score_s{si}_worst"] = float(worst_score)
                row[f"trades_s{si}_mean"] = float(np.mean(seg_trades)) if seg_trades else 0.0
                row[f"net_s{si}_mean"] = float(np.mean(seg_net)) if seg_net else 0.0
                row[f"tails_s{si}"] = int(seg_tails)
                row[f"long_s{si}"] = int(scen_long)
                row[f"short_s{si}"] = int(scen_short)

            score_cost_mean = float(np.mean(scenario_scores)) if scenario_scores else float("-inf")
            score_cost_worst = agg_worst(scenario_scores, args.cost_worst_agg, int(args.cost_worst_k), float(args.cost_worst_q))
            if len(scenario_scores) == 1:
                score_raw = float(scenario_scores[0])
            else:
                score_raw = float(args.w_cost_mean) * score_cost_mean + float(args.w_cost_worst) * score_cost_worst

            min_seg_seen = int(min(base_seg_trades)) if base_seg_trades else 0
            short_share_all = float(total_short / total_trades_all) if total_trades_all > 0 else 0.0
            row["min_seg_trades_seen"] = int(min_seg_seen)
            row["long_all"] = int(total_long)
            row["short_all"] = int(total_short)
            row["short_share_all"] = float(short_share_all)
            row["tails_all"] = int(total_tails)

            breakdown = assemble_objective(
                score_raw,
                score_cost_mean=score_cost_mean,
                score_cost_worst=score_cost_worst,
                min_seg_seen=min_seg_seen,
                total_short=total_short,
                short_share_all=short_share_all,
                regime_calm_frac=regime_calm_proxy,
                regime_active_frac=regime_active_proxy,
                seg_trades=base_seg_trades,
                cfg=objective_cfg,
            )
            final_score = float(breakdown.objective_final)

            row["score_base"] = float(base_seg_scores and (float(args.w_mean) * float(np.mean(base_seg_scores)) + float(args.w_worst) * agg_worst(base_seg_scores, args.worst_agg, int(args.worst_k), float(args.worst_q))) or float("-inf"))
            row["score_cost_mean"] = float(score_cost_mean)
            row["score_cost_worst"] = float(score_cost_worst)
            row["score_raw"] = float(breakdown.score_raw)
            row["penalty_min_seg_trades"] = float(breakdown.penalty_min_seg_trades)
            row["penalty_min_short_trades"] = float(breakdown.penalty_min_short_trades)
            row["penalty_min_short_share"] = float(breakdown.penalty_min_short_share)
            row["regime_extreme_penalty"] = float(breakdown.penalty_regime_extreme)
            row["penalty_bottom2_trades"] = float(breakdown.penalty_bottom2_trades)
            row["penalty_seg_trade_floor"] = float(breakdown.penalty_seg_trade_floor)
            row["penalty_trade_cv"] = float(breakdown.penalty_trade_cv)
            row["seg_trade_min"] = int(breakdown.seg_trade_min)
            row["bottom2_mean_trades"] = float(breakdown.bottom2_mean_trades)
            row["seg_trade_mean"] = float(breakdown.seg_trade_mean)
            row["seg_trade_std"] = float(breakdown.seg_trade_std)
            row["seg_trade_cv"] = float(breakdown.seg_trade_cv)
            row["objective_final"] = float(breakdown.objective_final)
            row["feasible_min_seg_trades"] = int(breakdown.feasible_min_seg_trades)
            row["feasible_short_trades"] = int(breakdown.feasible_short_trades)
            row["feasible_short_share"] = int(breakdown.feasible_short_share)
            row["feasible_all"] = int(breakdown.feasible_all)
            row["objective"] = float(final_score)
            row["status"] = "complete"
            row["base_seg_score_mean"] = float(np.mean(base_seg_scores)) if base_seg_scores else float("nan")
            row["base_seg_score_worst"] = agg_worst(base_seg_scores, args.worst_agg, int(args.worst_k), float(args.worst_q)) if base_seg_scores else float("nan")
            row["base_seg_trades_mean"] = float(np.mean(base_seg_trades)) if base_seg_trades else 0.0
            row["base_seg_net_mean"] = float(np.mean(base_seg_net)) if base_seg_net else 0.0

            trial_rows.append(row)
            _flush_log()
            return float(final_score)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
            trial_rows.append(row)
            _flush_log()
            raise

    print("[START] Single-tier v57 autotune")
    print(f"[INFO] rows={len(df_window)} window={window_oos} hist_extra={hist_extra} oos_len={args.oos_len} splits={args.splits}")
    print(f"[COST_SCENARIOS] {cost_scenarios}")

    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
    _flush_log(force=True)
    complete_rows = [r for r in trial_rows if r.get("status") == "complete"]
    feasible_complete = sum(int(bool(r.get("feasible_all", 0))) for r in complete_rows)
    if complete_rows and feasible_complete == 0:
        print("[WARN] No feasible complete trials under current global guards. objective_final is dominated by guard penalties.")

    best_trial = study.best_trial
    best_cfg, sampled_meta = _materialize_candidate(best_trial, base_cfg, ranges, args)

    # lookup logged best row for richer metadata
    best_row = None
    for r in trial_rows:
        if int(r.get("trial", -1)) == int(best_trial.number) and r.get("status") == "complete":
            best_row = r
            break
    tuned_meta = dict(sampled_meta)
    tuned_meta.update({
        "trial": int(best_trial.number),
        "score": float(best_trial.value),
        "cost_scenarios": cost_scenarios,
        "w_cost_mean": float(args.w_cost_mean),
        "w_cost_worst": float(args.w_cost_worst),
        "cost_worst_agg": str(args.cost_worst_agg),
        "cost_worst_k": int(args.cost_worst_k),
        "cost_worst_q": float(args.cost_worst_q),
        "w_mean": float(args.w_mean),
        "w_worst": float(args.w_worst),
        "worst_agg": str(args.worst_agg),
        "worst_k": int(args.worst_k),
        "worst_q": float(args.worst_q),
        "alpha_dd": float(args.alpha_dd),
        "beta_tail": float(args.beta_tail),
        "trade_mode": str(args.trade_mode),
        "trade_target": float(args.trade_target),
        "trade_band": float(args.trade_band),
        "min_seg_trades": int(args.min_seg_trades),
        "min_seg_trades_mode": str(args.min_seg_trades_mode),
        "min_seg_trades_penalty_k": float(args.min_seg_trades_penalty_k),
        "min_seg_trades_penalty_power": float(args.min_seg_trades_penalty_power),
        "side_balance_penalty_k": float(args.side_balance_penalty_k),
        "min_short_trades_global": int(args.min_short_trades_global),
        "short_trades_guard_mode": str(args.short_trades_guard_mode),
        "short_trades_penalty_k": float(args.short_trades_penalty_k),
        "short_trades_penalty_power": float(args.short_trades_penalty_power),
        "min_short_share_global": float(args.min_short_share_global),
        "short_share_guard_mode": str(args.short_share_guard_mode),
        "short_share_penalty_k": float(args.short_share_penalty_k),
        "short_share_penalty_power": float(args.short_share_penalty_power),
        "regime_extreme_max_frac": float(args.regime_extreme_max_frac),
        "regime_extreme_penalty_k": float(args.regime_extreme_penalty_k),
        "hard_guard_base": float(args.hard_guard_base),
        "hard_guard_step": float(args.hard_guard_step),
        "seg_bottom2_target": float(args.seg_bottom2_target),
        "seg_bottom2_penalty_k": float(args.seg_bottom2_penalty_k),
        "seg_floor_target": float(args.seg_floor_target),
        "seg_floor_penalty_k": float(args.seg_floor_penalty_k),
        "trade_cv_cap": float(args.trade_cv_cap),
        "trade_cv_penalty_k": float(args.trade_cv_penalty_k),
        "objective_version": "coverage_v40",
        "regime_lane_version": "active_bandpass_lane_v57",
    })
    if best_row:
        tuned_meta.update({
            "score_base": float(best_row.get("score_base", best_trial.value)),
            "score_cost_mean": float(best_row.get("score_cost_mean", best_trial.value)),
            "score_cost_worst": float(best_row.get("score_cost_worst", best_trial.value)),
            "score_raw": float(best_row.get("score_raw", best_trial.value)),
            "objective_final": float(best_row.get("objective_final", best_trial.value)),
            "penalty_min_seg_trades": float(best_row.get("penalty_min_seg_trades", 0.0)),
            "penalty_min_short_trades": float(best_row.get("penalty_min_short_trades", 0.0)),
            "penalty_min_short_share": float(best_row.get("penalty_min_short_share", 0.0)),
            "penalty_regime_extreme": float(best_row.get("regime_extreme_penalty", 0.0)),
            "penalty_bottom2_trades": float(best_row.get("penalty_bottom2_trades", 0.0)),
            "penalty_seg_trade_floor": float(best_row.get("penalty_seg_trade_floor", 0.0)),
            "penalty_trade_cv": float(best_row.get("penalty_trade_cv", 0.0)),
            "feasible_all": int(best_row.get("feasible_all", 0)),
            "long_all": int(best_row.get("long_all", 0)),
            "short_all": int(best_row.get("short_all", 0)),
            "short_share_all": float(best_row.get("short_share_all", 0.0)),
            "tails_all": int(best_row.get("tails_all", 0)),
            "base_seg_trades_mean": float(best_row.get("base_seg_trades_mean", 0.0)),
            "base_seg_net_mean": float(best_row.get("base_seg_net_mean", 0.0)),
            "seg_trade_min": int(best_row.get("seg_trade_min", 0)),
            "bottom2_mean_trades": float(best_row.get("bottom2_mean_trades", 0.0)),
            "seg_trade_mean": float(best_row.get("seg_trade_mean", 0.0)),
            "seg_trade_std": float(best_row.get("seg_trade_std", 0.0)),
            "seg_trade_cv": float(best_row.get("seg_trade_cv", 0.0)),
        })
        for k, v in best_row.items():
            if isinstance(k, str) and k.startswith("score_s"):
                tuned_meta[k] = v

    # persist canonical cost knobs into config for readability
    if cost_scenarios:
        best_cfg["cost_per_side"] = float(cost_scenarios[0]["taker"])
        best_cfg["slip_per_side"] = float(cost_scenarios[0]["slip"])
        best_cfg["maker_fee_per_side"] = float(cost_scenarios[0]["maker"])
    best_cfg["tail_cfg"]["stop_equity"] = float(args.stop_equity)
    best_cfg["tail_cfg"]["stop_dd"] = float(args.stop_dd)
    best_cfg["tail_cfg"]["warmup_steps"] = int(args.warmup_steps)
    best_cfg["tuned_meta"] = tuned_meta

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)
    _flush_log(force=True)

    print("[DONE] Saved best config ->", args.out_json)
    print("[DONE] Saved trial log   ->", args.log_csv)
    print(f"[BEST] trial={best_trial.number} score={best_trial.value:.6f}")


if __name__ == "__main__":
    main()
