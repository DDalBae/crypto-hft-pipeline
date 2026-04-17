# -*- coding: utf-8 -*-
"""
Single-tier RL-mode backtest (v58).

Key changes vs tiered backtests:
- no low/mid/top runtime tiers
- no multi_pos / pos_frac in core logic
- gate and direction are both weighted h1/h3/h5 mixtures
- split-hold parameters are independent
- BEP arm and BEP stop are separated
- same score logic as autotune is used for intrabar worst-case path selection
- coverage-aware objective metrics/penalties are aligned with autotune when present
- active_dense / active_sparse regime-lane diagnostics are exported when enabled, including band-pass sparse lane v2 metadata
- hidden config/tuned_meta fallback can be disabled with --allow-config-fallback 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hybrid_inference_scalp_v7 import HybridScalpInference
from hybrid_core_modified_v8 import (
    EXIT_NAMES,
    agg_worst,
    assemble_objective,
    evaluate_prepared_single_segment_hybrid,
    prepare_trial_context,
    prepare_single_segment_inputs_from_context,
    normalize_single_config_from_any,
    precompute_hybrids,
    safe_float,
    warmup_single_fast_core,
)


def derive_outputs_from_trade_log(trade_log_csv: str):
    trade_log_csv = trade_log_csv or "rolling_oos_tradelog_single_v50.csv"
    base, _ext = os.path.splitext(trade_log_csv)
    if "tradelog" in base:
        tpl = base.replace("tradelog", "{kind}")
    else:
        tpl = base + "_{kind}"
    exit_stats_csv = tpl.format(kind="exitstats") + ".csv"
    exit_stats_meta = tpl.format(kind="exitstats_meta") + ".json"
    single_stats_csv = tpl.format(kind="single_stats") + ".csv"
    results_csv = tpl.format(kind="results") + ".csv"
    return exit_stats_csv, exit_stats_meta, single_stats_csv, results_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="rolling_oos", choices=["single", "oos", "rolling_oos", "oos_seg"])
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--window", type=int, default=60000)
    ap.add_argument("--window-includes-hist-extra", dest="window_includes_hist_extra", type=int, default=0)
    ap.add_argument("--oos-len", dest="oos_len", type=int, default=30000)
    ap.add_argument("--splits", type=int, default=10)

    ap.add_argument("--cost-per-side", dest="cost_per_side", type=float, default=None)
    ap.add_argument("--slip-per-side", dest="slip_per_side", type=float, default=None)
    ap.add_argument("--maker-fee-per-side", dest="maker_fee_per_side", type=float, default=None)
    ap.add_argument("--fee-tp-mult", dest="fee_tp_mult", type=float, default=None)
    ap.add_argument("--fee-bep-mult", dest="fee_bep_mult", type=float, default=None, help="deprecated alias -> bep_arm_fee_mult")
    ap.add_argument("--atr-entry-mult", dest="atr_entry_mult", type=float, default=None)
    ap.add_argument("--range-entry-mult", dest="range_entry_mult", type=float, default=None)
    ap.add_argument("--low-vol-filter", dest="low_vol_filter", type=int, default=None)
    ap.add_argument("--trail-after-bep", dest="trail_after_bep", type=int, default=None)
    ap.add_argument("--risk-entry-mode", dest="risk_entry_mode", type=int, default=None)
    ap.add_argument("--use-atr-scaling", dest="use_atr_scaling", type=int, default=None)
    ap.add_argument("--leverage", type=float, default=None)
    ap.add_argument("--risk-lev-cap", dest="risk_lev_cap", type=float, default=None)
    ap.add_argument("--min-hold-tp", dest="min_hold_tp_bars", type=int, default=None)
    ap.add_argument("--tp-window-enabled", dest="tp_window_enabled", type=int, default=None)
    ap.add_argument("--tp-window-extend-bars", dest="tp_window_extend_bars", type=int, default=None)
    ap.add_argument("--tp-window-block-early-trail", dest="tp_window_block_early_trail", type=int, default=None)
    ap.add_argument("--tp-window-block-early-soft-sl", dest="tp_window_block_early_soft_sl", type=int, default=None)
    ap.add_argument("--tp-window-floor-trail-hold-to-tp", dest="tp_window_floor_trail_hold_to_tp", type=int, default=None)
    ap.add_argument("--tp-window-floor-soft-sl-hold-to-tp", dest="tp_window_floor_soft_sl_hold_to_tp", type=int, default=None)
    ap.add_argument("--tp-window-suspend-post-bep-shield", dest="tp_window_suspend_post_bep_shield_before_tp", type=int, default=None)
    ap.add_argument("--tp-window-progress-frac-arm", dest="tp_window_progress_frac_arm", type=float, default=None)
    ap.add_argument("--tp-window-expire-on-pullback-frac", dest="tp_window_expire_on_pullback_frac", type=float, default=None)
    ap.add_argument("--allow-soft-sl-before-trail", dest="allow_soft_sl_before_trail", type=int, default=None)
    ap.add_argument("--softsl-hold-floor", dest="softsl_hold_floor", type=int, default=None)
    ap.add_argument("--post-bep-shield-ignore-softsl-hold", dest="post_bep_shield_ignore_softsl_hold", type=int, default=None)
    ap.add_argument("--pre-bep-force-close-red-only", dest="pre_bep_force_close_red_only", type=int, default=None)
    ap.add_argument("--entry-episode-enabled", dest="entry_episode_enabled", type=int, default=None)
    ap.add_argument("--rearm-enabled", dest="rearm_enabled", type=int, default=None)
    ap.add_argument("--run-gap-reset-bars", dest="run_gap_reset_bars", type=int, default=None)
    ap.add_argument("--episode-max-entries-per-run", dest="episode_max_entries_per_run", type=int, default=None)
    ap.add_argument("--rearm-same-side-only", dest="rearm_same_side_only", type=int, default=None)
    ap.add_argument("--rearm-cooldown-bars", dest="rearm_cooldown_bars", type=int, default=None)
    ap.add_argument("--rearm-max-bars-after-exit", dest="rearm_max_bars_after_exit", type=int, default=None)
    ap.add_argument("--rearm-gate-reset-frac", dest="rearm_gate_reset_frac", type=float, default=None)
    ap.add_argument("--rearm-gate-refresh-frac", dest="rearm_gate_refresh_frac", type=float, default=None)
    ap.add_argument("--rearm-price-reset-frac", dest="rearm_price_reset_frac", type=float, default=None)
    ap.add_argument("--rearm-after-trail", dest="rearm_after_trail", type=int, default=None)
    ap.add_argument("--rearm-after-tp", dest="rearm_after_tp", type=int, default=None)
    ap.add_argument("--rearm-after-sl", dest="rearm_after_sl", type=int, default=None)
    ap.add_argument("--regime-detect-enabled", dest="regime_detect_enabled", type=int, default=None)
    ap.add_argument("--regime-threshold-enabled", dest="regime_threshold_enabled", type=int, default=None)
    ap.add_argument("--q-entry-calm", dest="q_entry_calm", type=float, default=None)
    ap.add_argument("--q-entry-mid", dest="q_entry_mid", type=float, default=None)
    ap.add_argument("--q-entry-active", dest="q_entry_active", type=float, default=None)
    ap.add_argument("--entry-th-calm", dest="entry_th_calm", type=float, default=None)
    ap.add_argument("--entry-th-mid", dest="entry_th_mid", type=float, default=None)
    ap.add_argument("--entry-th-active", dest="entry_th_active", type=float, default=None)
    ap.add_argument("--bucket-min-ready", dest="bucket_min_ready", type=int, default=None)
    ap.add_argument("--bucket-fallback-global", dest="bucket_fallback_global", type=int, default=None)
    ap.add_argument("--regime-filter-enabled", dest="regime_filter_enabled", type=int, default=None)
    ap.add_argument("--regime-filter-use-vol-split", dest="regime_filter_use_vol_split", type=int, default=None)
    ap.add_argument("--regime-filter-use-entry-mult-split", dest="regime_filter_use_entry_mult_split", type=int, default=None)
    ap.add_argument("--regime-filter-mid-interp-mode", dest="regime_filter_mid_interp_mode", type=str, default=None)
    ap.add_argument("--vol-low-th-calm", dest="vol_low_th_calm", type=float, default=None)
    ap.add_argument("--vol-low-th-mid", dest="vol_low_th_mid", type=float, default=None)
    ap.add_argument("--vol-low-th-active", dest="vol_low_th_active", type=float, default=None)
    ap.add_argument("--atr-entry-mult-calm", dest="atr_entry_mult_calm", type=float, default=None)
    ap.add_argument("--atr-entry-mult-active", dest="atr_entry_mult_active", type=float, default=None)
    ap.add_argument("--range-entry-mult-calm", dest="range_entry_mult_calm", type=float, default=None)
    ap.add_argument("--range-entry-mult-active", dest="range_entry_mult_active", type=float, default=None)
    ap.add_argument("--regime-lane-enabled", dest="regime_lane_enabled", type=int, default=None)
    ap.add_argument("--active-sparse-enabled", dest="active_sparse_enabled", type=int, default=None)
    ap.add_argument("--active-sparse-min-ready", dest="active_sparse_min_ready", type=int, default=None)
    ap.add_argument("--sparse-gate-q", dest="sparse_gate_q", type=float, default=None)
    ap.add_argument("--sparse-gate-floor-q", dest="sparse_gate_floor_q", type=float, default=None)
    ap.add_argument("--sparse-atr-q", dest="sparse_atr_q", type=float, default=None)
    ap.add_argument("--sparse-range-q", dest="sparse_range_q", type=float, default=None)
    ap.add_argument("--sparse-vol-q", dest="sparse_vol_q", type=float, default=None)
    ap.add_argument("--sparse-require-high-vol", dest="sparse_require_high_vol", type=int, default=None)
    ap.add_argument("--sparse-high-logic", dest="sparse_high_logic", type=str, default=None)
    ap.add_argument("--q-entry-active-sparse", dest="q_entry_active_sparse", type=float, default=None)
    ap.add_argument("--q-entry-active-sparse-delta", dest="q_entry_active_sparse_delta", type=float, default=None)
    ap.add_argument("--entry-th-active-sparse", dest="entry_th_active_sparse", type=float, default=None)
    ap.add_argument("--entry-th-active-sparse-delta", dest="entry_th_active_sparse_delta", type=float, default=None)
    ap.add_argument("--vol-low-th-active-sparse", dest="vol_low_th_active_sparse", type=float, default=None)
    ap.add_argument("--vol-low-th-active-sparse-delta", dest="vol_low_th_active_sparse_delta", type=float, default=None)
    ap.add_argument("--atr-entry-mult-active-sparse", dest="atr_entry_mult_active_sparse", type=float, default=None)
    ap.add_argument("--atr-entry-mult-active-sparse-delta", dest="atr_entry_mult_active_sparse_delta", type=float, default=None)
    ap.add_argument("--range-entry-mult-active-sparse", dest="range_entry_mult_active_sparse", type=float, default=None)
    ap.add_argument("--range-entry-mult-active-sparse-delta", dest="range_entry_mult_active_sparse_delta", type=float, default=None)
    ap.add_argument("--regime-weight-enabled", dest="regime_weight_enabled", type=int, default=None)
    ap.add_argument("--regime-stress-lo", dest="regime_stress_lo", type=float, default=None)
    ap.add_argument("--regime-stress-hi", dest="regime_stress_hi", type=float, default=None)
    ap.add_argument("--regime-alpha-ema", dest="regime_alpha_ema", type=float, default=None)
    ap.add_argument("--regime-alpha-hysteresis", dest="regime_alpha_hysteresis", type=float, default=None)
    ap.add_argument("--gate-calm-mix", dest="gate_calm_mix", type=float, default=None)
    ap.add_argument("--gate-active-mix", dest="gate_active_mix", type=float, default=None)
    ap.add_argument("--dir-calm-mix", dest="dir_calm_mix", type=float, default=None)
    ap.add_argument("--dir-active-mix", dest="dir_active_mix", type=float, default=None)

    ap.add_argument("--models_dir", type=str, default="")
    ap.add_argument("--seq_len", type=int, default=300)
    ap.add_argument("--cache_npz", dest="cache_npz", type=str, default="")
    ap.add_argument("--hybrid-batch-size", dest="hybrid_batch_size", type=int, default=2048)
    ap.add_argument("--hybrid-amp", type=int, default=1)

    ap.add_argument("--trade-log-csv", dest="trade_log_csv", type=str, default="rolling_oos_tradelog_single_v50.csv")
    ap.add_argument("--exit-stats-csv", dest="exit_stats_csv", type=str, default="")
    ap.add_argument("--exit-stats-meta", dest="exit_stats_meta", type=str, default="")
    ap.add_argument("--single-stats-csv", dest="single_stats_csv", type=str, default="")
    ap.add_argument("--results-csv", dest="results_csv", type=str, default="")

    ap.add_argument("--runner-policy-align", dest="runner_policy_align", type=int, default=0)
    ap.add_argument("--profit-floor-enabled", dest="profit_floor_enabled", type=int, default=None)
    ap.add_argument("--thesis-monitor-enabled", dest="thesis_monitor_enabled", type=int, default=None)
    ap.add_argument("--emit-runner-kpi-meta", dest="emit_runner_kpi_meta", type=int, default=1)

    ap.add_argument("--entry-q-lookback", dest="entry_q_lookback", type=int, default=6000)
    ap.add_argument("--entry-q-min-ready", dest="entry_q_min_ready", type=int, default=300)
    ap.add_argument("--vol-feature", dest="vol_feature", type=str, default="")
    ap.add_argument("--atr-feature", dest="atr_feature", type=str, default="")

    # scoring / worst-case alignment with autotune
    ap.add_argument("--trade_mode", default="soft")
    ap.add_argument("--trade_target", type=float, default=300.0)
    ap.add_argument("--trade_band", type=float, default=150.0)
    ap.add_argument("--barrier_k", type=float, default=2.0)
    ap.add_argument("--trade_shortage_penalty", type=float, default=0.05)
    ap.add_argument("--trade_excess_penalty", type=float, default=0.01)
    ap.add_argument("--side_balance_penalty_k", type=float, default=0.0)
    ap.add_argument("--min_short_trades_global", type=int, default=0)
    ap.add_argument("--min_short_share_global", type=float, default=0.0)
    ap.add_argument("--alpha_dd", type=float, default=0.9)
    ap.add_argument("--beta_tail", type=float, default=2.0)
    ap.add_argument("--w-mean", dest="w_mean", type=float, default=None)
    ap.add_argument("--w-worst", dest="w_worst", type=float, default=None)
    ap.add_argument("--worst-agg", dest="worst_agg", type=str, default=None)
    ap.add_argument("--worst-k", dest="worst_k", type=int, default=None)
    ap.add_argument("--worst-q", dest="worst_q", type=float, default=None)
    ap.add_argument("--min-seg-trades", dest="min_seg_trades", type=int, default=None)
    ap.add_argument("--min-seg-trades-mode", dest="min_seg_trades_mode", type=str, default=None, choices=["hard", "soft"])
    ap.add_argument("--min-seg-trades-penalty-k", dest="min_seg_trades_penalty_k", type=float, default=None)
    ap.add_argument("--min-seg-trades-penalty-power", dest="min_seg_trades_penalty_power", type=float, default=None)
    ap.add_argument("--short-trades-guard-mode", dest="short_trades_guard_mode", type=str, default=None, choices=["hard", "soft"])
    ap.add_argument("--short-trades-penalty-k", dest="short_trades_penalty_k", type=float, default=None)
    ap.add_argument("--short-trades-penalty-power", dest="short_trades_penalty_power", type=float, default=None)
    ap.add_argument("--short-share-guard-mode", dest="short_share_guard_mode", type=str, default=None, choices=["hard", "soft"])
    ap.add_argument("--short-share-penalty-k", dest="short_share_penalty_k", type=float, default=None)
    ap.add_argument("--short-share-penalty-power", dest="short_share_penalty_power", type=float, default=None)
    ap.add_argument("--regime-extreme-max-frac", dest="regime_extreme_max_frac", type=float, default=None)
    ap.add_argument("--regime-extreme-penalty-k", dest="regime_extreme_penalty_k", type=float, default=None)
    ap.add_argument("--hard-guard-base", dest="hard_guard_base", type=float, default=None)
    ap.add_argument("--hard-guard-step", dest="hard_guard_step", type=float, default=None)
    ap.add_argument("--seg_bottom2_target", "--seg-bottom2-target", dest="seg_bottom2_target", type=float, default=None)
    ap.add_argument("--seg_bottom2_penalty_k", "--seg-bottom2-penalty-k", dest="seg_bottom2_penalty_k", type=float, default=None)
    ap.add_argument("--seg_floor_target", "--seg-floor-target", dest="seg_floor_target", type=float, default=None)
    ap.add_argument("--seg_floor_penalty_k", "--seg-floor-penalty-k", dest="seg_floor_penalty_k", type=float, default=None)
    ap.add_argument("--trade_cv_cap", "--trade-cv-cap", dest="trade_cv_cap", type=float, default=None)
    ap.add_argument("--trade_cv_penalty_k", "--trade-cv-penalty-k", dest="trade_cv_penalty_k", type=float, default=None)
    ap.add_argument("--maxhold_ratio_free", type=float, default=1.0)
    ap.add_argument("--maxhold_penalty_k", type=float, default=0.0)
    ap.add_argument("--maxhold_penalty_power", type=float, default=2.0)

    # compatibility aliases (ignored in single-tier runtime)
    ap.add_argument("--tier_logic", type=str, default="")
    ap.add_argument("--multi_pos", type=int, default=0)
    ap.add_argument("--pos_frac", type=str, default="")
    ap.add_argument("--entry-q-lookback-top", dest="entry_q_lookback_top", type=int, default=0)
    ap.add_argument("--allow-config-fallback", dest="allow_config_fallback", type=int, default=0)
    ap.add_argument("--seg-id", dest="seg_id", type=int, default=4)

    args = ap.parse_args()

    d_exit, d_meta, d_single, d_results = derive_outputs_from_trade_log(args.trade_log_csv)
    if not args.exit_stats_csv:
        args.exit_stats_csv = d_exit
    if not args.exit_stats_meta:
        args.exit_stats_meta = d_meta
    if not args.single_stats_csv:
        args.single_stats_csv = d_single
    if not args.results_csv:
        args.results_csv = d_results

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    cfg = normalize_single_config_from_any(cfg_raw)
    cfg_defaults = normalize_single_config_from_any({})
    allow_config_fallback = bool(int(getattr(args, "allow_config_fallback", 0) or 0))
    tuned_meta = dict(cfg.get("tuned_meta", {})) if allow_config_fallback else {}

    runner_align = dict(cfg.get("runner_alignment_cfg", {}) or {})
    runner_align["enabled"] = int(args.runner_policy_align or 0)
    if args.profit_floor_enabled is not None:
        runner_align["profit_floor_enabled"] = int(args.profit_floor_enabled)
    if args.thesis_monitor_enabled is not None:
        runner_align["thesis_monitor_enabled"] = int(args.thesis_monitor_enabled)
    cfg["runner_alignment_cfg"] = runner_align

    if not allow_config_fallback:
        def _fill_if_none(name: str, value: Any) -> None:
            if getattr(args, name) is None:
                setattr(args, name, value)

        rd_def = dict(cfg_defaults.get("regime_detect_cfg", {}))
        rw_def = dict(cfg_defaults.get("regime_weight_cfg", {}))
        rt_def = dict(cfg_defaults.get("regime_threshold_cfg", {}))
        rf_def = dict(cfg_defaults.get("regime_filter_cfg", {}))
        dyn_def = dict(cfg_defaults.get("dynamic_cfg", {}))
        tpw_def = dict(cfg_defaults.get("tp_window_cfg", {}))
        ep_def = dict(cfg_defaults.get("entry_episode_cfg", {}))

        _fill_if_none("leverage", float(cfg_defaults.get("leverage", 10.0)))
        _fill_if_none("fee_tp_mult", float(cfg_defaults.get("fee_tp_mult", 1.0)))
        _fill_if_none("fee_bep_mult", float(cfg_defaults.get("bep_arm_fee_mult", 0.2)))
        _fill_if_none("atr_entry_mult", float(cfg_defaults.get("atr_entry_mult", 1.0)))
        _fill_if_none("range_entry_mult", float(cfg_defaults.get("range_entry_mult", 1.0)))
        _fill_if_none("low_vol_filter", int(cfg_defaults.get("low_vol_filter", 0)))
        _fill_if_none("trail_after_bep", int(cfg_defaults.get("trail_after_bep", 1)))
        _fill_if_none("risk_entry_mode", int(cfg_defaults.get("risk_entry_mode", 0)))
        _fill_if_none("use_atr_scaling", int(cfg_defaults.get("use_atr_scaling", 1)))
        _fill_if_none("risk_lev_cap", float(cfg_defaults.get("risk_cfg", {}).get("risk_lev_cap", 12.0)))
        _fill_if_none("min_hold_tp_bars", int(cfg_defaults.get("min_hold_tp_bars", cfg_defaults.get("min_hold_bars", 6))))

        _fill_if_none("tp_window_enabled", int(tpw_def.get("enabled", 0)))
        _fill_if_none("tp_window_extend_bars", int(tpw_def.get("extend_bars", 0)))
        _fill_if_none("tp_window_block_early_trail", int(tpw_def.get("block_early_trail", 0)))
        _fill_if_none("tp_window_block_early_soft_sl", int(tpw_def.get("block_early_soft_sl", 0)))
        _fill_if_none("tp_window_floor_trail_hold_to_tp", int(tpw_def.get("floor_trail_hold_to_tp", 0)))
        _fill_if_none("tp_window_floor_soft_sl_hold_to_tp", int(tpw_def.get("floor_soft_sl_hold_to_tp", 0)))
        _fill_if_none("tp_window_suspend_post_bep_shield_before_tp", int(tpw_def.get("suspend_post_bep_shield_before_tp", 0)))
        _fill_if_none("tp_window_progress_frac_arm", float(tpw_def.get("progress_frac_arm", 0.7)))
        _fill_if_none("tp_window_expire_on_pullback_frac", float(tpw_def.get("expire_on_pullback_frac", 0.0)))

        _fill_if_none("allow_soft_sl_before_trail", int(dyn_def.get("allow_soft_sl_before_trail", 0)))
        _fill_if_none("softsl_hold_floor", int(dyn_def.get("softsl_hold_floor", 0)))
        _fill_if_none("post_bep_shield_ignore_softsl_hold", int(dyn_def.get("post_bep_shield_ignore_softsl_hold", 0)))
        _fill_if_none("pre_bep_force_close_red_only", int(dyn_def.get("pre_bep_force_close_red_only", 1)))

        _fill_if_none("entry_episode_enabled", int(ep_def.get("enabled", 0)))
        _fill_if_none("rearm_enabled", int(ep_def.get("rearm_enabled", 0)))
        _fill_if_none("run_gap_reset_bars", int(ep_def.get("run_gap_reset_bars", 0)))
        _fill_if_none("episode_max_entries_per_run", int(ep_def.get("episode_max_entries_per_run", 1)))
        _fill_if_none("rearm_same_side_only", int(ep_def.get("rearm_same_side_only", 1)))
        _fill_if_none("rearm_cooldown_bars", int(ep_def.get("rearm_cooldown_bars", 0)))
        _fill_if_none("rearm_max_bars_after_exit", int(ep_def.get("rearm_max_bars_after_exit", 0)))
        _fill_if_none("rearm_gate_reset_frac", float(ep_def.get("rearm_gate_reset_frac", 1.0)))
        _fill_if_none("rearm_gate_refresh_frac", float(ep_def.get("rearm_gate_refresh_frac", 1.0)))
        _fill_if_none("rearm_price_reset_frac", float(ep_def.get("rearm_price_reset_frac", 0.0)))
        _fill_if_none("rearm_after_trail", int(ep_def.get("rearm_after_trail", 1)))
        _fill_if_none("rearm_after_tp", int(ep_def.get("rearm_after_tp", 0)))
        _fill_if_none("rearm_after_sl", int(ep_def.get("rearm_after_sl", 0)))

        _fill_if_none("regime_detect_enabled", int(rd_def.get("enabled", 0)))
        _fill_if_none("regime_threshold_enabled", int(rt_def.get("enabled", 0)))
        _fill_if_none("regime_weight_enabled", int(rw_def.get("enabled", 0)))
        _fill_if_none("regime_filter_enabled", int(rf_def.get("enabled", 0)))
        _fill_if_none("regime_stress_lo", float(rd_def.get("stress_lo", 0.25)))
        _fill_if_none("regime_stress_hi", float(rd_def.get("stress_hi", 0.65)))
        _fill_if_none("regime_alpha_ema", float(rd_def.get("alpha_ema", 0.15)))
        _fill_if_none("regime_alpha_hysteresis", float(rd_def.get("alpha_hysteresis", 0.03)))
        _fill_if_none("gate_calm_mix", float(rw_def.get("gate_calm_mix", 0.60)))
        _fill_if_none("gate_active_mix", float(rw_def.get("gate_active_mix", 0.55)))
        _fill_if_none("dir_calm_mix", float(rw_def.get("dir_calm_mix", 0.35)))
        _fill_if_none("dir_active_mix", float(rw_def.get("dir_active_mix", 0.50)))
        _fill_if_none("q_entry_calm", float(rt_def.get("q_entry_calm", cfg_defaults.get("q_entry", 0.85))))
        _fill_if_none("q_entry_mid", float(rt_def.get("q_entry_mid", cfg_defaults.get("q_entry", 0.85))))
        _fill_if_none("q_entry_active", float(rt_def.get("q_entry_active", cfg_defaults.get("q_entry", 0.85))))
        _fill_if_none("entry_th_calm", float(rt_def.get("entry_th_floor_calm", cfg_defaults.get("entry_th_floor", cfg_defaults.get("entry_th", 0.0)))))
        _fill_if_none("entry_th_mid", float(rt_def.get("entry_th_floor_mid", cfg_defaults.get("entry_th_floor", cfg_defaults.get("entry_th", 0.0)))))
        _fill_if_none("entry_th_active", float(rt_def.get("entry_th_floor_active", cfg_defaults.get("entry_th_floor", cfg_defaults.get("entry_th", 0.0)))))
        _fill_if_none("bucket_min_ready", int(rt_def.get("bucket_min_ready", 0)))
        _fill_if_none("bucket_fallback_global", int(rt_def.get("bucket_fallback_global", 1)))

        _fill_if_none("regime_filter_use_vol_split", int(rf_def.get("use_vol_split", 1)))
        _fill_if_none("regime_filter_use_entry_mult_split", int(rf_def.get("use_entry_mult_split", 1)))
        _fill_if_none("regime_filter_mid_interp_mode", str(rf_def.get("mid_interp_mode", "linear")))
        _fill_if_none("vol_low_th_calm", float(rf_def.get("vol_low_th_calm", cfg_defaults.get("risk_cfg", {}).get("vol_low_th", -1e9))))
        _fill_if_none("vol_low_th_mid", float(rf_def.get("vol_low_th_mid", cfg_defaults.get("risk_cfg", {}).get("vol_low_th", -1e9))))
        _fill_if_none("vol_low_th_active", float(rf_def.get("vol_low_th_active", cfg_defaults.get("risk_cfg", {}).get("vol_low_th", -1e9))))
        _fill_if_none("atr_entry_mult_calm", float(rf_def.get("atr_entry_mult_calm", cfg_defaults.get("atr_entry_mult", 1.0))))
        _fill_if_none("atr_entry_mult_active", float(rf_def.get("atr_entry_mult_active", cfg_defaults.get("atr_entry_mult", 1.0))))
        _fill_if_none("range_entry_mult_calm", float(rf_def.get("range_entry_mult_calm", cfg_defaults.get("range_entry_mult", 1.0))))
        _fill_if_none("range_entry_mult_active", float(rf_def.get("range_entry_mult_active", cfg_defaults.get("range_entry_mult", 1.0))))

        lane_def = dict(cfg_defaults.get("regime_lane_cfg", {}))
        _fill_if_none("regime_lane_enabled", int(lane_def.get("enabled", 0)))
        _fill_if_none("active_sparse_enabled", int(lane_def.get("active_sparse_enabled", 0)))
        _fill_if_none("active_sparse_min_ready", int(lane_def.get("active_sparse_min_ready", 160)))
        _fill_if_none("sparse_gate_q", float(lane_def.get("sparse_gate_q", 0.55)))
        _fill_if_none("sparse_gate_floor_q", float(lane_def.get("sparse_gate_floor_q", 0.0)))
        _fill_if_none("sparse_atr_q", float(lane_def.get("sparse_atr_q", 0.65)))
        _fill_if_none("sparse_range_q", float(lane_def.get("sparse_range_q", 0.65)))
        _fill_if_none("sparse_vol_q", float(lane_def.get("sparse_vol_q", 0.0)))
        _fill_if_none("sparse_require_high_vol", int(lane_def.get("sparse_require_high_vol", 0)))
        _fill_if_none("sparse_high_logic", str(lane_def.get("sparse_high_logic", "or")))
        _fill_if_none("q_entry_active_sparse", float(rt_def.get("q_entry_active_sparse", rt_def.get("q_entry_active", cfg_defaults.get("q_entry", 0.85)))))
        _fill_if_none("q_entry_active_sparse_delta", float(rt_def.get("q_entry_active_sparse_delta", 0.0)))
        _fill_if_none("entry_th_active_sparse", float(rt_def.get("entry_th_floor_active_sparse", rt_def.get("entry_th_floor_active", cfg_defaults.get("entry_th_floor", cfg_defaults.get("entry_th", 0.0))))))
        _fill_if_none("entry_th_active_sparse_delta", float(rt_def.get("entry_th_floor_active_sparse_delta", 0.0)))
        _fill_if_none("vol_low_th_active_sparse", float(rf_def.get("vol_low_th_active_sparse", rf_def.get("vol_low_th_active", cfg_defaults.get("risk_cfg", {}).get("vol_low_th", -1e9)))))
        _fill_if_none("vol_low_th_active_sparse_delta", float(rf_def.get("vol_low_th_active_sparse_delta", 0.0)))
        _fill_if_none("atr_entry_mult_active_sparse", float(rf_def.get("atr_entry_mult_active_sparse", rf_def.get("atr_entry_mult_active", cfg_defaults.get("atr_entry_mult", 1.0)))))
        _fill_if_none("atr_entry_mult_active_sparse_delta", float(rf_def.get("atr_entry_mult_active_sparse_delta", 0.0)))
        _fill_if_none("range_entry_mult_active_sparse", float(rf_def.get("range_entry_mult_active_sparse", rf_def.get("range_entry_mult_active", cfg_defaults.get("range_entry_mult", 1.0)))))
        _fill_if_none("range_entry_mult_active_sparse_delta", float(rf_def.get("range_entry_mult_active_sparse_delta", 0.0)))

        if not str(args.vol_feature).strip():
            args.vol_feature = str(cfg_defaults.get("runtime_feature_cfg", {}).get("vol_feature", "vol_z_60") or "vol_z_60")
        if not str(args.atr_feature).strip():
            args.atr_feature = str(cfg_defaults.get("runtime_feature_cfg", {}).get("atr_feature", "atr10_rel") or "atr10_rel")

    # CLI overrides
    if args.leverage is not None:
        cfg["leverage"] = float(args.leverage)
    if args.fee_tp_mult is not None:
        cfg["fee_tp_mult"] = float(args.fee_tp_mult)
    if args.fee_bep_mult is not None:
        cfg["bep_arm_fee_mult"] = float(args.fee_bep_mult)
    if args.atr_entry_mult is not None:
        cfg["atr_entry_mult"] = float(args.atr_entry_mult)
    if args.range_entry_mult is not None:
        cfg["range_entry_mult"] = float(args.range_entry_mult)
    if args.low_vol_filter is not None:
        cfg["low_vol_filter"] = int(args.low_vol_filter)
    if args.trail_after_bep is not None:
        cfg["trail_after_bep"] = int(args.trail_after_bep)
    if args.risk_entry_mode is not None:
        cfg["risk_entry_mode"] = int(args.risk_entry_mode)
    if args.use_atr_scaling is not None:
        cfg["use_atr_scaling"] = 1 if int(args.use_atr_scaling) != 0 else 0
    if args.risk_lev_cap is not None:
        cfg["risk_cfg"]["risk_lev_cap"] = float(args.risk_lev_cap)
    if args.min_hold_tp_bars is not None:
        cfg["min_hold_tp_bars"] = int(args.min_hold_tp_bars)
        cfg["min_hold_bars"] = int(args.min_hold_tp_bars)
    tpw = cfg.setdefault("tp_window_cfg", {})
    if args.tp_window_enabled is not None:
        tpw["enabled"] = int(args.tp_window_enabled)
    if args.tp_window_extend_bars is not None:
        tpw["extend_bars"] = int(args.tp_window_extend_bars)
    if args.tp_window_block_early_trail is not None:
        tpw["block_early_trail"] = int(args.tp_window_block_early_trail)
    if args.tp_window_block_early_soft_sl is not None:
        tpw["block_early_soft_sl"] = int(args.tp_window_block_early_soft_sl)
    if args.tp_window_floor_trail_hold_to_tp is not None:
        tpw["floor_trail_hold_to_tp"] = int(args.tp_window_floor_trail_hold_to_tp)
    if args.tp_window_floor_soft_sl_hold_to_tp is not None:
        tpw["floor_soft_sl_hold_to_tp"] = int(args.tp_window_floor_soft_sl_hold_to_tp)
    if args.tp_window_suspend_post_bep_shield_before_tp is not None:
        tpw["suspend_post_bep_shield_before_tp"] = int(args.tp_window_suspend_post_bep_shield_before_tp)
    if args.tp_window_progress_frac_arm is not None:
        tpw["progress_frac_arm"] = float(args.tp_window_progress_frac_arm)
    if args.tp_window_expire_on_pullback_frac is not None:
        tpw["expire_on_pullback_frac"] = float(args.tp_window_expire_on_pullback_frac)
    dyn = cfg.setdefault("dynamic_cfg", {})
    if args.allow_soft_sl_before_trail is not None:
        dyn["allow_soft_sl_before_trail"] = int(args.allow_soft_sl_before_trail)
    if args.softsl_hold_floor is not None:
        dyn["softsl_hold_floor"] = int(args.softsl_hold_floor)
    if args.post_bep_shield_ignore_softsl_hold is not None:
        dyn["post_bep_shield_ignore_softsl_hold"] = int(args.post_bep_shield_ignore_softsl_hold)
    if args.pre_bep_force_close_red_only is not None:
        dyn["pre_bep_force_close_red_only"] = int(args.pre_bep_force_close_red_only)
    epcfg = cfg.setdefault("entry_episode_cfg", {})
    if args.entry_episode_enabled is not None:
        epcfg["enabled"] = int(args.entry_episode_enabled)
    if args.rearm_enabled is not None:
        epcfg["rearm_enabled"] = int(args.rearm_enabled)
    if args.run_gap_reset_bars is not None:
        epcfg["run_gap_reset_bars"] = int(args.run_gap_reset_bars)
    if args.episode_max_entries_per_run is not None:
        epcfg["episode_max_entries_per_run"] = int(args.episode_max_entries_per_run)
    if args.rearm_same_side_only is not None:
        epcfg["rearm_same_side_only"] = int(args.rearm_same_side_only)
    if args.rearm_cooldown_bars is not None:
        epcfg["rearm_cooldown_bars"] = int(args.rearm_cooldown_bars)
    if args.rearm_max_bars_after_exit is not None:
        epcfg["rearm_max_bars_after_exit"] = int(args.rearm_max_bars_after_exit)
    if args.rearm_gate_reset_frac is not None:
        epcfg["rearm_gate_reset_frac"] = float(args.rearm_gate_reset_frac)
    if args.rearm_gate_refresh_frac is not None:
        epcfg["rearm_gate_refresh_frac"] = float(args.rearm_gate_refresh_frac)
    if args.rearm_price_reset_frac is not None:
        epcfg["rearm_price_reset_frac"] = float(args.rearm_price_reset_frac)
    if args.rearm_after_trail is not None:
        epcfg["rearm_after_trail"] = int(args.rearm_after_trail)
    if args.rearm_after_tp is not None:
        epcfg["rearm_after_tp"] = int(args.rearm_after_tp)
    if args.rearm_after_sl is not None:
        epcfg["rearm_after_sl"] = int(args.rearm_after_sl)
    regime_detect_cfg = cfg.setdefault("regime_detect_cfg", {})
    regime_weight_cfg = cfg.setdefault("regime_weight_cfg", {})
    regime_threshold_cfg = cfg.setdefault("regime_threshold_cfg", {})
    regime_filter_cfg = cfg.setdefault("regime_filter_cfg", {})
    regime_lane_cfg = cfg.setdefault("regime_lane_cfg", {})
    if args.regime_detect_enabled is not None:
        regime_detect_cfg["enabled"] = int(args.regime_detect_enabled)
    if args.regime_threshold_enabled is not None:
        regime_threshold_cfg["enabled"] = int(args.regime_threshold_enabled)
    if args.regime_weight_enabled is not None:
        regime_weight_cfg["enabled"] = int(args.regime_weight_enabled)
    if args.regime_filter_enabled is not None:
        regime_filter_cfg["enabled"] = int(args.regime_filter_enabled)
    if args.regime_stress_lo is not None:
        regime_detect_cfg["stress_lo"] = float(args.regime_stress_lo)
    if args.regime_stress_hi is not None:
        regime_detect_cfg["stress_hi"] = float(args.regime_stress_hi)
    if args.regime_alpha_ema is not None:
        regime_detect_cfg["alpha_ema"] = float(args.regime_alpha_ema)
    if args.regime_alpha_hysteresis is not None:
        regime_detect_cfg["alpha_hysteresis"] = float(args.regime_alpha_hysteresis)
    if args.gate_calm_mix is not None:
        regime_weight_cfg["gate_calm_mix"] = float(args.gate_calm_mix)
    if args.gate_active_mix is not None:
        regime_weight_cfg["gate_active_mix"] = float(args.gate_active_mix)
    if args.dir_calm_mix is not None:
        regime_weight_cfg["dir_calm_mix"] = float(args.dir_calm_mix)
    if args.dir_active_mix is not None:
        regime_weight_cfg["dir_active_mix"] = float(args.dir_active_mix)
    if args.q_entry_calm is not None:
        regime_threshold_cfg["q_entry_calm"] = float(args.q_entry_calm)
    if args.q_entry_mid is not None:
        regime_threshold_cfg["q_entry_mid"] = float(args.q_entry_mid)
    if args.q_entry_active is not None:
        regime_threshold_cfg["q_entry_active"] = float(args.q_entry_active)
    if args.entry_th_calm is not None:
        regime_threshold_cfg["entry_th_floor_calm"] = float(args.entry_th_calm)
    if args.entry_th_mid is not None:
        regime_threshold_cfg["entry_th_floor_mid"] = float(args.entry_th_mid)
    if args.entry_th_active is not None:
        regime_threshold_cfg["entry_th_floor_active"] = float(args.entry_th_active)
    if args.bucket_min_ready is not None:
        regime_threshold_cfg["bucket_min_ready"] = int(args.bucket_min_ready)
    if args.bucket_fallback_global is not None:
        regime_threshold_cfg["bucket_fallback_global"] = int(args.bucket_fallback_global)
    if args.regime_filter_use_vol_split is not None:
        regime_filter_cfg["use_vol_split"] = int(args.regime_filter_use_vol_split)
    if args.regime_filter_use_entry_mult_split is not None:
        regime_filter_cfg["use_entry_mult_split"] = int(args.regime_filter_use_entry_mult_split)
    if args.regime_filter_mid_interp_mode is not None:
        regime_filter_cfg["mid_interp_mode"] = str(args.regime_filter_mid_interp_mode)
    if args.vol_low_th_calm is not None:
        regime_filter_cfg["vol_low_th_calm"] = float(args.vol_low_th_calm)
    if args.vol_low_th_mid is not None:
        regime_filter_cfg["vol_low_th_mid"] = float(args.vol_low_th_mid)
    if args.vol_low_th_active is not None:
        regime_filter_cfg["vol_low_th_active"] = float(args.vol_low_th_active)
    if args.atr_entry_mult_calm is not None:
        regime_filter_cfg["atr_entry_mult_calm"] = float(args.atr_entry_mult_calm)
    if args.atr_entry_mult_active is not None:
        regime_filter_cfg["atr_entry_mult_active"] = float(args.atr_entry_mult_active)
    if args.range_entry_mult_calm is not None:
        regime_filter_cfg["range_entry_mult_calm"] = float(args.range_entry_mult_calm)
    if args.range_entry_mult_active is not None:
        regime_filter_cfg["range_entry_mult_active"] = float(args.range_entry_mult_active)
    if args.regime_lane_enabled is not None:
        regime_lane_cfg["enabled"] = int(args.regime_lane_enabled)
    if args.active_sparse_enabled is not None:
        regime_lane_cfg["active_sparse_enabled"] = int(args.active_sparse_enabled)
    if args.active_sparse_min_ready is not None:
        regime_lane_cfg["active_sparse_min_ready"] = int(args.active_sparse_min_ready)
    if args.sparse_gate_q is not None:
        regime_lane_cfg["sparse_gate_q"] = float(args.sparse_gate_q)
    if args.sparse_gate_floor_q is not None:
        regime_lane_cfg["sparse_gate_floor_q"] = float(args.sparse_gate_floor_q)
    if args.sparse_atr_q is not None:
        regime_lane_cfg["sparse_atr_q"] = float(args.sparse_atr_q)
    if args.sparse_range_q is not None:
        regime_lane_cfg["sparse_range_q"] = float(args.sparse_range_q)
    if args.sparse_vol_q is not None:
        regime_lane_cfg["sparse_vol_q"] = float(args.sparse_vol_q)
    if args.sparse_require_high_vol is not None:
        regime_lane_cfg["sparse_require_high_vol"] = int(args.sparse_require_high_vol)
    if args.sparse_high_logic is not None:
        regime_lane_cfg["sparse_high_logic"] = str(args.sparse_high_logic).strip().lower()
    if args.q_entry_active_sparse is not None:
        regime_threshold_cfg["q_entry_active_sparse"] = float(args.q_entry_active_sparse)
    if args.q_entry_active_sparse_delta is not None:
        regime_threshold_cfg["q_entry_active_sparse_delta"] = float(args.q_entry_active_sparse_delta)
    if args.entry_th_active_sparse is not None:
        regime_threshold_cfg["entry_th_floor_active_sparse"] = float(args.entry_th_active_sparse)
    if args.entry_th_active_sparse_delta is not None:
        regime_threshold_cfg["entry_th_floor_active_sparse_delta"] = float(args.entry_th_active_sparse_delta)
    if args.vol_low_th_active_sparse is not None:
        regime_filter_cfg["vol_low_th_active_sparse"] = float(args.vol_low_th_active_sparse)
    if args.vol_low_th_active_sparse_delta is not None:
        regime_filter_cfg["vol_low_th_active_sparse_delta"] = float(args.vol_low_th_active_sparse_delta)
    if args.atr_entry_mult_active_sparse is not None:
        regime_filter_cfg["atr_entry_mult_active_sparse"] = float(args.atr_entry_mult_active_sparse)
    if args.atr_entry_mult_active_sparse_delta is not None:
        regime_filter_cfg["atr_entry_mult_active_sparse_delta"] = float(args.atr_entry_mult_active_sparse_delta)
    if args.range_entry_mult_active_sparse is not None:
        regime_filter_cfg["range_entry_mult_active_sparse"] = float(args.range_entry_mult_active_sparse)
    if args.range_entry_mult_active_sparse_delta is not None:
        regime_filter_cfg["range_entry_mult_active_sparse_delta"] = float(args.range_entry_mult_active_sparse_delta)
    if str(args.vol_feature).strip():
        cfg.setdefault("runtime_feature_cfg", {})["vol_feature"] = str(args.vol_feature).strip()
    if str(args.atr_feature).strip():
        cfg.setdefault("runtime_feature_cfg", {})["atr_feature"] = str(args.atr_feature).strip()

    runtime_feature_cfg = dict(cfg.get("runtime_feature_cfg", {}))
    vol_feature = str(args.vol_feature).strip() if str(args.vol_feature).strip() else str(runtime_feature_cfg.get("vol_feature", "vol_z_60") or "vol_z_60")
    atr_feature = str(args.atr_feature).strip() if str(args.atr_feature).strip() else str(runtime_feature_cfg.get("atr_feature", "atr10_rel") or "atr10_rel")
    cfg.setdefault("runtime_feature_cfg", {})["vol_feature"] = str(vol_feature)
    cfg.setdefault("runtime_feature_cfg", {})["atr_feature"] = str(atr_feature)

    _soft_raw_allow = int(cfg.get("dynamic_cfg", {}).get("allow_soft_sl_before_trail", 0))
    _soft_raw_trail = int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))))
    _soft_raw_hold = int(cfg.get("min_hold_soft_sl_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))))
    cfg = normalize_single_config_from_any(cfg)
    runtime_feature_cfg = dict(cfg.get("runtime_feature_cfg", {}))
    vol_feature = str(runtime_feature_cfg.get("vol_feature", vol_feature) or vol_feature)
    atr_feature = str(runtime_feature_cfg.get("atr_feature", atr_feature) or atr_feature)
    cfg.setdefault("runtime_feature_cfg", {})["vol_feature"] = str(vol_feature)
    cfg.setdefault("runtime_feature_cfg", {})["atr_feature"] = str(atr_feature)
    softsl_guard_clamped = int(_soft_raw_allow == 0 and _soft_raw_hold < int(cfg.get("min_hold_soft_sl_bars", _soft_raw_hold)))
    softsl_guard_effective_soft = int(cfg.get("min_hold_soft_sl_bars", _soft_raw_hold))
    softsl_guard_effective_trail = int(cfg.get("min_hold_trail_bars", _soft_raw_trail))

    if allow_config_fallback:
        cfg_cost = safe_float(cfg_raw.get("cost_per_side", cfg_raw.get("taker_fee_per_side", 0.0005)), 0.0005)
        cfg_slip = safe_float(cfg_raw.get("slip_per_side", 0.0), 0.0)
        cfg_maker = safe_float(cfg_raw.get("maker_fee_per_side", cfg_cost), cfg_cost)
    else:
        cfg_cost = 0.00070
        cfg_slip = 0.00015
        cfg_maker = 0.00020

    args.cost_per_side = float(args.cost_per_side) if args.cost_per_side is not None else float(cfg_cost)
    args.slip_per_side = float(args.slip_per_side) if args.slip_per_side is not None else float(cfg_slip)
    args.maker_fee_per_side = float(args.maker_fee_per_side) if args.maker_fee_per_side is not None else float(cfg_maker)

    w_mean = float(args.w_mean) if args.w_mean is not None else float(tuned_meta.get("w_mean", 0.18))
    w_worst = float(args.w_worst) if args.w_worst is not None else float(tuned_meta.get("w_worst", 0.82))
    worst_agg = str(args.worst_agg) if args.worst_agg is not None else str(tuned_meta.get("worst_agg", "cvar"))
    worst_k = int(args.worst_k) if args.worst_k is not None else int(tuned_meta.get("worst_k", 4))
    worst_q = float(args.worst_q) if args.worst_q is not None else float(tuned_meta.get("worst_q", 0.2))
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
        "min_short_share_global": float(args.min_short_share_global),
        "maxhold_ratio_free": float(args.maxhold_ratio_free),
        "maxhold_penalty_k": float(args.maxhold_penalty_k),
        "maxhold_penalty_power": float(args.maxhold_penalty_power),
    }
    objective_cfg: Dict[str, Any] = {
        "min_seg_trades": int(args.min_seg_trades) if args.min_seg_trades is not None else int(tuned_meta.get("min_seg_trades", 0)),
        "min_seg_trades_mode": str(args.min_seg_trades_mode) if args.min_seg_trades_mode is not None else str(tuned_meta.get("min_seg_trades_mode", "soft")),
        "min_seg_trades_penalty_k": float(args.min_seg_trades_penalty_k) if args.min_seg_trades_penalty_k is not None else float(tuned_meta.get("min_seg_trades_penalty_k", 1.0)),
        "min_seg_trades_penalty_power": float(args.min_seg_trades_penalty_power) if args.min_seg_trades_penalty_power is not None else float(tuned_meta.get("min_seg_trades_penalty_power", 1.0)),
        "min_short_trades_global": int(args.min_short_trades_global),
        "short_trades_guard_mode": str(args.short_trades_guard_mode) if args.short_trades_guard_mode is not None else str(tuned_meta.get("short_trades_guard_mode", "hard")),
        "short_trades_penalty_k": float(args.short_trades_penalty_k) if args.short_trades_penalty_k is not None else float(tuned_meta.get("short_trades_penalty_k", 1.0)),
        "short_trades_penalty_power": float(args.short_trades_penalty_power) if args.short_trades_penalty_power is not None else float(tuned_meta.get("short_trades_penalty_power", 1.0)),
        "min_short_share_global": float(args.min_short_share_global),
        "short_share_guard_mode": str(args.short_share_guard_mode) if args.short_share_guard_mode is not None else str(tuned_meta.get("short_share_guard_mode", "hard")),
        "short_share_penalty_k": float(args.short_share_penalty_k) if args.short_share_penalty_k is not None else float(tuned_meta.get("short_share_penalty_k", 1000.0)),
        "short_share_penalty_power": float(args.short_share_penalty_power) if args.short_share_penalty_power is not None else float(tuned_meta.get("short_share_penalty_power", 1.0)),
        "regime_extreme_max_frac": float(args.regime_extreme_max_frac) if args.regime_extreme_max_frac is not None else float(tuned_meta.get("regime_extreme_max_frac", 1.0)),
        "regime_extreme_penalty_k": float(args.regime_extreme_penalty_k) if args.regime_extreme_penalty_k is not None else float(tuned_meta.get("regime_extreme_penalty_k", 0.0)),
        "hard_guard_base": float(args.hard_guard_base) if args.hard_guard_base is not None else float(tuned_meta.get("hard_guard_base", 1000000.0)),
        "hard_guard_step": float(args.hard_guard_step) if args.hard_guard_step is not None else float(tuned_meta.get("hard_guard_step", 1.0)),
        "seg_bottom2_target": float(args.seg_bottom2_target) if args.seg_bottom2_target is not None else float(tuned_meta.get("seg_bottom2_target", 0.0)),
        "seg_bottom2_penalty_k": float(args.seg_bottom2_penalty_k) if args.seg_bottom2_penalty_k is not None else float(tuned_meta.get("seg_bottom2_penalty_k", 0.0)),
        "seg_floor_target": float(args.seg_floor_target) if args.seg_floor_target is not None else float(tuned_meta.get("seg_floor_target", 0.0)),
        "seg_floor_penalty_k": float(args.seg_floor_penalty_k) if args.seg_floor_penalty_k is not None else float(tuned_meta.get("seg_floor_penalty_k", 0.0)),
        "trade_cv_cap": float(args.trade_cv_cap) if args.trade_cv_cap is not None else float(tuned_meta.get("trade_cv_cap", 0.0)),
        "trade_cv_penalty_k": float(args.trade_cv_penalty_k) if args.trade_cv_penalty_k is not None else float(tuned_meta.get("trade_cv_penalty_k", 0.0)),
    }

    objective_version = "coverage_v40+softsl_guard_fix"
    if int(cfg.get("regime_lane_cfg", {}).get("enabled", 0)) != 0:
        objective_version = "coverage_v40+regime_lane_v57+softsl_guard_fix"

    print("[START] RL_MODE Backtest single-tier v58")
    print(f"[CONFIG FALLBACK] enabled={int(allow_config_fallback)} policy={'config+tuned_meta' if allow_config_fallback else 'cli/default-only-for-exposed-fields'}")
    print(f"[COSTS] cost_per_side={args.cost_per_side} slip_per_side={args.slip_per_side} maker_fee_per_side={args.maker_fee_per_side} risk_lev_cap={cfg['risk_cfg']['risk_lev_cap']}")

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
    if args.rows and args.rows > 0:
        df = df.iloc[-args.rows:].reset_index(drop=True)

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
    print(f"[INFO] need={need} (window={window_oos} + hist_extra={hist_extra}) rows={len(df_window)} oos_len={args.oos_len} splits={args.splits} window_includes_hist_extra={int(args.window_includes_hist_extra)}")
    print(f"[RUNTIME FEATURES] vol_feature={vol_feature} atr_feature={atr_feature}")
    try:
        rd = dict(cfg.get("regime_detect_cfg", {}))
        rw = dict(cfg.get("regime_weight_cfg", {}))
        rt = dict(cfg.get("regime_threshold_cfg", {}))
        rf = dict(cfg.get("regime_filter_cfg", {}))
        print(f"[REGIME DETECT] enabled={int(rd.get('enabled', 0))} stress_lo={float(rd.get('stress_lo', 0.25)):.3f} stress_hi={float(rd.get('stress_hi', 0.65)):.3f} alpha_ema={float(rd.get('alpha_ema', 0.15)):.3f} alpha_hysteresis={float(rd.get('alpha_hysteresis', 0.03)):.3f}")
        print(f"[REGIME WEIGHTS] enabled={int(rw.get('enabled', 0))} gate_calm_mix={float(rw.get('gate_calm_mix', 0.60)):.3f} gate_active_mix={float(rw.get('gate_active_mix', 0.55)):.3f} dir_calm_mix={float(rw.get('dir_calm_mix', 0.35)):.3f} dir_active_mix={float(rw.get('dir_active_mix', 0.50)):.3f}")
        print(f"[REGIME THRESHOLDS] enabled={int(rt.get('enabled', 0))} q_calm={float(rt.get('q_entry_calm', cfg.get('q_entry', 0.85))):.4f} q_mid={float(rt.get('q_entry_mid', cfg.get('q_entry', 0.85))):.4f} q_active={float(rt.get('q_entry_active', cfg.get('q_entry', 0.85))):.4f} floor_calm={float(rt.get('entry_th_floor_calm', cfg.get('entry_th_floor', cfg.get('entry_th', 0.0)))):.6g} floor_mid={float(rt.get('entry_th_floor_mid', cfg.get('entry_th_floor', cfg.get('entry_th', 0.0)))):.6g} floor_active={float(rt.get('entry_th_floor_active', cfg.get('entry_th_floor', cfg.get('entry_th', 0.0)))):.6g} bucket_min_ready={int(rt.get('bucket_min_ready', 0))} fallback_global={int(rt.get('bucket_fallback_global', 1))}")
        print(f"[REGIME FILTERS] enabled={int(rf.get('enabled', 0))} use_vol_split={int(rf.get('use_vol_split', 1))} use_entry_mult_split={int(rf.get('use_entry_mult_split', 1))} mid_mode={str(rf.get('mid_interp_mode', 'linear'))} vol_calm={float(rf.get('vol_low_th_calm', cfg.get('risk_cfg', {}).get('vol_low_th', -1e9))):.6g} vol_mid={float(rf.get('vol_low_th_mid', cfg.get('risk_cfg', {}).get('vol_low_th', -1e9))):.6g} vol_active={float(rf.get('vol_low_th_active', cfg.get('risk_cfg', {}).get('vol_low_th', -1e9))):.6g} atr_calm={float(rf.get('atr_entry_mult_calm', cfg.get('atr_entry_mult', 1.0))):.6g} atr_active={float(rf.get('atr_entry_mult_active', cfg.get('atr_entry_mult', 1.0))):.6g} range_calm={float(rf.get('range_entry_mult_calm', cfg.get('range_entry_mult', 1.0))):.6g} range_active={float(rf.get('range_entry_mult_active', cfg.get('range_entry_mult', 1.0))):.6g}")
        rl = dict(cfg.get('regime_lane_cfg', {}))
        print(f"[REGIME LANE] enabled={int(rl.get('enabled', 0))} active_sparse={int(rl.get('active_sparse_enabled', 0))} min_ready={int(rl.get('active_sparse_min_ready', 160))} gate_floor_q={float(rl.get('sparse_gate_floor_q', 0.0)):.3f} gate_q={float(rl.get('sparse_gate_q', 0.55)):.3f} atr_q={float(rl.get('sparse_atr_q', 0.65)):.3f} range_q={float(rl.get('sparse_range_q', 0.65)):.3f} vol_q={float(rl.get('sparse_vol_q', 0.0)):.3f} require_high_vol={int(rl.get('sparse_require_high_vol', 0))} logic={str(rl.get('sparse_high_logic', 'or'))}")
        dynp = dict(cfg.get('dynamic_cfg', {}))
        print(f"[EXIT GUARD] allow_soft_sl_before_trail={int(dynp.get('allow_soft_sl_before_trail', 0))} softsl_hold_floor={int(dynp.get('softsl_hold_floor', 0))} min_hold_tp={int(cfg.get('min_hold_tp_bars', cfg.get('min_hold_bars', 0)))} min_hold_trail={int(cfg.get('min_hold_trail_bars', cfg.get('min_hold_tp_bars', cfg.get('min_hold_bars', 0))))} min_hold_soft_sl={int(cfg.get('min_hold_soft_sl_bars', cfg.get('min_hold_tp_bars', cfg.get('min_hold_bars', 0))))} guard_clamped={int(softsl_guard_clamped)}")
    except Exception:
        pass

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

    want_trade_logs = bool(str(args.trade_log_csv).strip())

    # Warm numba fast-core once so first segment does not pay compilation cost.
    warmup_single_fast_core()

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
        cfg=cfg,
    )

    def _run(seg_start: int, seg_end: int) -> Dict[str, Any]:
        prepared = prepare_single_segment_inputs_from_context(
            ctx=trial_ctx,
            seg_start=seg_start,
            seg_end=seg_end,
            entry_q_lookback=int(args.entry_q_lookback),
            entry_q_min_ready=int(args.entry_q_min_ready),
        )
        return evaluate_prepared_single_segment_hybrid(
            prepared=prepared,
            score_cfg=score_cfg,
            cost_per_side=float(args.cost_per_side),
            slip_per_side=float(args.slip_per_side),
            maker_fee_per_side=float(args.maker_fee_per_side),
            want_trade_logs=want_trade_logs,
        )


    if args.mode == "single":
        seg_start = int(window_start)
        seg_end = int(oos_start)
        if seg_end <= seg_start:
            raise ValueError("INS segment too small")
        res = _run(seg_start, seg_end)
        print("===== SINGLE (INS) Result =====")
        print(f"rows={seg_end-seg_start} net_ret={res['net_ret']:.4f} mdd={res['mdd_net']:.4f} trades={res['trades']} winrate={res['winrate_net']:.3f} tail={res['tail']}")
        return

    if args.mode == "oos":
        seg_start = int(oos_start)
        seg_end = int(oos_end)
        res = _run(seg_start, seg_end)
        print("===== OOS (FULL) Result =====")
        print(f"rows={seg_end-seg_start} net_ret={res['net_ret']:.4f} mdd={res['mdd_net']:.4f} trades={res['trades']} winrate={res['winrate_net']:.3f} tail={res['tail']}")
        return

    if args.mode == "oos_seg":
        seg_id = int(getattr(args, "seg_id", 4))
        if seg_id < 1 or seg_id > int(args.splits):
            seg_id = 4
        k = seg_id - 1
        seg_start = int(oos_start + k * seg_len)
        seg_end = int(seg_start + seg_len)
        res = _run(seg_start, seg_end)
        print("===== OOS SEGMENT Result =====")
        print(f"Segment: {seg_id}/{args.splits} rows={seg_end-seg_start}")
        print(f"Net Return : {res['net_ret']:.4f}")
        print(f"Net MDD    : {res['mdd_net']:.4f}")
        print(f"Net Winrate: {res['winrate_net']:.4f}")
        print(f"Trades     : {res['trades']}")
        print(f"Tail hit   : {res['tail']}")
        return

    rows: List[Dict[str, Any]] = []
    trade_rows_all: List[Dict[str, Any]] = []

    exit_cnt_tot = np.zeros(6, dtype=np.int64)
    exit_gross_tot = np.zeros(6, dtype=np.float64)
    exit_fee_tot = np.zeros(6, dtype=np.float64)
    exit_net_tot = np.zeros(6, dtype=np.float64)
    trail_before_tot = 0
    trail_after_tot = 0
    bep_armed_tot = 0
    ref_updates_tot = 0
    maxh_cnt_tot = 0
    single_rows: List[Dict[str, Any]] = []
    long_tot = 0
    short_tot = 0

    for k in range(int(args.splits)):
        seg_id = k + 1
        seg_start = int(oos_start + k * seg_len)
        seg_end = int(seg_start + seg_len)
        res = _run(seg_start, seg_end)

        exit_cnt_tot += res["exit_cnt"]
        exit_gross_tot += res["exit_gross_sum"]
        exit_fee_tot += res["exit_fee_sum"]
        exit_net_tot += res["exit_net_sum"]
        trail_before_tot += int(res["trail_before_bep"])
        trail_after_tot += int(res["trail_after_bep"])
        bep_armed_tot += int(res["bep_armed_trades"])
        ref_updates_tot += int(res["ref_updates"])
        maxh_cnt_tot += int(res["maxh_cnt"])
        long_tot += int(res["long_trades"])
        short_tot += int(res["short_trades"])

        if res.get("trade_logs"):
            for tr in res["trade_logs"]:
                tr["seg"] = int(seg_id)
                tr["thr_entry"] = float(res["thr_entry"])
                tr["score"] = float(res["score"])
                trade_rows_all.append(tr)

        row_seg = {
            "seg": int(seg_id),
            "rows": int(seg_end - seg_start),
            "net_ret": float(res["net_ret"]),
            "mdd_net": float(res["mdd_net"]),
            "winrate_net": float(res["winrate_net"]),
            "trades": int(res["trades"]),
            "long_trades": int(res["long_trades"]),
            "short_trades": int(res["short_trades"]),
            "short_share": float(res["short_share"]),
            "tail": int(res["tail"]),
            "score": float(res.get("score", float("nan"))),
            "regime_calm_frac": float(res.get("regime_calm_frac", 0.0)),
            "regime_active_frac": float(res.get("regime_active_frac", 0.0)),
            "tp_window_armed_trades": int(res.get("tp_window_armed_trades", 0)),
            "rearm_entries": int(res.get("rearm_entries", 0)),
        }
        for _k, _v in res.items():
            if isinstance(_k, str) and (_k.startswith("cand_") or _k.startswith("pass_") or _k.startswith("final_candidates") or _k.startswith("final_entries")):
                row_seg[_k] = _v
        rows.append(row_seg)
        single_rows.append({
            "seg": int(seg_id),
            "trades": int(res["trades"]),
            "long_trades": int(res["long_trades"]),
            "short_trades": int(res["short_trades"]),
            "short_share": float(res["short_share"]),
            "net": float(res["net_ret"]),
            "score": float(res["score"]),
            "maxh_cnt": int(res["maxh_cnt"]),
            "maxh_ratio": float(res["maxh_ratio"]),
            "thr_entry": float(res["thr_entry"]),
            "atr_high_th": float(res["atr_high_th"]),
            "side_penalty": float(res["side_penalty"]),
        })

    out_df = pd.DataFrame(rows)
    print("\n===== Rolling OOS Summary =====")
    print(out_df.to_string(index=False))

    print("\n[SUMMARY STATS]")
    print(f"segments evaluated : {len(out_df)}")
    print(f"net_ret mean       : {out_df['net_ret'].mean():.4f}")
    print(f"net_ret min/max    : {out_df['net_ret'].min():.4f} / {out_df['net_ret'].max():.4f}")
    print(f"mdd_net mean       : {out_df['mdd_net'].mean():.4f}")
    print(f"mdd_net max        : {out_df['mdd_net'].max():.4f}")
    print(f"trades mean        : {out_df['trades'].mean():.2f}")
    print(f"tail hits          : {int(out_df['tail'].sum())}")
    total_side_trades = int(long_tot + short_tot)
    short_share_tot = float(short_tot / total_side_trades) if total_side_trades > 0 else 0.0
    print(f"long/short trades  : {int(long_tot)} / {int(short_tot)} (short_share={short_share_tot:.3f})")

    score_raw_overall = float(w_mean) * float(out_df["score"].mean()) + float(w_worst) * float(agg_worst(out_df["score"].tolist(), worst_agg, int(worst_k), float(worst_q))) if not out_df.empty else float("-inf")
    breakdown = assemble_objective(
        score_raw_overall,
        score_cost_mean=score_raw_overall,
        score_cost_worst=score_raw_overall,
        min_seg_seen=int(out_df["trades"].min()) if not out_df.empty else 0,
        total_short=int(short_tot),
        short_share_all=float(short_share_tot),
        regime_calm_frac=float(out_df["regime_calm_frac"].mean()) if "regime_calm_frac" in out_df.columns and not out_df.empty else None,
        regime_active_frac=float(out_df["regime_active_frac"].mean()) if "regime_active_frac" in out_df.columns and not out_df.empty else None,
        seg_trades=out_df["trades"].tolist() if not out_df.empty else None,
        cfg=objective_cfg,
    )
    print(f"overall score_raw  : {breakdown.score_raw:.6f}")
    print(f"overall objective  : {breakdown.objective_final:.6f}")
    print(f"objective version  : {objective_version}")
    print(f"bottom2 mean trades: {breakdown.bottom2_mean_trades:.2f}")
    print(f"seg trade cv       : {breakdown.seg_trade_cv:.4f}")
    print(f"coverage penalties : bottom2={breakdown.penalty_bottom2_trades:.6f} seg_floor={breakdown.penalty_seg_trade_floor:.6f} trade_cv={breakdown.penalty_trade_cv:.6f}")
    print(f"feasible_all       : {int(breakdown.feasible_all)}")

    total_trades = int(exit_cnt_tot.sum())
    print("\n===== Exit Reason Breakdown (Aggregated) =====")
    print(f"total trades (by reason cnt sum) : {total_trades}")
    for i, name in enumerate(EXIT_NAMES):
        c = int(exit_cnt_tot[i])
        pct = (100.0 * c / total_trades) if total_trades > 0 else 0.0
        gross_s = float(exit_gross_tot[i])
        fee_s = float(exit_fee_tot[i])
        net_s = float(exit_net_tot[i])
        avg_net = (net_s / c) if c > 0 else 0.0
        print(f"{name:11s} | cnt={c:5d} ({pct:5.1f}%) | gross_sum={gross_s:+.6f} | fee_sum={fee_s:+.6f} | net_sum={net_s:+.6f} | avg_net={avg_net:+.6f}")

    print("\n[SINGLE STRATEGY BREAKDOWN] (Aggregated)")
    net_total = float(exit_net_tot.sum())
    gross_total = float(exit_gross_tot.sum())
    fee_total = float(exit_fee_tot.sum())
    avg_total = (net_total / total_trades) if total_trades > 0 else 0.0
    win_total = int((pd.DataFrame(trade_rows_all)["net_pnl"] > 0).sum()) if trade_rows_all else 0
    winrate_total = 100.0 * win_total / total_trades if total_trades > 0 else 0.0
    print(f"SINGLE | cnt={total_trades:4d} | long={int(long_tot):4d} short={int(short_tot):4d} short_share={short_share_tot:5.2f} | win%={winrate_total:5.1f} | gross={gross_total:+.6f} | fee={fee_total:+.6f} | net={net_total:+.6f} | avg_net={avg_total:+.6f}")

    print("\n[TRAIL DETAILS]")
    print(f"trail exits BEFORE BEP : {trail_before_tot}")
    print(f"trail exits AFTER  BEP : {trail_after_tot}")
    print(f"bep_armed trades       : {bep_armed_tot}")
    print(f"ref_price updates      : {ref_updates_tot}")

    try:
        out_exit = []
        for i, name in enumerate(EXIT_NAMES):
            c = int(exit_cnt_tot[i])
            net_s = float(exit_net_tot[i])
            out_exit.append({
                "reason": name,
                "count": c,
                "gross_sum": float(exit_gross_tot[i]),
                "fee_sum": float(exit_fee_tot[i]),
                "net_sum": net_s,
                "avg_net": (net_s / c) if c > 0 else 0.0,
            })
        pd.DataFrame(out_exit).to_csv(args.exit_stats_csv, index=False, encoding="utf-8")
        meta_payload = {
            "trail_before_bep": int(trail_before_tot),
            "trail_after_bep": int(trail_after_tot),
            "bep_armed_trades": int(bep_armed_tot),
            "ref_updates": int(ref_updates_tot),
            "risk_close_trades": int(exit_cnt_tot[EXIT_NAMES.index('RISK_CLOSE')]),
            "maxh_cnt": int(maxh_cnt_tot),
            "regime_detect_enabled": int(cfg.get("regime_detect_cfg", {}).get("enabled", 0)),
            "regime_weight_enabled": int(cfg.get("regime_weight_cfg", {}).get("enabled", 0)),
            "regime_threshold_enabled": int(cfg.get("regime_threshold_cfg", {}).get("enabled", 0)),
            "regime_filter_enabled": int(cfg.get("regime_filter_cfg", {}).get("enabled", 0)),
            "regime_lane_enabled": int(cfg.get("regime_lane_cfg", {}).get("enabled", 0)),
            "active_sparse_enabled": int(cfg.get("regime_lane_cfg", {}).get("active_sparse_enabled", 0)),
            "active_sparse_min_ready": int(cfg.get("regime_lane_cfg", {}).get("active_sparse_min_ready", 160)),
            "sparse_gate_q": float(cfg.get("regime_lane_cfg", {}).get("sparse_gate_q", 0.55)),
            "sparse_gate_floor_q": float(cfg.get("regime_lane_cfg", {}).get("sparse_gate_floor_q", 0.0)),
            "sparse_atr_q": float(cfg.get("regime_lane_cfg", {}).get("sparse_atr_q", 0.65)),
            "sparse_range_q": float(cfg.get("regime_lane_cfg", {}).get("sparse_range_q", 0.65)),
            "sparse_vol_q": float(cfg.get("regime_lane_cfg", {}).get("sparse_vol_q", 0.0)),
            "sparse_require_high_vol": int(cfg.get("regime_lane_cfg", {}).get("sparse_require_high_vol", 0)),
            "sparse_high_logic": str(cfg.get("regime_lane_cfg", {}).get("sparse_high_logic", "or")),
            "regime_lane_version": "active_bandpass_lane_v57",
            "softsl_guard_clamped": int(softsl_guard_clamped),
            "allow_soft_sl_before_trail": int(cfg.get("dynamic_cfg", {}).get("allow_soft_sl_before_trail", 0)),
            "softsl_hold_floor": int(cfg.get("dynamic_cfg", {}).get("softsl_hold_floor", 0)),
            "post_bep_shield_ignore_softsl_hold": int(cfg.get("dynamic_cfg", {}).get("post_bep_shield_ignore_softsl_hold", 0)),
            "min_hold_tp_bars_cfg": int(cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))),
            "min_hold_trail_bars_cfg": int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0)))),
            "min_hold_soft_sl_bars_cfg": int(cfg.get("min_hold_soft_sl_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0)))),
            "score_raw_overall": float(breakdown.score_raw),
            "objective_final_overall": float(breakdown.objective_final),
            "penalty_min_seg_trades": float(breakdown.penalty_min_seg_trades),
            "penalty_min_short_trades": float(breakdown.penalty_min_short_trades),
            "penalty_min_short_share": float(breakdown.penalty_min_short_share),
            "penalty_regime_extreme": float(breakdown.penalty_regime_extreme),
            "penalty_bottom2_trades": float(breakdown.penalty_bottom2_trades),
            "penalty_seg_trade_floor": float(breakdown.penalty_seg_trade_floor),
            "penalty_trade_cv": float(breakdown.penalty_trade_cv),
            "bottom2_mean_trades": float(breakdown.bottom2_mean_trades),
            "seg_trade_mean": float(breakdown.seg_trade_mean),
            "seg_trade_std": float(breakdown.seg_trade_std),
            "seg_trade_cv": float(breakdown.seg_trade_cv),
            "seg_trade_min": int(breakdown.seg_trade_min),
            "seg_trade_max": int(breakdown.seg_trade_max),
            "seg_bottom2_target": float(objective_cfg.get("seg_bottom2_target", 0.0)),
            "seg_bottom2_penalty_k": float(objective_cfg.get("seg_bottom2_penalty_k", 0.0)),
            "seg_floor_target": float(objective_cfg.get("seg_floor_target", 0.0)),
            "seg_floor_penalty_k": float(objective_cfg.get("seg_floor_penalty_k", 0.0)),
            "trade_cv_cap": float(objective_cfg.get("trade_cv_cap", 0.0)),
            "trade_cv_penalty_k": float(objective_cfg.get("trade_cv_penalty_k", 0.0)),
            "feasible_all": int(breakdown.feasible_all),
            "w_mean": float(w_mean),
            "w_worst": float(w_worst),
            "worst_agg": str(worst_agg),
            "worst_k": int(worst_k),
            "worst_q": float(worst_q),
            "objective_version": str(objective_version),
            "allow_config_fallback": int(allow_config_fallback),
            "config_fallback_policy": ("config+tuned_meta" if allow_config_fallback else "cli/default-only-for-exposed-fields"),
        }
        if not out_df.empty:
            for _col in out_df.columns:
                if isinstance(_col, str) and (_col.startswith("cand_") or _col.startswith("pass_") or _col.startswith("final_candidates") or _col.startswith("final_entries")):
                    meta_payload[f"agg_{_col}"] = float(pd.to_numeric(out_df[_col], errors="coerce").fillna(0).sum())
            if "active_sparse_fallback_dense_count" in out_df.columns:
                meta_payload["active_sparse_fallback_dense_cnt"] = int(pd.to_numeric(out_df["active_sparse_fallback_dense_count"], errors="coerce").fillna(0).sum())
            for _lane_col in (
                "active_sparse_after_gate_band_hist_count",
                "active_sparse_after_high_hist_count",
                "active_sparse_after_vol_hist_count",
                "active_sparse_after_gate_band_seg_count",
                "active_sparse_after_high_seg_count",
                "active_sparse_after_vol_seg_count",
            ):
                if _lane_col in out_df.columns:
                    meta_payload[f"agg_{_lane_col}"] = int(pd.to_numeric(out_df[_lane_col], errors="coerce").fillna(0).sum())
        if trade_rows_all:
            _trade_df_meta = pd.DataFrame(trade_rows_all)
            if "entry_regime_alpha" in _trade_df_meta.columns:
                _alpha = pd.to_numeric(_trade_df_meta["entry_regime_alpha"], errors="coerce").dropna()
                if len(_alpha) > 0:
                    meta_payload.update({
                        "entry_regime_alpha_mean": float(_alpha.mean()),
                        "entry_regime_alpha_p10": float(_alpha.quantile(0.10)),
                        "entry_regime_alpha_p50": float(_alpha.quantile(0.50)),
                        "entry_regime_alpha_p90": float(_alpha.quantile(0.90)),
                    })
            if "entry_regime_bucket" in _trade_df_meta.columns:
                _bucket = pd.to_numeric(_trade_df_meta["entry_regime_bucket"], errors="coerce").dropna().astype(int)
                if len(_bucket) > 0:
                    meta_payload.update({
                        "entry_regime_calm_frac": float((_bucket == 0).mean()),
                        "entry_regime_mid_frac": float((_bucket == 1).mean()),
                        "entry_regime_active_frac": float((_bucket == 2).mean()),
                        "entry_regime_calm_cnt": int((_bucket == 0).sum()),
                        "entry_regime_mid_cnt": int((_bucket == 1).sum()),
                        "entry_regime_active_cnt": int((_bucket == 2).sum()),
                    })
            if "entry_q_used" in _trade_df_meta.columns:
                _q_used = pd.to_numeric(_trade_df_meta["entry_q_used"], errors="coerce").dropna()
                if len(_q_used) > 0:
                    meta_payload.update({
                        "entry_q_used_mean": float(_q_used.mean()),
                        "entry_q_used_p10": float(_q_used.quantile(0.10)),
                        "entry_q_used_p50": float(_q_used.quantile(0.50)),
                        "entry_q_used_p90": float(_q_used.quantile(0.90)),
                    })
            if "entry_threshold_base_used" in _trade_df_meta.columns:
                _thr_base = pd.to_numeric(_trade_df_meta["entry_threshold_base_used"], errors="coerce").dropna()
                if len(_thr_base) > 0:
                    meta_payload.update({
                        "entry_threshold_base_used_mean": float(_thr_base.mean()),
                        "entry_threshold_base_used_p50": float(_thr_base.quantile(0.50)),
                        "entry_threshold_base_used_p90": float(_thr_base.quantile(0.90)),
                    })
            if "entry_th_used" in _trade_df_meta.columns:
                _thr_used = pd.to_numeric(_trade_df_meta["entry_th_used"], errors="coerce").dropna()
                if len(_thr_used) > 0:
                    meta_payload.update({
                        "entry_threshold_used_mean": float(_thr_used.mean()),
                        "entry_threshold_used_p50": float(_thr_used.quantile(0.50)),
                        "entry_threshold_used_p90": float(_thr_used.quantile(0.90)),
                    })
            if "entry_vol_low_th_used" in _trade_df_meta.columns:
                _vv = pd.to_numeric(_trade_df_meta["entry_vol_low_th_used"], errors="coerce").dropna()
                if len(_vv) > 0:
                    meta_payload.update({
                        "entry_vol_low_th_used_mean": float(_vv.mean()),
                        "entry_vol_low_th_used_p50": float(_vv.quantile(0.50)),
                        "entry_vol_low_th_used_p90": float(_vv.quantile(0.90)),
                    })
            if "entry_atr_entry_mult_used" in _trade_df_meta.columns:
                _aa = pd.to_numeric(_trade_df_meta["entry_atr_entry_mult_used"], errors="coerce").dropna()
                if len(_aa) > 0:
                    meta_payload.update({
                        "entry_atr_entry_mult_used_mean": float(_aa.mean()),
                        "entry_atr_entry_mult_used_p50": float(_aa.quantile(0.50)),
                        "entry_atr_entry_mult_used_p90": float(_aa.quantile(0.90)),
                    })
            if "entry_range_entry_mult_used" in _trade_df_meta.columns:
                _rr = pd.to_numeric(_trade_df_meta["entry_range_entry_mult_used"], errors="coerce").dropna()
                if len(_rr) > 0:
                    meta_payload.update({
                        "entry_range_entry_mult_used_mean": float(_rr.mean()),
                        "entry_range_entry_mult_used_p50": float(_rr.quantile(0.50)),
                        "entry_range_entry_mult_used_p90": float(_rr.quantile(0.90)),
                    })
            if "entry_profile_name" in _trade_df_meta.columns:
                _pname = _trade_df_meta["entry_profile_name"].astype(str).str.lower()
                meta_payload["entry_profile_active_dense_cnt"] = int((_pname == "active_dense").sum())
                meta_payload["entry_profile_active_sparse_cnt"] = int((_pname == "active_sparse").sum())
                meta_payload["entry_profile_active_sparse_frac"] = float((_pname == "active_sparse").mean()) if len(_pname) > 0 else 0.0
            if "entry_active_sparse_flag" in _trade_df_meta.columns:
                _as = pd.to_numeric(_trade_df_meta["entry_active_sparse_flag"], errors="coerce").fillna(0).astype(int)
                meta_payload["entry_active_sparse_flag_cnt"] = int((_as == 1).sum())
            if "entry_is_rearm" in _trade_df_meta.columns:
                _rearm = pd.to_numeric(_trade_df_meta["entry_is_rearm"], errors="coerce").fillna(0).astype(int)
                meta_payload["rearm_entries"] = int((_rearm == 1).sum())
            if "last_exit_reason" in _trade_df_meta.columns and "entry_is_rearm" in _trade_df_meta.columns:
                _last_reason = pd.to_numeric(_trade_df_meta["last_exit_reason"], errors="coerce").fillna(-1).astype(int)
                _rearm_mask = pd.to_numeric(_trade_df_meta["entry_is_rearm"], errors="coerce").fillna(0).astype(int) == 1
                meta_payload["rearm_entries_after_trail"] = int((_rearm_mask & (_last_reason == EXIT_NAMES.index("TRAIL"))).sum())
                meta_payload["rearm_entries_after_tp"] = int((_rearm_mask & (_last_reason == EXIT_NAMES.index("TP"))).sum())
                meta_payload["rearm_entries_after_sl"] = int((_rearm_mask & (_last_reason == EXIT_NAMES.index("SL"))).sum())
            if "tp_window_armed" in _trade_df_meta.columns:
                _tpw = pd.to_numeric(_trade_df_meta["tp_window_armed"], errors="coerce").fillna(0).astype(int)
                meta_payload["tp_window_armed_trades"] = int((_tpw == 1).sum())
            if "tp_window_live_bars" in _trade_df_meta.columns:
                _tpw_bars = pd.to_numeric(_trade_df_meta["tp_window_live_bars"], errors="coerce").fillna(0)
                meta_payload["tp_window_live_bars_total"] = int(_tpw_bars.sum())
                meta_payload["tp_window_live_bars_mean"] = float(_tpw_bars.mean()) if len(_tpw_bars) > 0 else 0.0
            if "entry_run_id" in _trade_df_meta.columns:
                _run = pd.to_numeric(_trade_df_meta["entry_run_id"], errors="coerce").dropna().astype(int)
                if len(_run) > 0:
                    _episodes_per_run = _run.value_counts()
                    meta_payload["run_count"] = int(_episodes_per_run.shape[0])
                    meta_payload["episodes_per_run_mean"] = float(_episodes_per_run.mean())
                    meta_payload["episodes_per_run_p90"] = float(_episodes_per_run.quantile(0.90))
        with open(args.exit_stats_meta, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        if bool(int(getattr(args, "emit_runner_kpi_meta", 1) or 0)) and trade_rows_all:
            _trade_df_meta = pd.DataFrame(trade_rows_all)

            if "runner_align_enabled" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["runner_align_enabled"], errors="coerce").fillna(0).astype(int)
                meta_payload["runner_align_enabled"] = int(_s.max())

            if "runner_align_profit_floor_enabled" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["runner_align_profit_floor_enabled"], errors="coerce").fillna(0).astype(int)
                meta_payload["runner_align_profit_floor_enabled"] = int(_s.max())

            if "runner_align_thesis_monitor_enabled" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["runner_align_thesis_monitor_enabled"], errors="coerce").fillna(0).astype(int)
                meta_payload["runner_align_thesis_monitor_enabled"] = int(_s.max())

            if "profit_floor_violation" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["profit_floor_violation"], errors="coerce").fillna(0).astype(int)
                meta_payload["profit_floor_violation_count"] = int(_s.sum())

            if "profit_floor_price" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["profit_floor_price"], errors="coerce")
                meta_payload["profit_floor_price_nonnull"] = int(_s.notna().sum())

            if "thesis_monitor_hit" in _trade_df_meta.columns:
                _s = pd.to_numeric(_trade_df_meta["thesis_monitor_hit"], errors="coerce").fillna(0).astype(int)
                meta_payload["thesis_monitor_hit_count"] = int(_s.sum())
                
        print(f"[DONE] Saved exit stats -> {args.exit_stats_csv} (+ meta json)")
    except Exception as e:
        print(f"[WARN] exit stats save failed: {e}")

    try:
        pd.DataFrame(single_rows).to_csv(args.single_stats_csv, index=False, encoding="utf-8")
        print(f"[DONE] Saved single stats -> {args.single_stats_csv}")
    except Exception as e:
        print(f"[WARN] single stats save failed: {e}")

    try:
        if trade_rows_all:
            trade_df = pd.DataFrame(trade_rows_all)
            if time_col is not None:
                times = pd.to_datetime(df_window[time_col], utc=True, errors="coerce")
                def _attach_time(col_idx: str, out_col: str):
                    idx = trade_df[col_idx].astype(int)
                    ok = (idx >= 0) & (idx < len(times))
                    vals = pd.Series(pd.NaT, index=trade_df.index, dtype="datetime64[ns, UTC]")
                    vals.loc[ok] = times.iloc[idx[ok].to_numpy()].to_numpy()
                    trade_df[out_col] = vals.astype(str)
                _attach_time("decision_idx", "decision_time")
                _attach_time("entry_idx", "entry_time")
                _attach_time("exit_idx", "exit_time")
            trade_df.to_csv(args.trade_log_csv, index=False, encoding="utf-8")
            print(f"[DONE] Saved trade log -> {args.trade_log_csv}")
    except Exception as e:
        print(f"[WARN] trade log save failed: {e}")

    out_df.to_csv(args.results_csv, index=False, encoding="utf-8")
    print(f"[DONE] Saved rolling OOS results -> {args.results_csv}")
    print("[END] Backtest Finished.")


if __name__ == "__main__":
    main()
