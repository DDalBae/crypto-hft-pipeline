# -*- coding: utf-8 -*-
"""
Authoritative v56 fast core with numba fast path plus tp_window, entry_episode/rearm structural fallback, active_sparse lane v2 support, and the soft-SL/trail guard fix.

This file removes the shadowed duplicate definitions that had accumulated in
`hybrid_rlmode_single_core_modified.py` and keeps a single public path for:
- regime-aware prepare helpers
- cost-aware candidate masking
- fast intrabar worst-case selection
- hybrid detailed backtests with optional Python fallback for structural
  features such as tp_window and entry-episode split / re-arm.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from hybrid_core_v7 import (
    EXIT_FORCE,
    EXIT_MAXH,
    EXIT_NAMES,
    EXIT_RISK,
    EXIT_SL,
    EXIT_TP,
    EXIT_TRAIL,
    HORIZONS_ALL,
    ObjectiveBreakdown,
    SegmentMetrics,
    agg_worst,
    apply_ranges_overrides,
    assemble_objective,
    build_bucketed_entry_threshold_pack,
    build_bucketed_filter_pack,
    build_dynamic_arrays,
    build_regime_adaptive_signal_bundle,
    build_single_best_config,
    build_regime_alpha_exogenous,
    normalize_horizon_weights,
    normalize_single_config_from_any,
    parse_float_list,
    precompute_hybrids,
    quantile_from_history,
    regime_bucket_name,
    safe_float,
    safe_int,
    segment_score,
    side_balance_penalty_component,
    simulate_trading_core_rl_single as simulate_trading_core_rl_single_detailed,
    weights_from_self_mix,
    weights_from_raw_vector,
)

_build_regime_adaptive_signal_bundle_v2 = build_regime_adaptive_signal_bundle
_regime_bucket_name_v2 = regime_bucket_name

@njit(cache=True)
def resolve_local_soft_sl_hold_numba(base_soft_hold: int, trail_hold: int, relax: int, allow_before_trail: int, hold_floor: int) -> int:
    v = int(base_soft_hold) - int(relax)
    floor_eff = int(hold_floor)
    if int(allow_before_trail) == 0:
        if int(trail_hold) > floor_eff:
            floor_eff = int(trail_hold)
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

@njit(cache=True)
def _bep_econ_fee_numba(taker_fee_side: float, maker_fee_per_side: float, mode_code: int) -> float:
    # 0=maker_be/scaled, 1=taker_be
    if int(mode_code) == 1:
        return float(taker_fee_side) + float(taker_fee_side)
    maker_exit = float(maker_fee_per_side) if float(maker_fee_per_side) > 0.0 else float(taker_fee_side)
    return float(taker_fee_side) + maker_exit

@njit(cache=True)
def _rearm_reason_code_numba(reason: int) -> int:
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

@njit(cache=True)
def _rearm_exit_allowed_numba(reason: int, rearm_after_trail: int, rearm_after_tp: int, rearm_after_sl: int) -> int:
    if int(reason) == int(EXIT_TRAIL):
        return 1 if int(rearm_after_trail) != 0 else 0
    if int(reason) == int(EXIT_TP):
        return 1 if int(rearm_after_tp) != 0 else 0
    if int(reason) == int(EXIT_SL):
        return 1 if int(rearm_after_sl) != 0 else 0
    return 0

@njit(cache=True)
def simulate_trading_core_rl_single_fast(
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    gate_strength: np.ndarray,
    dir_signal: np.ndarray,
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    range_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    atr_high_th: float,
    atr_med: float,
    range_med: float,
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
    bep_stop_mode_code: int,
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
    early_softsl_enabled: int,
    early_softsl_min_hold: int,
    early_softsl_progress_frac: float,
    early_trail_enabled: int,
    early_trail_min_hold: int,
    early_trail_progress_frac: float,
    early_trail_ref_updates_min: int,
    tp_window_enabled: int,
    tp_window_progress_frac_arm: float,
    tp_window_extend_bars: int,
    tp_window_block_early_trail: int,
    tp_window_block_early_soft_sl: int,
    tp_window_floor_trail_hold_to_tp: int,
    tp_window_floor_soft_sl_hold_to_tp: int,
    tp_window_suspend_post_bep_shield_before_tp: int,
    tp_window_expire_on_pullback_frac: float,
    entry_episode_enabled: int,
    rearm_enabled: int,
    run_gap_reset_bars: int,
    episode_max_entries_per_run: int,
    rearm_same_side_only: int,
    rearm_cooldown_bars: int,
    rearm_max_bars_after_exit: int,
    rearm_gate_reset_frac: float,
    rearm_gate_refresh_frac: float,
    rearm_price_reset_frac: float,
    rearm_after_trail: int,
    rearm_after_tp: int,
    rearm_after_sl: int,
    same_side_hold_enabled: int,
    same_side_hold_weak_enabled: int,
    same_side_hold_strong_ratio: float,
    same_side_hold_weak_ratio: float,
    same_side_hold_weak_min_progress_frac: float,
    same_side_hold_allow_pre_bep_weak: int,
    same_side_hold_pre_bep_max_bonus_bars: int,
    same_side_hold_bonus_bars_strong: int,
    same_side_hold_bonus_bars_weak: int,
    same_side_hold_max_extra_bars: int,
    same_side_hold_grace_after_bep_strong: int,
    same_side_hold_grace_after_bep_weak: int,
    same_side_hold_grace_after_unlock_strong: int,
    same_side_hold_grace_after_unlock_weak: int,
    support_strength_ratio_arr: np.ndarray,
    support_weak_eligible_mask: np.ndarray,
    support_pass_mask: np.ndarray,
    stop_equity: float,
    stop_dd: float,
    warmup_steps: int = 0,
    integer_leverage: int = 0,
    intrabar_mode: int = 1,
) -> Tuple[float, float, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int, int]:
    n = int(len(close))
    exit_cnt = np.zeros(6, dtype=np.int64)
    exit_gross_sum = np.zeros(6, dtype=np.float64)
    exit_fee_sum = np.zeros(6, dtype=np.float64)
    exit_net_sum = np.zeros(6, dtype=np.float64)

    if n < 2:
        return 0.0, 0.0, 0, 0, 0, exit_cnt, exit_gross_sum, exit_fee_sum, exit_net_sum, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if atr_med <= 0.0:
        atr_med = 1.0
    if range_med <= 0.0:
        range_med = 1.0

    taker_fee_side = float(cost_per_side) + float(slip_per_side)
    fee_roundtrip = 2.0 * taker_fee_side
    econ_be_fee = _bep_econ_fee_numba(taker_fee_side, maker_fee_per_side, bep_stop_mode_code)

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
    pend_is_rearm = 0
    pend_last_exit_reason = -1

    entry_bep_arm_val = 0.0
    entry_is_rearm = 0
    entry_last_exit_reason = -1
    mfe_rel = 0.0
    mae_rel = 0.0
    mfe_prev_rel = 0.0

    tp_window_armed_trades = 0
    tp_window_live_bars_total = 0
    tp_window_blocked_early_trail_total = 0
    tp_window_blocked_softsl_total = 0
    tpw_live = 0
    tpw_armed_at = -1
    tpw_peak_progress = 0.0
    tpw_progress_at_arm = 0.0
    tpw_blocked_early_trail_local = 0
    tpw_blocked_softsl_local = 0
    tpw_live_bars_local = 0

    rearm_entries = 0
    rearm_entries_after_trail = 0
    rearm_entries_after_tp = 0
    rearm_entries_after_sl = 0

    pos_same_side_hold_floor_trail = 0
    pos_same_side_grace_bep_until_i = -1
    pos_same_side_grace_unlock_until_i = -1
    same_side_hold_events = 0
    same_side_hold_strong_events = 0
    same_side_hold_weak_events = 0

    run_id = 0
    run_active = 0
    run_side = 0
    run_peak_gate = 0.0
    run_entries = 0
    run_gap_bars = 0
    run_reset_seen_after_exit = 0
    last_exit_i = -1000000000
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

            entry_bep_arm_val = pos_BEP_ARM if pos_BEP_ARM >= pos_bep_arm_fee else pos_bep_arm_fee
            pos_ref_updates_local = 0
            mfe_rel = 0.0
            mae_rel = 0.0
            mfe_prev_rel = 0.0
            tpw_live = 0
            tpw_armed_at = -1
            tpw_peak_progress = 0.0
            tpw_progress_at_arm = 0.0
            tpw_blocked_early_trail_local = 0
            tpw_blocked_softsl_local = 0
            tpw_live_bars_local = 0
            pos_same_side_hold_floor_trail = 0
            pos_same_side_grace_bep_until_i = -1
            pos_same_side_grace_unlock_until_i = -1
            entry_is_rearm = int(pend_is_rearm)
            entry_last_exit_reason = int(pend_last_exit_reason)
            if int(entry_is_rearm) == 1:
                rearm_entries += 1
                if int(entry_last_exit_reason) == int(EXIT_TRAIL):
                    rearm_entries_after_trail += 1
                elif int(entry_last_exit_reason) == int(EXIT_TP):
                    rearm_entries_after_tp += 1
                elif int(entry_last_exit_reason) == int(EXIT_SL):
                    rearm_entries_after_sl += 1

        hi = float(high[i])
        lo = float(low[i])
        c = float(close[i])

        if pos_side != 0:
            side = int(pos_side)
            ep = float(entry_price)
            if ep > 0.0:
                den = ep if ep > 1e-12 else 1e-12
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
                    min_tr_live = econ_be_fee if econ_be_fee >= fee_roundtrip * float(fee_tp_mult) else fee_roundtrip * float(fee_tp_mult)
                    if cand_tr < min_tr_live:
                        cand_tr = min_tr_live
                if cand_tr < pos_TR:
                    pos_TR = float(cand_tr)

                cand_soft = resolve_local_soft_sl_hold_numba(
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
                timeout_hit = (
                    hold >= int(pre_bep_timeout_bars)
                    and stress_prev >= float(pre_bep_stress_th)
                    and progress_prev < float(pre_bep_progress_frac)
                )
                if timeout_hit:
                    cand_sl = float(pos_SL_base) * float(pre_bep_degrade_sl_scale)
                    min_sl_live = 0.6 * fee_roundtrip
                    if cand_sl < min_sl_live:
                        cand_sl = min_sl_live
                    if cand_sl < pos_SL:
                        pos_SL = float(cand_sl)
                    if int(pre_bep_softsl_delta) > 0:
                        cand_soft = resolve_local_soft_sl_hold_numba(
                            int(pos_min_hold_soft_sl_base),
                            int(pos_min_hold_trail),
                            int(pre_bep_softsl_delta),
                            int(allow_soft_sl_before_trail),
                            int(softsl_hold_floor),
                        )
                        if cand_soft < pos_min_hold_soft_sl:
                            pos_min_hold_soft_sl = int(cand_soft)
                    if int(pre_bep_force_close_bars) > 0 and hold >= int(pre_bep_force_close_bars):
                        den = ep if ep > 1e-12 else 1e-12
                        unreal = side * (c - ep) / den
                        if (int(pre_bep_force_close_red_only) == 0) or (unreal <= 0.0):
                            force_close_now = True

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

            hard_sl_dist = float(SL_v) * float(hard_sl_mult_pre_unlock)
            if hard_sl_dist < 0.6 * fee_roundtrip:
                hard_sl_dist = 0.6 * fee_roundtrip
            hard_sl = ep * (1.0 - side * hard_sl_dist)

            do_exit = False
            exit_price = 0.0
            reason = EXIT_MAXH
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
                if float(maker_fee_per_side) > 0.0 and (reason == EXIT_TP or reason == EXIT_TRAIL):
                    exit_fee_side = float(maker_fee_per_side)
                den = ep if ep > 1e-12 else 1e-12
                fee_total = (taker_fee_side + exit_fee_side) * lev_now
                gross_pnl = (side * (exit_price - ep) / den) * lev_now
                net_pnl = gross_pnl - fee_total
                scaled = net_pnl
                if scaled < -0.999:
                    scaled = -0.999
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

                last_exit_i = int(i)
                last_exit_reason = int(reason)
                last_exit_side = int(side)
                last_exit_price = float(exit_price)
                last_run_id = int(run_id if run_active == 1 else -1)
                if run_active == 1 and last_run_id == run_id:
                    run_reset_seen_after_exit = 0

                pos_side = 0
                bep_armed = 0
                bep_armed_at = -1
                tpw_live = 0
                tpw_armed_at = -1
                tpw_peak_progress = 0.0
                tpw_progress_at_arm = 0.0
                tpw_blocked_early_trail_local = 0
                tpw_blocked_softsl_local = 0
                tpw_live_bars_local = 0
                pos_same_side_hold_floor_trail = 0
                pos_same_side_grace_bep_until_i = -1
                pos_same_side_grace_unlock_until_i = -1

            if pos_side != 0:
                den = ep if ep > 1e-12 else 1e-12
                fav_cur = 0.0
                if side == 1:
                    fav_cur = (hi - ep) / den
                else:
                    fav_cur = (ep - lo) / den
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
                        else:
                            if int(same_side_hold_grace_after_bep_weak) > 0:
                                pos_same_side_grace_bep_until_i = max(int(pos_same_side_grace_bep_until_i), int(i + int(same_side_hold_grace_after_bep_weak)))
                            if int(same_side_hold_grace_after_unlock_weak) > 0:
                                pos_same_side_grace_unlock_until_i = max(int(pos_same_side_grace_unlock_until_i), int(i + int(same_side_hold_grace_after_unlock_weak)))
                            same_side_hold_weak_events += 1

                        same_side_hold_events += 1

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
                    run_side = int(cand_side)
                    run_peak_gate = float(gsig)
                    run_entries = 0
                    run_gap_bars = 0
                    run_reset_seen_after_exit = 0
            else:
                if cand_on and int(cand_side) == int(run_side):
                    run_gap_bars = 0
                    if float(gsig) > float(run_peak_gate):
                        run_peak_gate = float(gsig)
                elif cand_on and int(cand_side) != int(run_side):
                    run_id += 1
                    run_active = 1
                    run_side = int(cand_side)
                    run_peak_gate = float(gsig)
                    run_entries = 0
                    run_gap_bars = 0
                    run_reset_seen_after_exit = 0
                else:
                    run_gap_bars += 1
                    if run_gap_bars > int(run_gap_reset_bars):
                        run_active = 0
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
            last_exit_reason_now = -1

            if int(entry_episode_enabled) == 1:
                if run_active == 0:
                    run_id += 1
                    run_active = 1
                    run_side = int(cand_side)
                    run_peak_gate = float(gsig)
                    run_entries = 1
                    run_gap_bars = 0
                    run_reset_seen_after_exit = 0
                else:
                    if run_entries == 0 or int(last_run_id) != int(run_id):
                        run_entries = int(run_entries) + 1
                    else:
                        if int(rearm_enabled) == 0:
                            allow_entry = False
                        else:
                            bars_since_last_exit_now = int(i - last_exit_i) if int(last_exit_i) > -1000000000 else -1
                            last_exit_reason_now = int(last_exit_reason)
                            price_reset_frac_now = 0.0
                            if int(last_exit_i) > -1000000000 and float(last_exit_price) > 0.0:
                                price_reset_frac_now = abs(float(c) - float(last_exit_price)) / max(abs(float(last_exit_price)), 1e-12)
                            reason_ok = _rearm_exit_allowed_numba(int(last_exit_reason), int(rearm_after_trail), int(rearm_after_tp), int(rearm_after_sl)) == 1
                            same_side_ok = (int(rearm_same_side_only) == 0) or (int(cand_side) == int(last_exit_side))
                            cooldown_ok = bars_since_last_exit_now >= int(rearm_cooldown_bars)
                            max_bars_ok = True if int(rearm_max_bars_after_exit) <= 0 else (bars_since_last_exit_now <= int(rearm_max_bars_after_exit))
                            episode_cap_ok = int(run_entries) < int(episode_max_entries_per_run)
                            gate_refresh_ok = float(gsig) >= float(rearm_gate_refresh_frac) * max(float(run_peak_gate), 1e-12)
                            reset_ok = bool(int(run_reset_seen_after_exit) == 1 or float(price_reset_frac_now) >= float(rearm_price_reset_frac))
                            allow_entry = bool(reason_ok and same_side_ok and cooldown_ok and max_bars_ok and episode_cap_ok and gate_refresh_ok and reset_ok)
                            if allow_entry:
                                is_rearm_now = 1
                                run_entries = int(run_entries) + 1
                        if not allow_entry:
                            continue
                    if int(run_entries) > int(episode_max_entries_per_run):
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
                min_tr = econ_be_fee if econ_be_fee >= fee_roundtrip * float(fee_tp_mult) else fee_roundtrip * float(fee_tp_mult)
                if tr_v < min_tr:
                    tr_v = min_tr

            local_soft_sl_hold = resolve_local_soft_sl_hold_numba(
                int(min_hold_soft_sl_bars),
                int(min_hold_trail_bars),
                int(dyn_softsl_relax_arr[i]),
                int(allow_soft_sl_before_trail),
                int(softsl_hold_floor),
            )

            pend_is_rearm = int(is_rearm_now)
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
        den = ep if ep > 1e-12 else 1e-12
        fee_total = (taker_fee_side + taker_fee_side) * lev_now
        gross_pnl = (side * (last_c - ep) / den) * lev_now
        net_pnl = gross_pnl - fee_total
        scaled = net_pnl
        if scaled < -0.999:
            scaled = -0.999
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
        pos_side = 0
        bep_armed = 0
        bep_armed_at = -1

    return (
        float(equity - 1.0),
        float(mdd),
        int(trade_cnt),
        int(win_cnt),
        int(tail_hit),
        exit_cnt,
        exit_gross_sum,
        exit_fee_sum,
        exit_net_sum,
        int(trail_before_bep_cnt),
        int(trail_after_bep_cnt),
        int(bep_armed_trades),
        int(ref_updates),
        int(long_trades),
        int(short_trades),
        int(maxh_cnt),
        int(tp_window_armed_trades),
        int(tp_window_live_bars_total),
        int(tp_window_blocked_early_trail_total),
        int(tp_window_blocked_softsl_total),
        int(rearm_entries),
        int(rearm_entries_after_trail),
        int(rearm_entries_after_tp),
        int(rearm_entries_after_sl),
        int(same_side_hold_events),
        int(same_side_hold_strong_events),
        int(same_side_hold_weak_events),
    )

def _bep_stop_mode_code(mode: str) -> int:
    mode = str(mode or "maker_be").strip().lower()
    return 1 if mode == "taker_be" else 0

def _available_horizons_from_signals(signals_by_h: Dict[int, np.ndarray]) -> Tuple[int, ...]:
    hs = []
    for h in HORIZONS_ALL:
        arr = signals_by_h.get(int(h))
        if arr is None:
            continue
        hs.append(int(h))
    return tuple(hs)

def _compose_weighted_signal(signals_by_h: Dict[int, np.ndarray], weights: Dict[str, float], available_horizons: Sequence[int]) -> np.ndarray:
    out = None
    for h in available_horizons:
        arr = np.asarray(signals_by_h[int(h)], dtype=np.float64)
        w = float(weights.get(f"w{int(h)}", 0.0))
        if out is None:
            out = np.zeros_like(arr, dtype=np.float64)
        out += w * arr
    if out is None:
        first = np.asarray(next(iter(signals_by_h.values())), dtype=np.float64)
        out = np.zeros_like(first, dtype=np.float64)
    return out

def _score_from_fast_tuple(res: Tuple[Any, ...], score_cfg: Dict[str, Any]) -> float:
    return segment_score(
        net_ret=float(res[0]),
        mdd=float(res[1]),
        tail_hit=int(res[4]),
        trades=int(res[2]),
        maxh_cnt=int(res[15]),
        long_trades=int(res[13]),
        short_trades=int(res[14]),
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

def _fast_tuple_to_result(res: Tuple[Any, ...], chosen_score: float, thr_entry: float, atr_high_th: float, score_cfg: Dict[str, Any]) -> Dict[str, Any]:
    trades = int(res[2])
    wins = int(res[3])
    long_trades = int(res[13])
    short_trades = int(res[14])
    short_share = float(short_trades / trades) if trades > 0 else 0.0
    side_penalty = side_balance_penalty_component(
        long_trades,
        short_trades,
        int(score_cfg.get("min_short_trades_global", 0)),
        float(score_cfg.get("min_short_share_global", 0.0)),
        float(score_cfg.get("side_balance_penalty_k", 0.0)),
    )
    maxh_cnt = int(res[15])
    maxh_ratio = float(maxh_cnt / trades) if trades > 0 else 0.0
    return {
        "net_ret": float(res[0]),
        "mdd_net": float(res[1]),
        "winrate_net": float(wins / trades) if trades > 0 else 0.0,
        "trades": trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "short_share": float(short_share),
        "trade_logs": [],
        "tail": int(res[4]),
        "thr_entry": float(thr_entry),
        "atr_high_th": float(atr_high_th),
        "exit_cnt": res[5].copy(),
        "exit_gross_sum": res[6].copy(),
        "exit_fee_sum": res[7].copy(),
        "exit_net_sum": res[8].copy(),
        "trail_before_bep": int(res[9]),
        "trail_after_bep": int(res[10]),
        "bep_armed_trades": int(res[11]),
        "ref_updates": int(res[12]),
        "maxh_cnt": int(maxh_cnt),
        "score": float(chosen_score),
        "side_penalty": float(side_penalty),
        "maxh_ratio": float(maxh_ratio),
        "tp_window_armed_trades": int(res[16]) if len(res) > 16 else 0,
        "tp_window_live_bars_total": int(res[17]) if len(res) > 17 else 0,
        "tp_window_blocked_early_trail": int(res[18]) if len(res) > 18 else 0,
        "tp_window_blocked_softsl": int(res[19]) if len(res) > 19 else 0,
        "rearm_entries": int(res[20]) if len(res) > 20 else 0,
        "rearm_entries_after_trail": int(res[21]) if len(res) > 21 else 0,
        "rearm_entries_after_tp": int(res[22]) if len(res) > 22 else 0,
        "rearm_entries_after_sl": int(res[23]) if len(res) > 23 else 0,
        "same_side_hold_events": int(res[24]) if len(res) > 24 else 0,
        "same_side_hold_strong_events": int(res[25]) if len(res) > 25 else 0,
        "same_side_hold_weak_events": int(res[26]) if len(res) > 26 else 0,
    }

def warmup_single_fast_core() -> None:
    x = np.linspace(1.0, 1.01, 8).astype(np.float64)
    z = np.zeros(8, dtype=np.float64)
    b = np.ones(8, dtype=np.bool_)
    simulate_trading_core_rl_single_fast(
        open_=x,
        close=x,
        high=x,
        low=x,
        gate_strength=z,
        dir_signal=z,
        ready=b,
        vol_z=z,
        atr_rel=np.ones(8, dtype=np.float64) * 0.001,
        range_rel=np.ones(8, dtype=np.float64) * 0.001,
        minutes_to_next_funding=np.ones(8, dtype=np.float64) * 999.0,
        atr_high_th=np.nan,
        atr_med=0.001,
        range_med=0.001,
        vol_low_th=-1e9,
        funding_near_min=0.0,
        risk_lev_cap=12.0,
        base_leverage=10.0,
        cost_per_side=0.0007,
        maker_fee_per_side=0.0002,
        slip_per_side=0.00015,
        fee_tp_mult=0.7,
        bep_arm_fee_mult=0.2,
        bep_stop_fee_mult=1.0,
        bep_stop_mode_code=0,
        atr_entry_mult=1.0,
        range_entry_mult=1.0,
        low_vol_filter=0,
        trail_after_bep=1,
        risk_entry_mode=0,
        use_atr_scaling=1,
        lev_mult=1.0,
        TP=0.01,
        SL=0.005,
        bep_arm_base=0.001,
        trailing=0.001,
        min_hold_bars=6,
        min_hold_trail_bars=8,
        min_hold_soft_sl_bars=6,
        max_hold_bars=32,
        dyn_lev_scale_arr=np.ones(8, dtype=np.float64),
        dyn_bep_scale_arr=np.ones(8, dtype=np.float64),
        dyn_trail_scale_arr=np.ones(8, dtype=np.float64),
        dyn_sl_scale_arr=np.ones(8, dtype=np.float64),
        dyn_softsl_relax_arr=np.zeros(8, dtype=np.int64),
        dyn_gate_mult_arr=np.ones(8, dtype=np.float64),
        dyn_stress_arr=np.zeros(8, dtype=np.float64),
        use_pre_bep_timeout=0,
        pre_bep_timeout_bars=3,
        pre_bep_stress_th=0.55,
        pre_bep_progress_frac=0.55,
        pre_bep_degrade_sl_scale=0.85,
        pre_bep_softsl_delta=0,
        pre_bep_force_close_bars=0,
        pre_bep_force_close_red_only=1,
        dyn_mode_code=0,
        allow_soft_sl_before_trail=0,
        softsl_hold_floor=0,
        post_bep_shield_ignore_softsl_hold=0,
        hard_sl_mult_pre_unlock=1.0,
        trail_grace_after_bep=0,
        trail_grace_after_unlock=0,
        early_softsl_enabled=0,
        early_softsl_min_hold=2,
        early_softsl_progress_frac=0.5,
        early_trail_enabled=0,
        early_trail_min_hold=3,
        early_trail_progress_frac=0.85,
        early_trail_ref_updates_min=1,
        tp_window_enabled=0,
        tp_window_progress_frac_arm=0.70,
        tp_window_extend_bars=0,
        tp_window_block_early_trail=1,
        tp_window_block_early_soft_sl=1,
        tp_window_floor_trail_hold_to_tp=1,
        tp_window_floor_soft_sl_hold_to_tp=1,
        tp_window_suspend_post_bep_shield_before_tp=1,
        tp_window_expire_on_pullback_frac=0.35,
        entry_episode_enabled=0,
        rearm_enabled=0,
        run_gap_reset_bars=1,
        episode_max_entries_per_run=1,
        rearm_same_side_only=1,
        rearm_cooldown_bars=1,
        rearm_max_bars_after_exit=8,
        rearm_gate_reset_frac=0.45,
        rearm_gate_refresh_frac=0.70,
        rearm_price_reset_frac=0.0004,
        rearm_after_trail=1,
        rearm_after_tp=1,
        rearm_after_sl=0,
        same_side_hold_enabled=0,
        same_side_hold_weak_enabled=1,
        same_side_hold_strong_ratio=1.0,
        same_side_hold_weak_ratio=0.82,
        same_side_hold_weak_min_progress_frac=0.35,
        same_side_hold_allow_pre_bep_weak=1,
        same_side_hold_pre_bep_max_bonus_bars=1,
        same_side_hold_bonus_bars_strong=2,
        same_side_hold_bonus_bars_weak=1,
        same_side_hold_max_extra_bars=4,
        same_side_hold_grace_after_bep_strong=1,
        same_side_hold_grace_after_bep_weak=0,
        same_side_hold_grace_after_unlock_strong=1,
        same_side_hold_grace_after_unlock_weak=0,
        support_strength_ratio_arr=np.zeros(8, dtype=np.float64),
        support_weak_eligible_mask=np.zeros(8, dtype=np.bool_),
        support_pass_mask=np.zeros(8, dtype=np.bool_),
        stop_equity=0.4,
        stop_dd=0.35,
        warmup_steps=0,
        integer_leverage=0,
        intrabar_mode=1,
    )
    # Warm regime-alpha numba path too, so trial #1 does not pay JIT compile cost.
    build_regime_alpha_exogenous(
        atr_arr=np.ones(8, dtype=np.float64) * 0.0010,
        range_arr=np.ones(8, dtype=np.float64) * 0.0011,
        vol_arr=np.zeros(8, dtype=np.float64),
        funding_arr=np.ones(8, dtype=np.float64) * 999.0,
        atr_high_th=0.0011,
        range_cut=0.0011,
        vol_low_th=-1e9,
        funding_soft_min=0.0,
        stress_lo=0.25,
        stress_hi=0.65,
        alpha_ema=0.15,
        alpha_hysteresis=0.03,
        w_atr=0.35,
        w_rng=0.20,
        w_vol=0.30,
        w_fund=0.15,
    )

def _bucket_count_dict(mask: np.ndarray, bucket_seg: np.ndarray, prefix: str) -> Dict[str, int]:
    mask = np.asarray(mask, dtype=bool)
    bucket_seg = np.asarray(bucket_seg, dtype=np.int8)
    return {
        prefix: int(np.sum(mask)),
        f"{prefix}_calm": int(np.sum(mask & (bucket_seg == 0))),
        f"{prefix}_mid": int(np.sum(mask & (bucket_seg == 1))),
        f"{prefix}_active": int(np.sum(mask & (bucket_seg == 2))),
    }

def _prepare_diag_summary(
    *,
    bucket_seg: np.ndarray,
    base_candidate_mask: np.ndarray,
    pass_qthr_mask: np.ndarray,
    cand_after_funding_mask: np.ndarray,
    vol_pass_mask: np.ndarray,
    atr_min_pass_mask: np.ndarray,
    range_min_pass_mask: np.ndarray,
    atr_high_pass_mask: np.ndarray,
    final_candidate_mask: np.ndarray,
) -> Dict[str, Any]:
    bucket_seg = np.asarray(bucket_seg, dtype=np.int8)
    out: Dict[str, Any] = {}
    out.update(_bucket_count_dict(base_candidate_mask, bucket_seg, "cand_total"))
    out.update(_bucket_count_dict(pass_qthr_mask, bucket_seg, "cand_after_qthr"))
    out.update(_bucket_count_dict(cand_after_funding_mask, bucket_seg, "cand_after_funding"))
    out.update(_bucket_count_dict(cand_after_funding_mask & vol_pass_mask, bucket_seg, "pass_vol"))
    out.update(_bucket_count_dict(cand_after_funding_mask & atr_min_pass_mask, bucket_seg, "pass_atr_min"))
    out.update(_bucket_count_dict(cand_after_funding_mask & range_min_pass_mask, bucket_seg, "pass_range_min"))
    out.update(_bucket_count_dict(cand_after_funding_mask & atr_high_pass_mask, bucket_seg, "pass_atr_high"))
    out.update(_bucket_count_dict(final_candidate_mask, bucket_seg, "final_candidates"))
    return out

def _ensure_normalized_single_cfg_once(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        try:
            if int(cfg.get("__normalized_single_v110__", 0) or 0) == 1:
                return dict(cfg)
        except Exception:
            pass

    out = normalize_single_config_from_any(dict(cfg) if isinstance(cfg, dict) else cfg)
    if isinstance(out, dict):
        out["__normalized_single_v110__"] = 1
    return out

def prepare_trial_context(
    open_px: np.ndarray,
    close_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    signals_by_h: Dict[int, np.ndarray],
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg_n = _ensure_normalized_single_cfg_once(cfg)
    if isinstance(cfg, dict):
        raw_same_side_hold_cfg = cfg.get("same_side_hold_cfg", {})
        if isinstance(raw_same_side_hold_cfg, dict) and raw_same_side_hold_cfg:
            cfg_n["same_side_hold_cfg"] = dict(raw_same_side_hold_cfg)
    avail = _available_horizons_from_signals(signals_by_h)
    gate_w = normalize_horizon_weights(cfg_n["gate_weights"], fallback={1:0.0,3:0.0,5:1.0,8:0.0,10:0.0}, available_horizons=avail)
    dir_w = normalize_horizon_weights(cfg_n["dir_weights"], fallback={1:0.0,3:0.0,5:1.0,8:0.0,10:0.0}, available_horizons=avail)
    range_rel_all = ((high_px - low_px) / np.maximum(close_px, 1e-12)).astype(np.float64, copy=False)

    risk_cfg = dict(cfg_n.get("risk_cfg", {}))
    atr_high_th = risk_cfg.get("atr_high_th", np.nan)
    try:
        atr_high_th_ = float(atr_high_th) if atr_high_th is not None else float("nan")
    except Exception:
        atr_high_th_ = float("nan")
    if np.isfinite(float(atr_high_th_)) and float(atr_high_th_) <= 0.0:
        atr_high_th_ = float("nan")
    if not np.isfinite(float(atr_high_th_)):
        atr_high_th_ = float(np.percentile(atr_rel, safe_float(risk_cfg.get("atr_percentile", 75.0), 75.0))) if len(atr_rel) > 10 else float("nan")

    range_med_all = float(np.median(range_rel_all)) if len(range_rel_all) else 1.0
    if range_med_all <= 0.0:
        range_med_all = 1.0
    range_cut_full = safe_float(cfg_n.get("range_entry_mult", 1.0), 1.0) * float(range_med_all)

    dyn_cfg = dict(cfg_n.get("dynamic_cfg", {}))
    regime_detect_cfg = dict(cfg_n.get("regime_detect_cfg", {}))
    regime_weight_cfg = dict(cfg_n.get("regime_weight_cfg", {}))
    regime_threshold_cfg = dict(cfg_n.get("regime_threshold_cfg", {}))
    regime_filter_cfg = dict(cfg_n.get("regime_filter_cfg", {}))
    regime_bundle = _build_regime_adaptive_signal_bundle_v2(
        signals_by_h=signals_by_h,
        available_horizons=avail,
        base_gate_weights=gate_w,
        base_dir_weights=dir_w,
        regime_detect_cfg=regime_detect_cfg,
        regime_weight_cfg=regime_weight_cfg,
        atr_arr=atr_rel,
        range_arr=range_rel_all,
        vol_arr=vol_z,
        funding_arr=minutes_to_next_funding,
        atr_high_th=float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        range_cut=float(range_cut_full),
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_soft_min=max(safe_float(dyn_cfg.get("funding_soft_min", 0.0), 0.0), safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0)),
        detect_required=bool(int(regime_threshold_cfg.get("enabled", 0)) != 0 or int(regime_filter_cfg.get("enabled", 0)) != 0),
    )

    return {
        "cfg": cfg_n,
        "available_horizons": tuple(avail),
        "gate_signal_all": regime_bundle["gate_signal_all"],
        "dir_signal_all": regime_bundle["dir_signal_all"],
        "regime_alpha_all": regime_bundle["alpha_arr"],
        "regime_bucket_all": regime_bundle["bucket_arr"],
        "regime_stress_all": regime_bundle["stress_arr"],
        "regime_profiles": {
            "gate_calm": regime_bundle["gate_calm_profile"],
            "gate_active": regime_bundle["gate_active_profile"],
            "dir_calm": regime_bundle["dir_calm_profile"],
            "dir_active": regime_bundle["dir_active_profile"],
        },
        "regime_detect_cfg": regime_detect_cfg,
        "regime_weight_cfg": regime_weight_cfg,
        "regime_threshold_cfg": regime_threshold_cfg,
        "regime_filter_cfg": regime_filter_cfg,
        "open_px": open_px,
        "close_px": close_px,
        "high_px": high_px,
        "low_px": low_px,
        "ready": ready,
        "vol_z": vol_z,
        "atr_rel": atr_rel,
        "minutes_to_next_funding": minutes_to_next_funding,
        "range_rel_all": range_rel_all,
        "signals_by_h": signals_by_h,
    }

def _prepare_single_segment_inputs_from_context_base(
    ctx: Dict[str, Any],
    seg_start: int,
    seg_end: int,
    entry_q_lookback: int,
    entry_q_min_ready: int,
    *,
    include_detailed: bool,
) -> Dict[str, Any]:
    cfg = ctx["cfg"]
    open_px = ctx["open_px"]
    close_px = ctx["close_px"]
    high_px = ctx["high_px"]
    low_px = ctx["low_px"]
    ready = ctx["ready"]
    vol_z = ctx["vol_z"]
    atr_rel = ctx["atr_rel"]
    minutes_to_next_funding = ctx["minutes_to_next_funding"]
    gate_signal_all = ctx["gate_signal_all"]
    dir_signal_all = ctx["dir_signal_all"]
    range_rel_all = ctx["range_rel_all"]

    seg = slice(int(seg_start), int(seg_end))
    hist_end = int(seg_start)
    hist_start = max(0, hist_end - int(entry_q_lookback))

    regime_bucket_all = ctx.get("regime_bucket_all")
    threshold_pack = build_bucketed_entry_threshold_pack(
        gate_signal_all=gate_signal_all,
        ready=ready,
        bucket_arr=regime_bucket_all,
        hist_start=hist_start,
        hist_end=hist_end,
        seg_start=int(seg_start),
        seg_end=int(seg_end),
        q_entry=safe_float(cfg.get("q_entry", 0.85), 0.85),
        entry_th_floor=safe_float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)), 0.0),
        entry_q_min_ready=int(entry_q_min_ready),
        regime_threshold_cfg=cfg.get("regime_threshold_cfg", {}),
    )
    thr_entry = float(threshold_pack["thr_entry_global"])

    risk_cfg = dict(cfg.get("risk_cfg", {}))
    atr_high_th = risk_cfg.get("atr_high_th", np.nan)
    try:
        atr_high_th_ = float(atr_high_th) if atr_high_th is not None else float("nan")
    except Exception:
        atr_high_th_ = float("nan")
    if np.isfinite(float(atr_high_th_)) and float(atr_high_th_) <= 0.0:
        atr_hist = atr_rel[hist_start:hist_end]
        atr_high_th_ = float(np.percentile(atr_hist, safe_float(risk_cfg.get("atr_percentile", 75.0), 75.0))) if atr_hist.size > 10 else 1e9

    range_hist = range_rel_all[hist_start:hist_end]
    range_med = float(np.median(range_hist)) if range_hist.size > 0 else 1.0
    if range_med <= 0.0:
        range_med = 1.0

    gate_strength_seg = gate_signal_all[seg].astype(np.float64, copy=False)
    dir_signal_seg = dir_signal_all[seg].astype(np.float64, copy=False)
    ready_seg = ready[seg].astype(bool, copy=False)
    atr_seg = atr_rel[seg].astype(np.float64, copy=False)
    vol_seg = vol_z[seg].astype(np.float64, copy=False)
    funding_seg = minutes_to_next_funding[seg].astype(np.float64, copy=False)
    range_seg = range_rel_all[seg].astype(np.float64, copy=False)
    entry_threshold_base_seg = np.asarray(threshold_pack["entry_threshold_base_seg"], dtype=np.float64)
    entry_q_used_seg = np.asarray(threshold_pack["entry_q_used_seg"], dtype=np.float64)
    entry_floor_used_seg = np.asarray(threshold_pack["entry_floor_used_seg"], dtype=np.float64)
    bucket_seg = np.asarray(threshold_pack.get("bucket_seg", np.full(len(gate_strength_seg), -1, dtype=np.int8)), dtype=np.int8)

    filter_pack = build_bucketed_filter_pack(
        bucket_arr=regime_bucket_all,
        seg_start=int(seg_start),
        seg_end=int(seg_end),
        regime_filter_cfg=cfg.get("regime_filter_cfg", {}),
        base_vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        base_atr_entry_mult=safe_float(cfg.get("atr_entry_mult", 1.0), 1.0),
        base_range_entry_mult=safe_float(cfg.get("range_entry_mult", 1.0), 1.0),
    )
    vol_low_th_used_seg = np.asarray(filter_pack["vol_low_th_arr"], dtype=np.float64)
    atr_entry_mult_used_seg = np.asarray(filter_pack["atr_entry_mult_arr"], dtype=np.float64)
    range_entry_mult_used_seg = np.asarray(filter_pack["range_entry_mult_arr"], dtype=np.float64)
    filter_bucket_seg = np.asarray(filter_pack["filter_bucket_seg"], dtype=np.int8)

    range_cut = safe_float(cfg.get("range_entry_mult", 1.0), 1.0) * float(range_med)
    dyn_gate_mult_arr, dyn_lev_scale_arr, dyn_bep_scale_arr, dyn_trail_scale_arr, dyn_sl_scale_arr, dyn_softsl_relax_arr, dyn_stress_arr = build_dynamic_arrays(
        dynamic_cfg=cfg["dynamic_cfg"],
        gate_strength_seg=gate_strength_seg,
        thr_entry=float(thr_entry),
        atr_seg=atr_seg,
        atr_high_th=float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        range_seg=range_seg,
        range_cut=float(range_cut),
        vol_seg=vol_seg,
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_seg=funding_seg,
        thr_entry_arr=entry_threshold_base_seg,
    )
    entry_threshold_eff_seg = entry_threshold_base_seg * dyn_gate_mult_arr

    atr_med = float(np.median(atr_seg)) if atr_seg.size else 1.0
    if atr_med <= 0.0:
        atr_med = 1.0

    fee_roundtrip = 2.0 * (safe_float(cfg.get("cost_per_side", 0.0), 0.0) + 0.0)
    # use zero here; actual runtime entry masks below use cost_per_side passed in evaluate-prepared stage only approximately via fee-multiple scalars.
    # To keep behavior consistent across prepare/autotune/backtest, use the taker+slip defaults from config if present when available.
    fee_roundtrip = 2.0 * (safe_float(cfg.get("cost_per_side", cfg.get("taker_fee_per_side", 0.0)), 0.0) + safe_float(cfg.get("slip_per_side", 0.0), 0.0))
    if fee_roundtrip <= 0.0:
        fee_roundtrip = 2.0 * 0.00085

    low_vol_enabled = int(cfg.get("low_vol_filter", 0)) != 0
    use_atr_scaling = int(cfg.get("use_atr_scaling", 1)) != 0
    risk_entry_mode = int(cfg.get("risk_entry_mode", 0))
    funding_near_min = safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0)

    base_candidate_mask = ready_seg & np.isfinite(gate_strength_seg) & np.isfinite(dir_signal_seg) & (gate_strength_seg > 0.0) & (dir_signal_seg != 0.0)
    pass_qthr_mask = base_candidate_mask & np.isfinite(entry_threshold_eff_seg) & (gate_strength_seg >= entry_threshold_eff_seg)
    funding_pass_mask = funding_seg >= float(funding_near_min)
    cand_after_funding_mask = pass_qthr_mask & funding_pass_mask
    if low_vol_enabled:
        vol_pass_mask = vol_seg > vol_low_th_used_seg
        atr_min_pass_mask = atr_seg >= (atr_entry_mult_used_seg * fee_roundtrip)
        range_min_pass_mask = range_seg >= (range_entry_mult_used_seg * fee_roundtrip)
    else:
        vol_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        atr_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        range_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if use_atr_scaling and np.isfinite(float(atr_high_th_)):
        atr_high_pass_mask = atr_seg <= float(atr_high_th_)
    else:
        atr_high_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if int(risk_entry_mode) == 3:
        risk_mode3_pass_mask = (atr_seg <= (atr_med * atr_entry_mult_used_seg)) & (range_seg <= (range_med * range_entry_mult_used_seg))
    else:
        risk_mode3_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    filter_pass_mask = vol_pass_mask & atr_min_pass_mask & range_min_pass_mask & atr_high_pass_mask & risk_mode3_pass_mask
    final_candidate_mask = cand_after_funding_mask & filter_pass_mask

    gate_strength_used = gate_strength_seg * final_candidate_mask.astype(np.float64)

    dyn = cfg["dynamic_cfg"]
    prog = cfg.get("progress_protect_cfg", {}) or {}
    fast_common = dict(
        open_=open_px[seg].astype(np.float64, copy=False),
        close=close_px[seg].astype(np.float64, copy=False),
        high=high_px[seg].astype(np.float64, copy=False),
        low=low_px[seg].astype(np.float64, copy=False),
        gate_strength=gate_strength_used.astype(np.float64, copy=False),
        dir_signal=dir_signal_seg.astype(np.float64, copy=False),
        ready=ready_seg,
        vol_z=vol_seg,
        atr_rel=atr_seg,
        range_rel=range_seg,
        minutes_to_next_funding=funding_seg,
        atr_high_th=float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        atr_med=float(atr_med),
        range_med=float(range_med),
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_near_min=0.0,
        risk_lev_cap=safe_float(risk_cfg.get("risk_lev_cap", 12.0), 12.0),
        base_leverage=safe_float(cfg.get("leverage", 10.0), 10.0),
        fee_tp_mult=safe_float(cfg.get("fee_tp_mult", 1.0), 1.0),
        bep_arm_fee_mult=safe_float(cfg.get("bep_arm_fee_mult", cfg.get("fee_bep_mult", 0.2)), 0.2),
        bep_stop_fee_mult=safe_float(cfg.get("bep_stop_fee_mult", 1.0), 1.0),
        bep_stop_mode_code=_bep_stop_mode_code(str(cfg.get("bep_stop_mode", "maker_be"))),
        atr_entry_mult=safe_float(cfg.get("atr_entry_mult", 1.0), 1.0),
        range_entry_mult=safe_float(cfg.get("range_entry_mult", 1.0), 1.0),
        low_vol_filter=0,
        trail_after_bep=safe_int(cfg.get("trail_after_bep", 1), 1),
        risk_entry_mode=0,
        use_atr_scaling=safe_int(cfg.get("use_atr_scaling", 1), 1),
        lev_mult=safe_float(cfg.get("lev_mult", 1.0), 1.0),
        TP=safe_float(cfg.get("TP", 0.0), 0.0),
        SL=safe_float(cfg.get("SL", 0.0), 0.0),
        bep_arm_base=safe_float(cfg.get("BEP_ARM", cfg.get("BEP", 0.0)), 0.0),
        trailing=safe_float(cfg.get("trailing", 0.0), 0.0),
        min_hold_bars=safe_int(cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0)), 0),
        min_hold_trail_bars=safe_int(cfg.get("min_hold_trail_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))), 0),
        min_hold_soft_sl_bars=safe_int(cfg.get("min_hold_soft_sl_bars", cfg.get("min_hold_tp_bars", cfg.get("min_hold_bars", 0))), 0),
        max_hold_bars=safe_int(cfg.get("max_hold_bars", 0), 0),
        dyn_lev_scale_arr=dyn_lev_scale_arr.astype(np.float64, copy=False),
        dyn_bep_scale_arr=dyn_bep_scale_arr.astype(np.float64, copy=False),
        dyn_trail_scale_arr=dyn_trail_scale_arr.astype(np.float64, copy=False),
        dyn_sl_scale_arr=dyn_sl_scale_arr.astype(np.float64, copy=False),
        dyn_softsl_relax_arr=dyn_softsl_relax_arr.astype(np.int64, copy=False),
        dyn_gate_mult_arr=dyn_gate_mult_arr.astype(np.float64, copy=False),
        dyn_stress_arr=dyn_stress_arr.astype(np.float64, copy=False),
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
        early_softsl_enabled=safe_int(prog.get("early_softsl_enabled", 0), 0),
        early_softsl_min_hold=safe_int(prog.get("early_softsl_min_hold", 2), 2),
        early_softsl_progress_frac=safe_float(prog.get("early_softsl_progress_frac", 0.5), 0.5),
        early_trail_enabled=safe_int(prog.get("early_trail_enabled", 0), 0),
        early_trail_min_hold=safe_int(prog.get("early_trail_min_hold", 3), 3),
        early_trail_progress_frac=safe_float(prog.get("early_trail_progress_frac", 0.85), 0.85),
        early_trail_ref_updates_min=safe_int(prog.get("early_trail_ref_updates_min", 1), 1),
        stop_equity=safe_float(cfg.get("tail_cfg", {}).get("stop_equity", 0.0), 0.0),
        stop_dd=safe_float(cfg.get("tail_cfg", {}).get("stop_dd", 1.0), 1.0),
        warmup_steps=safe_int(cfg.get("tail_cfg", {}).get("warmup_steps", 0), 0),
        integer_leverage=safe_int(cfg.get("integer_leverage", 0), 0),
    )

    regime_alpha_seg = np.asarray(ctx.get("regime_alpha_all", np.zeros(seg.stop - seg.start, dtype=np.float64))[seg], dtype=np.float64) if "regime_alpha_all" in ctx else np.zeros(seg.stop - seg.start, dtype=np.float64)
    regime_bucket_seg = np.asarray(ctx.get("regime_bucket_all", np.zeros(seg.stop - seg.start, dtype=np.int8))[seg], dtype=np.int8) if "regime_bucket_all" in ctx else np.zeros(seg.stop - seg.start, dtype=np.int8)
    diag_summary = _prepare_diag_summary(
        bucket_seg=bucket_seg,
        base_candidate_mask=base_candidate_mask,
        pass_qthr_mask=pass_qthr_mask,
        cand_after_funding_mask=cand_after_funding_mask,
        vol_pass_mask=vol_pass_mask,
        atr_min_pass_mask=atr_min_pass_mask,
        range_min_pass_mask=range_min_pass_mask,
        atr_high_pass_mask=atr_high_pass_mask,
        final_candidate_mask=final_candidate_mask,
    )
    out = {
        "cfg": cfg,
        "seg_start": int(seg_start),
        "seg_end": int(seg_end),
        "thr_entry": float(thr_entry),
        "atr_high_th": float(atr_high_th_) if np.isfinite(float(atr_high_th_)) else float("nan"),
        "fast_common": fast_common,
        "regime_alpha_mean": float(np.mean(regime_alpha_seg)) if regime_alpha_seg.size else 0.0,
        "regime_alpha_p50": float(np.quantile(regime_alpha_seg, 0.5)) if regime_alpha_seg.size else 0.0,
        "regime_active_frac": float(np.mean(regime_bucket_seg == 2)) if regime_bucket_seg.size else 0.0,
        "regime_calm_frac": float(np.mean(regime_bucket_seg == 0)) if regime_bucket_seg.size else 0.0,
        "threshold_enabled": int(threshold_pack.get("threshold_enabled", 0)),
        "threshold_bucket_min_ready": int(threshold_pack.get("bucket_min_ready", 0)),
        "threshold_bucket_fallback_global": int(threshold_pack.get("bucket_fallback_global", 1)),
        "filter_enabled": int(filter_pack.get("enabled", 0)),
        "filter_use_vol_split": int(filter_pack.get("use_vol_split", 1)),
        "filter_use_entry_mult_split": int(filter_pack.get("use_entry_mult_split", 1)),
        "entry_q_used_mean": float(np.mean(entry_q_used_seg)) if entry_q_used_seg.size else float(safe_float(cfg.get("q_entry", 0.85), 0.85)),
        "entry_q_used_p50": float(np.quantile(entry_q_used_seg, 0.5)) if entry_q_used_seg.size else float(safe_float(cfg.get("q_entry", 0.85), 0.85)),
        "entry_threshold_base_mean": float(np.mean(entry_threshold_base_seg)) if entry_threshold_base_seg.size else float(thr_entry),
        "entry_vol_low_th_used_mean": float(np.mean(vol_low_th_used_seg)) if vol_low_th_used_seg.size else float(safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9)),
        "entry_atr_entry_mult_used_mean": float(np.mean(atr_entry_mult_used_seg)) if atr_entry_mult_used_seg.size else float(safe_float(cfg.get("atr_entry_mult", 1.0), 1.0)),
        "entry_range_entry_mult_used_mean": float(np.mean(range_entry_mult_used_seg)) if range_entry_mult_used_seg.size else float(safe_float(cfg.get("range_entry_mult", 1.0), 1.0)),
        "entry_threshold_eff_seg": entry_threshold_eff_seg,
        "entry_threshold_base_seg": entry_threshold_base_seg,
        "entry_q_used_seg": entry_q_used_seg,
        "entry_floor_used_seg": entry_floor_used_seg,
        "vol_low_th_used_seg": vol_low_th_used_seg,
        "atr_entry_mult_used_seg": atr_entry_mult_used_seg,
        "range_entry_mult_used_seg": range_entry_mult_used_seg,
        "filter_bucket_seg": filter_bucket_seg,
        "diag_summary": diag_summary,
    }
    if include_detailed:
        detailed_kwargs = dict(fast_common)
        detailed_kwargs.pop("range_rel", None)
        detailed_kwargs.pop("atr_med", None)
        detailed_kwargs.pop("range_med", None)
        detailed_kwargs.pop("bep_stop_mode_code", None)
        detailed_kwargs["bep_stop_mode"] = str(cfg.get("bep_stop_mode", "maker_be"))
        detailed_kwargs["cost_per_side"] = 0.0
        detailed_kwargs["maker_fee_per_side"] = 0.0
        detailed_kwargs["slip_per_side"] = 0.0
        detailed_kwargs["seg_start"] = int(seg_start)
        regime_alpha_all = ctx.get("regime_alpha_all")
        regime_bucket_all = ctx.get("regime_bucket_all")
        if regime_alpha_all is not None:
            detailed_kwargs["regime_alpha_arr"] = np.asarray(regime_alpha_all[seg], dtype=np.float64)
        if regime_bucket_all is not None:
            detailed_kwargs["regime_bucket_arr"] = np.asarray(regime_bucket_all[seg], dtype=np.int64)
        out["detailed_kwargs"] = detailed_kwargs
    return out




def _inject_v33_structural_kwargs(prepared: Dict[str, Any]) -> Dict[str, Any]:
    # cfg is already normalized once in prepare_trial_context().
    # Do NOT normalize again per-segment; that slows autotune.
    cfg = dict(prepared.get("cfg", {}) or {})

    tpw = dict(cfg.get("tp_window_cfg", {}) or {})
    epcfg = dict(cfg.get("entry_episode_cfg", {}) or {})
    ssh = dict(cfg.get("same_side_hold_cfg", {}) or {})

    structural_kwargs = {
        "tp_window_enabled": int(tpw.get("enabled", 0)),
        "tp_window_progress_frac_arm": float(tpw.get("progress_frac_arm", 0.70)),
        "tp_window_extend_bars": int(tpw.get("extend_bars", 0)),
        "tp_window_block_early_trail": int(tpw.get("block_early_trail", 1)),
        "tp_window_block_early_soft_sl": int(tpw.get("block_early_soft_sl", 1)),
        "tp_window_floor_trail_hold_to_tp": int(tpw.get("floor_trail_hold_to_tp", 1)),
        "tp_window_floor_soft_sl_hold_to_tp": int(tpw.get("floor_soft_sl_hold_to_tp", 1)),
        "tp_window_suspend_post_bep_shield_before_tp": int(tpw.get("suspend_post_bep_shield_before_tp", 1)),
        "tp_window_expire_on_pullback_frac": float(tpw.get("expire_on_pullback_frac", 0.35)),
        "entry_episode_enabled": int(epcfg.get("enabled", 0)),
        "rearm_enabled": int(epcfg.get("rearm_enabled", 0)),
        "run_gap_reset_bars": int(epcfg.get("run_gap_reset_bars", 1)),
        "episode_max_entries_per_run": int(epcfg.get("episode_max_entries_per_run", 1)),
        "rearm_same_side_only": int(epcfg.get("rearm_same_side_only", 1)),
        "rearm_cooldown_bars": int(epcfg.get("rearm_cooldown_bars", 1)),
        "rearm_max_bars_after_exit": int(epcfg.get("rearm_max_bars_after_exit", 8)),
        "rearm_gate_reset_frac": float(epcfg.get("rearm_gate_reset_frac", 0.45)),
        "rearm_gate_refresh_frac": float(epcfg.get("rearm_gate_refresh_frac", 0.70)),
        "rearm_price_reset_frac": float(epcfg.get("rearm_price_reset_frac", 0.0004)),
        "rearm_after_trail": int(epcfg.get("rearm_after_trail", 1)),
        "rearm_after_tp": int(epcfg.get("rearm_after_tp", 1)),
        "rearm_after_sl": int(epcfg.get("rearm_after_sl", 0)),
    }

    same_side_kwargs = {
        "same_side_hold_enabled": int(ssh.get("enabled", 0)),
        "same_side_hold_weak_enabled": int(ssh.get("weak_enabled", 1)),
        "same_side_hold_strong_ratio": float(ssh.get("strong_ratio", 1.0)),
        "same_side_hold_weak_ratio": float(ssh.get("weak_ratio", 0.82)),
        "same_side_hold_weak_min_progress_frac": float(ssh.get("weak_min_progress_frac", 0.35)),
        "same_side_hold_allow_pre_bep_weak": int(ssh.get("allow_pre_bep_weak", 1)),
        "same_side_hold_pre_bep_max_bonus_bars": int(ssh.get("pre_bep_max_bonus_bars", 1)),
        "same_side_hold_bonus_bars_strong": int(ssh.get("strong_bonus_bars", 2)),
        "same_side_hold_bonus_bars_weak": int(ssh.get("weak_bonus_bars", 1)),
        "same_side_hold_max_extra_bars": int(ssh.get("max_extra_bars", 4)),
        "same_side_hold_grace_after_bep_strong": int(ssh.get("strong_grace_after_bep_bars", 1)),
        "same_side_hold_grace_after_bep_weak": int(ssh.get("weak_grace_after_bep_bars", 0)),
        "same_side_hold_grace_after_unlock_strong": int(ssh.get("strong_grace_after_unlock_bars", 1)),
        "same_side_hold_grace_after_unlock_weak": int(ssh.get("weak_grace_after_unlock_bars", 0)),
    }

    prepared["v33_structural_kwargs"] = structural_kwargs
    prepared["same_side_hold_cfg"] = same_side_kwargs
    prepared["has_v33_structural_features"] = False

    if "fast_common" in prepared:
        prepared["fast_common"].update({
            "tp_window_enabled": int(structural_kwargs["tp_window_enabled"]),
            "tp_window_progress_frac_arm": float(structural_kwargs["tp_window_progress_frac_arm"]),
            "tp_window_extend_bars": int(structural_kwargs["tp_window_extend_bars"]),
            "tp_window_block_early_trail": int(structural_kwargs["tp_window_block_early_trail"]),
            "tp_window_block_early_soft_sl": int(structural_kwargs["tp_window_block_early_soft_sl"]),
            "tp_window_floor_trail_hold_to_tp": int(structural_kwargs["tp_window_floor_trail_hold_to_tp"]),
            "tp_window_floor_soft_sl_hold_to_tp": int(structural_kwargs["tp_window_floor_soft_sl_hold_to_tp"]),
            "tp_window_suspend_post_bep_shield_before_tp": int(structural_kwargs["tp_window_suspend_post_bep_shield_before_tp"]),
            "tp_window_expire_on_pullback_frac": float(structural_kwargs["tp_window_expire_on_pullback_frac"]),
            "entry_episode_enabled": int(structural_kwargs["entry_episode_enabled"]),
            "rearm_enabled": int(structural_kwargs["rearm_enabled"]),
            "run_gap_reset_bars": int(structural_kwargs["run_gap_reset_bars"]),
            "episode_max_entries_per_run": int(structural_kwargs["episode_max_entries_per_run"]),
            "rearm_same_side_only": int(structural_kwargs["rearm_same_side_only"]),
            "rearm_cooldown_bars": int(structural_kwargs["rearm_cooldown_bars"]),
            "rearm_max_bars_after_exit": int(structural_kwargs["rearm_max_bars_after_exit"]),
            "rearm_gate_reset_frac": float(structural_kwargs["rearm_gate_reset_frac"]),
            "rearm_gate_refresh_frac": float(structural_kwargs["rearm_gate_refresh_frac"]),
            "rearm_price_reset_frac": float(structural_kwargs["rearm_price_reset_frac"]),
            "rearm_after_trail": int(structural_kwargs["rearm_after_trail"]),
            "rearm_after_tp": int(structural_kwargs["rearm_after_tp"]),
            "rearm_after_sl": int(structural_kwargs["rearm_after_sl"]),
        })
        prepared["fast_common"].update(same_side_kwargs)

    if "detailed_kwargs" in prepared:
        prepared["detailed_kwargs"].update(structural_kwargs)
        prepared["detailed_kwargs"].update(same_side_kwargs)

    return prepared

# --- runner-alignment audit helpers (detailed-path only) ---
def _runner_align_cfg(prepared: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(prepared.get("cfg", {}) or {})
    raw = prepared.get("runner_alignment_cfg", cfg.get("runner_alignment_cfg", {})) or {}
    return dict(raw if isinstance(raw, dict) else {})

def _annotate_trade_logs_runner_alignment(
    trade_logs: Any,
    prepared: Dict[str, Any],
    cost_per_side: float,
    slip_per_side: float,
    maker_fee_per_side: float,
) -> Tuple[list, Dict[str, Any]]:
    logs = list(trade_logs or [])
    align = _runner_align_cfg(prepared)
    meta = {
        "runner_align_enabled": int(bool(int(align.get("enabled", 0) or 0))),
        "runner_align_profit_floor_needed": 0,
        "runner_align_profit_floor_violation": 0,
        "runner_align_thesis_hits": 0,
    }
    if (not logs) or (meta["runner_align_enabled"] == 0):
        return logs, meta

    cfg = dict(prepared.get("cfg", {}) or {})
    profit_on = int(bool(int(align.get("profit_floor_enabled", 0) or 0)))
    thesis_on = int(bool(int(align.get("thesis_monitor_enabled", 0) or 0)))

    taker_fee_side = float(cost_per_side) + float(slip_per_side)
    maker_exit_fee = float(maker_fee_per_side) if float(maker_fee_per_side) > 0.0 else taker_fee_side
    econ_be_fee = taker_fee_side + maker_exit_fee
    bep_stop_fee_mult = float(cfg.get("bep_stop_fee_mult", 1.0) or 1.0)

    out: list = []
    for row0 in logs:
        row = dict(row0 or {})
        entry_price = safe_float(row.get("entry_price", np.nan), np.nan)
        exit_price = safe_float(row.get("exit_price", np.nan), np.nan)
        reason = str(row.get("exit_reason", "") or "").upper().strip()

        side = int(row.get("side", 0) or 0)
        if side == 0:
            entry_side = str(row.get("entry_side", "") or "").upper().strip()
            side = 1 if entry_side == "BUY" else (-1 if entry_side == "SELL" else 0)

        row["runner_align_enabled"] = int(meta["runner_align_enabled"])
        row["runner_align_profit_floor_enabled"] = int(profit_on)
        row["runner_align_thesis_monitor_enabled"] = int(thesis_on)
        row["profit_floor_price"] = np.nan
        row["profit_floor_violation"] = 0
        row["thesis_monitor_hit"] = 0

        if profit_on and side != 0 and np.isfinite(entry_price):
            floor_px = entry_price * (1.0 + float(side) * econ_be_fee * bep_stop_fee_mult)
            row["profit_floor_price"] = float(floor_px)

            if reason in {"TP", "TRAIL"}:
                meta["runner_align_profit_floor_needed"] += 1
                violated = (
                    (side > 0 and np.isfinite(exit_price) and exit_price < floor_px - 1e-12) or
                    (side < 0 and np.isfinite(exit_price) and exit_price > floor_px + 1e-12)
                )
                if violated:
                    row["profit_floor_violation"] = 1
                    meta["runner_align_profit_floor_violation"] += 1

        if thesis_on:
            subreason = str(
                row.get("thesis_subreason", row.get("thesis_reason", row.get("risk_close_subreason", "")))
                or ""
            ).strip()
            if subreason:
                row["thesis_monitor_hit"] = 1
                meta["runner_align_thesis_hits"] += 1

        out.append(row)

    return out, meta


def prepare_single_segment_fast_inputs_from_context(
    ctx: Dict[str, Any],
    seg_start: int,
    seg_end: int,
    entry_q_lookback: int,
    entry_q_min_ready: int,
) -> Dict[str, Any]:
    prepared = _prepare_single_segment_inputs_from_context_base(
        ctx=ctx,
        seg_start=seg_start,
        seg_end=seg_end,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
        include_detailed=False,
    )
    seg = slice(int(seg_start), int(seg_end))
    cfg = prepared.get("cfg", ctx.get("cfg", {}))
    risk_cfg = dict(cfg.get("risk_cfg", {}))
    prepared["raw_gate_strength_seg"] = np.asarray(ctx.get("gate_signal_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_dir_signal_seg"] = np.asarray(ctx.get("dir_signal_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_ready_seg"] = np.asarray(ctx.get("ready", np.zeros(seg.stop-seg.start, dtype=bool))[seg], dtype=bool)
    prepared["raw_funding_seg"] = np.asarray(ctx.get("minutes_to_next_funding", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_vol_seg"] = np.asarray(ctx.get("vol_z", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_atr_seg"] = np.asarray(ctx.get("atr_rel", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_range_seg"] = np.asarray(ctx.get("range_rel_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["funding_near_min"] = float(safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0))
    prepared["low_vol_enabled"] = int(cfg.get("low_vol_filter", 0))
    prepared["use_atr_scaling_flag"] = int(cfg.get("use_atr_scaling", 1))
    prepared["risk_entry_mode_flag"] = int(cfg.get("risk_entry_mode", 0))
    prepared["atr_high_th_value"] = float(prepared.get("atr_high_th", np.nan)) if np.isfinite(float(prepared.get("atr_high_th", np.nan))) else float("nan")
    prepared["atr_med"] = float(prepared.get("fast_common", {}).get("atr_med", 1.0))
    prepared["range_med"] = float(prepared.get("fast_common", {}).get("range_med", 1.0))
    return _inject_v33_structural_kwargs(prepared)


def prepare_single_segment_inputs_from_context(
    ctx: Dict[str, Any],
    seg_start: int,
    seg_end: int,
    entry_q_lookback: int,
    entry_q_min_ready: int,
) -> Dict[str, Any]:
    prepared = _prepare_single_segment_inputs_from_context_base(
        ctx=ctx,
        seg_start=seg_start,
        seg_end=seg_end,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
        include_detailed=True,
    )
    seg = slice(int(seg_start), int(seg_end))
    cfg = prepared.get("cfg", ctx.get("cfg", {}))
    risk_cfg = dict(cfg.get("risk_cfg", {}))
    prepared["raw_gate_strength_seg"] = np.asarray(ctx.get("gate_signal_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_dir_signal_seg"] = np.asarray(ctx.get("dir_signal_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_ready_seg"] = np.asarray(ctx.get("ready", np.zeros(seg.stop-seg.start, dtype=bool))[seg], dtype=bool)
    prepared["raw_funding_seg"] = np.asarray(ctx.get("minutes_to_next_funding", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_vol_seg"] = np.asarray(ctx.get("vol_z", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_atr_seg"] = np.asarray(ctx.get("atr_rel", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["raw_range_seg"] = np.asarray(ctx.get("range_rel_all", np.zeros(seg.stop-seg.start, dtype=np.float64))[seg], dtype=np.float64)
    prepared["funding_near_min"] = float(safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0))
    prepared["low_vol_enabled"] = int(cfg.get("low_vol_filter", 0))
    prepared["use_atr_scaling_flag"] = int(cfg.get("use_atr_scaling", 1))
    prepared["risk_entry_mode_flag"] = int(cfg.get("risk_entry_mode", 0))
    prepared["atr_high_th_value"] = float(prepared.get("atr_high_th", np.nan)) if np.isfinite(float(prepared.get("atr_high_th", np.nan))) else float("nan")
    prepared["atr_med"] = float(prepared.get("fast_common", {}).get("atr_med", 1.0))
    prepared["range_med"] = float(prepared.get("fast_common", {}).get("range_med", 1.0))
    prepared["runner_alignment_cfg"] = dict(cfg.get("runner_alignment_cfg", {}) or {})
    return _inject_v33_structural_kwargs(prepared)


def prepare_single_segment_inputs(
    seg_start: int,
    seg_end: int,
    open_px: np.ndarray,
    close_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    signals_by_h: Dict[int, np.ndarray],
    ready: np.ndarray,
    vol_z: np.ndarray,
    atr_rel: np.ndarray,
    minutes_to_next_funding: np.ndarray,
    cfg: Dict[str, Any],
    entry_q_lookback: int,
    entry_q_min_ready: int,
) -> Dict[str, Any]:
    ctx = prepare_trial_context(
        open_px=open_px,
        close_px=close_px,
        high_px=high_px,
        low_px=low_px,
        signals_by_h=signals_by_h,
        ready=ready,
        vol_z=vol_z,
        atr_rel=atr_rel,
        minutes_to_next_funding=minutes_to_next_funding,
        cfg=cfg,
    )
    return prepare_single_segment_inputs_from_context(
        ctx=ctx,
        seg_start=seg_start,
        seg_end=seg_end,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
    )


def _use_python_structural_fallback(prepared: Dict[str, Any]) -> bool:
    return False


def _detailed_result_to_segment_result(
    res: Dict[str, Any],
    chosen_score: float,
    thr_entry: float,
    atr_high_th: float,
    score_cfg: Dict[str, Any],
    diag_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trades = int(res["trades"])
    wins = int(res["wins"])
    long_trades = int(res["long_trades"])
    short_trades = int(res["short_trades"])
    short_share = float(short_trades / trades) if trades > 0 else 0.0
    side_penalty = side_balance_penalty_component(
        long_trades,
        short_trades,
        int(score_cfg.get("min_short_trades_global", 0)),
        float(score_cfg.get("min_short_share_global", 0.0)),
        float(score_cfg.get("side_balance_penalty_k", 0.0)),
    )
    maxh_ratio = (float(res["maxh_cnt"]) / float(trades)) if trades > 0 else 0.0
    out = {
        "net_ret": float(res["net_ret"]),
        "mdd_net": float(res["mdd"]),
        "winrate_net": float(wins / trades) if trades > 0 else 0.0,
        "trades": trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "short_share": float(short_share),
        "trade_logs": list(res.get("trade_logs", [])),
        "tail": int(res["tail_hit"]),
        "thr_entry": float(thr_entry),
        "atr_high_th": float(atr_high_th),
        "exit_cnt": res["exit_cnt"],
        "exit_gross_sum": res["exit_gross_sum"],
        "exit_fee_sum": res["exit_fee_sum"],
        "exit_net_sum": res["exit_net_sum"],
        "trail_before_bep": int(res["trail_before_bep"]),
        "trail_after_bep": int(res["trail_after_bep"]),
        "bep_armed_trades": int(res["bep_armed_trades"]),
        "ref_updates": int(res["ref_updates"]),
        "maxh_cnt": int(res["maxh_cnt"]),
        "score": float(chosen_score),
        "side_penalty": float(side_penalty),
        "maxh_ratio": float(maxh_ratio),
        "tp_window_armed_trades": int(res.get("tp_window_armed_trades", 0)),
        "tp_window_live_bars_total": int(res.get("tp_window_live_bars_total", 0)),
        "tp_window_blocked_early_trail": int(res.get("tp_window_blocked_early_trail", 0)),
        "tp_window_blocked_softsl": int(res.get("tp_window_blocked_softsl", 0)),
        "rearm_entries": int(res.get("rearm_entries", 0)),
        "rearm_entries_after_trail": int(res.get("rearm_entries_after_trail", 0)),
        "rearm_entries_after_tp": int(res.get("rearm_entries_after_tp", 0)),
        "rearm_entries_after_sl": int(res.get("rearm_entries_after_sl", 0)),
        "same_side_hold_events": int(res.get("same_side_hold_events", 0)),
        "same_side_hold_strong_events": int(res.get("same_side_hold_strong_events", 0)),
        "same_side_hold_weak_events": int(res.get("same_side_hold_weak_events", 0)),
    }
    if diag_summary:
        out.update(diag_summary)
    out["final_entries"] = int(out.get("trades", 0))
    return out


def _run_detailed_with_mode(prepared: Dict[str, Any], intrabar_mode: int, cost_per_side: float, slip_per_side: float, maker_fee_per_side: float, final_candidate_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    detailed_kwargs = dict(prepared["detailed_kwargs"])
    detailed_kwargs["cost_per_side"] = float(cost_per_side)
    detailed_kwargs["maker_fee_per_side"] = float(maker_fee_per_side)
    detailed_kwargs["slip_per_side"] = float(slip_per_side)
    if final_candidate_mask is None:
        final_candidate_mask, _diag_summary = _runtime_candidate_masks(prepared, cost_per_side=float(cost_per_side), slip_per_side=float(slip_per_side), need_diag=False)
    raw_gate_strength_seg = np.asarray(prepared.get("raw_gate_strength_seg", detailed_kwargs.get("gate_strength", [])), dtype=np.float64)
    detailed_kwargs["gate_strength"] = raw_gate_strength_seg * np.asarray(final_candidate_mask, dtype=np.float64)
    return simulate_trading_core_rl_single_detailed(**detailed_kwargs, intrabar_mode=int(intrabar_mode))


def evaluate_prepared_single_segment_fast(
    prepared: Dict[str, Any],
    score_cfg: Dict[str, Any],
    cost_per_side: float,
    slip_per_side: float,
    maker_fee_per_side: float,
    need_diag: bool = False,
) -> Dict[str, Any]:
    final_candidate_mask, diag_summary = _runtime_candidate_masks(prepared, cost_per_side=float(cost_per_side), slip_per_side=float(slip_per_side), need_diag=bool(need_diag))
    if _use_python_structural_fallback(prepared):
        res_ohlc = _run_detailed_with_mode(prepared, 1, cost_per_side, slip_per_side, maker_fee_per_side, final_candidate_mask=final_candidate_mask)
        res_olhc = _run_detailed_with_mode(prepared, 2, cost_per_side, slip_per_side, maker_fee_per_side, final_candidate_mask=final_candidate_mask)
        score_ohlc = segment_score(
            net_ret=float(res_ohlc["net_ret"]), mdd=float(res_ohlc["mdd"]), tail_hit=int(res_ohlc["tail_hit"]),
            trades=int(res_ohlc["trades"]), maxh_cnt=int(res_ohlc["maxh_cnt"]), long_trades=int(res_ohlc["long_trades"]), short_trades=int(res_ohlc["short_trades"]),
            alpha_dd=float(score_cfg.get("alpha_dd", 0.9)), beta_tail=float(score_cfg.get("beta_tail", 2.0)), trade_mode=str(score_cfg.get("trade_mode", "none")),
            trade_target=float(score_cfg.get("trade_target", 0.0)), trade_band=float(score_cfg.get("trade_band", 0.0)), barrier_k=float(score_cfg.get("barrier_k", 2.0)),
            shortage_penalty=float(score_cfg.get("trade_shortage_penalty", 0.05)), excess_penalty=float(score_cfg.get("trade_excess_penalty", 0.01)),
            maxhold_ratio_free=float(score_cfg.get("maxhold_ratio_free", 1.0)), maxhold_penalty_k=float(score_cfg.get("maxhold_penalty_k", 0.0)), maxhold_penalty_power=float(score_cfg.get("maxhold_penalty_power", 2.0)),
            side_balance_penalty_k=float(score_cfg.get("side_balance_penalty_k", 0.0)), min_short_trades=int(score_cfg.get("min_short_trades_global", 0)), min_short_share=float(score_cfg.get("min_short_share_global", 0.0)),
        )
        score_olhc = segment_score(
            net_ret=float(res_olhc["net_ret"]), mdd=float(res_olhc["mdd"]), tail_hit=int(res_olhc["tail_hit"]),
            trades=int(res_olhc["trades"]), maxh_cnt=int(res_olhc["maxh_cnt"]), long_trades=int(res_olhc["long_trades"]), short_trades=int(res_olhc["short_trades"]),
            alpha_dd=float(score_cfg.get("alpha_dd", 0.9)), beta_tail=float(score_cfg.get("beta_tail", 2.0)), trade_mode=str(score_cfg.get("trade_mode", "none")),
            trade_target=float(score_cfg.get("trade_target", 0.0)), trade_band=float(score_cfg.get("trade_band", 0.0)), barrier_k=float(score_cfg.get("barrier_k", 2.0)),
            shortage_penalty=float(score_cfg.get("trade_shortage_penalty", 0.05)), excess_penalty=float(score_cfg.get("trade_excess_penalty", 0.01)),
            maxhold_ratio_free=float(score_cfg.get("maxhold_ratio_free", 1.0)), maxhold_penalty_k=float(score_cfg.get("maxhold_penalty_k", 0.0)), maxhold_penalty_power=float(score_cfg.get("maxhold_penalty_power", 2.0)),
            side_balance_penalty_k=float(score_cfg.get("side_balance_penalty_k", 0.0)), min_short_trades=int(score_cfg.get("min_short_trades_global", 0)), min_short_share=float(score_cfg.get("min_short_share_global", 0.0)),
        )
        chosen = res_ohlc
        chosen_score = score_ohlc
        chosen_mode = 1
        if (score_olhc < score_ohlc) or (score_olhc == score_ohlc and (float(res_olhc["net_ret"]) < float(res_ohlc["net_ret"]) or (float(res_olhc["net_ret"]) == float(res_ohlc["net_ret"]) and float(res_olhc["mdd"]) > float(res_ohlc["mdd"])))):
            chosen = res_olhc
            chosen_score = score_olhc
            chosen_mode = 2
        out = _detailed_result_to_segment_result(chosen, chosen_score, prepared["thr_entry"], prepared["atr_high_th"], score_cfg, diag_summary if need_diag else None)
        out["chosen_intrabar_mode"] = int(chosen_mode)
        out["trade_logs"] = []
        return out

    common = dict(prepared["fast_common"])
    common["cost_per_side"] = float(cost_per_side)
    common["slip_per_side"] = float(slip_per_side)
    common["maker_fee_per_side"] = float(maker_fee_per_side)
    raw_gate_strength_seg = np.asarray(prepared.get("raw_gate_strength_seg", common.get("gate_strength", [])), dtype=np.float64)
    common["gate_strength"] = raw_gate_strength_seg * final_candidate_mask.astype(np.float64)

    res_ohlc = simulate_trading_core_rl_single_fast(**common, intrabar_mode=1)
    res_olhc = simulate_trading_core_rl_single_fast(**common, intrabar_mode=2)
    score_ohlc = _score_from_fast_tuple(res_ohlc, score_cfg)
    score_olhc = _score_from_fast_tuple(res_olhc, score_cfg)

    chosen = res_ohlc
    chosen_score = score_ohlc
    chosen_mode = 1
    if (score_olhc < score_ohlc) or (
        score_olhc == score_ohlc and (
            float(res_olhc[0]) < float(res_ohlc[0])
            or (float(res_olhc[0]) == float(res_ohlc[0]) and float(res_olhc[1]) > float(res_ohlc[1]))
        )
    ):
        chosen = res_olhc
        chosen_score = score_olhc
        chosen_mode = 2

    out = _fast_tuple_to_result(chosen, chosen_score, prepared["thr_entry"], prepared["atr_high_th"], score_cfg)
    out["chosen_intrabar_mode"] = int(chosen_mode)
    if need_diag:
        out.update(diag_summary)
    out["lane_enabled"] = int(prepared.get("lane_enabled", 0))
    out["active_sparse_enabled"] = int(prepared.get("active_sparse_enabled", 0))
    out["active_sparse_fallback_dense_count"] = int(prepared.get("active_sparse_fallback_dense_count", 0))
    out["sparse_gate_cut"] = prepared.get("sparse_gate_cut", None)
    out["sparse_gate_floor_cut"] = prepared.get("sparse_gate_floor_cut", None)
    out["sparse_atr_cut"] = prepared.get("sparse_atr_cut", None)
    out["sparse_range_cut"] = prepared.get("sparse_range_cut", None)
    out["sparse_vol_cut"] = prepared.get("sparse_vol_cut", None)
    out["sparse_high_logic"] = prepared.get("sparse_high_logic", "or")
    out["sparse_require_high_vol"] = int(prepared.get("sparse_require_high_vol", 0))
    out["active_sparse_after_gate_band_hist_count"] = int(prepared.get("active_sparse_after_gate_band_hist_count", 0))
    out["active_sparse_after_high_hist_count"] = int(prepared.get("active_sparse_after_high_hist_count", 0))
    out["active_sparse_after_vol_hist_count"] = int(prepared.get("active_sparse_after_vol_hist_count", 0))
    out["active_sparse_after_gate_band_seg_count"] = int(prepared.get("active_sparse_after_gate_band_seg_count", 0))
    out["active_sparse_after_high_seg_count"] = int(prepared.get("active_sparse_after_high_seg_count", 0))
    out["active_sparse_after_vol_seg_count"] = int(prepared.get("active_sparse_after_vol_seg_count", 0))
    out["final_entries"] = int(out.get("trades", 0))
    return out



def evaluate_single_segment_fast(
    seg_start: int,
    seg_end: int,
    open_px: np.ndarray,
    close_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    signals_by_h: Dict[int, np.ndarray],
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
    prepared = prepare_single_segment_inputs(
        seg_start=seg_start,
        seg_end=seg_end,
        open_px=open_px,
        close_px=close_px,
        high_px=high_px,
        low_px=low_px,
        signals_by_h=signals_by_h,
        ready=ready,
        vol_z=vol_z,
        atr_rel=atr_rel,
        minutes_to_next_funding=minutes_to_next_funding,
        cfg=cfg,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
    )
    return evaluate_prepared_single_segment_fast(prepared=prepared, score_cfg=score_cfg, cost_per_side=cost_per_side, slip_per_side=slip_per_side, maker_fee_per_side=maker_fee_per_side)


def evaluate_single_segment_hybrid(
    seg_start: int,
    seg_end: int,
    open_px: np.ndarray,
    close_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    signals_by_h: Dict[int, np.ndarray],
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
    want_trade_logs: bool = False,
) -> Dict[str, Any]:
    prepared = prepare_single_segment_inputs(
        seg_start=seg_start,
        seg_end=seg_end,
        open_px=open_px,
        close_px=close_px,
        high_px=high_px,
        low_px=low_px,
        signals_by_h=signals_by_h,
        ready=ready,
        vol_z=vol_z,
        atr_rel=atr_rel,
        minutes_to_next_funding=minutes_to_next_funding,
        cfg=cfg,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
    )
    return evaluate_prepared_single_segment_hybrid(prepared=prepared, score_cfg=score_cfg, cost_per_side=cost_per_side, slip_per_side=slip_per_side, maker_fee_per_side=maker_fee_per_side, want_trade_logs=want_trade_logs)


__all__ = [
    "ObjectiveBreakdown",
    "SegmentMetrics",
    "agg_worst",
    "apply_ranges_overrides",
    "assemble_objective",
    "build_single_best_config",
    "prepare_trial_context",
    "prepare_single_segment_fast_inputs_from_context",
    "prepare_single_segment_inputs_from_context",
    "prepare_single_segment_inputs",
    "evaluate_prepared_single_segment_fast",
    "evaluate_prepared_single_segment_hybrid",
    "evaluate_single_segment_fast",
    "evaluate_single_segment_hybrid",
    "normalize_single_config_from_any",
    "parse_float_list",
    "precompute_hybrids",
    "safe_float",
    "safe_int",
    "weights_from_self_mix",
    "weights_from_raw_vector",
    "normalize_horizon_weights",
    "warmup_single_fast_core",
    "EXIT_NAMES",
]


# --- v110 modified prepare/runtime active dual-lane overlay ---
import hybrid_core_v7 as _core_v110


def _active_profile_count_dict(mask: np.ndarray, profile_seg: np.ndarray, prefix: str) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    profile_seg = np.asarray(profile_seg, dtype=np.int8)
    return {
        f"{prefix}_active_dense": int(np.sum(mask & (profile_seg == 2))),
        f"{prefix}_active_sparse": int(np.sum(mask & (profile_seg == 3))),
    }


_prepare_diag_summary_v110_base = _prepare_diag_summary

def _prepare_diag_summary(
    *,
    bucket_seg: np.ndarray,
    base_candidate_mask: np.ndarray,
    pass_qthr_mask: np.ndarray,
    cand_after_funding_mask: np.ndarray,
    vol_pass_mask: np.ndarray,
    atr_min_pass_mask: np.ndarray,
    range_min_pass_mask: np.ndarray,
    atr_high_pass_mask: np.ndarray,
    final_candidate_mask: np.ndarray,
    profile_seg: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    out = _prepare_diag_summary_v110_base(
        bucket_seg=np.asarray(bucket_seg, dtype=np.int8),
        base_candidate_mask=np.asarray(base_candidate_mask, dtype=bool),
        pass_qthr_mask=np.asarray(pass_qthr_mask, dtype=bool),
        cand_after_funding_mask=np.asarray(cand_after_funding_mask, dtype=bool),
        vol_pass_mask=np.asarray(vol_pass_mask, dtype=bool),
        atr_min_pass_mask=np.asarray(atr_min_pass_mask, dtype=bool),
        range_min_pass_mask=np.asarray(range_min_pass_mask, dtype=bool),
        atr_high_pass_mask=np.asarray(atr_high_pass_mask, dtype=bool),
        final_candidate_mask=np.asarray(final_candidate_mask, dtype=bool),
    )
    if profile_seg is not None:
        profile_seg = np.asarray(profile_seg, dtype=np.int8)
        out.update(_active_profile_count_dict(base_candidate_mask, profile_seg, "cand_total"))
        out.update(_active_profile_count_dict(pass_qthr_mask, profile_seg, "cand_after_qthr"))
        out.update(_active_profile_count_dict(cand_after_funding_mask, profile_seg, "cand_after_funding"))
        out.update(_active_profile_count_dict(cand_after_funding_mask & vol_pass_mask, profile_seg, "pass_vol"))
        out.update(_active_profile_count_dict(cand_after_funding_mask & atr_min_pass_mask, profile_seg, "pass_atr_min"))
        out.update(_active_profile_count_dict(cand_after_funding_mask & range_min_pass_mask, profile_seg, "pass_range_min"))
        out.update(_active_profile_count_dict(cand_after_funding_mask & atr_high_pass_mask, profile_seg, "pass_atr_high"))
        out.update(_active_profile_count_dict(final_candidate_mask, profile_seg, "final_candidates"))
    return out


_prepare_single_segment_inputs_from_context_base_v110_base = _prepare_single_segment_inputs_from_context_base

def _prepare_single_segment_inputs_from_context_base(
    ctx: Dict[str, Any],
    seg_start: int,
    seg_end: int,
    entry_q_lookback: int,
    entry_q_min_ready: int,
    *,
    include_detailed: bool,
) -> Dict[str, Any]:
    prepared = _prepare_single_segment_inputs_from_context_base_v110_base(
        ctx=ctx,
        seg_start=seg_start,
        seg_end=seg_end,
        entry_q_lookback=entry_q_lookback,
        entry_q_min_ready=entry_q_min_ready,
        include_detailed=include_detailed,
    )
    cfg = _ensure_normalized_single_cfg_once(prepared.get("cfg", ctx.get("cfg", {})))
    src_cfg_for_same_side = prepared.get("cfg", ctx.get("cfg", {}))
    if isinstance(src_cfg_for_same_side, dict):
        raw_same_side_hold_cfg = src_cfg_for_same_side.get("same_side_hold_cfg", {})
        if isinstance(raw_same_side_hold_cfg, dict) and raw_same_side_hold_cfg:
            cfg["same_side_hold_cfg"] = dict(raw_same_side_hold_cfg)
    prepared["cfg"] = cfg
    seg = slice(int(seg_start), int(seg_end))
    hist_end = int(seg_start)
    hist_start = max(0, hist_end - int(entry_q_lookback))

    gate_signal_all = np.asarray(ctx.get("gate_signal_all", np.zeros(len(ctx.get("open_px", [])), dtype=np.float64)), dtype=np.float64)
    dir_signal_all = np.asarray(ctx.get("dir_signal_all", np.zeros_like(gate_signal_all)), dtype=np.float64)
    ready_all = np.asarray(ctx.get("ready", np.zeros_like(gate_signal_all, dtype=bool)), dtype=bool)
    atr_rel_all = np.asarray(ctx.get("atr_rel", np.zeros_like(gate_signal_all)), dtype=np.float64)
    vol_z_all = np.asarray(ctx.get("vol_z", np.zeros_like(gate_signal_all)), dtype=np.float64)
    funding_all = np.asarray(ctx.get("minutes_to_next_funding", np.zeros_like(gate_signal_all)), dtype=np.float64)
    range_rel_all = np.asarray(ctx.get("range_rel_all", np.zeros_like(gate_signal_all)), dtype=np.float64)
    regime_bucket_all = np.asarray(ctx.get("regime_bucket_all", np.full_like(gate_signal_all, -1, dtype=np.int8)), dtype=np.int8)

    lane_pack = _core_v110.build_active_sparse_lane_pack(
        gate_signal_all=gate_signal_all,
        ready=ready_all,
        atr_rel_all=atr_rel_all,
        range_rel_all=range_rel_all,
        vol_z_all=vol_z_all,
        bucket_arr=regime_bucket_all,
        hist_start=hist_start,
        hist_end=hist_end,
        seg_start=int(seg_start),
        seg_end=int(seg_end),
        regime_lane_cfg=cfg.get("regime_lane_cfg", {}),
    )
    profile_seg = np.asarray(lane_pack.get("profile_seg", np.full(seg.stop - seg.start, -1, dtype=np.int8)), dtype=np.int8)
    profile_hist = np.asarray(lane_pack.get("profile_hist", np.full(max(0, hist_end - hist_start), -1, dtype=np.int8)), dtype=np.int8)
    bucket_seg = np.asarray(lane_pack.get("bucket_seg", regime_bucket_all[seg]), dtype=np.int8)
    active_sparse_flag_seg = np.asarray(lane_pack.get("active_sparse_flag_seg", np.zeros(seg.stop - seg.start, dtype=np.bool_)), dtype=np.bool_)
    lane_meta = dict(lane_pack.get("profile_meta", {}))

    threshold_pack = _core_v110.build_profiled_entry_threshold_pack(
        gate_signal_all=gate_signal_all,
        ready=ready_all,
        bucket_arr=regime_bucket_all,
        profile_hist=profile_hist,
        profile_seg=profile_seg,
        hist_start=hist_start,
        hist_end=hist_end,
        seg_start=int(seg_start),
        seg_end=int(seg_end),
        q_entry=safe_float(cfg.get("q_entry", 0.85), 0.85),
        entry_th_floor=safe_float(cfg.get("entry_th_floor", cfg.get("entry_th", 0.0)), 0.0),
        entry_q_min_ready=int(entry_q_min_ready),
        regime_threshold_cfg=cfg.get("regime_threshold_cfg", {}),
    )
    filter_pack = _core_v110.build_profiled_filter_pack(
        profile_seg=profile_seg,
        regime_filter_cfg=cfg.get("regime_filter_cfg", {}),
        base_vol_low_th=safe_float(cfg.get("risk_cfg", {}).get("vol_low_th", -1e9), -1e9),
        base_atr_entry_mult=safe_float(cfg.get("atr_entry_mult", 1.0), 1.0),
        base_range_entry_mult=safe_float(cfg.get("range_entry_mult", 1.0), 1.0),
    )

    gate_strength_seg = np.asarray(gate_signal_all[seg], dtype=np.float64)
    dir_signal_seg = np.asarray(dir_signal_all[seg], dtype=np.float64)
    ready_seg = np.asarray(ready_all[seg], dtype=bool)
    atr_seg = np.asarray(atr_rel_all[seg], dtype=np.float64)
    vol_seg = np.asarray(vol_z_all[seg], dtype=np.float64)
    funding_seg = np.asarray(funding_all[seg], dtype=np.float64)
    range_seg = np.asarray(range_rel_all[seg], dtype=np.float64)
    entry_threshold_base_seg = np.asarray(threshold_pack.get("entry_threshold_base_seg", prepared.get("entry_threshold_base_seg", [])), dtype=np.float64)
    entry_q_used_seg = np.asarray(threshold_pack.get("entry_q_used_seg", prepared.get("entry_q_used_seg", [])), dtype=np.float64)
    entry_floor_used_seg = np.asarray(threshold_pack.get("entry_floor_used_seg", prepared.get("entry_floor_used_seg", [])), dtype=np.float64)
    vol_low_th_used_seg = np.asarray(filter_pack.get("vol_low_th_arr", prepared.get("vol_low_th_used_seg", [])), dtype=np.float64)
    atr_entry_mult_used_seg = np.asarray(filter_pack.get("atr_entry_mult_arr", prepared.get("atr_entry_mult_used_seg", [])), dtype=np.float64)
    range_entry_mult_used_seg = np.asarray(filter_pack.get("range_entry_mult_arr", prepared.get("range_entry_mult_used_seg", [])), dtype=np.float64)

    risk_cfg = dict(cfg.get("risk_cfg", {}))
    atr_high_th_val = float(prepared.get("atr_high_th", np.nan)) if np.isfinite(float(prepared.get("atr_high_th", np.nan))) else float("nan")
    range_med = float(prepared.get("range_med", 1.0))
    if range_med <= 0.0:
        range_med = 1.0
    atr_med = float(prepared.get("atr_med", 1.0))
    if atr_med <= 0.0:
        atr_med = 1.0
    dyn_gate_mult_arr, dyn_lev_scale_arr, dyn_bep_scale_arr, dyn_trail_scale_arr, dyn_sl_scale_arr, dyn_softsl_relax_arr, dyn_stress_arr = build_dynamic_arrays(
        dynamic_cfg=cfg["dynamic_cfg"],
        gate_strength_seg=gate_strength_seg,
        thr_entry=float(threshold_pack.get("thr_entry_global", prepared.get("thr_entry", 0.0))),
        atr_seg=atr_seg,
        atr_high_th=float(atr_high_th_val) if np.isfinite(float(atr_high_th_val)) else float("nan"),
        range_seg=range_seg,
        range_cut=float(safe_float(cfg.get("range_entry_mult", 1.0), 1.0) * float(range_med)),
        vol_seg=vol_seg,
        vol_low_th=safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9),
        funding_seg=funding_seg,
        thr_entry_arr=entry_threshold_base_seg,
    )
    entry_threshold_eff_seg = entry_threshold_base_seg * dyn_gate_mult_arr

    funding_near_min = float(safe_float(risk_cfg.get("funding_near_min", 0.0), 0.0))
    low_vol_enabled = int(cfg.get("low_vol_filter", 0)) != 0
    use_atr_scaling = int(cfg.get("use_atr_scaling", 1)) != 0
    risk_entry_mode = int(cfg.get("risk_entry_mode", 0))
    base_candidate_mask = ready_seg & np.isfinite(gate_strength_seg) & np.isfinite(dir_signal_seg) & (gate_strength_seg > 0.0) & (dir_signal_seg != 0.0)
    pass_qthr_mask = base_candidate_mask & np.isfinite(entry_threshold_eff_seg) & (gate_strength_seg >= entry_threshold_eff_seg)
    funding_pass_mask = funding_seg >= float(funding_near_min)
    cand_after_funding_mask = pass_qthr_mask & funding_pass_mask
    fee_roundtrip = 2.0 * (safe_float(cfg.get("cost_per_side", cfg.get("taker_fee_per_side", 0.0)), 0.0) + safe_float(cfg.get("slip_per_side", 0.0), 0.0))
    if fee_roundtrip <= 0.0:
        fee_roundtrip = 2.0 * 0.00085
    if low_vol_enabled:
        vol_pass_mask = vol_seg > vol_low_th_used_seg
        atr_min_pass_mask = atr_seg >= (atr_entry_mult_used_seg * fee_roundtrip)
        range_min_pass_mask = range_seg >= (range_entry_mult_used_seg * fee_roundtrip)
    else:
        vol_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        atr_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        range_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if use_atr_scaling and np.isfinite(float(atr_high_th_val)):
        atr_high_pass_mask = atr_seg <= float(atr_high_th_val)
    else:
        atr_high_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if int(risk_entry_mode) == 3:
        risk_mode3_pass_mask = (atr_seg <= (atr_med * atr_entry_mult_used_seg)) & (range_seg <= (range_med * range_entry_mult_used_seg))
    else:
        risk_mode3_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    filter_pass_mask = vol_pass_mask & atr_min_pass_mask & range_min_pass_mask & atr_high_pass_mask & risk_mode3_pass_mask
    final_candidate_mask = cand_after_funding_mask & filter_pass_mask

    support_strength_ratio_seg = np.zeros_like(gate_strength_seg, dtype=np.float64)
    valid_thr = np.isfinite(entry_threshold_eff_seg) & (entry_threshold_eff_seg > 1e-12)
    support_strength_ratio_seg[valid_thr] = gate_strength_seg[valid_thr] / entry_threshold_eff_seg[valid_thr]

    support_hard_block_mask = (~funding_pass_mask) | (~atr_high_pass_mask) | (~risk_mode3_pass_mask)
    support_weak_eligible_mask = base_candidate_mask & (~support_hard_block_mask)
    support_pass_mask = final_candidate_mask.copy()

    fast_common = dict(prepared.get("fast_common", {}))
    fast_common.update({
        "gate_strength": gate_strength_seg.astype(np.float64, copy=False),
        "dir_signal": dir_signal_seg.astype(np.float64, copy=False),
        "ready": ready_seg,
        "vol_z": vol_seg.astype(np.float64, copy=False),
        "atr_rel": atr_seg.astype(np.float64, copy=False),
        "range_rel": range_seg.astype(np.float64, copy=False),
        "minutes_to_next_funding": funding_seg.astype(np.float64, copy=False),
        "atr_med": float(atr_med),
        "range_med": float(range_med),
        "dyn_lev_scale_arr": dyn_lev_scale_arr.astype(np.float64, copy=False),
        "dyn_bep_scale_arr": dyn_bep_scale_arr.astype(np.float64, copy=False),
        "dyn_trail_scale_arr": dyn_trail_scale_arr.astype(np.float64, copy=False),
        "dyn_sl_scale_arr": dyn_sl_scale_arr.astype(np.float64, copy=False),
        "dyn_softsl_relax_arr": dyn_softsl_relax_arr.astype(np.int64, copy=False),
        "dyn_gate_mult_arr": dyn_gate_mult_arr.astype(np.float64, copy=False),
        "dyn_stress_arr": dyn_stress_arr.astype(np.float64, copy=False),
        "support_strength_ratio_arr": support_strength_ratio_seg.astype(np.float64, copy=False),
        "support_weak_eligible_mask": support_weak_eligible_mask.astype(np.bool_, copy=False),
        "support_pass_mask": support_pass_mask.astype(np.bool_, copy=False),
    })

    regime_alpha_all = ctx.get("regime_alpha_all")
    regime_bucket_all = ctx.get("regime_bucket_all")
    regime_alpha_seg = np.asarray(regime_alpha_all[seg], dtype=np.float64) if regime_alpha_all is not None else np.zeros(seg.stop - seg.start, dtype=np.float64)
    regime_bucket_seg = np.asarray(regime_bucket_all[seg], dtype=np.int8) if regime_bucket_all is not None else np.full(seg.stop - seg.start, -1, dtype=np.int8)
    diag_summary = _prepare_diag_summary(
        bucket_seg=bucket_seg,
        profile_seg=profile_seg,
        base_candidate_mask=base_candidate_mask,
        pass_qthr_mask=pass_qthr_mask,
        cand_after_funding_mask=cand_after_funding_mask,
        vol_pass_mask=vol_pass_mask,
        atr_min_pass_mask=atr_min_pass_mask,
        range_min_pass_mask=range_min_pass_mask,
        atr_high_pass_mask=atr_high_pass_mask,
        final_candidate_mask=final_candidate_mask,
    )

    prepared.update({
        "cfg": cfg,
        "thr_entry": float(threshold_pack.get("thr_entry_global", prepared.get("thr_entry", 0.0))),
        "fast_common": fast_common,
        "threshold_enabled": int(threshold_pack.get("threshold_enabled", 0)),
        "threshold_bucket_min_ready": int(threshold_pack.get("bucket_min_ready", 0)),
        "threshold_bucket_fallback_global": int(threshold_pack.get("bucket_fallback_global", 1)),
        "filter_enabled": int(filter_pack.get("enabled", 0)),
        "filter_use_vol_split": int(filter_pack.get("use_vol_split", 1)),
        "filter_use_entry_mult_split": int(filter_pack.get("use_entry_mult_split", 1)),
        "lane_enabled": int(lane_meta.get("lane_enabled", lane_pack.get("enabled", 0))),
        "active_sparse_enabled": int(lane_meta.get("active_sparse_enabled", 0)),
        "active_sparse_fallback_dense_count": int(lane_meta.get("active_sparse_fallback_dense_cnt", 0)),
        "sparse_gate_cut": lane_meta.get("sparse_gate_cut", None),
        "sparse_gate_floor_cut": lane_meta.get("sparse_gate_floor_cut", None),
        "sparse_atr_cut": lane_meta.get("sparse_atr_cut", None),
        "sparse_range_cut": lane_meta.get("sparse_range_cut", None),
        "sparse_vol_cut": lane_meta.get("sparse_vol_cut", None),
        "sparse_high_logic": lane_meta.get("sparse_high_logic", cfg.get("regime_lane_cfg", {}).get("sparse_high_logic", "or")),
        "sparse_require_high_vol": int(lane_meta.get("sparse_require_high_vol", cfg.get("regime_lane_cfg", {}).get("sparse_require_high_vol", 0))),
        "active_sparse_after_gate_band_hist_count": int(lane_meta.get("active_sparse_after_gate_band_hist_count", 0)),
        "active_sparse_after_high_hist_count": int(lane_meta.get("active_sparse_after_high_hist_count", 0)),
        "active_sparse_after_vol_hist_count": int(lane_meta.get("active_sparse_after_vol_hist_count", 0)),
        "active_sparse_after_gate_band_seg_count": int(lane_meta.get("active_sparse_after_gate_band_seg_count", 0)),
        "active_sparse_after_high_seg_count": int(lane_meta.get("active_sparse_after_high_seg_count", 0)),
        "active_sparse_after_vol_seg_count": int(lane_meta.get("active_sparse_after_vol_seg_count", 0)),
        "entry_q_used_mean": float(np.mean(entry_q_used_seg)) if entry_q_used_seg.size else float(safe_float(cfg.get("q_entry", 0.85), 0.85)),
        "entry_q_used_p50": float(np.quantile(entry_q_used_seg, 0.5)) if entry_q_used_seg.size else float(safe_float(cfg.get("q_entry", 0.85), 0.85)),
        "entry_threshold_base_mean": float(np.mean(entry_threshold_base_seg)) if entry_threshold_base_seg.size else float(prepared.get("thr_entry", 0.0)),
        "entry_vol_low_th_used_mean": float(np.mean(vol_low_th_used_seg)) if vol_low_th_used_seg.size else float(safe_float(risk_cfg.get("vol_low_th", -1e9), -1e9)),
        "entry_atr_entry_mult_used_mean": float(np.mean(atr_entry_mult_used_seg)) if atr_entry_mult_used_seg.size else float(safe_float(cfg.get("atr_entry_mult", 1.0), 1.0)),
        "entry_range_entry_mult_used_mean": float(np.mean(range_entry_mult_used_seg)) if range_entry_mult_used_seg.size else float(safe_float(cfg.get("range_entry_mult", 1.0), 1.0)),
        "entry_threshold_eff_seg": entry_threshold_eff_seg,
        "entry_threshold_base_seg": entry_threshold_base_seg,
        "entry_q_used_seg": entry_q_used_seg,
        "entry_floor_used_seg": entry_floor_used_seg,
        "vol_low_th_used_seg": vol_low_th_used_seg,
        "atr_entry_mult_used_seg": atr_entry_mult_used_seg,
        "range_entry_mult_used_seg": range_entry_mult_used_seg,
        "filter_bucket_seg": bucket_seg,
        "profile_seg": profile_seg,
        "entry_profile_used_seg": np.asarray(threshold_pack.get("profile_seg", profile_seg), dtype=np.int8),
        "filter_profile_used_seg": np.asarray(filter_pack.get("profile_seg", profile_seg), dtype=np.int8),
        "active_sparse_flag_seg": active_sparse_flag_seg,
        "support_strength_ratio_seg": support_strength_ratio_seg,
        "support_weak_eligible_mask": support_weak_eligible_mask,
        "support_pass_mask": support_pass_mask,
        "same_side_hold_cfg": dict(cfg.get("same_side_hold_cfg", {}) or {}),
        "diag_summary": diag_summary,
        "regime_alpha_mean": float(np.mean(regime_alpha_seg)) if regime_alpha_seg.size else 0.0,
        "regime_alpha_p50": float(np.quantile(regime_alpha_seg, 0.5)) if regime_alpha_seg.size else 0.0,
        "regime_active_frac": float(np.mean(regime_bucket_seg == 2)) if regime_bucket_seg.size else 0.0,
        "regime_calm_frac": float(np.mean(regime_bucket_seg == 0)) if regime_bucket_seg.size else 0.0,
    })
    if include_detailed:
        detailed_kwargs = dict(fast_common)
        detailed_kwargs.pop("range_rel", None)
        detailed_kwargs.pop("atr_med", None)
        detailed_kwargs.pop("range_med", None)
        detailed_kwargs.pop("bep_stop_mode_code", None)
        detailed_kwargs["bep_stop_mode"] = str(cfg.get("bep_stop_mode", "maker_be"))
        detailed_kwargs["cost_per_side"] = 0.0
        detailed_kwargs["maker_fee_per_side"] = 0.0
        detailed_kwargs["slip_per_side"] = 0.0
        detailed_kwargs["seg_start"] = int(seg_start)
        if regime_alpha_all is not None:
            detailed_kwargs["regime_alpha_arr"] = np.asarray(regime_alpha_all[seg], dtype=np.float64)
        if regime_bucket_all is not None:
            detailed_kwargs["regime_bucket_arr"] = np.asarray(regime_bucket_all[seg], dtype=np.int64)
        prepared["detailed_kwargs"] = detailed_kwargs
    return prepared


def _runtime_candidate_masks(prepared: Dict[str, Any], cost_per_side: float, slip_per_side: float, need_diag: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    gate_strength_seg = np.asarray(prepared.get("raw_gate_strength_seg", prepared.get("fast_common", {}).get("gate_strength", [])), dtype=np.float64)
    dir_signal_seg = np.asarray(prepared.get("raw_dir_signal_seg", prepared.get("fast_common", {}).get("dir_signal", [])), dtype=np.float64)
    ready_seg = np.asarray(prepared.get("raw_ready_seg", prepared.get("fast_common", {}).get("ready", [])), dtype=bool)
    funding_seg = np.asarray(prepared.get("raw_funding_seg", prepared.get("fast_common", {}).get("minutes_to_next_funding", [])), dtype=np.float64)
    vol_seg = np.asarray(prepared.get("raw_vol_seg", prepared.get("fast_common", {}).get("vol_z", [])), dtype=np.float64)
    atr_seg = np.asarray(prepared.get("raw_atr_seg", prepared.get("fast_common", {}).get("atr_rel", [])), dtype=np.float64)
    range_seg = np.asarray(prepared.get("raw_range_seg", prepared.get("fast_common", {}).get("range_rel", [])), dtype=np.float64)
    entry_threshold_eff_seg = np.asarray(prepared.get("entry_threshold_eff_seg", []), dtype=np.float64)
    vol_low_th_used_seg = np.asarray(prepared.get("vol_low_th_used_seg", []), dtype=np.float64)
    atr_entry_mult_used_seg = np.asarray(prepared.get("atr_entry_mult_used_seg", []), dtype=np.float64)
    range_entry_mult_used_seg = np.asarray(prepared.get("range_entry_mult_used_seg", []), dtype=np.float64)
    bucket_seg = np.asarray(prepared.get("filter_bucket_seg", np.full(len(gate_strength_seg), -1, dtype=np.int8)), dtype=np.int8)
    profile_seg = np.asarray(prepared.get("profile_seg", np.full(len(gate_strength_seg), -1, dtype=np.int8)), dtype=np.int8)
    atr_med = float(prepared.get("atr_med", 1.0))
    range_med = float(prepared.get("range_med", 1.0))
    funding_near_min = float(prepared.get("funding_near_min", 0.0))
    low_vol_enabled = bool(int(prepared.get("low_vol_enabled", 0)) != 0)
    use_atr_scaling_flag = bool(int(prepared.get("use_atr_scaling_flag", 0)) != 0)
    risk_entry_mode_flag = int(prepared.get("risk_entry_mode_flag", 0))
    atr_high_th_val = float(prepared.get("atr_high_th_value", np.nan))

    base_candidate_mask = ready_seg & np.isfinite(gate_strength_seg) & np.isfinite(dir_signal_seg) & (gate_strength_seg > 0.0) & (dir_signal_seg != 0.0)
    pass_qthr_mask = base_candidate_mask & np.isfinite(entry_threshold_eff_seg) & (gate_strength_seg >= entry_threshold_eff_seg)
    funding_pass_mask = funding_seg >= funding_near_min
    cand_after_funding_mask = pass_qthr_mask & funding_pass_mask
    fee_roundtrip = 2.0 * (float(cost_per_side) + float(slip_per_side))
    if low_vol_enabled:
        vol_pass_mask = vol_seg > vol_low_th_used_seg
        atr_min_pass_mask = atr_seg >= (atr_entry_mult_used_seg * fee_roundtrip)
        range_min_pass_mask = range_seg >= (range_entry_mult_used_seg * fee_roundtrip)
    else:
        vol_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        atr_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
        range_min_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if use_atr_scaling_flag and np.isfinite(atr_high_th_val):
        atr_high_pass_mask = atr_seg <= atr_high_th_val
    else:
        atr_high_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    if risk_entry_mode_flag == 3:
        risk_mode3_pass_mask = (atr_seg <= (atr_med * atr_entry_mult_used_seg)) & (range_seg <= (range_med * range_entry_mult_used_seg))
    else:
        risk_mode3_pass_mask = np.ones_like(cand_after_funding_mask, dtype=bool)
    filter_pass_mask = vol_pass_mask & atr_min_pass_mask & range_min_pass_mask & atr_high_pass_mask & risk_mode3_pass_mask
    final_candidate_mask = cand_after_funding_mask & filter_pass_mask
    if need_diag:
        diag_summary = _prepare_diag_summary(
            bucket_seg=bucket_seg,
            profile_seg=profile_seg,
            base_candidate_mask=base_candidate_mask,
            pass_qthr_mask=pass_qthr_mask,
            cand_after_funding_mask=cand_after_funding_mask,
            vol_pass_mask=vol_pass_mask,
            atr_min_pass_mask=atr_min_pass_mask,
            range_min_pass_mask=range_min_pass_mask,
            atr_high_pass_mask=atr_high_pass_mask,
            final_candidate_mask=final_candidate_mask,
        )
    else:
        diag_summary = {}
    return final_candidate_mask, diag_summary


def evaluate_prepared_single_segment_hybrid(
    prepared: Dict[str, Any],
    score_cfg: Dict[str, Any],
    cost_per_side: float,
    slip_per_side: float,
    maker_fee_per_side: float,
    want_trade_logs: bool = False,
) -> Dict[str, Any]:
    fast_out = evaluate_prepared_single_segment_fast(
        prepared=prepared,
        score_cfg=score_cfg,
        cost_per_side=cost_per_side,
        slip_per_side=slip_per_side,
        maker_fee_per_side=maker_fee_per_side,
        need_diag=True,
    )
    if not want_trade_logs:
        return fast_out

    chosen_mode = int(fast_out.get("chosen_intrabar_mode", 1))
    chosen_detailed = _run_detailed_with_mode(prepared, chosen_mode, cost_per_side, slip_per_side, maker_fee_per_side)

    aligned_logs, align_meta = _annotate_trade_logs_runner_alignment(
        chosen_detailed.get("trade_logs", []),
        prepared,
        cost_per_side,
        slip_per_side,
        maker_fee_per_side,
    )
    chosen_detailed["trade_logs"] = list(aligned_logs)

    seg_start = int(prepared.get("seg_start", 0))
    eff_arr = np.asarray(prepared.get("entry_threshold_eff_seg", []), dtype=np.float64)
    base_arr = np.asarray(prepared.get("entry_threshold_base_seg", []), dtype=np.float64)
    q_arr = np.asarray(prepared.get("entry_q_used_seg", []), dtype=np.float64)
    floor_arr = np.asarray(prepared.get("entry_floor_used_seg", []), dtype=np.float64)
    vol_arr = np.asarray(prepared.get("vol_low_th_used_seg", []), dtype=np.float64)
    atr_mult_arr = np.asarray(prepared.get("atr_entry_mult_used_seg", []), dtype=np.float64)
    range_mult_arr = np.asarray(prepared.get("range_entry_mult_used_seg", []), dtype=np.float64)
    bucket_arr = np.asarray(prepared.get("filter_bucket_seg", []), dtype=np.int8)
    profile_arr = np.asarray(prepared.get("profile_seg", np.full(len(bucket_arr), -1, dtype=np.int8)), dtype=np.int8)
    sparse_flag_arr = np.asarray(prepared.get("active_sparse_flag_seg", np.zeros(len(profile_arr), dtype=np.bool_)), dtype=np.bool_)

    final_entries_by_bucket = {0: 0, 1: 0, 2: 0}
    final_entries_by_profile = {0: 0, 1: 0, 2: 0, 3: 0}
    for tr in chosen_detailed.get("trade_logs", []):
        try:
            local = int(tr.get("decision_idx", -1)) - seg_start
        except Exception:
            local = -1
        if 0 <= local < len(eff_arr):
            tr["entry_th_used"] = float(eff_arr[local])
            tr["entry_threshold_base_used"] = float(base_arr[local])
            tr["entry_q_used"] = float(q_arr[local])
            tr["entry_floor_used"] = float(floor_arr[local])
            tr["entry_vol_low_th_used"] = float(vol_arr[local])
            tr["entry_atr_entry_mult_used"] = float(atr_mult_arr[local])
            tr["entry_range_entry_mult_used"] = float(range_mult_arr[local])
            b = int(bucket_arr[local]) if local < len(bucket_arr) else -1
            p = int(profile_arr[local]) if local < len(profile_arr) else -1
            tr["entry_filter_bucket"] = int(b)
            tr["entry_filter_bucket_name"] = regime_bucket_name(int(b))
            tr["entry_profile_id"] = int(p)
            tr["entry_profile_name"] = _core_v110.regime_profile_name(int(p))
            tr["entry_active_sparse_flag"] = int(bool(sparse_flag_arr[local])) if local < len(sparse_flag_arr) else 0
            tr["entry_q_profile_used"] = float(q_arr[local])
            tr["entry_threshold_profile_used"] = float(base_arr[local])
            tr["entry_vol_low_th_profile_used"] = float(vol_arr[local])
            tr["entry_atr_entry_mult_profile_used"] = float(atr_mult_arr[local])
            tr["entry_range_entry_mult_profile_used"] = float(range_mult_arr[local])
            if b in final_entries_by_bucket:
                final_entries_by_bucket[b] += 1
            if p in final_entries_by_profile:
                final_entries_by_profile[p] += 1
        else:
            tr.setdefault("entry_filter_bucket", -1)
            tr.setdefault("entry_filter_bucket_name", "unknown")
            tr.setdefault("entry_profile_id", -1)
            tr.setdefault("entry_profile_name", "unknown")
            tr.setdefault("entry_active_sparse_flag", 0)

    out = _detailed_result_to_segment_result(
        chosen_detailed,
        float(fast_out["score"]),
        prepared["thr_entry"],
        prepared["atr_high_th"],
        score_cfg,
        {k: v for k, v in fast_out.items() if isinstance(k, str) and (k.startswith("cand_") or k.startswith("pass_") or k.startswith("final_candidates"))},
    )
    out["trade_logs"] = list(chosen_detailed.get("trade_logs", []))
    out["chosen_intrabar_mode"] = int(chosen_mode)
    out["final_entries"] = int(chosen_detailed.get("trades", 0))
    out["final_entries_calm"] = int(final_entries_by_bucket[0])
    out["final_entries_mid"] = int(final_entries_by_bucket[1])
    out["final_entries_active"] = int(final_entries_by_bucket[2])
    out["final_entries_active_dense"] = int(final_entries_by_profile[2])
    out["final_entries_active_sparse"] = int(final_entries_by_profile[3])
    out["lane_enabled"] = int(prepared.get("lane_enabled", 0))
    out["active_sparse_enabled"] = int(prepared.get("active_sparse_enabled", 0))
    out["active_sparse_fallback_dense_count"] = int(prepared.get("active_sparse_fallback_dense_count", 0))
    out["sparse_gate_cut"] = prepared.get("sparse_gate_cut", None)
    out["sparse_gate_floor_cut"] = prepared.get("sparse_gate_floor_cut", None)
    out["sparse_atr_cut"] = prepared.get("sparse_atr_cut", None)
    out["sparse_range_cut"] = prepared.get("sparse_range_cut", None)
    out["sparse_vol_cut"] = prepared.get("sparse_vol_cut", None)
    out["sparse_high_logic"] = prepared.get("sparse_high_logic", "or")
    out["sparse_require_high_vol"] = int(prepared.get("sparse_require_high_vol", 0))
    out["active_sparse_after_gate_band_hist_count"] = int(prepared.get("active_sparse_after_gate_band_hist_count", 0))
    out["active_sparse_after_high_hist_count"] = int(prepared.get("active_sparse_after_high_hist_count", 0))
    out["active_sparse_after_vol_hist_count"] = int(prepared.get("active_sparse_after_vol_hist_count", 0))
    out["active_sparse_after_gate_band_seg_count"] = int(prepared.get("active_sparse_after_gate_band_seg_count", 0))
    out["active_sparse_after_high_seg_count"] = int(prepared.get("active_sparse_after_high_seg_count", 0))
    out["active_sparse_after_vol_seg_count"] = int(prepared.get("active_sparse_after_vol_seg_count", 0))
    out.update({k: v for k, v in align_meta.items() if str(k).startswith("runner_align_")})
    return out
