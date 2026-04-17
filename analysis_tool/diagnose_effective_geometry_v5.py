
import argparse, json
import pandas as pd
import numpy as np

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

def _is_close(a, b, tol=1e-12):
    return np.abs(a - b) <= tol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tradelog", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cost-per-side", type=float, default=0.00070)
    ap.add_argument("--slip-per-side", type=float, default=0.00015)
    ap.add_argument("--maker-fee-per-side", type=float, default=0.0002)
    ap.add_argument("--fee-tp-mult", type=float, default=0.70)
    ap.add_argument("--trail-exit-mode", choices=["maker", "taker"], default="maker")
    args = ap.parse_args()

    tr = pd.read_csv(args.tradelog)
    cfg = json.load(open(args.config, "r", encoding="utf-8"))

    sl_coef = float(cfg.get("SL", np.nan))
    tp_coef = float(cfg.get("TP", np.nan))
    bep_coef = float(cfg.get("BEP_ARM", cfg.get("BEP", np.nan)))
    tr_coef = float(cfg.get("trailing", np.nan))

    risk_cfg = cfg.get("risk_cfg", {}) or {}
    atr_high_th = _safe_float(risk_cfg.get("atr_high_th", np.nan))
    vol_low_th = _safe_float(risk_cfg.get("vol_low_th", np.nan))
    funding_near_min = _safe_float(risk_cfg.get("funding_near_min", np.nan))

    atr_entry_mult = _safe_float(cfg.get("atr_entry_mult", np.nan))
    range_entry_mult = _safe_float(cfg.get("range_entry_mult", np.nan))
    trail_after_bep = int(cfg.get("trail_after_bep", 1))
    fee_tp_mult = _safe_float(cfg.get("fee_tp_mult", args.fee_tp_mult))
    bep_arm_fee_mult = _safe_float(cfg.get("bep_arm_fee_mult", np.nan))
    bep_stop_fee_mult = _safe_float(cfg.get("bep_stop_fee_mult", np.nan))
    bep_stop_mode = str(cfg.get("bep_stop_mode", "taker_be"))

    atr = tr["atr_rel_entry"].to_numpy(dtype=float) if len(tr) else np.array([], dtype=float)
    mfe = tr["mfe"].to_numpy(dtype=float) if len(tr) else np.array([], dtype=float)
    mae = np.abs(tr["mae"].to_numpy(dtype=float)) if len(tr) else np.array([], dtype=float)
    vol_z = tr["vol_z_60_entry"].to_numpy(dtype=float) if (len(tr) and "vol_z_60_entry" in tr.columns) else None

    taker_fee_side = float(args.cost_per_side) + float(args.slip_per_side)
    maker_fee_side = float(args.maker_fee_per_side)
    fee_roundtrip_taker = 2.0 * taker_fee_side
    fee_roundtrip_maker = taker_fee_side + maker_fee_side

    min_tp = max(fee_roundtrip_maker * fee_tp_mult, fee_roundtrip_taker * fee_tp_mult)
    min_sl = 0.6 * fee_roundtrip_taker

    dyn_sl_scale = tr["entry_dyn_sl_scale"].to_numpy(dtype=float) if ("entry_dyn_sl_scale" in tr.columns and len(tr)) else np.ones_like(atr)
    raw_sl = sl_coef * atr * dyn_sl_scale
    raw_tp = tp_coef * atr
    sl_eff = np.maximum(raw_sl, min_sl)
    tp_eff = np.maximum(raw_tp, min_tp)

    dyn_bep_scale = tr["entry_dyn_bep_scale"].to_numpy(dtype=float) if ("entry_dyn_bep_scale" in tr.columns and len(tr)) else np.ones_like(atr)
    raw_bep = bep_coef * atr * dyn_bep_scale

    arm_floor = fee_roundtrip_maker * bep_arm_fee_mult if np.isfinite(bep_arm_fee_mult) else np.nan
    if "entry_bep_arm_fee" in tr.columns:
        bep_arm_fee_eff = tr["entry_bep_arm_fee"].to_numpy(dtype=float)
    else:
        bep_arm_fee_eff = np.full_like(atr, arm_floor, dtype=float)

    if "bep_arm_value" in tr.columns:
        bep_eff = tr["bep_arm_value"].to_numpy(dtype=float)
    else:
        bep_eff = np.maximum(raw_bep, bep_arm_fee_eff)

    dyn_trail_scale = tr["entry_dyn_trail_scale"].to_numpy(dtype=float) if ("entry_dyn_trail_scale" in tr.columns and len(tr)) else np.ones_like(atr)
    raw_tr = tr_coef * atr * dyn_trail_scale
    exit_fee_for_trail = maker_fee_side if args.trail_exit_mode == "maker" else taker_fee_side
    min_tr = max(taker_fee_side + exit_fee_for_trail, fee_roundtrip_maker * fee_tp_mult)
    tr_eff = raw_tr if trail_after_bep == 1 else np.maximum(raw_tr, min_tr)

    if bep_stop_mode == "maker_be":
        stop_floor = fee_roundtrip_maker * bep_stop_fee_mult
    else:
        stop_floor = fee_roundtrip_taker * bep_stop_fee_mult

    out = {
        "mode": "single",
        "n_trades": int(len(tr)),
        "trail_after_bep": int(trail_after_bep),
        "trail_exit_mode": str(args.trail_exit_mode),
        "taker_fee_side": taker_fee_side,
        "maker_fee_side": maker_fee_side,
        "fee_roundtrip_taker": fee_roundtrip_taker,
        "fee_roundtrip_maker": fee_roundtrip_maker,
        "bep_stop_mode": bep_stop_mode,
        "median_atr_rel_entry": float(np.median(atr)) if len(tr) else np.nan,
        "median_sl_eff": float(np.median(sl_eff)) if len(tr) else np.nan,
        "median_tp_eff": float(np.median(tp_eff)) if len(tr) else np.nan,
        "median_bep_eff": float(np.median(bep_eff)) if len(tr) else np.nan,
        "median_tr_eff": float(np.median(tr_eff)) if len(tr) else np.nan,
        "median_bep_stop_fee": float(np.median(tr["entry_bep_stop_fee"].to_numpy(dtype=float))) if ("entry_bep_stop_fee" in tr.columns and len(tr)) else stop_floor,
        "median_tp_sl_ratio_eff": float(np.median(tp_eff / np.maximum(sl_eff, 1e-12))) if len(tr) else np.nan,
        "mfe_ge_bep_frac": float(np.mean(mfe >= bep_eff)) if len(tr) else np.nan,
        "mfe_ge_tp_frac": float(np.mean(mfe >= tp_eff)) if len(tr) else np.nan,
        "mae_ge_sl_frac": float(np.mean(mae >= sl_eff)) if len(tr) else np.nan,
        "mfe_ge_tr_frac": float(np.mean(mfe >= tr_eff)) if len(tr) else np.nan,
        "atr_high_th": atr_high_th,
        "vol_low_th": vol_low_th,
        "funding_near_min": funding_near_min,
        "atr_entry_mult": atr_entry_mult,
        "range_entry_mult": range_entry_mult,
        "uses_tradelevel_bep": bool(("bep_arm_value" in tr.columns) or ("entry_bep_arm_fee" in tr.columns)),
        "uses_tradelevel_trail": bool("entry_dyn_trail_scale" in tr.columns),
        "uses_tradelevel_sl": bool("entry_dyn_sl_scale" in tr.columns),
        "uses_tradelevel_soft_sl": bool("entry_min_hold_soft_sl_local" in tr.columns),
    }

    if "entry_dyn_gate_mult" in tr.columns and len(tr):
        out["median_dyn_gate_mult"] = float(np.median(tr["entry_dyn_gate_mult"].to_numpy(dtype=float)))
    if "entry_dyn_lev_scale" in tr.columns and len(tr):
        out["median_dyn_lev_scale"] = float(np.median(tr["entry_dyn_lev_scale"].to_numpy(dtype=float)))
    if "entry_dyn_bep_scale" in tr.columns and len(tr):
        out["median_dyn_bep_scale"] = float(np.median(tr["entry_dyn_bep_scale"].to_numpy(dtype=float)))
    if "entry_dyn_trail_scale" in tr.columns and len(tr):
        out["median_dyn_trail_scale"] = float(np.median(tr["entry_dyn_trail_scale"].to_numpy(dtype=float)))
    if "entry_dyn_sl_scale" in tr.columns and len(tr):
        out["median_dyn_sl_scale"] = float(np.median(tr["entry_dyn_sl_scale"].to_numpy(dtype=float)))
    if "entry_min_hold_soft_sl_local" in tr.columns and len(tr):
        out["median_entry_min_hold_soft_sl_local"] = float(np.median(tr["entry_min_hold_soft_sl_local"].to_numpy(dtype=float)))

    atr_entry_cut = atr_entry_mult * fee_roundtrip_taker if np.isfinite(atr_entry_mult) else np.nan
    range_entry_cut = range_entry_mult * fee_roundtrip_taker if np.isfinite(range_entry_mult) else np.nan
    out["atr_entry_cut"] = atr_entry_cut
    out["range_entry_cut"] = range_entry_cut
    out["atr_filter_impossible"] = bool(np.isfinite(atr_high_th) and np.isfinite(atr_entry_cut) and (atr_high_th < atr_entry_cut))

    if len(tr):
        out["trade_atr_ge_entry_cut_frac"] = float(np.mean(atr >= atr_entry_cut)) if np.isfinite(atr_entry_cut) else np.nan
        out["trade_atr_le_high_th_frac"] = float(np.mean(atr <= atr_high_th)) if np.isfinite(atr_high_th) else np.nan
        if vol_z is not None and np.isfinite(vol_low_th):
            out["trade_vol_gt_low_th_frac"] = float(np.mean(vol_z > vol_low_th))

    sl = tr[tr.get("exit_reason", pd.Series([""] * len(tr))) == "SL"].copy() if len(tr) else pd.DataFrame()
    if len(sl):
        sl_atr = sl["atr_rel_entry"].to_numpy(dtype=float)
        sl_mfe = sl["mfe"].to_numpy(dtype=float)
        dyn_sl_s = sl["entry_dyn_sl_scale"].to_numpy(dtype=float) if "entry_dyn_sl_scale" in sl.columns else np.ones_like(sl_atr)
        raw_sl_s = sl_coef * sl_atr * dyn_sl_s
        raw_tp_s = tp_coef * sl_atr
        sl_eff_s = np.maximum(raw_sl_s, min_sl)
        tp_eff_s = np.maximum(raw_tp_s, min_tp)

        if "bep_arm_value" in sl.columns:
            bep_eff_s = sl["bep_arm_value"].to_numpy(dtype=float)
        else:
            dyn_bep_s = sl["entry_dyn_bep_scale"].to_numpy(dtype=float) if "entry_dyn_bep_scale" in sl.columns else np.ones_like(sl_atr)
            raw_bep_s = bep_coef * sl_atr * dyn_bep_s
            bep_fee_s = sl["entry_bep_arm_fee"].to_numpy(dtype=float) if "entry_bep_arm_fee" in sl.columns else np.full_like(sl_atr, arm_floor, dtype=float)
            bep_eff_s = np.maximum(raw_bep_s, bep_fee_s)

        dyn_tr_s = sl["entry_dyn_trail_scale"].to_numpy(dtype=float) if "entry_dyn_trail_scale" in sl.columns else np.ones_like(sl_atr)
        raw_tr_s = tr_coef * sl_atr * dyn_tr_s
        tr_eff_s = raw_tr_s if trail_after_bep == 1 else np.maximum(raw_tr_s, min_tr)

        out["sl_n"] = int(len(sl))
        out["sl_mfe_ge_bep_frac"] = float(np.mean(sl_mfe >= bep_eff_s))
        out["sl_mfe_ge_tp_frac"] = float(np.mean(sl_mfe >= tp_eff_s))
        out["sl_mfe_ge_tr_frac"] = float(np.mean(sl_mfe >= tr_eff_s))
        out["sl_mfe_ge_2xtr_frac"] = float(np.mean(sl_mfe >= 2.0 * tr_eff_s))
        out["sl_mfe_ge_half_tp_frac"] = float(np.mean(sl_mfe >= 0.5 * tp_eff_s))
        if "entry_min_hold_soft_sl_local" in sl.columns and "hold_bars" in sl.columns:
            sl_soft = sl["entry_min_hold_soft_sl_local"].to_numpy(dtype=float)
            sl_hold = sl["hold_bars"].to_numpy(dtype=float)
            out["sl_hold_ge_softsl_frac"] = float(np.mean(sl_hold >= sl_soft))
            out["sl_hold_lt_softsl_frac"] = float(np.mean(sl_hold < sl_soft))
            out["sl_median_local_softsl_hold"] = float(np.median(sl_soft)) if len(sl_soft) else np.nan
    else:
        out["sl_n"] = 0

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()