
# -*- coding: utf-8 -*-
"""
Single-tier aware regime summary + filter pass-rates.
Backward compatible with old tiered configs, but if config schema starts with "single_"
or tier params / pos_frac are absent, it treats the strategy as SINGLE and does not
filter tradelog by tier_name.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIER_NAMES = ["low", "mid", "top"]

def _pick_active_tier_from_pos_frac(pos_frac):
    if isinstance(pos_frac, dict):
        vals = [float(pos_frac.get(k, 0.0)) for k in TIER_NAMES]
    else:
        vals = [float(x) for x in (pos_frac or [1.0, 0.0, 0.0])]
        if len(vals) != 3:
            vals = (vals + [0.0, 0.0, 0.0])[:3]
    if max(vals) <= 0:
        return "low"
    return TIER_NAMES[int(np.argmax(vals))]

def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _safe_bool_mean(s: pd.Series):
    if s is None or len(s) == 0:
        return np.nan
    return float(np.mean(s.astype(bool)))

def _q(arr, q):
    if arr is None or len(arr) == 0:
        return np.nan
    try:
        return float(np.quantile(np.asarray(arr, dtype=np.float64), q))
    except Exception:
        return np.nan

def _median(arr):
    if arr is None or len(arr) == 0:
        return np.nan
    try:
        return float(np.median(np.asarray(arr, dtype=np.float64)))
    except Exception:
        return np.nan

def _load_tradelog_summary(tradelog_path: Path, tier: str, single_mode: bool):
    tdf = pd.read_csv(tradelog_path)
    if (not single_mode) and ("tier_name" in tdf.columns):
        tdf = tdf[tdf["tier_name"].astype(str).str.lower() == str(tier).lower()].copy()

    if "seg" not in tdf.columns:
        raise ValueError("tradelog missing required column: seg")

    if "exit_reason" not in tdf.columns and "exit_reason_id" in tdf.columns:
        id_map = {1: "SL", 2: "TRAIL", 3: "TP", 4: "MAX_HOLD", 5: "FORCE_CLOSE", 6: "RISK_CLOSE"}
        tdf["exit_reason"] = tdf["exit_reason_id"].map(id_map).fillna("UNK")
    if "hold_bars" not in tdf.columns:
        tdf["hold_bars"] = np.nan
    if "final_min_hold_soft_sl_local" not in tdf.columns:
        tdf["final_min_hold_soft_sl_local"] = tdf.get("entry_min_hold_soft_sl_local", np.nan)
    if "entry_min_hold_soft_sl_local" not in tdf.columns:
        tdf["entry_min_hold_soft_sl_local"] = tdf.get("final_min_hold_soft_sl_local", np.nan)
    if "entry_post_bep_shield_ignore_softsl_hold" not in tdf.columns:
        tdf["entry_post_bep_shield_ignore_softsl_hold"] = 0

    tdf["exit_reason"] = tdf["exit_reason"].astype(str)
    tdf["is_sl"] = tdf["exit_reason"].eq("SL")
    tdf["is_trail"] = tdf["exit_reason"].eq("TRAIL")
    tdf["is_maxh"] = tdf["exit_reason"].eq("MAX_HOLD")
    tdf["hold0"] = pd.to_numeric(tdf["hold_bars"], errors="coerce").fillna(np.nan) <= 0
    tdf["hold_le1"] = pd.to_numeric(tdf["hold_bars"], errors="coerce").fillna(np.nan) <= 1

    bep_arm = pd.to_numeric(tdf.get("bep_arm_value", np.nan), errors="coerce")
    mfe = pd.to_numeric(tdf.get("mfe", np.nan), errors="coerce")
    final_softsl = pd.to_numeric(tdf.get("final_min_hold_soft_sl_local", np.nan), errors="coerce")
    entry_softsl = pd.to_numeric(tdf.get("entry_min_hold_soft_sl_local", np.nan), errors="coerce")
    hold = pd.to_numeric(tdf.get("hold_bars", np.nan), errors="coerce")

    tdf["mfe_ge_bep"] = np.where(np.isfinite(bep_arm) & np.isfinite(mfe), mfe >= (bep_arm - 1e-12), False)
    tdf["pre_softsl_final"] = np.where(np.isfinite(final_softsl) & np.isfinite(hold), hold < final_softsl, False)
    tdf["pre_softsl_entry"] = np.where(np.isfinite(entry_softsl) & np.isfinite(hold), hold < entry_softsl, False)
    tdf["final_lt_entry_local"] = np.where(
        np.isfinite(final_softsl) & np.isfinite(entry_softsl), final_softsl < entry_softsl, False
    )
    tdf["shield_on"] = pd.to_numeric(tdf["entry_post_bep_shield_ignore_softsl_hold"], errors="coerce").fillna(0).astype(int) == 1

    rows = []
    for seg, g in tdf.groupby("seg", sort=True):
        sl = g[g["is_sl"]]
        rows.append(
            {
                "seg": int(seg),
                "trades_tlog": int(len(g)),
                "sl_n_tlog": int(len(sl)),
                "trail_n_tlog": int(np.sum(g["is_trail"])),
                "maxh_n_tlog": int(np.sum(g["is_maxh"])),
                "risk_n_tlog": int(np.sum(g["exit_reason"].eq("RISK_CLOSE"))),
                "tp_n_tlog": int(np.sum(g["exit_reason"].eq("TP"))),
                "hold_p50_tlog": _median(pd.to_numeric(g["hold_bars"], errors="coerce").dropna().to_numpy()),
                "shield_on%": _safe_bool_mean(g["shield_on"]) * 100.0 if len(g) else np.nan,
                "final_lt_entry_softsl%": _safe_bool_mean(g["final_lt_entry_local"]) * 100.0 if len(g) else np.nan,
                "sl_hold0%": _safe_bool_mean(sl["hold0"]) * 100.0 if len(sl) else np.nan,
                "sl_hold_le1%": _safe_bool_mean(sl["hold_le1"]) * 100.0 if len(sl) else np.nan,
                "sl_mfe_ge_bep%": _safe_bool_mean(sl["mfe_ge_bep"]) * 100.0 if len(sl) else np.nan,
                "sl_pre_softsl_entry_bep%": _safe_bool_mean(sl["mfe_ge_bep"] & sl["pre_softsl_entry"]) * 100.0 if len(sl) else np.nan,
                "sl_pre_softsl_final_bep%": _safe_bool_mean(sl["mfe_ge_bep"] & sl["pre_softsl_final"]) * 100.0 if len(sl) else np.nan,
                "sl_pre_softsl_final_bep_n": int(np.sum(sl["mfe_ge_bep"] & sl["pre_softsl_final"])),
            }
        )
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_clean_final.csv")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tradelog", default=None)
    ap.add_argument("--window", type=int, default=60000)
    ap.add_argument("--window-includes-hist-extra", dest="window_includes_hist_extra", type=int, default=0)
    ap.add_argument("--oos_len", type=int, default=30000)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=300)
    ap.add_argument("--entry-q-lookback", dest="entry_q_lookback", type=int, default=6000)
    ap.add_argument("--entry-q-min-ready", dest="entry_q_min_ready", type=int, default=300)
    ap.add_argument("--tier", default="auto", choices=["auto", "single", "low", "mid", "top"])
    ap.add_argument("--cost_per_side", type=float, default=None)
    ap.add_argument("--slip_per_side", type=float, default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cols0 = pd.read_csv(args.csv, nrows=0).columns
    required_cols = ["close", "high", "low", "vol_z_60", "atr10_rel"]
    missing_cols = [c for c in required_cols if c not in cols0]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    df = pd.read_csv(args.csv, usecols=required_cols)

    hist_extra = int(max(args.entry_q_lookback, args.seq_len, args.entry_q_min_ready))
    if int(args.window_includes_hist_extra) != 0:
        need = int(args.window)
        WINDOW = int(max(1, need - hist_extra))
    else:
        WINDOW = int(args.window)
        need = int(WINDOW + hist_extra)

    OOS_LEN = int(args.oos_len)
    SPLITS = int(args.splits)
    SEG_LEN = OOS_LEN // SPLITS
    if SEG_LEN * SPLITS != OOS_LEN:
        raise ValueError("oos_len must be divisible by splits")
    if len(df) < need:
        raise ValueError(f"rows={len(df)} < need={need} (window={WINDOW} + hist_extra={hist_extra})")

    dfw = df.iloc[-need:].reset_index(drop=True)
    window_start = hist_extra
    oos_end = window_start + WINDOW
    oos_start = max(window_start, oos_end - OOS_LEN)

    close = dfw["close"].to_numpy(dtype=np.float64)
    high = dfw["high"].to_numpy(dtype=np.float64)
    low = dfw["low"].to_numpy(dtype=np.float64)
    atr_rel = dfw["atr10_rel"].to_numpy(dtype=np.float64)
    vol_z = dfw["vol_z_60"].to_numpy(dtype=np.float64)
    range_rel = (high - low) / np.maximum(close, 1e-12)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    schema = str(cfg.get("schema", "") or "")
    single_mode = schema.startswith("single_") or (not isinstance(cfg.get("tier_params"), dict))
    if single_mode:
        tier = "single"
        low_vol_filter = int(cfg.get("low_vol_filter", 0))
        use_atr_scaling = int(cfg.get("use_atr_scaling", 0))
        risk_cfg = cfg.get("risk_cfg", {}) or {}
        vol_th = float(risk_cfg.get("vol_low_th", -1e9))
        atr_mult = float(cfg.get("atr_entry_mult", 1.0))
        range_mult = float(cfg.get("range_entry_mult", 1.0))
        atr_high_raw = risk_cfg.get("atr_high_th", cfg.get("atr_high_th", 1e9))
    else:
        pos_frac = cfg.get("pos_frac", [1.0, 0.0, 0.0])
        active_tier = _pick_active_tier_from_pos_frac(pos_frac)
        tier = active_tier if args.tier == "auto" else args.tier
        low_vol_filter = int(cfg.get("low_vol_filter", 0))
        use_atr_scaling = int(cfg.get("use_atr_scaling", 0))
        risk_cfg = cfg.get("risk_cfg", {}) or {}
        vol_low_th_tier = risk_cfg.get("vol_low_th_tier", {
            "low": cfg.get("vol_low_th_low", -1e9),
            "mid": cfg.get("vol_low_th_mid", -1e9),
            "top": cfg.get("vol_low_th_top", -1e9),
        })
        atr_entry_mult_tier = cfg.get("atr_entry_mult_tier", {
            "low": cfg.get("atr_entry_mult_low", 1.0),
            "mid": cfg.get("atr_entry_mult_mid", 1.0),
            "top": cfg.get("atr_entry_mult_top", 1.0),
        })
        range_entry_mult_tier = cfg.get("range_entry_mult_tier", {
            "low": cfg.get("range_entry_mult_low", 1.0),
            "mid": cfg.get("range_entry_mult_mid", 1.0),
            "top": cfg.get("range_entry_mult_top", 1.0),
        })
        vol_th = float(vol_low_th_tier.get(tier, -1e9))
        atr_mult = float(atr_entry_mult_tier.get(tier, 1.0))
        range_mult = float(range_entry_mult_tier.get(tier, 1.0))
        atr_high_raw = risk_cfg.get("atr_high_th", cfg.get("atr_high_th", 1e9))

    cps = float(cfg.get("cost_per_side", 0.0)) if args.cost_per_side is None else float(args.cost_per_side)
    sps = float(cfg.get("slip_per_side", 0.0)) if args.slip_per_side is None else float(args.slip_per_side)
    fee_roundtrip = 2.0 * (float(cps) + float(sps))
    atr_th = atr_mult * fee_roundtrip
    range_th = range_mult * fee_roundtrip
    atr_high_th = np.inf if atr_high_raw is None else float(atr_high_raw)

    rows = []
    for k in range(1, SPLITS + 1):
        s = oos_start + (k - 1) * SEG_LEN
        e = s + SEG_LEN
        seg_atr = atr_rel[s:e]
        seg_vol = vol_z[s:e]
        seg_range = range_rel[s:e]

        vol_pass = seg_vol > vol_th
        atr_pass = seg_atr >= atr_th
        range_pass = seg_range >= range_th
        all_pass = vol_pass & atr_pass & range_pass
        atr_after_vol = atr_pass[vol_pass]
        range_after_vol_atr = range_pass[vol_pass & atr_pass]
        atr_scaling_pass = seg_atr <= atr_high_th

        rows.append(
            {
                "seg": k,
                "n": int(len(seg_atr)),
                "atr_rel_p50": _median(seg_atr),
                "atr_rel_p90": _q(seg_atr, 0.90),
                "range_rel_p50": _median(seg_range),
                "range_rel_p90": _q(seg_range, 0.90),
                "vol_z_p10": _q(seg_vol, 0.10),
                "vol_z_p50": _median(seg_vol),
                "vol_z_p90": _q(seg_vol, 0.90),
                "pass_vol%": float(np.mean(vol_pass)) * 100.0,
                "pass_atr%": float(np.mean(atr_pass)) * 100.0,
                "pass_range%": float(np.mean(range_pass)) * 100.0,
                "pass_all%": float(np.mean(all_pass)) * 100.0,
                "pass_atr_given_vol%": (float(np.mean(atr_after_vol)) * 100.0) if len(atr_after_vol) else np.nan,
                "pass_range_given_vol_atr%": (float(np.mean(range_after_vol_atr)) * 100.0) if len(range_after_vol_atr) else np.nan,
                "pass_atr_high%": float(np.mean(atr_scaling_pass)) * 100.0 if np.isfinite(atr_high_th) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if args.tradelog:
        tl = _load_tradelog_summary(Path(args.tradelog), tier=tier, single_mode=single_mode)
        out = out.merge(tl, on="seg", how="left")

    disp = out.copy()
    for c in disp.columns:
        if c in ["seg", "n", "trades_tlog", "sl_n_tlog", "trail_n_tlog", "maxh_n_tlog", "risk_n_tlog", "tp_n_tlog", "sl_pre_softsl_final_bep_n"]:
            continue
        if c.endswith("%"):
            disp[c] = disp[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
        else:
            disp[c] = disp[c].map(lambda x: x if pd.isna(x) else round(float(x), 6))

    print("===== Regime Summary + Filter Pass-Rates =====")
    print(f"[CONFIG] {args.config}")
    print(f"[MODE] {'single-tier' if single_mode else 'tiered'} using={tier}")
    print(
        f"[FILTERS] low_vol_filter={low_vol_filter} use_atr_scaling={use_atr_scaling} "
        f"vol_th={vol_th:.6f} atr_mult={atr_mult:.6f} range_mult={range_mult:.6f} "
        f"fee_roundtrip={fee_roundtrip:.6f} atr_th={atr_th:.6f} range_th={range_th:.6f} "
        f"atr_high_th={atr_high_th if np.isfinite(atr_high_th) else float('nan'):.6f}"
    )
    print(disp.to_string(index=False))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"[DONE] Saved -> {args.out_csv}")

if __name__ == "__main__":
    main()