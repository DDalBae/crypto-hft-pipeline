
import argparse, json
import pandas as pd
import numpy as np

def q(x, p):
    if len(x)==0:
        return np.nan
    return float(np.quantile(x, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tradelog", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.tradelog)
    if "hold_bars" not in df.columns:
        raise SystemExit("missing hold_bars")
    out = {}
    out["n"] = int(len(df))
    out["hold_all"] = {
        "p25": q(df["hold_bars"], 0.25),
        "p50": q(df["hold_bars"], 0.50),
        "p75": q(df["hold_bars"], 0.75),
        "p90": q(df["hold_bars"], 0.90),
        "mean": float(df["hold_bars"].mean()),
    }
    if "exit_reason" in df.columns:
        by_reason = {}
        for reason, sub in df.groupby("exit_reason"):
            payload = {
                "n": int(len(sub)),
                "p25": q(sub["hold_bars"], 0.25),
                "p50": q(sub["hold_bars"], 0.50),
                "p75": q(sub["hold_bars"], 0.75),
                "p90": q(sub["hold_bars"], 0.90),
                "mean": float(sub["hold_bars"].mean()),
                "mfe_mean": float(sub["mfe"].mean()) if "mfe" in sub.columns else np.nan,
                "mae_mean": float(sub["mae"].mean()) if "mae" in sub.columns else np.nan,
            }
            if str(reason) == "SL" and "entry_min_hold_soft_sl_local" in sub.columns:
                soft = sub["entry_min_hold_soft_sl_local"].to_numpy(dtype=float)
                hold = sub["hold_bars"].to_numpy(dtype=float)
                payload["softsl_local_p25"] = q(soft, 0.25)
                payload["softsl_local_p50"] = q(soft, 0.50)
                payload["softsl_local_p75"] = q(soft, 0.75)
                payload["softsl_local_p90"] = q(soft, 0.90)
                payload["softsl_local_mean"] = float(np.mean(soft)) if len(soft) else np.nan
                payload["hold_minus_softsl_mean"] = float(np.mean(hold - soft)) if len(soft) else np.nan
                payload["hold_ge_softsl_frac"] = float(np.mean(hold >= soft)) if len(soft) else np.nan
                payload["hold_lt_softsl_frac"] = float(np.mean(hold < soft)) if len(soft) else np.nan
            by_reason[str(reason)] = payload
        out["by_reason"] = by_reason
    if "seg" in df.columns:
        seg = {}
        for k, sub in df.groupby("seg"):
            seg[str(int(k))] = {
                "n": int(len(sub)),
                "hold_p50": q(sub["hold_bars"],0.50),
                "hold_p90": q(sub["hold_bars"],0.90),
                "tp_rate": float(np.mean(sub["exit_reason"]=="TP")) if "exit_reason" in sub.columns else np.nan,
                "trail_rate": float(np.mean(sub["exit_reason"]=="TRAIL")) if "exit_reason" in sub.columns else np.nan,
                "sl_rate": float(np.mean(sub["exit_reason"]=="SL")) if "exit_reason" in sub.columns else np.nan,
            }
        out["by_seg"] = seg
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()