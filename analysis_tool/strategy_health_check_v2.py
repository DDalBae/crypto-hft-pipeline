
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tradelog", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.tradelog)
    print(f"[INFO] Analyzing: {args.tradelog}")

    print("\n===== [A] Failure Reason Analysis =====")
    if 'exit_reason' in df.columns:
        sl = df[df.exit_reason=="SL"].copy()
        if not sl.empty:
            sl["mfe_to_bep"] = sl["mfe"] / sl["bep_arm_value"].replace(0, np.nan)
            sl["sl_class"] = np.select(
                [sl.mfe_to_bep >= 0.70, sl.mfe_to_bep <= 0.20],
                ["near_bep", "early_fail"],
                default="mid_fail"
            )
            sl_p = sl.groupby(["seg","sl_class"]).size().unstack(fill_value=0)
        else:
            sl_p = pd.DataFrame()

        mx = df[df.exit_reason=="MAX_HOLD"].copy()
        if not mx.empty:
            mx["mfe_to_bep"] = mx["mfe"] / mx["bep_arm_value"].replace(0, np.nan)
            mx["maxh_class"] = np.select(
                [mx.mfe_to_bep >= 0.80, mx.mfe_to_bep <= 0.30],
                ["almost_bep", "no_follow"],
                default="mid"
            )
            mx_p = mx.groupby(["seg","maxh_class"]).size().unstack(fill_value=0)
        else:
            mx_p = pd.DataFrame()

        seg_stats = pd.DataFrame({"trades": df.groupby("seg").size()})
        seg_stats["SL_cnt"] = df[df.exit_reason=="SL"].groupby("seg").size()
        seg_stats["MAXH_cnt"] = df[df.exit_reason=="MAX_HOLD"].groupby("seg").size()
        seg_stats = seg_stats.fillna(0).astype(int)

        if not sl_p.empty: seg_stats = seg_stats.join(sl_p.add_prefix("SL_"))
        if not mx_p.empty: seg_stats = seg_stats.join(mx_p.add_prefix("MH_"))

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(seg_stats)
    else:
        print("[WARN] 'exit_reason' column missing.")

    print("\n===== [B] Profit Concentration Analysis =====")
    pnl_col = "net_pnl_alloc" if "net_pnl_alloc" in df.columns else ("net_pnl" if "net_pnl" in df.columns else None)
    if pnl_col is not None:
        p = df[pnl_col].astype(float)
        net_total = p.sum()
        pos_trades = p[p > 0]
        pos_sum = pos_trades.sum()
        win_rate = len(pos_trades) / len(p) if len(p) > 0 else 0.0
        print(f"Total Net PnL: {net_total}")
        print(f"Gross Profit (Wins only): {pos_sum}")
        print(f"Win Rate: {win_rate}")

        if pos_sum > 0:
            pos_sorted = pos_trades.sort_values(ascending=False)
            def get_share(q):
                k = max(1, int(len(pos_sorted) * q))
                return pos_sorted.iloc[:k].sum() / pos_sum
            print(f"- Top 1% Share: {get_share(0.01)}")
            print(f"- Top 5% Share: {get_share(0.05)}")
            print(f"- Top 10% Share: {get_share(0.10)}")
            w = (pos_sorted / pos_sum).values
            hhi = np.sum(w*w)
            print(f"- Profit HHI: {hhi}")
        else:
            print("[INFO] No positive trades to analyze.")
    else:
        print("[WARN] no net_pnl column found.")

if __name__ == "__main__":
    main()