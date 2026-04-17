import argparse, json
import numpy as np
import pandas as pd


def q(arr, p):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.quantile(arr, p))


def frac(mask):
    mask = np.asarray(mask)
    if mask.size == 0:
        return float('nan')
    return float(np.mean(mask.astype(float)))


def main():
    ap = argparse.ArgumentParser(description='Quick v24 smoke check for entry vs final soft-SL hold separation.')
    ap.add_argument('--tradelog', required=True)
    ap.add_argument('--out-json', default='')
    args = ap.parse_args()

    df = pd.read_csv(args.tradelog)
    out = {
        'n_trades': int(len(df)),
        'has_entry_local': bool('entry_min_hold_soft_sl_local' in df.columns),
        'has_final_local': bool('final_min_hold_soft_sl_local' in df.columns),
        'has_exit_reason': bool('exit_reason' in df.columns),
    }

    if ('entry_min_hold_soft_sl_local' in df.columns) and ('final_min_hold_soft_sl_local' in df.columns):
        entry = df['entry_min_hold_soft_sl_local'].to_numpy(dtype=float)
        final = df['final_min_hold_soft_sl_local'].to_numpy(dtype=float)
        delta = final - entry
        out.update({
            'entry_softsl_hold_p50': q(entry, 0.50),
            'final_softsl_hold_p50': q(final, 0.50),
            'entry_softsl_hold_p90': q(entry, 0.90),
            'final_softsl_hold_p90': q(final, 0.90),
            'final_lt_entry_frac': frac(final < entry),
            'final_eq_entry_frac': frac(final == entry),
            'delta_softsl_hold_mean': float(np.nanmean(delta)) if np.isfinite(delta).any() else float('nan'),
            'delta_softsl_hold_min': float(np.nanmin(delta)) if np.isfinite(delta).any() else float('nan'),
            'delta_softsl_hold_max': float(np.nanmax(delta)) if np.isfinite(delta).any() else float('nan'),
        })

    if ('exit_reason' in df.columns) and ('hold_bars' in df.columns):
        sl = df[df['exit_reason'].astype(str) == 'SL'].copy()
        out['sl_n'] = int(len(sl))
        if len(sl) and ('entry_min_hold_soft_sl_local' in sl.columns):
            hold = sl['hold_bars'].to_numpy(dtype=float)
            entry = sl['entry_min_hold_soft_sl_local'].to_numpy(dtype=float)
            out['sl_hold_ge_entry_softsl_frac'] = frac(hold >= entry)
            out['sl_hold_lt_entry_softsl_frac'] = frac(hold < entry)
        if len(sl) and ('final_min_hold_soft_sl_local' in sl.columns):
            hold = sl['hold_bars'].to_numpy(dtype=float)
            final = sl['final_min_hold_soft_sl_local'].to_numpy(dtype=float)
            out['sl_hold_ge_final_softsl_frac'] = frac(hold >= final)
            out['sl_hold_lt_final_softsl_frac'] = frac(hold < final)
            out['sl_final_lt_entry_frac'] = frac(final < sl['entry_min_hold_soft_sl_local'].to_numpy(dtype=float)) if 'entry_min_hold_soft_sl_local' in sl.columns else float('nan')

    txt = json.dumps(out, indent=2, ensure_ascii=False)
    print(txt)
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            f.write(txt)


if __name__ == '__main__':
    main()