
import argparse, json
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tradelog', required=True)
    ap.add_argument('--maker-fee-per-side', type=float, default=0.0002)
    ap.add_argument('--cost-per-side', type=float, default=0.0007)
    ap.add_argument('--slip-per-side', type=float, default=0.00015)
    ap.add_argument('--current-trail-mode', choices=['maker','taker'], default='maker')
    ap.add_argument('--counterfactual-trail-mode', choices=['maker','taker'], default='taker')
    args = ap.parse_args()

    df = pd.read_csv(args.tradelog)
    if 'exit_reason' not in df.columns:
        raise SystemExit('missing exit_reason column')
    if 'lev' not in df.columns or 'gross_pnl' not in df.columns:
        raise SystemExit('missing lev/gross_pnl columns')

    taker_fee_side = args.cost_per_side + args.slip_per_side
    maker_fee_side = args.maker_fee_per_side

    def exit_fee_side(reason: str, trail_mode: str) -> float:
        if reason == 'TP':
            return maker_fee_side
        if reason == 'TRAIL':
            return maker_fee_side if trail_mode == 'maker' else taker_fee_side
        # current engine uses taker-style exit accounting for SL / MAX_HOLD / RISK_CLOSE / FORCE_CLOSE
        return taker_fee_side

    reasons = list(df['exit_reason'].fillna('UNKNOWN').astype(str).unique())
    reasons = sorted(reasons, key=lambda x: ['TP','SL','TRAIL','MAX_HOLD','FORCE_CLOSE','RISK_CLOSE'].index(x) if x in ['TP','SL','TRAIL','MAX_HOLD','FORCE_CLOSE','RISK_CLOSE'] else 999)

    out = {}
    for reason in reasons:
        sub = df[df['exit_reason'] == reason].copy()
        if sub.empty:
            out[reason] = {'count': 0}
            continue

        fee_current = (taker_fee_side + exit_fee_side(reason, args.current_trail_mode)) * sub['lev'].to_numpy()
        fee_cf = (taker_fee_side + exit_fee_side(reason, args.counterfactual_trail_mode)) * sub['lev'].to_numpy()

        gross = sub['gross_pnl'].to_numpy()
        net_current = gross - fee_current
        net_cf = gross - fee_cf

        out[reason] = {
            'count': int(len(sub)),
            'gross_sum': float(gross.sum()),
            'fee_current_sum': float(fee_current.sum()),
            'net_current_sum': float(net_current.sum()),
            'avg_current': float(net_current.mean()),
            'fee_counterfactual_sum': float(fee_cf.sum()),
            'net_counterfactual_sum': float(net_cf.sum()),
            'avg_counterfactual': float(net_cf.mean()),
            'cf_sensitive': bool(reason == 'TRAIL')
        }

    total_current = sum(float(v.get('net_current_sum', 0.0)) for v in out.values())
    total_cf = sum(float(v.get('net_counterfactual_sum', 0.0)) for v in out.values())

    summary = {
        'taker_fee_side': taker_fee_side,
        'maker_fee_side': maker_fee_side,
        'current_trail_mode': args.current_trail_mode,
        'counterfactual_trail_mode': args.counterfactual_trail_mode,
        'total_net_current': total_current,
        'total_net_counterfactual': total_cf,
        'delta_cf_minus_current': total_cf - total_current,
        'by_reason': out,
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()