# -*- coding: utf-8 -*-
"""
sanitize_feat22_full.py

현재는 FEAT22 전용이 아니라, **Collector_v8 FEAT30 + 5-horizon 계약**까지 처리하는
계약 인지(contract-aware) 데이터 살균(정화) 스크립트다.
파일명은 호환성을 위해 유지한다.

지원 범위
---------
- 시간 정렬 + 중복 timestamp 제거
- gap(분봉 불연속) 감지 및 gap 영향 완화
- 미래 라벨 시간 정합성 검증 후 불량은 NaN 처리
  * y_next1/3/5/8/10
  * y1/3/5/8/10_class
  * legacy alias: y_class == y5_class
  * optional *_class_soft / y_class_soft 도 있으면 함께 처리
- 과거 수익률 피처 시간 정합성 검증 후 불량은 0.0 중립화
  * r1/r3/r5/r8/r10
- prev_close 의존 피처(gap_open/high_ext/low_ext) 갭 구간 중립화
- FEAT30/legacy FEAT22 피처 NaN/inf 처리
  * 기본: ffill only (no bfill)
  * 남은 NaN은 선택적으로 0.0 fill
- minutes_to_next_funding가 있으면 timestamp 기준으로 재계산
- 리포트 출력(중복/갭/라벨 분포/피처 결측/극단값)

권장 사용
---------
collector(clean csv) -> split(past/future, no overlap) -> sanitize -> probe/trainer/backtest

주의
----
- 이 스크립트는 tail label 행 삭제를 하지 않는다.
  y_next* 끝부분 NaN은 shift(-h) 때문에 자연스럽다.
- Collector_v8 clean output이 이미 상당히 정제되어 있으면,
  sanitize는 주로 split 후 시간 정합성 최종 확인용이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


# =========================
# Contract constants
# =========================
FEAT30: List[str] = [
    "gap_open", "high_ext", "low_ext", "body_ret",
    "r1", "r3", "r5", "r8", "r10",
    "atr1_rel", "atr3_rel", "atr5_rel", "atr8_rel", "atr10_rel",
    "vol_z_3", "vol_z_5", "vol_z_8", "vol_z_10", "vol_z_60",
    "spread_proxy", "taker_buy_ratio",
    "upper_wick_rel", "lower_wick_rel", "wick_ratio",
    "funding_diff",
    "session_vwap_dist", "session_vwap_slope", "session_range_pct",
    "bb_pctb_20", "efficiency_ratio_10",
]

# Legacy subset is still supported if a file only has old contract columns.
FEAT22: List[str] = [
    "gap_open", "high_ext", "low_ext", "body_ret",
    "r1", "r3", "r5", "r10",
    "atr1_rel", "atr3_rel", "atr5_rel", "atr10_rel",
    "vol_z_3", "vol_z_5", "vol_z_10", "vol_z_60",
    "spread_proxy", "taker_buy_ratio",
    "upper_wick_rel", "lower_wick_rel", "wick_ratio",
    "funding_diff",
]

FEATURES_ALL: List[str] = list(dict.fromkeys(FEAT30 + FEAT22))
RETURN_LAGS: List[int] = [1, 3, 5, 8, 10]
TARGET_HORIZONS: List[int] = [1, 3, 5, 8, 10]
PREV_CLOSE_DEP: List[str] = ["gap_open", "high_ext", "low_ext"]
CLASS_COLS: Dict[int, str] = {1: "y1_class", 3: "y3_class", 5: "y5_class", 8: "y8_class", 10: "y10_class"}
LEGACY_CLASS_ALIAS: Dict[int, str] = {5: "y_class"}
SOFT_CLASS_COLS: Dict[int, str] = {1: "y1_class_soft", 3: "y3_class_soft", 5: "y5_class_soft", 8: "y8_class_soft", 10: "y10_class_soft"}
LEGACY_SOFT_CLASS_ALIAS: Dict[int, str] = {5: "y_class_soft"}


def _detect_time_col(df: pd.DataFrame) -> str:
    if "time" in df.columns:
        return "time"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError("No 'time' or 'timestamp' column found.")


def _to_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _min_diff(ts_a: pd.Series, ts_b: pd.Series) -> pd.Series:
    return (ts_a - ts_b).dt.total_seconds() / 60.0


def _as_float_col(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _present_horizons(df: pd.DataFrame) -> List[int]:
    out: List[int] = []
    for h in TARGET_HORIZONS:
        cand = [f"y_next{h}", CLASS_COLS[h], SOFT_CLASS_COLS[h]]
        if h in LEGACY_CLASS_ALIAS:
            cand.append(LEGACY_CLASS_ALIAS[h])
        if h in LEGACY_SOFT_CLASS_ALIAS:
            cand.append(LEGACY_SOFT_CLASS_ALIAS[h])
        if any(c in df.columns for c in cand):
            out.append(int(h))
    return out


def _present_return_lags(df: pd.DataFrame) -> List[int]:
    return [int(h) for h in RETURN_LAGS if f"r{int(h)}" in df.columns]


def _recompute_minutes_to_next_funding(ts: pd.Series) -> np.ndarray:
    out: List[float] = []
    funding_hours = [0, 8, 16]
    for t in ts:
        if pd.isna(t):
            out.append(np.nan)
            continue
        h = int(t.hour)
        future_hours = [fh for fh in funding_hours if fh > h]
        if future_hours:
            nh = min(future_hours)
            nxt = t.replace(hour=int(nh), minute=0, second=0, microsecond=0)
        else:
            nxt = (t + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out.append(float((nxt - t).total_seconds() / 60.0))
    return np.asarray(out, dtype=np.float64)


def _safe_contract_name(df: pd.DataFrame) -> str:
    has_feat30 = all(c in df.columns for c in FEAT30)
    has_feat22 = all(c in df.columns for c in FEAT22)
    if has_feat30:
        return "FEAT30"
    if has_feat22:
        return "FEAT22"
    return "PARTIAL"


def _report_targets(df: pd.DataFrame, horizons: Iterable[int]) -> None:
    print("\n" + "=" * 88)
    print("TARGET REPORT")
    print("=" * 88)

    for h in horizons:
        y_col = f"y_next{h}"
        if y_col in df.columns:
            s = pd.to_numeric(df[y_col], errors="coerce").astype("float64")
            nanp = float(s.isna().mean())
            if nanp >= 1.0:
                print(f"- {y_col}: all NaN")
            else:
                print(
                    f"- {y_col:8s} | nan%={nanp:.6f} "
                    f"min={np.nanmin(s.values):+.6f} max={np.nanmax(s.values):+.6f} "
                    f"mean={np.nanmean(s.values):+.6e} std={np.nanstd(s.values):.6f}"
                )

    print("\n[CLS] (NaN=neutral, long=1, short=0)")
    for h in horizons:
        cls_candidates = [CLASS_COLS[h]]
        if h in LEGACY_CLASS_ALIAS:
            cls_candidates.append(LEGACY_CLASS_ALIAS[h])
        for c in cls_candidates:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce").astype("float64")
            neutral = float(s.isna().mean())
            p0 = float(np.nanmean(s.values == 0.0))
            p1 = float(np.nanmean(s.values == 1.0))
            p05 = float(np.nanmean(s.values == 0.5))
            print(f"- {c:12s} | neutral={neutral:.6f} p0={p0:.6f} p0.5={p05:.6f} p1={p1:.6f}")


def _report_features(df: pd.DataFrame, features: List[str], contract_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"FEATURE REPORT ({contract_name})")
    print("=" * 88)

    exist = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Missing contract feature columns ({len(missing)}): {missing}")
    if not exist:
        print("[WARN] No contract feature columns found to report.")
        return

    nan_rates = []
    for c in exist:
        s = pd.to_numeric(df[c], errors="coerce").astype("float64")
        nan_rates.append((c, float(s.isna().mean())))
    nan_rates.sort(key=lambda x: x[1], reverse=True)
    print("[Top NaN% features]")
    for c, r in nan_rates[:10]:
        print(f"  - {c:20s} nan%={r:.6f}")

    key = [
        c for c in [
            "r1", "r3", "r5", "r8", "r10",
            "atr1_rel", "atr3_rel", "atr5_rel", "atr8_rel", "atr10_rel",
            "vol_z_3", "vol_z_5", "vol_z_8", "vol_z_10", "vol_z_60",
            "spread_proxy", "taker_buy_ratio",
            "session_vwap_dist", "session_vwap_slope", "session_range_pct",
            "bb_pctb_20", "efficiency_ratio_10",
        ] if c in df.columns
    ]
    if key:
        print("\n[Key feature min/max]")
        for c in key:
            s = pd.to_numeric(df[c], errors="coerce").astype("float64")
            if s.isna().all():
                print(f"  - {c:20s} all NaN")
                continue
            print(
                f"  - {c:20s} "
                f"min={np.nanmin(s.values):+.6f} max={np.nanmax(s.values):+.6f} "
                f"mean={np.nanmean(s.values):+.6e}"
            )


def sanitize_one_file(
    path: Path,
    tol_min: float,
    inplace: bool,
    out_suffix: str,
    report_only: bool,
    ffill_features: bool,
    fillna_zero: bool,
    sync_y5_alias: bool,
) -> None:
    print(f"\n🚀 Processing: {path}")

    df = pd.read_csv(path, low_memory=False)
    time_col = _detect_time_col(df)

    # parse time
    df[time_col] = _to_datetime_utc(df[time_col])
    before_drop_bad_time = len(df)
    df = df.dropna(subset=[time_col]).copy()
    dropped_bad_time = before_drop_bad_time - len(df)

    # sort + dedup
    n0 = len(df)
    df = df.sort_values(time_col).copy()
    dup_cnt = int(df.duplicated(subset=[time_col]).sum())
    if dup_cnt > 0:
        df = df.drop_duplicates(subset=[time_col], keep="last").copy()
    df = df.reset_index(drop=True)
    n1 = len(df)

    ts = df[time_col]
    contract_name = _safe_contract_name(df)
    features = [c for c in FEAT30 if c in df.columns]
    if not features:
        features = [c for c in FEAT22 if c in df.columns]
    if not features:
        features = [c for c in FEATURES_ALL if c in df.columns]

    # keep y5 alias synchronized when possible
    if sync_y5_alias:
        if "y5_class" not in df.columns and "y_class" in df.columns:
            df["y5_class"] = df["y_class"]
        if "y5_class" in df.columns and "y_class" not in df.columns:
            df["y_class"] = df["y5_class"]

    # recompute minutes_to_next_funding if present
    if "minutes_to_next_funding" in df.columns:
        df["minutes_to_next_funding"] = _recompute_minutes_to_next_funding(ts)

    # gap stats
    dt1 = _min_diff(ts, ts.shift(1))
    gap_mask = dt1.notna() & ((dt1 - 1.0).abs() > tol_min)
    gap_cnt = int(gap_mask.sum())
    max_gap = float(dt1[gap_mask].max()) if gap_cnt > 0 else 0.0

    print(f"  - contract_guess: {contract_name}")
    print(f"  - rows: {n0} -> {n1} (removed dup={dup_cnt}, dropped bad time={dropped_bad_time})")
    print(f"  - gaps: cnt={gap_cnt}  max_gap(min)={max_gap:.3f}  tol={tol_min}")

    # ----------------------------
    # (1) Future labels sanitize
    # ----------------------------
    horizons = _present_horizons(df)
    bad_future_total: Dict[int, int] = {}
    for h in horizons:
        dt_future = _min_diff(ts.shift(-h), ts)  # (t+h) - t
        bad_future = dt_future.isna() | ((dt_future - float(h)).abs() > tol_min)
        bad_future_total[h] = int(bad_future.sum())

        y_col = f"y_next{h}"
        if y_col in df.columns:
            _as_float_col(df, y_col)
            df.loc[bad_future, y_col] = np.nan

        cls_candidates = [CLASS_COLS[h]]
        if h in LEGACY_CLASS_ALIAS:
            cls_candidates.append(LEGACY_CLASS_ALIAS[h])
        soft_candidates = [SOFT_CLASS_COLS[h]]
        if h in LEGACY_SOFT_CLASS_ALIAS:
            soft_candidates.append(LEGACY_SOFT_CLASS_ALIAS[h])

        for c in cls_candidates + soft_candidates:
            if c in df.columns:
                _as_float_col(df, c)
                df.loc[bad_future, c] = np.nan

    if bad_future_total:
        print("  - bad_future counts: " + ", ".join([f"h{h}={bad_future_total[h]}" for h in horizons]))
    else:
        print("  - bad_future counts: (no label horizons detected)")

    # ----------------------------
    # (2) Past-return features sanitize
    # ----------------------------
    lags = _present_return_lags(df)
    bad_past_total: Dict[int, int] = {}
    for lag in lags:
        dt_past = _min_diff(ts, ts.shift(lag))
        bad_past = dt_past.isna() | ((dt_past - float(lag)).abs() > tol_min)
        bad_past_total[lag] = int(bad_past.sum())

        r_col = f"r{lag}"
        _as_float_col(df, r_col)
        df.loc[bad_past, r_col] = 0.0

    if bad_past_total:
        print("  - bad_past counts: " + ", ".join([f"r{lag}={bad_past_total[lag]}" for lag in lags]))
    else:
        print("  - bad_past counts: (no return lag columns detected)")

    # ----------------------------
    # (3) prev_close-dependent features sanitize on dt1 gap
    # ----------------------------
    bad_1 = dt1.isna() | ((dt1 - 1.0).abs() > tol_min)
    bad1_cnt = int(bad_1.sum())
    for c in PREV_CLOSE_DEP:
        if c in df.columns:
            _as_float_col(df, c)
            df.loc[bad_1, c] = 0.0
    print(f"  - prev_close_dep neutralized rows: {bad1_cnt}")

    # ----------------------------
    # (4) Feature NaN/inf safety (no lookahead)
    # ----------------------------
    for c in features:
        _as_float_col(df, c)
        s = df[c].astype("float64").replace([np.inf, -np.inf], np.nan)
        df[c] = s

    if ffill_features and features:
        # safe: only uses past, no lookahead
        df[features] = df[features].ffill()

    if fillna_zero and features:
        # remaining NaN (leading warmup etc.) -> 0.0 fallback
        df[features] = df[features].fillna(0.0)

    # aux numeric columns
    for c in ["minutes_to_next_funding", "funding_diff", "taker_buy_ratio", "spread_proxy"]:
        if c in df.columns:
            _as_float_col(df, c)
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    for c in ["funding_diff", "taker_buy_ratio", "spread_proxy"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    if sync_y5_alias and "y5_class" in df.columns:
        df["y_class"] = df["y5_class"]

    # ----------------------------
    # Report
    # ----------------------------
    t0 = ts.iloc[0] if len(ts) else None
    t1 = ts.iloc[-1] if len(ts) else None
    print(f"  - time range: {t0}  ->  {t1}")
    _report_targets(df, horizons=horizons)
    _report_features(df, features=features, contract_name=contract_name)

    # ----------------------------
    # Save
    # ----------------------------
    if report_only:
        print("🧾 report-only mode: not saving.")
        return

    out_path = path if inplace else path.with_name(path.stem + out_suffix + path.suffix)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collector_v8 FEAT30 + legacy FEAT22 contract-aware sanitizer")
    ap.add_argument(
        "--files",
        nargs="+",
        default=["./train_clean_final_past.csv", "./train_clean_final_future.csv"],
        help="정화할 CSV 파일들 (기본: past/future)",
    )
    ap.add_argument("--tol-min", type=float, default=0.1, help="시간 정합성 허용 오차(분). 기본 0.1")
    ap.add_argument("--inplace", action="store_true", help="원본 파일 덮어쓰기")
    ap.add_argument("--out-suffix", type=str, default="_feat30_sanitized", help="inplace가 아닐 때 붙일 suffix")
    ap.add_argument("--report-only", action="store_true", help="리포트만 출력하고 저장하지 않음")
    ap.add_argument("--no-ffill-features", action="store_true", help="contract feature ffill 비활성화")
    ap.add_argument("--no-fillna-zero", action="store_true", help="contract feature 남은 NaN 0-fill 비활성화")
    ap.add_argument("--no-sync-y5-alias", action="store_true", help="y5_class <-> y_class alias 동기화 비활성화")

    args = ap.parse_args()
    ffill_features = not args.no_ffill_features
    fillna_zero = not args.no_fillna_zero
    sync_y5_alias = not args.no_sync_y5_alias

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"⚠️ File not found: {p} (skip)")
            continue
        sanitize_one_file(
            path=p,
            tol_min=float(args.tol_min),
            inplace=bool(args.inplace),
            out_suffix=str(args.out_suffix),
            report_only=bool(args.report_only),
            ffill_features=ffill_features,
            fillna_zero=fillna_zero,
            sync_y5_alias=sync_y5_alias,
        )

    print("\n🎉 Contract-aware sanitize 완료!")


if __name__ == "__main__":
    main()