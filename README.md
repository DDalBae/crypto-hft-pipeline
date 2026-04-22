# Hybrid Scalp Quant Pipeline: TCN-LSTM Architecture

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides a deep learning-based quantitative trading pipeline designed for high-frequency scalping (HFT) on 1-minute (1m) cryptocurrency intervals, such as Binance ETHUSDT.

Going beyond simple time-series prediction, this project utilizes a hybrid architecture that combines the local feature extraction of **TCN (Temporal Convolutional Networks)** with the long-term memory capabilities of **LSTM**. It also includes a realistic backtesting environment that reflects fees and slippage, along with an automated hyperparameter tuning system powered by **Optuna**.

## Core Features

### 1. Hybrid TCN-LSTM Architecture  
**Files:** `hybrid_core_v7.py`, `hybrid_core_modified_v8.py`

A sophisticated hybrid neural network designed to capture non-linear patterns while controlling noise in financial data.

- **TCN (Temporal Convolutional Network):** Rapidly extracts local chart patterns such as tick micro-movements and short-term support/resistance using dilated causal convolutions.
- **LSTM (Long Short-Term Memory):** Learns long-term momentum trends and historical trajectories based on features extracted by the TCN.
- **Feature Attention (v8 Modified):** Implements an attention mechanism to assign dynamic weights to critical time steps, improving focus on high-alpha signals.

### 2. Multi-Objective Trainer  
**File:** `trainer_hybrid_all_ethusdt_v7.py`

A comprehensive training module that optimizes for multiple trading-specific objectives simultaneously.

- **Multi-Loss Optimization:** Jointly optimizes regression (magnitude), classification (direction), and hybrid entry scores.
- **Mixed Precision (AMP) & Gradient Clipping:** Supports Automatic Mixed Precision for GPU memory efficiency and implements gradient clipping for stable convergence.

### 3. RL-Mode Optuna Autotuning  
**File:** `autotune_hybrid_net_optuna_rlmode_v59.py`

An automated engine that searches for the optimal model backbone and execution thresholds.

- **Architecture Search:** Explores parameters such as `d_model`, `tcn_channels`, `lstm_hidden`, and dropout rates.
- **Reward-Driven Tuning:** Uses net profit and win rate derived from a simulated backtest environment (RL-mode) as primary optimization objectives instead of standard loss minimization.

### 4. Realistic Backtesting Engine  
**File:** `backtest_hybrid_scalp_with_costs_rlmode_v59.py`

A conservative backtester designed to reduce backtest illusions and verify real-world profitability.

- **Strict Transaction Costs:** Explicitly incorporates maker/taker fees and slippage into every trade execution.
- **Risk Management:** Implements dynamic Take-Profit (TP), Stop-Loss (SL), and maximum holding time constraints to reduce 1-minute market whipsaws.
- **Comprehensive Metrics:** Outputs key indicators including net profit, maximum drawdown (MDD), win rate, and Sharpe ratio.

## System Architecture & Workflow

The pipeline is designed as an end-to-end framework:

1. **Auto-Tuning**  
   Run the Optuna engine to derive optimal hyperparameters and execution gates.

2. **Model Training**  
   Execute a full-train run using the optimized configuration and save the best weights.

3. **Historical Backtest**  
   Verify performance on out-of-sample data with strict fee/slippage settings.

4. **Live Inference**  
   Load the trained weights into the streaming inference engine to generate real-time Long/Short/Hold signals.

## Quick Start

### 1. Hyperparameter Tuning

Optimize the architecture and thresholds using Optuna.

```bash
python autotune_hybrid_net_optuna_rlmode_v59.py   --data "data/ethusdt_features.parquet"   --n-trials 50
```

### 2. Model Training

Train the model with the optimized hyperparameters.

```bash
python trainer_hybrid_all_ethusdt_v7.py   --data "data/ethusdt_features.parquet"   --epochs 30   --batch-size 512
```

### 3. Backtesting

Validate the trained model in a realistic cost environment.

```bash
python backtest_hybrid_scalp_with_costs_rlmode_v59.py   --data "data/ethusdt_valid.parquet"   --model "weights/best_model_v7.pt"   --fee 0.0004   --slippage 0.0001
```

## Environment Requirements

- Python 3.9+
- PyTorch 2.x (CUDA-enabled recommended)
- Pandas
- NumPy
- Optuna
- Numba
- Optional: Binance API keys for live data streaming

## Quantitative Analysis & Data Pipeline (`/analysis_tools`)

### Strategy Diagnostics & Health Check
* **`analyze_trail_counterfactual_v3.py`**: A counterfactual analysis tool that simulates PnL impact under different fee scenarios (Maker vs. Taker) for Trailing Stop exits.
* **`diagnose_effective_geometry_v5.py`**: Analyzes trade geometry, including MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) relative to local Soft-SL levels.
* **`strategy_health_check_v2.py`**: Audits strategy robustness by analyzing profit concentration (Top 1%/5% share) and categorizing failure reasons (Early fail vs. Near-BEP fail).
* **`summarize_motion_profile_v3.py`**: Provides statistical breakdowns of holding times and price motion profiles categorized by exit reasons (TP, SL, TRAIL, MAX_HOLD).

### System Utilities
* **`regime_summary_with_filters_v5.py`**: Summarizes strategy pass-rates and performance metrics across different market regimes (Volatility, ATR, and Volume filters).
* **`smoke_check_final_softsl.py`**: Validates the consistency between entry-time Soft-SL logic and final exit execution.

## Disclaimer

This repository is intended for educational and research purposes only. High-frequency trading involves significant financial risk. The provided models and backtest results do not guarantee future performance. Use at your own risk.
