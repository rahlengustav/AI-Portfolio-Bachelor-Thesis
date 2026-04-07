# Backtest Results Guide

This folder contains the cleaned backtest outputs.

Files:
- `summary.csv`: high-level metrics and sanity checks.
- `strategy_returns.csv`: one row per rebalance date and strategy.
- `predictions.csv`: model rankings on rebalance dates only.
- `random_baseline_summary.csv`: one row per random trial.

Important notes:
- `predictions.csv` only contains rebalance dates, not every trading day.
- `static_random_forest` is train-once but leakage-safe.
- `walk_forward_random_forest` is optional and only appears when enabled in the config.
- `equal_weight_universe` is a naive baseline that holds all stocks equally.
- `omxs30_benchmark` is the index benchmark over the same holding horizon.
- The stock universe uses the current OMXS30 names across history, which introduces survivorship bias.
- Because OMXS30 is market-cap weighted while `equal_weight_universe` is equal-weighted, beating the index alone is not enough to prove alpha.
- A more honest comparison is often `static_random_forest` versus `equal_weight_universe` and the random baseline distribution.
