from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn could not be imported. Please run the script with "
        "'./.venv/bin/python random_forest_portfolio.py'."
    ) from exc


@dataclass
class PortfolioModelConfig:
    """
    Central configuration for the backtest.

    The most important knobs are:
    - prediction_horizon_days: forecast horizon
    - top_n_stocks: number of holdings
    - rebalance_every_n_days: rebalance frequency
    """

    dataset_file: Path = Path("data/omxs30_features_dataset.csv")
    index_file: Path = Path("data/omxs30_index_close.xlsx")
    index_sheet: str = "Close Price"
    output_dir: Path = Path("data/backtest_results")
    prediction_horizon_days: int = 5
    execution_lag_days: int = 1
    top_n_stocks: int = 5
    rebalance_every_n_days: int = 5
    test_size: float = 0.20
    random_state: int = 42
    random_strategy_trials: int = 10
    n_estimators: int = 5
    max_depth: int | None = 8
    min_samples_leaf: int = 5
    n_jobs: int = -1
    initial_portfolio_value: float = 1.0
    include_walk_forward: bool = True
    walk_forward_refit_every_n_rebalances: int = 5
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "daily_return",
            "return_5d",
            "return_20d",
            "return_60d",
            "return_120d",
            "volatility_20d",
            "volatility_60d",
            "distance_to_ma_20d",
            "distance_to_ma_60d",
            "max_drawdown_60d",
            "beta_vs_omxs30_60d",
            "relative_strength_vs_omxs30_20d",
        ]
    )


class RandomForestPortfolioBacktester:
    """
    Backtest a Random Forest strategy with clear, leakage-aware logic.
    """

    def __init__(self, config: PortfolioModelConfig | None = None) -> None:
        self.config = config or PortfolioModelConfig()

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Run the full workflow and return the main output tables.
        """

        dataset = self._load_dataset()
        index_data = self._load_index_data()
        model_data = self._prepare_model_dataset(dataset)

        train_end_date, test_start_date = self._calculate_split_dates(model_data)
        leakage_audit = self._audit_original_split_leakage(model_data, train_end_date, test_start_date)
        rebalance_dates = self._select_rebalance_dates(model_data, test_start_date)

        static_train_data = model_data[model_data["target_end_date"] <= test_start_date].copy()
        rebalance_test_data = model_data[model_data["date"].isin(rebalance_dates)].copy()

        static_predictions = self._run_static_model(static_train_data, rebalance_test_data)

        if self.config.include_walk_forward:
            walk_forward_predictions = self._run_walk_forward_model(model_data, rebalance_dates)
            walk_forward_portfolio = self._build_model_portfolio(
                walk_forward_predictions,
                strategy_name="walk_forward_random_forest",
            )
        else:
            walk_forward_predictions = pd.DataFrame()
            walk_forward_portfolio = pd.DataFrame()

        static_portfolio = self._build_model_portfolio(
            static_predictions,
            strategy_name="static_random_forest",
        )
        equal_weight_portfolio = self._build_equal_weight_portfolio(model_data, rebalance_dates)
        benchmark_portfolio = self._build_benchmark_portfolio(index_data, rebalance_dates)
        random_trial_summary = self._run_random_strategy_trials(model_data, rebalance_dates)

        prediction_frames = [static_predictions]
        if not walk_forward_predictions.empty:
            prediction_frames.append(walk_forward_predictions)
        predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["model_type", "date", "prediction_rank"]
        ).reset_index(drop=True)

        strategy_frames = [static_portfolio, equal_weight_portfolio, benchmark_portfolio]
        if not walk_forward_portfolio.empty:
            strategy_frames.append(walk_forward_portfolio)
        strategy_returns = pd.concat(strategy_frames, ignore_index=True).sort_values(
            ["strategy_name", "rebalance_date"]
        ).reset_index(drop=True)

        summary = self._build_summary(
            leakage_audit=leakage_audit,
            static_train_data=static_train_data,
            static_predictions=static_predictions,
            walk_forward_predictions=walk_forward_predictions,
            strategy_returns=strategy_returns,
            random_trial_summary=random_trial_summary,
        )

        results = {
            "summary": summary,
            "strategy_returns": strategy_returns,
            "predictions": predictions,
            "random_trial_summary": random_trial_summary,
        }
        self._save_outputs(results)
        return results

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load the feature dataset.
        """

        dataset = pd.read_csv(self.config.dataset_file)
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset = dataset.sort_values(["ticker", "date"]).reset_index(drop=True)
        return dataset

    def _load_index_data(self) -> pd.DataFrame:
        """
        Load OMXS30 index prices and build future benchmark returns.
        """

        index_data = pd.read_excel(self.config.index_file, sheet_name=self.config.index_sheet)
        index_data["Date"] = pd.to_datetime(index_data["Date"])
        index_data["Close price"] = pd.to_numeric(index_data["Close price"], errors="coerce")
        index_data = index_data.dropna(subset=["Close price"]).sort_values("Date").reset_index(drop=True)

        horizon = self.config.prediction_horizon_days
        entry_lag = self.config.execution_lag_days
        index_data["entry_close_price"] = index_data["Close price"].shift(-entry_lag)
        index_data["future_close_price"] = index_data["Close price"].shift(-(entry_lag + horizon))
        index_data["benchmark_return"] = (
            index_data["future_close_price"] / index_data["entry_close_price"]
        ) - 1.0
        return index_data.rename(columns={"Date": "date", "Close price": "benchmark_close_price"})

    def _prepare_model_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create the leakage-aware supervised learning table.

        target_end_date tells us when the future return becomes known.
        """

        model_data = dataset.copy()
        horizon = self.config.prediction_horizon_days
        entry_lag = self.config.execution_lag_days

        model_data["entry_close_price"] = model_data.groupby("ticker")["close_price"].shift(-entry_lag)
        model_data["future_close_price"] = model_data.groupby("ticker")["close_price"].shift(
            -(entry_lag + horizon)
        )
        model_data["target_end_date"] = model_data.groupby("ticker")["date"].shift(
            -(entry_lag + horizon)
        )
        model_data["target_return"] = (
            model_data["future_close_price"] / model_data["entry_close_price"]
        ) - 1.0

        required_columns = [
            "date",
            "ticker",
            "close_price",
            "entry_close_price",
            "target_return",
            "target_end_date",
            *self.config.feature_columns,
        ]
        model_data = model_data.dropna(subset=required_columns).reset_index(drop=True)
        return model_data.sort_values(["date", "ticker"]).reset_index(drop=True)

    def _calculate_split_dates(self, model_data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Calculate chronological train/test split dates.
        """

        unique_dates = np.array(sorted(model_data["date"].unique()))
        split_index = int(len(unique_dates) * (1.0 - self.config.test_size))
        split_index = min(max(split_index, 1), len(unique_dates) - 1)
        return pd.Timestamp(unique_dates[split_index - 1]), pd.Timestamp(unique_dates[split_index])

    def _audit_original_split_leakage(
        self,
        model_data: pd.DataFrame,
        train_end_date: pd.Timestamp,
        test_start_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Audit how many rows would have leaked in the original train-once setup.
        """

        original_train = model_data[model_data["date"] <= train_end_date].copy()
        leaky_rows = original_train[original_train["target_end_date"] >= test_start_date].copy()

        return pd.DataFrame(
            [
                {"metric": "train_end_date", "value": train_end_date.date().isoformat()},
                {"metric": "test_start_date", "value": test_start_date.date().isoformat()},
                {"metric": "original_train_rows", "value": len(original_train)},
                {"metric": "leaky_train_rows", "value": len(leaky_rows)},
                {"metric": "leaky_train_dates", "value": leaky_rows["date"].nunique()},
            ]
        )

    def _select_rebalance_dates(
        self, model_data: pd.DataFrame, test_start_date: pd.Timestamp
    ) -> list[pd.Timestamp]:
        """
        Select only the rebalance dates from the test period.
        """

        test_dates = np.array(sorted(model_data.loc[model_data["date"] >= test_start_date, "date"].unique()))
        return [pd.Timestamp(value) for value in test_dates[:: self.config.rebalance_every_n_days]]

    def _make_model(self) -> RandomForestRegressor:
        """
        Create a fresh Random Forest model instance.
        """

        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def _run_static_model(
        self, static_train_data: pd.DataFrame, rebalance_test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Train one static model and predict only on rebalance dates.
        """

        model = self._make_model()
        model.fit(static_train_data[self.config.feature_columns], static_train_data["target_return"])

        predictions = rebalance_test_data.copy()
        predictions["predicted_return"] = model.predict(predictions[self.config.feature_columns])
        predictions["model_type"] = "static_random_forest"
        predictions["training_rows_used"] = len(static_train_data)
        predictions["prediction_rank"] = predictions.groupby("date")["predicted_return"].rank(
            method="first",
            ascending=False,
        )
        return predictions.sort_values(["date", "prediction_rank"]).reset_index(drop=True)

    def _run_walk_forward_model(
        self, model_data: pd.DataFrame, rebalance_dates: list[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Re-train the model at each rebalance date using only already-known data.
        """

        prediction_rows: list[pd.DataFrame] = []
        fitted_model: RandomForestRegressor | None = None
        latest_training_size = -1
        feature_matrix = model_data[self.config.feature_columns].to_numpy(dtype=np.float32, copy=False)
        target_vector = model_data["target_return"].to_numpy(dtype=np.float32, copy=False)
        sorted_by_target_end = model_data.sort_values("target_end_date", kind="mergesort").reset_index()
        sorted_train_indices = sorted_by_target_end["index"].to_numpy()
        sorted_target_end_ns = sorted_by_target_end["target_end_date"].astype("int64").to_numpy()
        prediction_index_by_date = {
            pd.Timestamp(date_value): np.asarray(index_values, dtype=np.int64)
            for date_value, index_values in model_data.groupby("date").indices.items()
        }
        metadata_columns = ["date", "ticker", "close_price", "target_return"]

        for rebalance_index, rebalance_date in enumerate(rebalance_dates):
            train_limit = np.searchsorted(
                sorted_target_end_ns,
                rebalance_date.value,
                side="right",
            )
            train_indices = sorted_train_indices[:train_limit]
            prediction_indices = prediction_index_by_date.get(rebalance_date)

            if train_limit == 0 or prediction_indices is None or len(prediction_indices) == 0:
                continue

            should_refit = (
                fitted_model is None
                or rebalance_index % self.config.walk_forward_refit_every_n_rebalances == 0
                or train_limit != latest_training_size
            )

            if should_refit:
                fitted_model = self._make_model()
                fitted_model.fit(feature_matrix[train_indices], target_vector[train_indices])
                latest_training_size = train_limit

            prediction_universe = model_data.iloc[prediction_indices][metadata_columns].copy()
            prediction_universe["predicted_return"] = fitted_model.predict(
                feature_matrix[prediction_indices]
            )
            prediction_universe["model_type"] = "walk_forward_random_forest"
            prediction_universe["training_rows_used"] = train_limit
            prediction_universe["prediction_rank"] = prediction_universe["predicted_return"].rank(
                method="first",
                ascending=False,
            )
            prediction_rows.append(prediction_universe)

        if not prediction_rows:
            return pd.DataFrame()

        return pd.concat(prediction_rows, ignore_index=True).sort_values(
            ["date", "prediction_rank"]
        ).reset_index(drop=True)

    def _build_model_portfolio(
        self, prediction_frame: pd.DataFrame, strategy_name: str
    ) -> pd.DataFrame:
        """
        Convert ranked predictions into an equal-weight top-N portfolio.
        """

        portfolio_rows: list[dict[str, object]] = []
        portfolio_value = self.config.initial_portfolio_value

        for rebalance_date, daily_frame in prediction_frame.groupby("date"):
            selected = daily_frame.nlargest(self.config.top_n_stocks, "predicted_return").copy()
            if selected.empty:
                continue

            period_return = float(selected["target_return"].mean())
            portfolio_value *= 1.0 + period_return

            portfolio_rows.append(
                {
                    "strategy_name": strategy_name,
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "holding_period_days": self.config.prediction_horizon_days,
                    "number_of_holdings": len(selected),
                    "selection_note": ",".join(selected["ticker"].tolist()),
                    "signal_value": float(selected["predicted_return"].mean()),
                    "realized_period_return": period_return,
                    "cumulative_value": portfolio_value,
                }
            )

        return pd.DataFrame(portfolio_rows)

    def _build_equal_weight_portfolio(
        self, model_data: pd.DataFrame, rebalance_dates: list[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Build an equal-weight portfolio of all available stocks.
        """

        portfolio_rows: list[dict[str, object]] = []
        portfolio_value = self.config.initial_portfolio_value

        for rebalance_date in rebalance_dates:
            daily_frame = model_data[model_data["date"] == rebalance_date].copy()
            if daily_frame.empty:
                continue

            period_return = float(daily_frame["target_return"].mean())
            portfolio_value *= 1.0 + period_return

            portfolio_rows.append(
                {
                    "strategy_name": "equal_weight_universe",
                    "rebalance_date": rebalance_date,
                    "holding_period_days": self.config.prediction_horizon_days,
                    "number_of_holdings": len(daily_frame),
                    "selection_note": "ALL_AVAILABLE_STOCKS",
                    "signal_value": np.nan,
                    "realized_period_return": period_return,
                    "cumulative_value": portfolio_value,
                }
            )

        return pd.DataFrame(portfolio_rows)

    def _build_benchmark_portfolio(
        self, index_data: pd.DataFrame, rebalance_dates: list[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Build the OMXS30 benchmark on the same rebalance schedule.
        """

        benchmark_frame = index_data[index_data["date"].isin(pd.to_datetime(rebalance_dates))].copy()
        benchmark_frame = benchmark_frame.dropna(subset=["benchmark_return"]).sort_values("date")

        benchmark_value = self.config.initial_portfolio_value
        rows: list[dict[str, object]] = []

        for _, row in benchmark_frame.iterrows():
            period_return = float(row["benchmark_return"])
            benchmark_value *= 1.0 + period_return
            rows.append(
                {
                    "strategy_name": "omxs30_benchmark",
                    "rebalance_date": pd.Timestamp(row["date"]),
                    "holding_period_days": self.config.prediction_horizon_days,
                    "number_of_holdings": 1,
                    "selection_note": "OMXS30_INDEX",
                    "signal_value": float(row["benchmark_close_price"]),
                    "realized_period_return": period_return,
                    "cumulative_value": benchmark_value,
                }
            )

        return pd.DataFrame(rows)

    def _run_random_strategy_trials(
        self, model_data: pd.DataFrame, rebalance_dates: list[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Run random top-N selection many times to create a baseline distribution.
        """

        trial_rows: list[dict[str, object]] = []

        for trial_id in range(self.config.random_strategy_trials):
            rng = np.random.default_rng(self.config.random_state + trial_id)
            portfolio_value = self.config.initial_portfolio_value

            for rebalance_date in rebalance_dates:
                daily_frame = model_data[model_data["date"] == rebalance_date].copy()
                if daily_frame.empty:
                    continue

                sample_size = min(self.config.top_n_stocks, len(daily_frame))
                chosen_index = rng.choice(daily_frame.index.to_numpy(), size=sample_size, replace=False)
                selected = daily_frame.loc[chosen_index]
                portfolio_value *= 1.0 + float(selected["target_return"].mean())

            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "final_portfolio_value": portfolio_value,
                    "total_return": portfolio_value / self.config.initial_portfolio_value - 1.0,
                }
            )

        return pd.DataFrame(trial_rows)

    def _calculate_prediction_metrics(
        self, prediction_frame: pd.DataFrame, label: str
    ) -> list[dict[str, object]]:
        """
        Calculate error metrics for one prediction table.
        """

        if prediction_frame.empty:
            return [
                {"metric": f"{label}_mse", "value": np.nan},
                {"metric": f"{label}_rmse", "value": np.nan},
                {"metric": f"{label}_mae", "value": np.nan},
                {"metric": f"{label}_r2", "value": np.nan},
            ]

        y_true = prediction_frame["target_return"]
        y_pred = prediction_frame["predicted_return"]
        mse = mean_squared_error(y_true, y_pred)

        return [
            {"metric": f"{label}_mse", "value": mse},
            {"metric": f"{label}_rmse", "value": float(np.sqrt(mse))},
            {"metric": f"{label}_mae", "value": mean_absolute_error(y_true, y_pred)},
            {"metric": f"{label}_r2", "value": r2_score(y_true, y_pred)},
        ]

    def _calculate_strategy_metrics(
        self, strategy_returns: pd.DataFrame, strategy_name: str
    ) -> list[dict[str, object]]:
        """
        Calculate summary metrics for one strategy.
        """

        sub = strategy_returns[strategy_returns["strategy_name"] == strategy_name].copy()
        if sub.empty:
            return [
                {"metric": f"{strategy_name}_rebalances", "value": 0},
                {"metric": f"{strategy_name}_average_period_return", "value": np.nan},
                {"metric": f"{strategy_name}_total_return", "value": np.nan},
            ]

        total_return = float(sub["cumulative_value"].iloc[-1] / self.config.initial_portfolio_value - 1.0)

        return [
            {"metric": f"{strategy_name}_rebalances", "value": len(sub)},
            {"metric": f"{strategy_name}_average_period_return", "value": float(sub["realized_period_return"].mean())},
            {"metric": f"{strategy_name}_total_return", "value": total_return},
        ]

    def _build_summary(
        self,
        leakage_audit: pd.DataFrame,
        static_train_data: pd.DataFrame,
        static_predictions: pd.DataFrame,
        walk_forward_predictions: pd.DataFrame,
        strategy_returns: pd.DataFrame,
        random_trial_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build one compact summary table for the whole backtest.
        """

        benchmark_total_return = self._get_strategy_total_return(strategy_returns, "omxs30_benchmark")
        static_total_return = self._get_strategy_total_return(strategy_returns, "static_random_forest")
        walk_forward_total_return = self._get_strategy_total_return(
            strategy_returns, "walk_forward_random_forest"
        )

        rows: list[dict[str, object]] = [
            {"metric": "prediction_horizon_days", "value": self.config.prediction_horizon_days},
            {"metric": "execution_lag_days", "value": self.config.execution_lag_days},
            {"metric": "top_n_stocks", "value": self.config.top_n_stocks},
            {"metric": "rebalance_every_n_days", "value": self.config.rebalance_every_n_days},
            {"metric": "random_strategy_trials", "value": self.config.random_strategy_trials},
            {"metric": "include_walk_forward", "value": self.config.include_walk_forward},
            {
                "metric": "walk_forward_refit_every_n_rebalances",
                "value": self.config.walk_forward_refit_every_n_rebalances,
            },
            {"metric": "static_training_rows", "value": len(static_train_data)},
        ]

        rows.extend(leakage_audit.to_dict(orient="records"))
        rows.extend(self._calculate_prediction_metrics(static_predictions, "static_model"))
        rows.extend(self._calculate_prediction_metrics(walk_forward_predictions, "walk_forward_model"))
        rows.extend(self._calculate_strategy_metrics(strategy_returns, "static_random_forest"))
        rows.extend(self._calculate_strategy_metrics(strategy_returns, "walk_forward_random_forest"))
        rows.extend(self._calculate_strategy_metrics(strategy_returns, "equal_weight_universe"))
        rows.extend(self._calculate_strategy_metrics(strategy_returns, "omxs30_benchmark"))

        rows.extend(
            [
                {"metric": "random_mean_total_return", "value": float(random_trial_summary["total_return"].mean())},
                {"metric": "random_median_total_return", "value": float(random_trial_summary["total_return"].median())},
                {"metric": "random_std_total_return", "value": float(random_trial_summary["total_return"].std())},
                {"metric": "random_min_total_return", "value": float(random_trial_summary["total_return"].min())},
                {"metric": "random_max_total_return", "value": float(random_trial_summary["total_return"].max())},
                {
                    "metric": "static_excess_vs_benchmark",
                    "value": static_total_return - benchmark_total_return,
                },
                {
                    "metric": "walk_forward_excess_vs_benchmark",
                    "value": walk_forward_total_return - benchmark_total_return,
                },
                {
                    "metric": "static_excess_vs_equal_weight_universe",
                    "value": static_total_return - self._get_strategy_total_return(
                        strategy_returns, "equal_weight_universe"
                    ),
                },
                {
                    "metric": "static_percentile_vs_random",
                    "value": float((random_trial_summary["total_return"] <= static_total_return).mean()),
                },
                {
                    "metric": "walk_forward_percentile_vs_random",
                    "value": float((random_trial_summary["total_return"] <= walk_forward_total_return).mean()),
                },
            ]
        )

        return pd.DataFrame(rows)

    def _get_strategy_total_return(
        self, strategy_returns: pd.DataFrame, strategy_name: str
    ) -> float:
        """
        Safely read the final total return for one strategy.
        """

        sub = strategy_returns[strategy_returns["strategy_name"] == strategy_name]
        if sub.empty:
            return np.nan

        return float(sub["cumulative_value"].iloc[-1] / self.config.initial_portfolio_value - 1.0)

    def _build_output_guide(self) -> str:
        """
        Create a short human-readable guide for the output folder.
        """

        return (
            "# Backtest Results Guide\n\n"
            "This folder contains the cleaned backtest outputs.\n\n"
            "Files:\n"
            "- `summary.csv`: high-level metrics and sanity checks.\n"
            "- `strategy_returns.csv`: one row per rebalance date and strategy.\n"
            "- `predictions.csv`: model rankings on rebalance dates only.\n"
            "- `random_baseline_summary.csv`: one row per random trial.\n\n"
            "Important notes:\n"
            "- `predictions.csv` only contains rebalance dates, not every trading day.\n"
            "- By default, the backtest uses a 1-day execution lag: signals from date t are traded from t+1 onward.\n"
            "- `static_random_forest` is train-once but leakage-safe.\n"
            "- `walk_forward_random_forest` is optional and only appears when enabled in the config.\n"
            "- `equal_weight_universe` is a naive baseline that holds all stocks equally.\n"
            "- `omxs30_benchmark` is the index benchmark over the same holding horizon.\n"
            "- The stock universe uses the current OMXS30 names across history, which introduces survivorship bias.\n"
            "- Because OMXS30 is market-cap weighted while `equal_weight_universe` is equal-weighted, beating the index alone is not enough to prove alpha.\n"
            "- A more honest comparison is often `static_random_forest` versus `equal_weight_universe` and the random baseline distribution.\n"
        )

    def _save_outputs(self, results: dict[str, pd.DataFrame]) -> None:
        """
        Save a smaller and clearer output set.
        """

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        results["summary"].to_csv(self.config.output_dir / "summary.csv", index=False)
        results["strategy_returns"].to_csv(self.config.output_dir / "strategy_returns.csv", index=False)

        prediction_columns = [
            "date",
            "ticker",
            "model_type",
            "training_rows_used",
            "predicted_return",
            "target_return",
            "prediction_rank",
        ]
        results["predictions"][prediction_columns].to_csv(
            self.config.output_dir / "predictions.csv",
            index=False,
        )
        results["random_trial_summary"].to_csv(
            self.config.output_dir / "random_baseline_summary.csv",
            index=False,
        )
        (self.config.output_dir / "README.md").write_text(self._build_output_guide(), encoding="utf-8")


def main() -> None:
    """
    Run the cleaned backtest pipeline and print a concise overview.
    """

    parser = argparse.ArgumentParser(description="Run the OMXS30 Random Forest backtest.")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable full walk-forward evaluation.",
    )
    parser.add_argument(
        "--walk-forward-refit-every",
        type=int,
        default=1,
        help="Refit frequency for walk-forward mode. Use 1 for true full walk-forward.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=None,
        help="Override the number of random baseline trials.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override the number of trees in the Random Forest.",
    )
    args = parser.parse_args()

    config = PortfolioModelConfig()
    if args.walk_forward:
        config.include_walk_forward = True
        config.walk_forward_refit_every_n_rebalances = max(1, args.walk_forward_refit_every)
    if args.random_trials is not None:
        config.random_strategy_trials = max(0, args.random_trials)
    if args.n_estimators is not None:
        config.n_estimators = max(1, args.n_estimators)

    backtester = RandomForestPortfolioBacktester(config)
    results = backtester.run()

    print(f"Saved outputs to: {backtester.config.output_dir}")
    print("\nSummary sample:")
    print(results["summary"].head(20).to_string(index=False))
    print("\nStrategy return sample:")
    print(results["strategy_returns"].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
