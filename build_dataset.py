from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """
    Central configuration for all rolling windows and output settings.

    The windows are expressed in trading days.
    """

    price_file: Path = Path("data/omxs30_close_prices_raw.xlsx")
    price_sheet: str = "Closing Prices"
    index_file: Path = Path("data/omxs30_index_close.xlsx")
    index_sheet: str = "Close Price"
    output_file: Path = Path("data/omxs30_features_dataset.csv")
    volatility_windows: tuple[int, ...] = (20, 60)
    moving_average_windows: tuple[int, ...] = (20, 60)
    return_windows: tuple[int, ...] = (1, 5, 20, 60, 120)
    drawdown_window: int = 60
    beta_window: int = 60
    relative_strength_window: int = 20
    annualize_volatility: bool = False
    trading_days_per_year: int = 252


class OMXS30FeatureBuilder:
    """
    Build a structured feature dataset from stock prices and OMXS30 index prices.

    The final dataset is returned in tidy format:
    one row per date and ticker.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def build_feature_dataframe(self) -> pd.DataFrame:
        """
        Run the full pipeline and return the final feature dataframe.
        """

        stock_prices_wide = self._load_stock_prices()
        index_prices = self._load_index_prices()

        stock_returns = self._calculate_return_features(stock_prices_wide)
        volatility_features = self._calculate_volatility_features(stock_returns["daily_return"])
        moving_average_features = self._calculate_moving_average_features(stock_prices_wide)
        drawdown_features = self._calculate_drawdown_feature(stock_prices_wide)
        beta_feature = self._calculate_beta_feature(stock_returns["daily_return"], index_prices)
        relative_strength_feature = self._calculate_relative_strength_feature(
            stock_prices_wide, index_prices
        )

        wide_features = {
            "close_price": stock_prices_wide,
            **stock_returns,
            **volatility_features,
            **moving_average_features,
            **drawdown_features,
            **beta_feature,
            **relative_strength_feature,
        }

        long_features = []

        for feature_name, feature_frame in wide_features.items():
            stacked = (
                feature_frame.stack(future_stack=True)
                .rename(feature_name)
                .reset_index()
            )
            stacked.columns = ["date", "ticker", feature_name]
            long_features.append(stacked)

        feature_dataframe = long_features[0]

        for feature_frame in long_features[1:]:
            feature_dataframe = feature_dataframe.merge(
                feature_frame, on=["date", "ticker"], how="left"
            )

        feature_dataframe = feature_dataframe.sort_values(["date", "ticker"]).reset_index(drop=True)
        return feature_dataframe

    def save_feature_dataframe(self, dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Save the final dataframe to CSV and also return it for direct use in code.
        """

        final_dataframe = dataframe if dataframe is not None else self.build_feature_dataframe()
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        final_dataframe.to_csv(self.config.output_file, index=False)
        return final_dataframe

    def _load_stock_prices(self) -> pd.DataFrame:
        """
        Load the 'Closing Prices' sheet and normalize it into a clean price matrix.

        The Excel sheet contains:
        - row 1: metadata text and repeated 'Price Close'
        - row 2: the actual header row with Date + ticker symbols
        - row 3 onward: the actual data

        This method explicitly rebuilds the correct headers before converting values.
        """

        raw = pd.read_excel(self.config.price_file, sheet_name=self.config.price_sheet, header=None)

        actual_header = raw.iloc[1].tolist()
        data = raw.iloc[2:].copy()
        data.columns = actual_header

        data["Date"] = pd.to_datetime(data["Date"])

        ticker_columns = [column for column in data.columns if column != "Date"]
        for ticker in ticker_columns:
            data[ticker] = pd.to_numeric(data[ticker], errors="coerce")

        stock_prices = data.set_index("Date").sort_index()
        stock_prices.index.name = "date"
        return stock_prices

    def _load_index_prices(self) -> pd.Series:
        """
        Load OMXS30 index prices as a clean time series.
        """

        index_prices = pd.read_excel(
            self.config.index_file,
            sheet_name=self.config.index_sheet,
        )

        index_prices["Date"] = pd.to_datetime(index_prices["Date"])
        index_prices["Close price"] = pd.to_numeric(index_prices["Close price"], errors="coerce")
        index_prices = index_prices.dropna(subset=["Close price"])

        index_series = (
            index_prices.set_index("Date")["Close price"].sort_index().rename("omxs30_close_price")
        )
        index_series.index.name = "date"
        return index_series

    def _calculate_return_features(self, stock_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate simple percentage returns over multiple horizons.

        Formula:
        return_t = (price_t / price_t-n) - 1
        """

        features: dict[str, pd.DataFrame] = {}

        for window in self.config.return_windows:
            feature_name = "daily_return" if window == 1 else f"return_{window}d"
            features[feature_name] = stock_prices.pct_change(periods=window, fill_method=None)

        return features

    def _calculate_volatility_features(
        self, daily_returns: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate rolling volatility from daily returns.

        By default, this is the rolling standard deviation of daily returns.
        If annualize_volatility=True, the result is multiplied by sqrt(252).
        """

        features: dict[str, pd.DataFrame] = {}
        annualization_factor = (
            np.sqrt(self.config.trading_days_per_year) if self.config.annualize_volatility else 1.0
        )

        for window in self.config.volatility_windows:
            rolling_volatility = daily_returns.rolling(window=window, min_periods=window).std()
            features[f"volatility_{window}d"] = rolling_volatility * annualization_factor

        return features

    def _calculate_moving_average_features(
        self, stock_prices: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate distance to rolling moving averages.

        Formula:
        distance_to_ma = (price / moving_average) - 1
        """

        features: dict[str, pd.DataFrame] = {}

        for window in self.config.moving_average_windows:
            moving_average = stock_prices.rolling(window=window, min_periods=window).mean()
            features[f"distance_to_ma_{window}d"] = (stock_prices / moving_average) - 1.0

        return features

    def _calculate_drawdown_feature(self, stock_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate the maximum drawdown observed inside the latest rolling window.

        Step 1:
        For every day, calculate the current drawdown relative to the running peak
        inside the trailing window.

        Step 2:
        Inside the same trailing window, keep the most negative drawdown value.

        The result is negative or zero:
        - 0.00 means no drawdown
        - -0.25 means a 25% maximum drawdown
        """

        window = self.config.drawdown_window
        rolling_peak = stock_prices.rolling(window=window, min_periods=window).max()
        current_drawdown = (stock_prices / rolling_peak) - 1.0
        maximum_drawdown = current_drawdown.rolling(window=window, min_periods=window).min()
        return {f"max_drawdown_{window}d": maximum_drawdown}

    def _calculate_beta_feature(
        self, daily_returns: pd.DataFrame, index_prices: pd.Series
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate rolling beta against OMXS30.

        Beta is computed as:
        covariance(stock_return, index_return) / variance(index_return)

        The result is aligned by trading date, and the same date index is used
        for every stock column.
        """

        window = self.config.beta_window
        index_returns = index_prices.pct_change(fill_method=None).rename("omxs30_daily_return")

        aligned_stock_returns = daily_returns.reindex(index_returns.index)
        rolling_index_variance = index_returns.rolling(window=window, min_periods=window).var()

        beta_frame = pd.DataFrame(index=aligned_stock_returns.index, columns=aligned_stock_returns.columns)

        for ticker in aligned_stock_returns.columns:
            rolling_covariance = aligned_stock_returns[ticker].rolling(
                window=window, min_periods=window
            ).cov(index_returns)
            beta_frame[ticker] = rolling_covariance / rolling_index_variance

        return {f"beta_vs_omxs30_{window}d": beta_frame}

    def _calculate_relative_strength_feature(
        self, stock_prices: pd.DataFrame, index_prices: pd.Series
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate relative strength against OMXS30.

        Assumption used here:
        relative_strength = stock_n_day_return - index_n_day_return

        This makes the feature easy to interpret:
        - positive value  -> the stock outperformed the index over the window
        - negative value  -> the stock underperformed the index over the window

        If you later want the pure price ratio instead, this method is the only
        place that needs to be changed.
        """

        window = self.config.relative_strength_window
        stock_window_returns = stock_prices.pct_change(periods=window, fill_method=None)
        index_window_return = index_prices.pct_change(periods=window, fill_method=None)

        aligned_index_return = index_window_return.reindex(stock_window_returns.index)
        relative_strength = stock_window_returns.sub(aligned_index_return, axis=0)

        return {f"relative_strength_vs_omxs30_{window}d": relative_strength}


def main() -> None:
    """
    Build the feature dataset, save it, and print a short preview.
    """

    builder = OMXS30FeatureBuilder()
    dataset = builder.save_feature_dataframe()

    print(f"Saved dataset to: {builder.config.output_file}")
    print(f"Dataset shape: {dataset.shape}")
    print(dataset.head(10).to_string())


if __name__ == "__main__":
    main()
