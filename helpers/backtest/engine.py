from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest.config import BacktestConfig, BenchmarkConfig
from backtest.factor_processor import ProcessedFactor


@dataclass(frozen=True)
class BacktestResult:
    factor_name: str
    nav_df: pd.DataFrame
    period_return_df: pd.DataFrame
    benchmark_return_df: pd.DataFrame
    factor_return_df: pd.DataFrame
    meta: dict[str, Any]


class BacktestEngine:
    """执行分组回测主流程。"""

    def __init__(
        self,
        backtest_cfg: BacktestConfig,
        benchmark_cfg: BenchmarkConfig,
        index_quotes: pd.DataFrame | None,
    ):
        self.backtest_cfg = backtest_cfg
        self.benchmark_cfg = benchmark_cfg
        self.index_quotes = index_quotes

    def Run(self, processed: ProcessedFactor) -> BacktestResult:
        prepared = self._prepare_rebalance_frame(processed)
        group_labels = [f"Group{i}" for i in range(1, self.backtest_cfg.groups + 1)]
        all_return_index = group_labels + self._benchmark_columns() + self._excess_columns()

        period_return_history: dict[pd.Timestamp, pd.Series] = {}
        nav_history: dict[pd.Timestamp, pd.Series] = {}
        nav_state = pd.Series(1.0, index=all_return_index, dtype=float)
        previous_group_df: pd.DataFrame | None = None
        factor_return_rows: list[pd.DataFrame] = []

        trading_dates = [pd.Timestamp(date) for date in sorted(prepared["TradingDate"].dropna().unique())]
        if len(trading_dates) < 2:
            raise ValueError("At least two rebalance dates are required.")

        benchmark_returns = self._build_benchmark_returns(prepared, processed)

        for current_date in trading_dates[:-1]:

            today = prepared[prepared["TradingDate"] == current_date].copy()
            today = self._assign_groups(today, group_labels)
            if today.empty:
                continue

            if previous_group_df is None:
                today = today[(today["Suspension"] == 0) & (today["LimitStatus"] == 0)].copy()
            else:
                today = today.merge(previous_group_df, on="Stkcd", how="left")
                blocked = (today["Suspension"] == 1) | today["LimitStatus"].isin([1, -1])
                today.loc[blocked, "quantile_group"] = today.loc[blocked, "previous_group"]

            today = today.dropna(subset=["quantile_group"]).copy()
            if today.empty:
                continue

            factor_return_rows.append(
                today[["TradingDate", "Stkcd", processed.factor_name, "ret", "quantile_group"]].copy()
            )

            r_mean = pd.Series(0.0, index=all_return_index, name=current_date)
            group_returns = today.groupby("quantile_group", observed=False)["ret"].mean()
            for label, value in group_returns.items():
                r_mean.loc[str(label)] = value

            if previous_group_df is not None:
                r_mean = self._apply_transaction_cost(today, group_labels, r_mean)

            for col in self._benchmark_columns():
                r_mean.loc[col] = benchmark_returns.get(col, pd.Series(dtype=float)).get(current_date, 0.0)

            high_group = group_labels[-1]
            low_group = group_labels[0]
            if "Benchmark_pool" in r_mean.index:
                r_mean.loc["Excess_pool_high"] = r_mean.loc[high_group] - r_mean.loc["Benchmark_pool"]
                r_mean.loc["Excess_pool_low"] = r_mean.loc[low_group] - r_mean.loc["Benchmark_pool"]
            for code in self.benchmark_cfg.index_codes:
                benchmark_col = f"Benchmark_index_{code}"
                if benchmark_col not in r_mean.index:
                    continue
                r_mean.loc[f"Excess_index_{code}_high"] = r_mean.loc[high_group] - r_mean.loc[benchmark_col]
                r_mean.loc[f"Excess_index_{code}_low"] = r_mean.loc[low_group] - r_mean.loc[benchmark_col]

            period_return_history[current_date] = r_mean.copy()
            nav_state = nav_state * (1.0 + r_mean.fillna(0.0))
            nav_history[current_date] = nav_state.copy()
            previous_group_df = today[["Stkcd", "quantile_group"]].rename(
                columns={"quantile_group": "previous_group"}
            )

        nav_df = pd.DataFrame(nav_history).T.sort_index()
        period_return_df = pd.DataFrame(period_return_history).T.sort_index()
        benchmark_return_df = period_return_df[self._benchmark_columns()].copy()
        factor_return_df = (
            pd.concat(factor_return_rows, ignore_index=True)
            if factor_return_rows
            else pd.DataFrame(columns=["TradingDate", "Stkcd", processed.factor_name, "ret", "quantile_group"])
        )

        nav_df = self._select_directional_excess(nav_df, group_labels)
        period_return_df = self._select_directional_excess(period_return_df, group_labels)

        return BacktestResult(
            factor_name=processed.factor_name,
            nav_df=nav_df,
            period_return_df=period_return_df,
            benchmark_return_df=benchmark_return_df,
            factor_return_df=factor_return_df,
            meta={
                "factor_name": processed.factor_name,
                "start_date": processed.start_date,
                "end_date": processed.end_date,
                "frequency": self.backtest_cfg.frequency,
                "groups": self.backtest_cfg.groups,
                "risk_free_rate": self.backtest_cfg.risk_free_rate,
                "advantage_group": self._advantage_group(nav_df, group_labels),
            },
        )

    def _prepare_rebalance_frame(self, processed: ProcessedFactor) -> pd.DataFrame:
        freq = self._pandas_frequency()
        factor_name = processed.factor_name
        price = self._resample_last(processed.price_df, freq, ["price"])
        factor = self._resample_last(processed.factor_df, freq, [factor_name])
        limit = self._resample_last(processed.suspension_limit_df, freq, ["Suspension", "LimitStatus"])

        df = factor.merge(limit, on=["TradingDate", "Stkcd"], how="right")
        df = df.merge(price, on=["TradingDate", "Stkcd"], how="right")
        df = df.sort_values(["Stkcd", "TradingDate"])
        df["ret"] = df.groupby("Stkcd")["price"].pct_change()
        df["ret"] = df.groupby("Stkcd")["ret"].shift(-1)
        df["ret"] = df.groupby("Stkcd")["ret"].transform(lambda x: x.fillna(0.0))
        df["Suspension"] = df["Suspension"].fillna(0)
        df["LimitStatus"] = df["LimitStatus"].fillna(0)
        return df.sort_values(["TradingDate", "Stkcd"]).copy()

    def _resample_last(self, df: pd.DataFrame, freq: str, value_cols: list[str]) -> pd.DataFrame:
        if self.backtest_cfg.frequency == "daily":
            return df[["TradingDate", "Stkcd"] + value_cols].copy()
        return (
            df.groupby([pd.Grouper(key="TradingDate", freq=freq), "Stkcd"])[value_cols]
            .last()
            .reset_index()
        )

    def _pandas_frequency(self) -> str:
        mapping = {
            "daily": "D",
            "weekly": "W-Fri",
            "monthly": "MS",
            "quarterly": "QS",
        }
        if self.backtest_cfg.frequency not in mapping:
            raise ValueError(f"Unsupported backtest frequency: {self.backtest_cfg.frequency}")
        return mapping[self.backtest_cfg.frequency]

    def _assign_groups(self, today: pd.DataFrame, group_labels: list[str]) -> pd.DataFrame:
        factor_name = [c for c in today.columns if c not in {
            "TradingDate",
            "Stkcd",
            "Suspension",
            "LimitStatus",
            "price",
            "ret",
        }][0]
        today = today.dropna(subset=[factor_name]).copy()
        if today.empty:
            return today
        try:
            today["quantile"] = today[factor_name].rank(method="first", pct=True)
            today["quantile_group"] = pd.qcut(
                today["quantile"],
                q=self.backtest_cfg.groups,
                labels=group_labels,
                duplicates="drop",
            )
            if today["quantile_group"].nunique(dropna=True) < self.backtest_cfg.groups:
                return today.iloc[0:0].copy()
        except ValueError:
            return today.iloc[0:0].copy()
        return today

    def _apply_transaction_cost(
        self,
        today: pd.DataFrame,
        group_labels: list[str],
        r_mean: pd.Series,
    ) -> pd.Series:
        for label in group_labels:
            group = today[today["quantile_group"] == label].copy()
            if group.empty:
                continue
            changed = group["quantile_group"] != group["previous_group"]
            transaction_cost = changed.sum() / len(group) * self.backtest_cfg.transaction_cost.total
            r_mean.loc[label] -= transaction_cost
        return r_mean

    def _build_benchmark_returns(
        self,
        prepared: pd.DataFrame,
        processed: ProcessedFactor,
    ) -> dict[str, pd.Series]:
        benchmark_returns: dict[str, pd.Series] = {}
        if self.benchmark_cfg.has_pool_equal_weight:
            benchmark_returns["Benchmark_pool"] = self._pool_benchmark_returns(processed)
        if self.index_quotes is not None:
            benchmark_returns.update(self._index_benchmark_returns(processed))
        return benchmark_returns

    def _pool_benchmark_returns(self, processed: ProcessedFactor) -> pd.Series:
        freq = self._pandas_frequency()
        price = self._resample_last(processed.benchmark_price_df, freq, ["price"])
        price = price.sort_values(["Stkcd", "TradingDate"])
        price["ret"] = price.groupby("Stkcd")["price"].pct_change()
        price["ret"] = price.groupby("Stkcd")["ret"].shift(-1)
        price["ret"] = price.groupby("Stkcd")["ret"].transform(lambda x: x.fillna(0.0))
        return price.groupby("TradingDate")["ret"].mean()

    def _index_benchmark_returns(self, processed: ProcessedFactor) -> dict[str, pd.Series]:
        result: dict[str, pd.Series] = {}
        quotes = self.index_quotes.copy()
        quotes["Indexcd"] = quotes["Indexcd"].astype(str)
        quotes = quotes[
            (quotes["TradingDate"] >= processed.start_date)
            & (quotes["TradingDate"] <= processed.end_date)
            & quotes["Indexcd"].isin(self.benchmark_cfg.index_codes)
        ].copy()
        if quotes.empty:
            return result
        freq = self._pandas_frequency()
        if self.backtest_cfg.frequency != "daily":
            quotes = (
                quotes.groupby([pd.Grouper(key="TradingDate", freq=freq), "Indexcd"])["Close"]
                .last()
                .reset_index()
            )
        quotes = quotes.sort_values(["Indexcd", "TradingDate"])
        quotes["ret"] = quotes.groupby("Indexcd")["Close"].pct_change()
        quotes["ret"] = quotes.groupby("Indexcd")["ret"].shift(-1)
        quotes["ret"] = quotes.groupby("Indexcd")["ret"].transform(lambda x: x.fillna(0.0))
        for code in self.benchmark_cfg.index_codes:
            series = quotes[quotes["Indexcd"] == code].set_index("TradingDate")["ret"]
            result[f"Benchmark_index_{code}"] = series
        return result

    def _benchmark_columns(self) -> list[str]:
        cols: list[str] = []
        if self.benchmark_cfg.has_pool_equal_weight:
            cols.append("Benchmark_pool")
        cols.extend([f"Benchmark_index_{code}" for code in self.benchmark_cfg.index_codes])
        return cols

    def _excess_columns(self) -> list[str]:
        cols: list[str] = []
        if self.benchmark_cfg.has_pool_equal_weight:
            cols.extend(["Excess_pool_high", "Excess_pool_low"])
        for code in self.benchmark_cfg.index_codes:
            cols.extend([f"Excess_index_{code}_high", f"Excess_index_{code}_low"])
        return cols

    def _select_directional_excess(self, df: pd.DataFrame, group_labels: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        advantage = self._advantage_group(df, group_labels)
        suffix = "high" if advantage == group_labels[-1] else "low"
        if f"Excess_pool_{suffix}" in df.columns:
            df["Excess_pool"] = df[f"Excess_pool_{suffix}"]
        for code in self.benchmark_cfg.index_codes:
            col = f"Excess_index_{code}_{suffix}"
            if col in df.columns:
                df[f"Excess_index_{code}"] = df[col]
        drop_cols = [c for c in df.columns if isinstance(c, str) and (c.endswith("_high") or c.endswith("_low"))]
        return df.drop(columns=drop_cols)

    def _advantage_group(self, df: pd.DataFrame, group_labels: list[str]) -> str:
        if df.empty:
            return group_labels[-1]
        low = group_labels[0]
        high = group_labels[-1]
        if high in df.columns and low in df.columns and df[high].mean() > df[low].mean():
            return high
        return low
