from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.config import BacktestConfig


EXCLUDED_BOARDS = {"chinext", "star"}


class EmptyFactorDataError(ValueError):
    """因子在有效区间内没有可回测数据。"""


@dataclass(frozen=True)
class ProcessedFactor:
    factor_name: str
    factor_df: pd.DataFrame
    price_df: pd.DataFrame
    benchmark_price_df: pd.DataFrame
    suspension_limit_df: pd.DataFrame
    start_date: str
    end_date: str


class FactorProcessor:
    """读取并清洗单个因子。"""

    def __init__(
        self,
        backtest_cfg: BacktestConfig,
        filter_stock_pool: pd.DataFrame,
        benchmark_stock_pool: pd.DataFrame,
        market_value: pd.DataFrame | None,
        logger: logging.Logger,
    ):
        self.backtest_cfg = backtest_cfg
        self.filter_stock_pool = filter_stock_pool
        self.benchmark_stock_pool = benchmark_stock_pool
        self.market_value = market_value
        self.logger = logger

    def LoadAndProcess(
        self,
        factor_path: Path,
        close_price: pd.DataFrame,
        suspension_limit: pd.DataFrame,
    ) -> ProcessedFactor:
        raw_factor, factor_name = self._load_factor_df(factor_path)
        raw_factor = self._standardize_columns(raw_factor)
        factor_name = self._resolve_factor_name(raw_factor, factor_name)

        if factor_name != factor_path.stem:
            raw_factor = raw_factor.rename(columns={factor_name: factor_path.stem})
            factor_name = factor_path.stem

        raw_factor["TradingDate"] = self._parse_trading_date(raw_factor["TradingDate"])
        start_date, end_date = self._resolve_effective_period(raw_factor)

        factor = self._clip_factor(raw_factor, factor_name, start_date, end_date)
        if factor.empty:
            raise EmptyFactorDataError(f"Factor data is empty after date filtering: {factor_path}")
        if self._is_discrete_signal_factor(factor, factor_name):
            raise EmptyFactorDataError(f"Discrete signal factor skipped: {factor_path}")

        price = self._prepare_price(close_price, start_date, end_date)
        benchmark_price = price.copy()
        limit = self._prepare_suspension_limit(suspension_limit, start_date, end_date)
        filter_pool_long = self._prepare_filter_stock_pool(start_date, end_date)
        benchmark_pool_long = self._prepare_benchmark_stock_pool(start_date, end_date)

        factor = self._filter_by_pool(factor, filter_pool_long)
        price = self._filter_by_pool(price, filter_pool_long)
        limit = self._filter_by_pool(limit, filter_pool_long)
        benchmark_price = self._filter_by_pool(benchmark_price, benchmark_pool_long)
        if factor.empty:
            raise EmptyFactorDataError(f"Factor data is empty after universe filtering: {factor_path}")

        factor[factor_name] = pd.to_numeric(factor[factor_name], errors="coerce")
        factor[factor_name] = factor[factor_name].replace([np.inf, -np.inf], np.nan)
        factor = factor.dropna(subset=[factor_name]).copy()
        factor[factor_name] = factor.groupby("TradingDate")[factor_name].transform(self._zscore)
        factor = factor.dropna(subset=[factor_name]).copy()

        if self.backtest_cfg.neutralize == "cap":
            factor = self._neutralize_by_cap(factor, factor_name)

        return ProcessedFactor(
            factor_name=factor_name,
            factor_df=factor[["TradingDate", "Stkcd", factor_name]].copy(),
            price_df=price,
            benchmark_price_df=benchmark_price,
            suspension_limit_df=limit,
            start_date=start_date,
            end_date=end_date,
        )

    def _load_factor_df(self, factor_path: Path) -> tuple[pd.DataFrame, str]:
        suffix = factor_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(factor_path)
        elif suffix in {".parquet", ".pq"}:
            filters = self._build_parquet_filters()
            try:
                df = pd.read_parquet(factor_path, engine="pyarrow", filters=filters)
            except Exception as err:
                self.logger.debug("Filtered parquet read failed for %s: %s", factor_path, err)
                df = pd.read_parquet(factor_path, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported factor file type: {factor_path}")
        if df.empty:
            raise EmptyFactorDataError(f"Factor file is empty: {factor_path}")
        return df, factor_path.stem

    def _build_parquet_filters(self) -> list[tuple[str, str, Any]] | None:
        filters: list[tuple[str, str, Any]] = []
        if self.backtest_cfg.start_date:
            filters.append(("TradingDate", ">=", pd.Timestamp(self.backtest_cfg.start_date)))
        if self.backtest_cfg.end_date:
            filters.append(("TradingDate", "<=", pd.Timestamp(self.backtest_cfg.end_date)))
        return filters or None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        lower_map = {str(col).lower(): col for col in df.columns}
        rename_map: dict[Any, str] = {}
        for candidate in ["tradingdate", "date", "time", "accper"]:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = "TradingDate"
                break
        for candidate in ["stkcd", "code", "symbol"]:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = "Stkcd"
                break
        return df.rename(columns=rename_map)

    def _resolve_factor_name(self, df: pd.DataFrame, file_basename: str) -> str:
        if file_basename in df.columns:
            return file_basename
        if "TradingDate" not in df.columns or "Stkcd" not in df.columns:
            raise KeyError(f"Factor file must contain TradingDate and Stkcd columns: {list(df.columns)}")
        raise KeyError(f"Factor column '{file_basename}' not found. Available columns: {list(df.columns)}")

    def _resolve_effective_period(self, df: pd.DataFrame) -> tuple[str, str]:
        factor_dates = df["TradingDate"].dropna()
        if factor_dates.empty:
            raise EmptyFactorDataError("Factor has no valid TradingDate values.")
        factor_start = pd.Timestamp(factor_dates.min())
        factor_end = pd.Timestamp(factor_dates.max())
        cfg_start = pd.Timestamp(self.backtest_cfg.start_date) if self.backtest_cfg.start_date else None
        cfg_end = pd.Timestamp(self.backtest_cfg.end_date) if self.backtest_cfg.end_date else None
        start = factor_start if cfg_start is None else max(cfg_start, factor_start)
        end = factor_end if cfg_end is None else min(cfg_end, factor_end)
        if start > end:
            raise EmptyFactorDataError("Configured period has no overlap with factor data.")
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _clip_factor(self, df: pd.DataFrame, factor_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        clipped = df[(df["TradingDate"] >= start_date) & (df["TradingDate"] <= end_date)].copy()
        clipped["Stkcd"] = clipped["Stkcd"].astype(int)
        return clipped[["TradingDate", "Stkcd", factor_name]]

    def _is_discrete_signal_factor(self, df: pd.DataFrame, factor_name: str) -> bool:
        values = pd.to_numeric(df[factor_name], errors="coerce").dropna().unique()
        if len(values) == 0:
            return True
        if len(values) > 3:
            return False
        normalized: set[int] = set()
        for value in values:
            if np.isclose(value, -1.0, atol=1e-12, rtol=0):
                normalized.add(-1)
            elif np.isclose(value, 0.0, atol=1e-12, rtol=0):
                normalized.add(0)
            elif np.isclose(value, 1.0, atol=1e-12, rtol=0):
                normalized.add(1)
            else:
                return False
        return normalized in [{0, 1}, {-1, 0}, {-1, 0, 1}]

    def _prepare_price(self, close_price: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        price = close_price.copy()
        price = self._standardize_columns(price)
        price["TradingDate"] = self._parse_trading_date(price["TradingDate"])
        price["Stkcd"] = price["Stkcd"].astype(int)
        price = price[(price["TradingDate"] >= start_date) & (price["TradingDate"] <= end_date)].copy()
        return price[["TradingDate", "Stkcd", "price"]]

    def _prepare_suspension_limit(
        self,
        suspension_limit: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        limit = suspension_limit.copy()
        limit = self._standardize_columns(limit)
        limit["TradingDate"] = self._parse_trading_date(limit["TradingDate"])
        limit["Stkcd"] = limit["Stkcd"].astype(int)
        limit = limit.sort_values(["Stkcd", "TradingDate"])
        limit[["Suspension", "LimitStatus"]] = (
            limit.groupby("Stkcd")[["Suspension", "LimitStatus"]].shift(-1)
        )
        limit = limit[(limit["TradingDate"] >= start_date) & (limit["TradingDate"] <= end_date)].copy()
        return limit[["TradingDate", "Stkcd", "Suspension", "LimitStatus"]]

    def _prepare_filter_stock_pool(self, start_date: str, end_date: str) -> pd.DataFrame:
        pool = self.filter_stock_pool.copy()
        pool.index = pd.to_datetime(pool.index)
        pool = pool.loc[start_date:end_date]
        pool.columns = [int(str(col).split(".")[0]) for col in pool.columns]
        for board in EXCLUDED_BOARDS:
            for code in self._board_codes(pool.columns, board):
                pool.loc[:, code] = 0
        pool_long = pool.stack().reset_index()
        pool_long.columns = ["TradingDate", "Stkcd", "in_pool"]
        pool_long["Stkcd"] = pool_long["Stkcd"].astype(int)
        return pool_long[pool_long["in_pool"] == 1][["TradingDate", "Stkcd"]]

    def _prepare_benchmark_stock_pool(self, start_date: str, end_date: str) -> pd.DataFrame:
        pool = self.benchmark_stock_pool.copy()
        pool.index = pd.to_datetime(pool.index)
        pool = pool.loc[start_date:end_date]
        pool.columns = [int(str(col).split(".")[0]) for col in pool.columns]
        pool_long = pool.stack().reset_index()
        pool_long.columns = ["TradingDate", "Stkcd", "in_pool"]
        pool_long["Stkcd"] = pool_long["Stkcd"].astype(int)
        return pool_long[pool_long["in_pool"] == 1][["TradingDate", "Stkcd"]]

    def _board_codes(self, columns: pd.Index, board: str) -> list[int]:
        if board == "chinext":
            return [col for col in columns if 300000 <= int(col) <= 301999]
        if board == "star":
            return [col for col in columns if 688000 <= int(col) <= 689999]
        return []

    def _filter_by_pool(self, df: pd.DataFrame, pool_long: pd.DataFrame) -> pd.DataFrame:
        return df.merge(pool_long, on=["TradingDate", "Stkcd"], how="inner")

    def _zscore(self, values: pd.Series) -> pd.Series:
        std = values.std()
        if pd.isna(std) or np.isclose(std, 0):
            return pd.Series(np.nan, index=values.index)
        return (values - values.mean()) / std

    def _neutralize_by_cap(self, factor: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        if self.market_value is None:
            raise ValueError("market_value is required when neutralize='cap'.")
        market_value = self.market_value.copy()
        market_value["Stkcd"] = market_value["Stkcd"].astype(int)
        df = factor.merge(market_value, on=["TradingDate", "Stkcd"], how="inner")

        def neutralize_one_day(group: pd.DataFrame) -> pd.DataFrame:
            group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_name, "MarketValue"])
            if len(group) < 3:
                group["neutralized_factor"] = np.nan
                return group[["TradingDate", "Stkcd", "neutralized_factor"]]
            x = pd.DataFrame(
                {
                    "log_mv": np.log(group["MarketValue"]),
                    "const": 1.0,
                },
                index=group.index,
            )
            model = sm.OLS(group[factor_name], x).fit()
            group["neutralized_factor"] = model.resid
            return group[["TradingDate", "Stkcd", "neutralized_factor"]]

        neutralized = df.groupby("TradingDate", group_keys=False).apply(neutralize_one_day)
        neutralized = neutralized.rename(columns={"neutralized_factor": factor_name})
        return neutralized.dropna(subset=[factor_name]).copy()

    def _parse_trading_date(self, series: pd.Series) -> pd.Series:
        """用固定格式解析日期，避免 pandas 在超大 object 列上做昂贵去重缓存。"""
        return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce", cache=False)
