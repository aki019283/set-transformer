from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.config import MetricConfig
from backtest.engine import BacktestResult


def ApplyPlotStyle() -> None:
    """设置统一图表风格。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


class MetricBase:
    """指标分析基类。"""

    NAME = "metric"

    def __init__(self, **params: Any):
        self.params = params
        self.result: Any = None

    def Analyze(self, backtest_result: BacktestResult) -> Any:
        raise NotImplementedError

    def Display(self, logger) -> None:
        logger.info("%s metric result is ready.", self.NAME)

    def Save(self, output_dir: Path) -> None:
        raise NotImplementedError

    def _ensure_output_dir(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, fig: plt.Figure, output_path: Path) -> None:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    def _periods_per_year(self, frequency: str) -> int:
        return {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}.get(frequency, 12)

    def _group_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if str(c).startswith("Group")]

    def _advantage_group(self, backtest_result: BacktestResult) -> str:
        return str(backtest_result.meta.get("advantage_group", "Group10"))

    def _format_summary(self, df: pd.DataFrame, max_rows: int = 8) -> str:
        return "<empty>" if df.empty else df.head(max_rows).to_string()


class NetValueMetric(MetricBase):
    NAME = "net_value"

    def Analyze(self, backtest_result: BacktestResult) -> pd.DataFrame:
        self.backtest_result = backtest_result
        self.result = backtest_result.nav_df.copy()
        return self.result

    def Display(self, logger) -> None:
        logger.info("Net value tail:\n%s", self._format_summary(self.result.tail()))

    def Save(self, output_dir: Path) -> None:
        self._ensure_output_dir(output_dir)
        self.result.to_csv(output_dir / "net_value.csv", encoding="utf-8-sig")
        ApplyPlotStyle()
        fig, ax = plt.subplots(figsize=(14, 7))
        advantage = self._advantage_group(self.backtest_result)
        for col in self._group_columns(self.result):
            ax.plot(
                self.result.index,
                self.result[col],
                label=col,
                linewidth=2.4 if col == advantage else 1.0,
                alpha=0.95 if col == advantage else 0.45,
            )
        for col in [c for c in self.result.columns if str(c).startswith("Benchmark")]:
            ax.plot(self.result.index, self.result[col], label=col, linewidth=2.0, linestyle="--")
        for col in [c for c in self.result.columns if str(c).startswith("Excess")]:
            ax.plot(self.result.index, self.result[col], label=col, linewidth=2.0, linestyle=":")
        ax.set_title(f"{self.backtest_result.factor_name} Net Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Value")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout()
        self._save_figure(fig, output_dir / "net_value.png")


class PeriodReturnMetric(MetricBase):
    NAME = "period_return"

    def Analyze(self, backtest_result: BacktestResult) -> pd.DataFrame:
        self.backtest_result = backtest_result
        self.result = backtest_result.period_return_df.copy()
        return self.result

    def Display(self, logger) -> None:
        logger.info("Period return tail:\n%s", self._format_summary(self.result.tail()))

    def Save(self, output_dir: Path) -> None:
        self._ensure_output_dir(output_dir)
        self.result.to_csv(output_dir / "period_return.csv", encoding="utf-8-sig")
        advantage = self._advantage_group(self.backtest_result)
        plot_cols = [advantage] + [c for c in self.result.columns if str(c).startswith("Excess")]
        plot_df = self.result[[c for c in plot_cols if c in self.result.columns]].copy()
        ApplyPlotStyle()
        fig, axes = plt.subplots(len(plot_df.columns), 1, figsize=(14, max(3.0, 2.6 * len(plot_df.columns))), sharex=True)
        axes = [axes] if len(plot_df.columns) == 1 else axes
        for ax, col in zip(axes, plot_df.columns):
            colors = ["#2ca02c" if value >= 0 else "#d62728" for value in plot_df[col]]
            ax.bar(plot_df.index, plot_df[col], color=colors, width=18)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(col)
            ax.set_ylabel("Return")
        axes[-1].set_xlabel("Date")
        fig.suptitle(f"{self.backtest_result.factor_name} Period Returns", y=1.01)
        fig.tight_layout()
        self._save_figure(fig, output_dir / "period_return.png")


class PerformanceMetric(MetricBase):
    NAME = "performance"

    def Analyze(self, backtest_result: BacktestResult) -> pd.DataFrame:
        self.backtest_result = backtest_result
        nav = backtest_result.nav_df.copy()
        ret = backtest_result.period_return_df.copy()
        periods = self._periods_per_year(backtest_result.meta["frequency"])
        risk_free_rate = float(backtest_result.meta["risk_free_rate"])
        cumulative_return = nav.iloc[-1] / nav.iloc[0] - 1
        annual_yield = (np.power(np.abs(nav.iloc[-1] / nav.iloc[0]), periods / max(len(ret), 1)) - 1) * np.sign(
            nav.iloc[-1] / nav.iloc[0]
        )
        annual_std = ret.std() * np.sqrt(periods)
        rf = pd.Series((1 + risk_free_rate) ** (1 / periods) - 1, index=ret.columns)
        rf.loc[rf.index.str.startswith("Excess")] = 0.0
        sharpe = (ret.mean() - rf) * periods / annual_std.replace(0, np.nan)
        result = pd.concat(
            [cumulative_return, annual_yield, annual_std, sharpe, nav.apply(self._max_drawdown, axis=0)],
            axis=1,
        )
        result.columns = ["CumulativeReturn", "AnnualYield", "AnnualStd", "Sharpe", "MaxDrawdown"]
        result["Monotonicity"] = self._monotonicity(nav)
        self.result = result
        return result

    def Display(self, logger) -> None:
        advantage = self._advantage_group(self.backtest_result)
        sharpe = self.result.loc[advantage, "Sharpe"] if advantage in self.result.index else np.nan
        logger.info("Performance summary. Advantage=%s Sharpe=%.6f", advantage, sharpe)
        logger.info("\n%s", self._format_summary(self.result))

    def Save(self, output_dir: Path) -> None:
        self._ensure_output_dir(output_dir)
        with pd.ExcelWriter(output_dir / "performance.xlsx", engine="openpyxl") as writer:
            self.result.to_excel(writer, sheet_name="Performance")
        self._plot(output_dir / "performance.png")

    def _max_drawdown(self, nav: pd.Series) -> float:
        values = pd.to_numeric(nav, errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return np.nan
        values = np.concatenate([np.array([1.0]), values])
        peak = np.maximum.accumulate(values)
        valid = ~np.isclose(peak, 0)
        drawdown = np.full(values.shape, np.nan, dtype=float)
        drawdown[valid] = (peak[valid] - values[valid]) / peak[valid]
        return float(np.nanmax(drawdown)) if np.isfinite(drawdown).any() else np.nan

    def _monotonicity(self, nav: pd.DataFrame) -> float:
        group_cols = self._group_columns(nav)
        if not group_cols:
            return np.nan
        return float(pd.Series(nav[group_cols].iloc[-1].to_numpy(dtype=float)).corr(pd.Series(np.arange(1, len(group_cols) + 1))))

    def _plot(self, output_path: Path) -> None:
        ApplyPlotStyle()
        group_df = self.result.loc[self._group_columns(self.result)].copy()
        advantage = self._advantage_group(self.backtest_result)
        monotonicity = group_df["Monotonicity"].dropna().iloc[0] if group_df["Monotonicity"].notna().any() else np.nan
        x = np.arange(len(group_df))
        highlight = ["#ff7f0e" if idx == advantage else "#4c78a8" for idx in group_df.index]
        fig, axes = plt.subplots(3, 1, figsize=(13, 13))
        fig.suptitle(f"{self.backtest_result.factor_name} Performance | Monotonicity={monotonicity:.4f}")
        self._dual_axis_plot(axes[0], x, group_df, "CumulativeReturn", "AnnualYield", highlight, advantage)
        axes[0].set_title("Cumulative Return and Annual Yield")
        self._dual_axis_plot(axes[1], x, group_df, "AnnualStd", "Sharpe", highlight, advantage)
        axes[1].set_title("Annual Std and Sharpe")
        axes[2].bar(x, group_df["MaxDrawdown"], color=highlight, alpha=0.85)
        axes[2].set_ylabel("Max Drawdown")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(group_df.index)
        axes[2].set_title("Max Drawdown")
        fig.tight_layout()
        self._save_figure(fig, output_path)

    def _dual_axis_plot(self, ax, x: np.ndarray, group_df: pd.DataFrame, bar_col: str, line_col: str, colors: list[str], advantage: str) -> None:
        ax.bar(x, group_df[bar_col], color=colors, alpha=0.85)
        ax.set_ylabel(bar_col)
        ax2 = ax.twinx()
        ax2.plot(x, group_df[line_col], color="#9467bd", marker="o")
        ax2.set_ylabel(line_col)
        if advantage in group_df.index:
            idx = list(group_df.index).index(advantage)
            value = group_df.loc[advantage, line_col]
            ax2.annotate(f"{value:.3f}", xy=(x[idx], value), xytext=(0, 10), textcoords="offset points", ha="center", color="#d62728", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(group_df.index)


class ICMetric(MetricBase):
    NAME = "ic"

    def Analyze(self, backtest_result: BacktestResult) -> dict[str, pd.DataFrame]:
        self.backtest_result = backtest_result
        factor_name = backtest_result.factor_name
        rows = []
        for date, group in backtest_result.factor_return_df.groupby("TradingDate"):
            clean = group[[factor_name, "ret"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 3 or clean[factor_name].nunique() < 2 or clean["ret"].nunique() < 2:
                ic, rank_ic = np.nan, np.nan
            else:
                ic = clean[factor_name].corr(clean["ret"], method="pearson")
                rank_ic = clean[factor_name].corr(clean["ret"], method="spearman")
            rows.append({"TradingDate": date, "IC": ic, "RankIC": rank_ic})
        series = pd.DataFrame(rows).set_index("TradingDate").sort_index() if rows else pd.DataFrame(columns=["IC", "RankIC"])
        series["CumulativeIC"] = series["IC"].fillna(0).cumsum()
        series["CumulativeRankIC"] = series["RankIC"].fillna(0).cumsum()
        self.result = {"summary": self._summary(series), "series": series}
        return self.result

    def Display(self, logger) -> None:
        logger.info("IC summary:\n%s", self._format_summary(self.result["summary"]))

    def Save(self, output_dir: Path) -> None:
        self._ensure_output_dir(output_dir)
        with pd.ExcelWriter(output_dir / "ic.xlsx", engine="openpyxl") as writer:
            self.result["summary"].to_excel(writer, sheet_name="Summary")
            self.result["series"].to_excel(writer, sheet_name="Series")
        self._plot(output_dir / "ic.png")

    def _summary(self, series_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in ["IC", "RankIC"]:
            series = series_df[col].dropna() if col in series_df.columns else pd.Series(dtype=float)
            mean, std, count = series.mean(), series.std(), series.count()
            rows.append({
                "Metric": col,
                "Mean": mean,
                "Std": std,
                "ICIR": mean / std if std and not np.isclose(std, 0) else np.nan,
                "TStat": mean / std * np.sqrt(count) if std and not np.isclose(std, 0) else np.nan,
                "PositiveRatio": (series > 0).mean() if count else np.nan,
                "Obs": count,
            })
        return pd.DataFrame(rows)

    def _plot(self, output_path: Path) -> None:
        ApplyPlotStyle()
        series = self.result["series"]
        summary = self.result["summary"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{self.backtest_result.factor_name} IC Diagnostics")
        if not series.empty:
            axes[0, 0].plot(series.index, series["IC"], label="IC")
            axes[0, 0].plot(series.index, series["RankIC"], label="RankIC")
            axes[0, 1].plot(series.index, series["CumulativeIC"], label="CumulativeIC")
            axes[0, 1].plot(series.index, series["CumulativeRankIC"], label="CumulativeRankIC")
            series[["IC", "RankIC"]].plot(kind="hist", bins=30, alpha=0.55, ax=axes[1, 0])
        for ax in axes[0]:
            ax.axhline(0, color="black", linewidth=0.8)
            ax.legend(frameon=False)
        axes[0, 0].set_title("IC Time Series")
        axes[0, 1].set_title("Cumulative IC")
        axes[1, 0].set_title("IC Distribution")
        if not summary.empty:
            summary.set_index("Metric")[["Mean", "ICIR", "PositiveRatio"]].plot(kind="bar", ax=axes[1, 1])
        axes[1, 1].set_title("Summary")
        axes[1, 1].tick_params(axis="x", rotation=0)
        fig.tight_layout()
        self._save_figure(fig, output_path)


class ExtraDiagnosticsMetric(MetricBase):
    NAME = "extra_diagnostics"

    def __init__(self, rolling_years: int = 3, **params: Any):
        super().__init__(rolling_years=rolling_years, **params)
        self.rolling_years = int(rolling_years)

    def Analyze(self, backtest_result: BacktestResult) -> dict[str, pd.DataFrame]:
        self.backtest_result = backtest_result
        ret = backtest_result.period_return_df.copy()
        periods = self._periods_per_year(backtest_result.meta["frequency"])
        risk_free_rate = float(backtest_result.meta["risk_free_rate"])
        rolling_window = self.rolling_years * periods
        rf = pd.Series((1 + risk_free_rate) ** (1 / periods) - 1, index=ret.columns)
        rf.loc[rf.index.str.startswith("Excess")] = 0.0
        rolling_sharpe = (ret.rolling(rolling_window).mean() - rf) * periods / (ret.rolling(rolling_window).std() * np.sqrt(periods))
        rolling_summary = rolling_sharpe.agg(["mean", "std", "count"]).T.reset_index()
        rolling_summary.columns = ["Group", f"Rolling_{self.rolling_years}Y_Sharpe_Mean", f"Rolling_{self.rolling_years}Y_Sharpe_SD", f"Rolling_{self.rolling_years}Y_Obs"]
        self.result = {
            "rolling_sharpe": rolling_summary,
            "annual_win_rate": self._win_rate_table(ret, "Y", "Annual"),
            "quarterly_win_rate": self._win_rate_table(ret, "Q", "Quarterly"),
            "monthly_win_rate": self._win_rate_table(ret, "M", "Monthly"),
            "return_concentration": self._return_concentration(ret),
            "exclude_top_months": self._exclude_top_months_metrics(ret, risk_free_rate),
        }
        return self.result

    def Display(self, logger) -> None:
        logger.info("Extra diagnostics rolling sharpe:\n%s", self._format_summary(self.result["rolling_sharpe"]))

    def Save(self, output_dir: Path) -> None:
        self._ensure_output_dir(output_dir)
        with pd.ExcelWriter(output_dir / "extra_diagnostics.xlsx", engine="openpyxl") as writer:
            self.result["rolling_sharpe"].to_excel(writer, sheet_name="RollingSharpe", index=False)
            self.result["annual_win_rate"].to_excel(writer, sheet_name="AnnualWinRate", index=False)
            self.result["quarterly_win_rate"].to_excel(writer, sheet_name="QuarterlyWinRate", index=False)
            self.result["monthly_win_rate"].to_excel(writer, sheet_name="MonthlyWinRate", index=False)
            self.result["return_concentration"].to_excel(writer, sheet_name="ReturnConcentration", index=False)
            self.result["exclude_top_months"].to_excel(writer, sheet_name="ExcludeTopMonths", index=False)
        self._plot(output_dir / "extra_diagnostics.png")

    def _win_rate_table(self, ret: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
        resample_rule = {"Y": "YE", "Q": "QE", "M": "ME"}.get(rule, rule)
        period_ret = (1 + ret).resample(resample_rule).prod(min_count=1) - 1
        return pd.DataFrame({
            "Group": period_ret.columns,
            f"{prefix}_N": period_ret.count().values,
            f"{prefix}_Positive_N": (period_ret > 0).sum().values,
            f"{prefix}_Positive_Ratio": ((period_ret > 0).sum() / period_ret.count()).values,
            f"{prefix}_Return_Mean": period_ret.mean().values,
            f"{prefix}_Return_SD": period_ret.std().values,
        })

    def _return_concentration(self, ret: pd.DataFrame) -> pd.DataFrame:
        rows = []
        monthly_ret = (1 + ret).resample("ME").prod(min_count=1) - 1
        for col in monthly_ret.columns:
            series = monthly_ret[col].dropna().sort_values(ascending=False)
            total_simple_return = series.sum()
            top3, top5 = series.head(3), series.head(5)
            positive = series[series > 0]
            positive_sum = positive.sum()
            top3_positive, top5_positive = positive.head(3), positive.head(5)
            positive_weights = positive / positive_sum if not np.isclose(positive_sum, 0) else pd.Series(dtype=float)
            rows.append({
                "Group": col,
                "Total_Simple_Return_Sum": total_simple_return,
                "Total_Compound_Return": (1 + series).prod() - 1,
                "Top3_Month_Return_Sum": top3.sum(),
                "Top3_Contribution_to_TotalSimpleReturn": np.nan if np.isclose(total_simple_return, 0) else top3.sum() / total_simple_return,
                "Top3_Months": "; ".join([f"{idx.strftime('%Y-%m')}:{value:.6f}" for idx, value in top3.items()]),
                "Top5_Month_Return_Sum": top5.sum(),
                "Top5_Contribution_to_TotalSimpleReturn": np.nan if np.isclose(total_simple_return, 0) else top5.sum() / total_simple_return,
                "Top5_Months": "; ".join([f"{idx.strftime('%Y-%m')}:{value:.6f}" for idx, value in top5.items()]),
                "Positive_Return_Sum": positive_sum,
                "Top3_Positive_Return_Sum": top3_positive.sum(),
                "Top3_Positive_Contribution": np.nan if np.isclose(positive_sum, 0) else top3_positive.sum() / positive_sum,
                "Top3_Positive_Months": "; ".join([f"{idx.strftime('%Y-%m')}:{value:.6f}" for idx, value in top3_positive.items()]),
                "Top5_Positive_Return_Sum": top5_positive.sum(),
                "Top5_Positive_Contribution": np.nan if np.isclose(positive_sum, 0) else top5_positive.sum() / positive_sum,
                "Top5_Positive_Months": "; ".join([f"{idx.strftime('%Y-%m')}:{value:.6f}" for idx, value in top5_positive.items()]),
                "Positive_Return_HHI": positive_weights.pow(2).sum() if not positive_weights.empty else np.nan,
            })
        return pd.DataFrame(rows)

    def _exclude_top_months_metrics(self, ret: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
        monthly_ret = (1 + ret).resample("ME").prod(min_count=1) - 1
        frames = [
            self._exclude_top_n_months(monthly_ret, 3, risk_free_rate),
            self._exclude_top_n_months(monthly_ret, 5, risk_free_rate),
        ]
        return pd.concat(frames, ignore_index=True)

    def _exclude_top_n_months(self, monthly_ret: pd.DataFrame, remove_top_n: int, risk_free_rate: float) -> pd.DataFrame:
        rows = []
        adjusted_cumulative_returns: dict[str, float] = {}
        for col in monthly_ret.columns:
            series = pd.to_numeric(monthly_ret[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            removed_months = ""
            cumulative_return = annual_yield = annual_std = sharpe = max_drawdown = np.nan
            if not series.empty:
                top_index = series.nlargest(min(remove_top_n, len(series)), keep="first").index
                top_values = series.loc[top_index]
                removed_months = "; ".join([f"{idx.strftime('%Y-%m')}:{value:.4f}" for idx, value in top_values.items()])
                adjusted = series.copy()
                adjusted.loc[top_index] = 0.0
                adjusted = adjusted.sort_index()
                nav = (1 + adjusted).cumprod()
                cumulative_return = nav.iloc[-1] - 1.0
                annual_yield = self._annual_yield_from_nav(nav, 12)
                annual_std = adjusted.std() * np.sqrt(12)
                if pd.notna(annual_std) and not np.isclose(annual_std, 0):
                    rf = 0.0 if str(col).startswith("Excess") else (1 + risk_free_rate) ** (1 / 12) - 1
                    sharpe = (adjusted.mean() - rf) * 12 / annual_std
                max_drawdown = self._max_drawdown_from_nav(nav)
            adjusted_cumulative_returns[col] = cumulative_return
            rows.append({
                "Scenario": f"Exclude_Top{remove_top_n}_Months",
                "Group": col,
                "Removed_Months": removed_months,
                "CumulativeReturn": cumulative_return,
                "AnnualYield": annual_yield,
                "AnnualStd": annual_std,
                "Sharpe": sharpe,
                "MaxDrawdown": max_drawdown,
            })

        result = pd.DataFrame(rows)
        result["Monotonicity"] = self._monotonicity_from_returns(adjusted_cumulative_returns)
        return result

    def _annual_yield_from_nav(self, nav: pd.Series, periods_per_year: int) -> float:
        if nav.empty:
            return np.nan
        ending_wealth = nav.iloc[-1]
        n_periods = len(nav)
        return float((np.power(np.abs(ending_wealth), periods_per_year / n_periods) - 1) * np.sign(ending_wealth))

    def _max_drawdown_from_nav(self, nav: pd.Series) -> float:
        values = np.concatenate([np.array([1.0]), nav.to_numpy(dtype=float)])
        peak = np.maximum.accumulate(values)
        valid = ~np.isclose(peak, 0)
        drawdown = np.full(values.shape, np.nan, dtype=float)
        drawdown[valid] = (peak[valid] - values[valid]) / peak[valid]
        return float(np.nanmax(drawdown)) if np.isfinite(drawdown).any() else np.nan

    def _monotonicity_from_returns(self, cumulative_returns: dict[str, float]) -> float:
        group_cols = [f"Group{i}" for i in range(1, 11)]
        if not all(col in cumulative_returns for col in group_cols):
            return np.nan
        values = pd.Series([cumulative_returns[col] for col in group_cols], index=range(1, 11), dtype=float)
        valid = values.notna()
        if valid.sum() < 2:
            return np.nan
        return float(values[valid].corr(pd.Series(range(1, 11), index=range(1, 11), dtype=float)[valid]))

    def _plot(self, output_path: Path) -> None:
        ApplyPlotStyle()
        fig, axes = plt.subplots(3, 2, figsize=(15, 13))
        fig.suptitle(f"{self.backtest_result.factor_name} Extra Diagnostics")
        rolling = self.result["rolling_sharpe"].set_index("Group")
        rolling[[f"Rolling_{self.rolling_years}Y_Sharpe_Mean", f"Rolling_{self.rolling_years}Y_Sharpe_SD"]].plot(kind="bar", ax=axes[0, 0])
        self.result["annual_win_rate"].set_index("Group")["Annual_Positive_Ratio"].plot(kind="bar", ax=axes[0, 1], color="#2ca02c")
        self.result["quarterly_win_rate"].set_index("Group")["Quarterly_Positive_Ratio"].plot(kind="bar", ax=axes[1, 0], color="#1f77b4")
        self.result["monthly_win_rate"].set_index("Group")["Monthly_Positive_Ratio"].plot(kind="bar", ax=axes[1, 1], color="#54a24b")
        self.result["return_concentration"].set_index("Group")[["Top3_Positive_Contribution", "Top5_Positive_Contribution"]].plot(kind="bar", ax=axes[2, 0])
        exclude_pivot = self.result["exclude_top_months"].pivot(index="Group", columns="Scenario", values="Sharpe")
        exclude_pivot.plot(kind="bar", ax=axes[2, 1])
        axes[0, 0].set_title("Rolling Sharpe")
        axes[0, 1].set_title("Annual Positive Ratio")
        axes[0, 1].set_ylim(0, 1)
        axes[1, 0].set_title("Quarterly Positive Ratio")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 1].set_title("Monthly Positive Ratio")
        axes[1, 1].set_ylim(0, 1)
        axes[2, 0].set_title("Positive Return Concentration")
        axes[2, 0].set_ylim(0, 1)
        axes[2, 1].set_title("Sharpe after Excluding Top Months")
        fig.tight_layout()
        self._save_figure(fig, output_path)


_METRIC_REGISTRY = {
    "net_value": NetValueMetric,
    "period_return": PeriodReturnMetric,
    "performance": PerformanceMetric,
    "ic": ICMetric,
    "extra_diagnostics": ExtraDiagnosticsMetric,
}


def MakeMetric(metric_cfg: MetricConfig) -> MetricBase:
    metric_cls = _METRIC_REGISTRY.get(metric_cfg.name)
    if metric_cls is None:
        raise ValueError(f"Unsupported metric: {metric_cfg.name}")
    return metric_cls(**metric_cfg.params)
