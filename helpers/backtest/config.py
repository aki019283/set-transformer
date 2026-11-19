from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime import load_yaml_config, normalize_frequency, resolve_path


@dataclass(frozen=True)
class DataSourceConfig:
    adj_close_path: Path
    suspension_limit_path: Path
    filter_stock_pool_path: Path
    benchmark_stock_pool_path: Path
    market_value_path: Path | None
    index_quotes_path: Path | None


@dataclass(frozen=True)
class FactorInputConfig:
    patterns: list[str]


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    save: bool = True


@dataclass(frozen=True)
class TransactionCostConfig:
    stamp_tax: float = 0.001
    commission_buy: float = 0.0001
    commission_sell: float = 0.0001
    transfer_buy: float = 0.00001
    transfer_sell: float = 0.00001

    @property
    def total(self) -> float:
        return sum((self.stamp_tax, self.commission_buy, self.commission_sell, self.transfer_buy, self.transfer_sell))


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str | None
    end_date: str | None
    frequency: str
    groups: int
    neutralize: str | None
    risk_free_rate: float
    transaction_cost: TransactionCostConfig


@dataclass(frozen=True)
class BenchmarkItemConfig:
    mode: str
    weights: str | None = None
    index_code: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    items: list[BenchmarkItemConfig]

    @property
    def has_pool_equal_weight(self) -> bool:
        return any(item.mode == "pool" and item.weights == "equal" for item in self.items)

    @property
    def index_codes(self) -> list[str]:
        return [str(item.index_code) for item in self.items if item.mode == "index" and item.index_code]


@dataclass(frozen=True)
class MetricConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogConfig:
    log_file: Path | None
    level: str


@dataclass(frozen=True)
class BacktestRuntimeConfig:
    data: DataSourceConfig
    factor: FactorInputConfig
    output: OutputConfig
    backtest: BacktestConfig
    benchmark: BenchmarkConfig
    metrics: list[MetricConfig]
    log: LogConfig


DEFAULT_METRICS = ("net_value", "period_return", "performance", "ic", "extra_diagnostics")


def LoadBacktestConfig(config_path: str | Path) -> BacktestRuntimeConfig:
    raw = load_yaml_config(config_path)
    source = _as_dict(_as_dict(raw.get("data")).get("source_data"))
    backtest = _as_dict(raw.get("backtest"))
    period = _as_dict(backtest.get("period"))
    transaction = _as_dict(backtest.get("transaction_cost"))
    factors = _as_list(raw.get("factors"))
    if not factors or not all(isinstance(item, str) and item.strip() for item in factors):
        raise ValueError("factors must be a non-empty list of factor files or glob patterns")
    patterns = [str(item) for item in factors]

    runtime = BacktestRuntimeConfig(
        data=DataSourceConfig(
            adj_close_path=_required_path(source.get("adj_close"), "data.source_data.adj_close"),
            suspension_limit_path=_required_path(source.get("suspension_limit"), "data.source_data.suspension_limit"),
            filter_stock_pool_path=_required_path(source.get("filter_stock_pool"), "data.source_data.filter_stock_pool"),
            benchmark_stock_pool_path=_required_path(source.get("benchmark_stock_pool"), "data.source_data.benchmark_stock_pool"),
            market_value_path=_optional_path(source.get("market_value")),
            index_quotes_path=_optional_path(source.get("index_quotes")),
        ),
        factor=FactorInputConfig(patterns=patterns),
        output=OutputConfig(
            output_dir=_required_path(_as_dict(raw.get("output")).get("output_dir"), "output.output_dir"),
            save=bool(_as_dict(raw.get("output")).get("save", True)),
        ),
        backtest=BacktestConfig(
            start_date=_optional_date(period.get("start_date")),
            end_date=_optional_date(period.get("end_date")),
            frequency=normalize_frequency(str(backtest.get("frequency", "monthly"))),
            groups=int(backtest.get("groups", 10)),
            neutralize=_optional_str(backtest.get("neutralize")),
            risk_free_rate=float(backtest.get("risk_free_rate", 0.02)),
            transaction_cost=TransactionCostConfig(
                stamp_tax=float(transaction.get("stamp_tax", 0.001)),
                commission_buy=float(transaction.get("commission_buy", 0.0001)),
                commission_sell=float(transaction.get("commission_sell", 0.0001)),
                transfer_buy=float(transaction.get("transfer_buy", 0.00001)),
                transfer_sell=float(transaction.get("transfer_sell", 0.00001)),
            ),
        ),
        benchmark=BenchmarkConfig(items=[_parse_benchmark_item(item) for item in _as_list(raw.get("benchmark"))]),
        metrics=[MetricConfig(name=name) for name in DEFAULT_METRICS],
        log=LogConfig(
            log_file=_optional_path(_as_dict(raw.get("log")).get("log_file")),
            level=str(_as_dict(raw.get("log")).get("level", "INFO")).upper(),
        ),
    )
    _validate(runtime)
    return runtime


load_backtest_config = LoadBacktestConfig


def _validate(cfg: BacktestRuntimeConfig) -> None:
    if cfg.backtest.groups < 2:
        raise ValueError("backtest.groups must be at least 2")
    if cfg.backtest.neutralize not in {None, "cap"}:
        raise ValueError("backtest.neutralize only supports null or 'cap'")
    if not cfg.benchmark.items:
        raise ValueError("benchmark must provide at least one item")
    if cfg.benchmark.index_codes and cfg.data.index_quotes_path is None:
        raise ValueError("data.source_data.index_quotes is required for index benchmarks")
    for pattern in cfg.factor.patterns:
        if not any(char in pattern for char in "*?[") and Path(pattern).suffix.lower() not in {
            ".csv", ".parquet", ".pq",
        }:
            raise ValueError(f"Factor entry must be a supported factor file or glob pattern: {pattern}")


def _parse_benchmark_item(item: Any) -> BenchmarkItemConfig:
    if not isinstance(item, dict):
        raise TypeError("benchmark item must be a mapping")
    mode = str(item.get("mode", "")).strip().lower()
    if mode == "pool":
        weights = str(item.get("weights", "equal")).strip().lower()
        if weights != "equal":
            raise ValueError("pool benchmark only supports equal weights")
        return BenchmarkItemConfig(mode=mode, weights=weights)
    if mode == "index" and item.get("index_code") is not None:
        return BenchmarkItemConfig(mode=mode, index_code=str(item["index_code"]))
    raise ValueError("benchmark.mode must be pool or index")


def _required_path(value: Any, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required config field: {field_name}")
    return resolve_path(value)


def _optional_path(value: Any) -> Path | None:
    return None if value is None or not str(value).strip() else resolve_path(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_date(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    parts = text.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return text


__all__ = [
    "BacktestConfig", "BacktestRuntimeConfig", "BenchmarkConfig", "DataSourceConfig",
    "DEFAULT_METRICS", "FactorInputConfig", "LoadBacktestConfig", "MetricConfig",
    "TransactionCostConfig",
]
