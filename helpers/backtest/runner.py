from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.config import BacktestRuntimeConfig, FactorInputConfig, MetricConfig
from backtest.engine import BacktestEngine, BacktestResult
from backtest.factor_processor import EmptyFactorDataError, FactorProcessor
from backtest.metrics import MakeMetric
from runtime import resolve_path


class BacktestRunner:
    """统一编排批量因子回测。"""

    def __init__(self, runtime_cfg: BacktestRuntimeConfig):
        self.runtime_cfg = runtime_cfg

    def Run(self) -> None:
        factor_files = _FactorLocator(self.runtime_cfg.factor).Locate()
        with _BacktestDataContext(self.runtime_cfg) as context:
            logger = context.logger
            logger.info("Total factor files: %s", len(factor_files))
            processor = FactorProcessor(
                backtest_cfg=self.runtime_cfg.backtest,
                filter_stock_pool=context.filter_stock_pool,
                benchmark_stock_pool=context.benchmark_stock_pool,
                market_value=context.market_value,
                logger=logger,
            )
            engine = BacktestEngine(
                backtest_cfg=self.runtime_cfg.backtest,
                benchmark_cfg=self.runtime_cfg.benchmark,
                index_quotes=context.index_quotes,
            )
            metric_runner = _MetricRunner(self.runtime_cfg.metrics, logger)

            for idx, factor_path in enumerate(factor_files, start=1):
                self._run_single_factor(idx, len(factor_files), factor_path, context, processor, engine, metric_runner)

    def _run_single_factor(
        self,
        idx: int,
        total: int,
        factor_path: Path,
        context: "_BacktestDataContext",
        processor: FactorProcessor,
        engine: BacktestEngine,
        metric_runner: "_MetricRunner",
    ) -> None:
        try:
            context.logger.info("Backtesting %s (%s/%s)", factor_path.name, idx, total)
            processed = processor.LoadAndProcess(
                factor_path=factor_path,
                close_price=context.close_price,
                suspension_limit=context.suspension_limit,
            )
            result = engine.Run(processed)
            metric_runner.Run(result, self._make_output_dir(processed.factor_name))
            context.logger.info("SUCCESS: %s", factor_path)
        except EmptyFactorDataError as err:
            context.logger.warning("%s", err)
        except Exception as err:
            context.AddFailure(factor_path, err)

    def _make_output_dir(self, factor_name: str) -> Path | None:
        if not self.runtime_cfg.output.save:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.runtime_cfg.output.output_dir / factor_name / timestamp
        suffix = 1
        while output_dir.exists():
            output_dir = self.runtime_cfg.output.output_dir / factor_name / f"{timestamp}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class _FactorLocator:
    """按配置定位因子文件。"""

    _VALID_EXTENSIONS = {".csv", ".parquet", ".pq"}

    def __init__(self, cfg: FactorInputConfig):
        self.cfg = cfg

    def Locate(self) -> list[Path]:
        files = self._resolve_patterns()
        deduped = sorted({p.resolve() for p in files}, key=lambda x: str(x))
        if not deduped:
            raise FileNotFoundError("No valid factor files found.")
        return deduped

    def _resolve_patterns(self) -> list[Path]:
        files: list[Path] = []
        for pattern in self.cfg.patterns:
            path_text = str(resolve_path(pattern))
            matches = [Path(value) for value in glob.glob(path_text, recursive=True)]
            if not matches:
                raise FileNotFoundError(f"Factor pattern matched no files: {pattern}")
            if any(item.is_dir() for item in matches):
                raise ValueError(f"Factor pattern matched a directory: {pattern}")
            files.extend(item for item in matches if self._is_factor_file(item))
        return files

    def _is_factor_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self._VALID_EXTENSIONS


class _MetricRunner:
    """按配置顺序执行指标分析。"""

    def __init__(self, metric_configs: list[MetricConfig], logger):
        self.metric_configs = metric_configs
        self.logger = logger

    def Run(self, backtest_result: BacktestResult, output_dir: Path | None) -> None:
        for idx, metric_cfg in enumerate(self.metric_configs, start=1):
            metric = MakeMetric(metric_cfg)
            self.logger.info("Analyzing metric [%s] %s", idx, metric_cfg.name)
            metric.Analyze(backtest_result)
            metric.Display(self.logger)
            if output_dir is not None:
                metric_dir = output_dir / metric_cfg.name
                metric.Save(metric_dir)
                self.logger.info("Saved metric [%s] %s to %s", idx, metric_cfg.name, metric_dir)


class _BacktestDataContext:
    """管理回测共享数据与日志生命周期。"""

    def __init__(self, runtime_cfg: BacktestRuntimeConfig):
        self.runtime_cfg = runtime_cfg
        self.logger = logging.getLogger(f"backtest.{id(self)}")
        self.failed_file_path: Path | None = None
        self.close_price: pd.DataFrame | None = None
        self.suspension_limit: pd.DataFrame | None = None
        self.filter_stock_pool: pd.DataFrame | None = None
        self.benchmark_stock_pool: pd.DataFrame | None = None
        self.market_value: pd.DataFrame | None = None
        self.index_quotes: pd.DataFrame | None = None
        self.failed_factors: list[dict[str, str]] = []

    def __enter__(self) -> "_BacktestDataContext":
        self._setup_logger()
        self._load_shared_tables()
        self.logger.info("Loaded shared backtest tables.")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.logger.exception("Fatal error in backtest data context", exc_info=(exc_type, exc, tb))
        self._write_failed_summary()
        self._teardown_logger()
        return False

    def AddFailure(self, factor_path: Path, error: Exception) -> None:
        self.failed_factors.append({"name": str(factor_path), "error": str(error)})
        self.logger.error("FAILED: %s", factor_path)
        self.logger.exception("Error processing factor %s", factor_path)

    def _setup_logger(self) -> None:
        cfg = self.runtime_cfg.log
        level = getattr(logging, cfg.level, logging.INFO)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(stream_handler)

        if cfg.log_file is None:
            return

        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.failed_file_path = cfg.log_file.with_suffix(".failed")
        file_handler = logging.FileHandler(cfg.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.info("Log file enabled: %s", cfg.log_file)

    def _teardown_logger(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def _load_shared_tables(self) -> None:
        data_cfg = self.runtime_cfg.data
        self.close_price = pd.read_parquet(data_cfg.adj_close_path, engine="pyarrow")
        self.suspension_limit = pd.read_parquet(data_cfg.suspension_limit_path, engine="pyarrow")
        self.filter_stock_pool = pd.read_parquet(data_cfg.filter_stock_pool_path, engine="pyarrow")
        self.benchmark_stock_pool = pd.read_parquet(data_cfg.benchmark_stock_pool_path, engine="pyarrow")

        if data_cfg.market_value_path is not None:
            market_value = pd.read_parquet(data_cfg.market_value_path, engine="pyarrow")
            self.market_value = market_value[["TradingDate", "Stkcd", "MarketValue"]].copy()
            self.market_value["TradingDate"] = pd.to_datetime(self.market_value["TradingDate"])

        if data_cfg.index_quotes_path is not None:
            self.index_quotes = pd.read_parquet(data_cfg.index_quotes_path, engine="pyarrow")
            self.index_quotes["TradingDate"] = pd.to_datetime(self.index_quotes["TradingDate"])

    def _write_failed_summary(self) -> None:
        if not self.failed_factors:
            return
        self.logger.info("FAILED FACTORS SUMMARY")
        for row in self.failed_factors:
            self.logger.info("Factor: %s", row["name"])
            self.logger.info("Error: %s", row["error"])
        if self.failed_file_path is None:
            return
        with open(self.failed_file_path, "w", encoding="utf-8") as stream:
            for row in self.failed_factors:
                stream.write(f"{row['name']}\n")
