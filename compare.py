"""Replay the archived FT seed-42 book with same-date flags and scheduled closes."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / "helpers")]
from standard_sharpe import standard_sharpe
import calendar_recheck as cal
from audit_months import read_slice
from valuation_fix import adjacent_returns, make_engine



def replay(data, output, signal, schedule):
    from backtest.config import BacktestConfig, BenchmarkConfig, TransactionCostConfig
    from backtest.engine import BacktestEngine
    output.mkdir(parents=True, exist_ok=True)
    keys = ["TradingDate", "Stkcd"]
    cal.START, cal.VALUE_END = "2024-01-01", "2025-11-30"
    lo, hi = pd.Timestamp(cal.START), pd.Timestamp("2025-12-05")
    prices = read_slice(data/"adj_close.parquet", keys+["price"], lo, hi)
    prices = prices.loc[prices.TradingDate <= cal.VALUE_END]
    flags = read_slice(data/"Suspension_Limit.parquet", keys+["Suspension", "LimitStatus"], lo, hi)
    pools = []
    for name in ["history-new.pq", "history-1800.pq"]:
        pool = pd.read_parquet(data/name)
        pool.index = pd.to_datetime(pool.index)
        aligned, _ = cal.complete_month_ends(pool.loc[cal.START:cal.VALUE_END], prices.loc[prices.price.gt(0), "TradingDate"])
        pools.append(aligned)
    cfg = BacktestConfig(cal.START, "2025-10-31", "monthly", 10, None, .02, TransactionCostConfig())
    prep = cal.make_processor(cfg, *pools).LoadAndProcess(signal, prices, flags)
    flags.Stkcd = flags.Stkcd.astype(int)
    same = prep.suspension_limit_df[keys].merge(flags, on=keys, how="left", validate="one_to_one", indicator=True)
    if not same._merge.eq("both").all():
        raise ValueError("Missing same-date trading status")
    # Keep the original status resampling/default convention in both replays and disclose its use.
    (output/"status_field_audit.json").write_text(json.dumps({
        "source_missing_fields_before_monthly_resampling":same[["Suspension","LimitStatus"]].isna().sum().to_dict(),
        "convention":"Archived M0 engine uses last observed monthly status and treats remaining missing flags as 0."
    }, indent=2), encoding="utf8")
    adjusted = make_engine(BacktestEngine, prices, output)(cfg, BenchmarkConfig([]), None).Run(
        replace(prep, suspension_limit_df=same.drop(columns="_merge")))
    legacy = make_engine(BacktestEngine, prices, output)(cfg, BenchmarkConfig([]), None).Run(prep)
    # 固定修正交易状态后的最高十分组持仓，再按计划成交日读取价格。
    held = adjusted.factor_return_df.query("quantile_group == 'Group10'")[keys].merge(
        adjacent_returns(prices), on=keys, validate="one_to_one")
    held["month"] = (held.TradingDate.dt.to_period("M")+1).astype(str)
    held = held.merge(schedule[["holding_month","entry","exit"]], left_on="month", right_on="holding_month", validate="many_to_one")
    quotes = read_slice(data/"adj_close.parquet", keys+["price"], lo, hi).set_index(keys, verify_integrity=True).price
    for date in ["entry", "exit"]:
        values = quotes.reindex(pd.MultiIndex.from_arrays([held[date],held.Stkcd])).to_numpy()
        if not (np.isfinite(values) & (values > 0)).all():
            raise ValueError(f"Missing held price at {date}")
        held[date+"_price"] = values
    held["new_return"] = held.exit_price/held.entry_price-1
    book = held.groupby("month").agg(holdings=("Stkcd","size"), old_gross=("ret","mean"), new_gross=("new_return","mean"))
    legacy_net = adjusted.period_return_df.Group10.copy()
    legacy_net.index = (legacy_net.index.to_period("M")+1).astype(str)
    expected = list(pd.period_range("2024-02","2025-11",freq="M").astype(str))
    if list(book.index)!=expected or list(legacy_net.index)!=expected:
        raise ValueError("Incomplete or misaligned monthly series")
    book["legacy_cost"] = book.old_gross - legacy_net
    # 用新的持有期毛收益减去原组合简化成本，得到比较用月度净收益。
    book["repriced_net"] = book.new_gross - book.legacy_cost
    legacy_series = legacy.period_return_df.Group10.copy()
    legacy_series.index = (legacy_series.index.to_period("M")+1).astype(str)
    if list(legacy_series.index)!=expected:
        raise ValueError("Incomplete M0 series")
    book["historical_m0_next_flags"] = legacy_series
    book.to_csv(output/"monthly.csv")
    held[keys+["month","entry","exit","entry_price","exit_price","new_return"]].to_parquet(output/"positions.parquet", index=False)
    return book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(1, str(args.project_root.resolve()/"src"))
    report = args.output/"report"
    identity = json.loads((args.output/"identity.json").read_text(encoding="utf8"))
    from hashlib import file_digest
    def verify_market():
        for filename in ["adj_close.parquet","Suspension_Limit.parquet","history-new.pq","history-1800.pq"]:
            source = (args.project_root/"data/source_data"/filename).resolve()
            with source.open("rb") as stream:
                if file_digest(stream,"sha256").hexdigest() != identity["hashes"][str(source)]:
                    raise ValueError(f"Backtest inputs have changed since training: {source}")
    verify_market()
    ref = BASE/"reference"
    schedule = pd.read_csv(ref/"schedule.csv", parse_dates=["entry","exit"])
    current = pd.read_csv(report/"holding_month_schedule.csv", parse_dates=["entry","exit"])
    current = current[current.holding_month.isin(schedule.holding_month)].reset_index(drop=True)
    pd.testing.assert_frame_equal(current[schedule.columns], schedule)
    signal_paths = [ref/"ft_transformer_s42.parquet", args.output/"factors/dcn_set_s42.parquet"]
    match_keys = None
    metrics = []
    for path in signal_paths:
        scores = pd.read_parquet(path)
        keys = scores.loc[pd.to_datetime(scores.TradingDate).isin(pd.to_datetime(schedule.signal_label)),
                          ["TradingDate","Stkcd"]].copy()
        keys.Stkcd = keys.Stkcd.astype(int)
        keys = keys.sort_values(list(keys)).reset_index(drop=True)
        if keys.duplicated().any() or keys.empty:
            raise ValueError(f"Invalid prediction keys: {path.name}")
        if match_keys is None:
            match_keys = keys
        else:
            pd.testing.assert_frame_equal(keys, match_keys, check_dtype=False)
        dest = report/"comparison_22m"/path.stem
        book = replay(args.project_root/"data/source_data", dest, path, schedule)
        if path == signal_paths[0]:
            historical = pd.read_csv(ref/"monthly.csv", index_col="month")
            np.testing.assert_array_equal(book.index, historical.index)
            np.testing.assert_allclose(book.repriced_net, historical.repriced_net, atol=1e-12, rtol=1e-12)
        # Keep the historical M0 convention separate from the fixed-holdings FT comparison.
        for rule in ["historical_m0_next_flags", "repriced_net"]:
            r = book[rule]
            if len(r)!=22 or not np.isfinite(r).all() or (r<=-1).any():
                raise ValueError("Invalid net monthly returns")
            nav = np.r_[1.,(1+r).cumprod()]
            metrics.append(dict(model=path.stem, rule=rule, months=len(r),
                                standard_sharpe=standard_sharpe(r), cagr=nav[-1]**(12/len(r))-1,
                                max_drawdown=float((1-nav/np.maximum.accumulate(nav)).max())))
    comparison = pd.DataFrame(metrics)
    comparison["identical_input_panel_to_archived_ft"] = json.loads(
        (report/"panel_comparison.json").read_text(encoding="utf8"))["identical"]
    verify_market()
    comparison.to_csv(report/"comparison_22m/summary.csv", index=False)
    status = json.loads((report/"status.json").read_text(encoding="utf8"))
    status.update(state="HISTORICAL_COMPARISON_COMPLETE", paired_return_months=22)
    stage = report/"status.tmp"
    stage.write_text(json.dumps(status,indent=2),encoding="utf8")
    stage.replace(report/"status.json")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
