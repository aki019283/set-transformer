"""Read-only input audit. Does not train, modify datasets, or publish factors."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd

KEYS=['TradingDate','Stkcd']

def month_counts(frame):
    if frame.empty: return pd.Series(dtype=int)
    return frame.groupby(pd.to_datetime(frame.TradingDate).dt.to_period('M')).size()

def normalize(frame):
    out=frame.copy()
    out['TradingDate']=pd.to_datetime(out.TradingDate,errors='raise').dt.normalize()
    out['Stkcd']=pd.to_numeric(out.Stkcd,errors='raise').astype('int64')
    if out.duplicated(KEYS).any(): raise ValueError('Duplicate date-stock source keys')
    return out

def monthly_join(frame,pool):
    """Diagnostic only: compare month-key overlap without generating a return."""
    left=frame.copy(); right=pool[KEYS].copy()
    left['_month']=left.TradingDate.dt.to_period('M')
    right['_month']=right.TradingDate.dt.to_period('M')
    right=right[['_month','Stkcd']].drop_duplicates()
    return left.merge(right,on=['_month','Stkcd'],how='inner',validate='many_to_one').drop(columns='_month')

def adjacent_prices(frame):
    """Count t and t+1 price pairs explicitly; never bridge a missing month."""
    prices=frame.sort_values('TradingDate').copy()
    prices['_month']=prices.TradingDate.dt.to_period('M')
    prices=prices.groupby(['_month','Stkcd'],as_index=False).tail(1)
    next_prices=prices[['_month','Stkcd','price']].copy()
    next_prices['_month']=next_prices['_month']-1
    pairs=prices.merge(next_prices,on=['_month','Stkcd'],how='left',suffixes=('','_next'),validate='one_to_one')
    pairs['valid_next_price']=np.isfinite(pairs.price)&np.isfinite(pairs.price_next)&pairs.price.gt(0)&pairs.price_next.gt(0)
    return pairs

def read_slice(path,columns,start,end):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pq.read_schema(path)
    field=schema.field('TradingDate').type
    lo,hi=(str(start.date()),str(end.date())) if pa.types.is_string(field) or pa.types.is_large_string(field) else (start.to_pydatetime(),end.to_pydatetime())
    print('Reading '+str(path),flush=True)
    try: frame=pd.read_parquet(path,columns=columns,filters=[('TradingDate','>=',lo),('TradingDate','<=',hi)])
    except (pa.ArrowInvalid,pa.ArrowNotImplementedError,TypeError):
        print('Filtered read unsupported; reading selected columns only.',flush=True)
        frame=pd.read_parquet(path,columns=columns)
    frame=normalize(frame)
    return frame.loc[frame.TradingDate.between(start,end)].copy()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--study',default='screen_seed42')
    parser.add_argument('--export-dir',type=Path)
    args=parser.parse_args(); root=args.project_root.resolve()
    if not args.study.replace('_','').replace('-','').isalnum(): raise ValueError('Invalid study')
    sys.path.insert(0,str(root/'src'))
    from backtest.config import LoadBacktestConfig
    from backtest.factor_processor import FactorProcessor
    from backtest.engine import BacktestEngine
    exp=root/'data/experiments/nn_interpretable_oct'/args.study
    config_path=exp/'00_tools/backtest_training_M0.yaml'
    cfg=LoadBacktestConfig(config_path)
    plan=json.loads((exp/'01_audit/plan.json').read_text(encoding='utf-8'))
    dates=pd.to_datetime(plan['test_months']); start=dates.min().to_period('M').start_time; end=dates.max()
    out=root/'data/experiments/nn_month_coverage_audit'/datetime.now().strftime('%Y%m%dT%H%M%S%f')
    out.mkdir(parents=True)
    factor_path=exp/'04_factors/training/M0.parquet'
    factor=normalize(pd.read_parquet(factor_path))
    price=read_slice(cfg.data.adj_close_path,KEYS+['price'],start,end+pd.offsets.MonthEnd(1))
    limit=read_slice(cfg.data.suspension_limit_path,KEYS+['Suspension','LimitStatus'],start,end+pd.offsets.MonthEnd(1))
    pool=pd.read_parquet(cfg.data.filter_stock_pool_path)
    bench=pd.read_parquet(cfg.data.benchmark_stock_pool_path)
    processor=FactorProcessor(cfg.backtest,pool,bench,None,logging.getLogger('month_audit'))
    engine=BacktestEngine(cfg.backtest,cfg.benchmark,None)
    loaded,_=processor._load_factor_df(factor_path)
    loaded=processor._standardize_columns(loaded)
    loaded['TradingDate']=processor._parse_trading_date(loaded.TradingDate)
    old_start,old_end=processor._resolve_effective_period(loaded)
    clipped=processor._clip_factor(loaded,'M0',old_start,old_end)
    eligible=processor._prepare_filter_stock_pool(old_start,old_end)
    eligible_full=processor._prepare_filter_stock_pool(str(start.date()),str(end.date()))
    filtered_factor=processor._filter_by_pool(clipped,eligible)
    price_clipped=processor._prepare_price(price,old_start,old_end)
    price_exact=processor._filter_by_pool(price_clipped,eligible)
    limit_clipped=processor._prepare_suspension_limit(limit,old_start,old_end)
    limit_exact=processor._filter_by_pool(limit_clipped,eligible)
    processed=processor.LoadAndProcess(factor_path,price,limit)
    prepared=engine._prepare_rebalance_frame(processed)
    prepared_dates=sorted(prepared.TradingDate.dropna().unique())
    skipped_last=pd.Timestamp(prepared_dates[-1]).to_period('M') if prepared_dates else None
    month_price=monthly_join(price,eligible_full)
    pairs=adjacent_prices(price)
    valid_pairs=pairs.loc[pairs.valid_next_price]
    signal_pairs=factor.assign(_month=factor.TradingDate.dt.to_period('M')).merge(
        valid_pairs[['_month','Stkcd']],on=['_month','Stkcd'],how='inner',validate='one_to_one')
    count_frames={'factor_input':factor,'factor_after_config':clipped,'factor_after_pool':filtered_factor,
        'price_raw':price,'price_after_config':price_clipped,'price_exact_pool':price_exact,
        'limit_exact_pool':limit_exact,'prepared_factor_nonnull':prepared.dropna(subset=['M0']),
        'price_month_key_overlap':month_price,'signal_with_adjacent_price':signal_pairs}
    counts={k:month_counts(v) for k,v in count_frames.items()}
    results=sorted((exp/'05_backtests/training/M0').glob('*/period_return/period_return.csv'),key=lambda p:p.stat().st_mtime)
    if not results: raise FileNotFoundError('Original M0 return CSV missing')
    old_returns=pd.read_csv(results[-1],index_col=0,parse_dates=True)
    observed=set(old_returns['Group10'].dropna().index.to_period('M'))
    rows=[]
    for date in dates:
        period=date.to_period('M'); row={'month':str(period),'calendar_month_end':str(date.date()),
            'month_end_weekend':date.dayofweek>=5,'observed_in_old_returns':period in observed}
        row.update({key:int(value.get(period,0)) for key,value in counts.items()})
        row['loop_discards_last_prepared_month']=period==skipped_last
        row['price_month_latest_date']=str(price.loc[price.TradingDate.dt.to_period('M').eq(period),'TradingDate'].max())
        reasons=[]
        if not row['factor_after_config'] and row['factor_input']: reasons.append('factor_removed_by_config_end_filter')
        if row['factor_after_config'] and not row['factor_after_pool']: reasons.append('factor_pool_date_or_stock_mismatch')
        if row['price_after_config'] and not row['price_exact_pool'] and row['price_month_key_overlap']:
            reasons.append('daily_price_exact_date_pool_join_loses_month')
        if row['loop_discards_last_prepared_month'] and row['prepared_factor_nonnull']: reasons.append('engine_skips_last_prepared_month')
        if not row['signal_with_adjacent_price']: reasons.append('no_valid_adjacent_month_price_pairs')
        if not row['observed_in_old_returns'] and not reasons: reasons.append('requires_further_trade_or_group_audit')
        row['diagnosis']='; '.join(reasons) or 'observed'
        rows.append(row)
    table=pd.DataFrame(rows); table.to_csv(out/'month_coverage_audit.csv',index=False,encoding='utf-8-sig')
    scope={'factor_config_path':str(config_path),'requested_start':str(start.date()),'requested_factor_end':str(end.date()),
        'parsed_backtest_end':cfg.backtest.end_date,'effective_factor_end':old_end,
        'raw_price_last_date_in_audit':str(price.TradingDate.max()),'old_observed_months':len(observed),
        'planned_months':len(dates),'no_training_performed':True,'no_inputs_modified':True,
        'month_key_overlap_is_diagnostic_not_new_backtest':True,
        'next_step':'Review month loss evidence before preparing corrected backtests or starting seed123/2026.',
        'source_files':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [factor_path,cfg.data.adj_close_path,cfg.data.suspension_limit_path,cfg.data.filter_stock_pool_path,results[-1]]},
        'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'src/backtest/factor_processor.py',root/'src/backtest/engine.py',root/'src/backtest/config.py']}}
    (out/'audit_summary.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table[['month','observed_in_old_returns','diagnosis']].to_string(index=False),flush=True)
    print('Completed. Audit: '+str(out),flush=True)
    if args.export_dir:
        try:
            destination=args.export_dir/out.name
            destination.mkdir(parents=True,exist_ok=False)
            for name in ['month_coverage_audit.csv','audit_summary.json']:
                shutil.copy2(out/name,destination/name)
            print('Copied audit to: '+str(destination),flush=True)
        except OSError as exc:
            print('Audit succeeded, but export failed. Copy the audit folder manually: '+str(exc),flush=True)

if __name__=='__main__': main()
