"""Isolated calendar-only replay and paired Huber replication. No production writes."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import training_base as base
from audit_months import adjacent_prices, normalize, read_slice

KEYS = base.KEYS
START = '2024-01-01'
SIGNAL_END = '2026-07-31'
VALUE_END = '2026-08-31'
EXPECTED = pd.period_range('2024-01', '2026-07', freq='M')
FAMILIES = {'training': ['M0', 'H1', 'H2', 'F50', 'F70'], 'size': ['T0', 'T2']}


def complete_month_ends(pool, trading_dates=None):
    """Repair absent/nontrading-empty month ends, not intentional trading-day empty pools."""
    out = pool.copy()
    out.index = pd.to_datetime(out.index).normalize()
    if out.index.has_duplicates or out.index.isna().any():
        raise ValueError('Pool has duplicate or invalid snapshot dates')
    out = out.sort_index()
    sessions = None if trading_dates is None else pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique()
    audit = []
    for period, part in out.groupby(out.index.to_period('M')):
        end = period.to_timestamp('M')
        exists = end in part.index
        original_members = int(part.loc[end].eq(1).sum()) if exists else 0
        # A dated nonempty snapshot is authoritative even on a nontrading day.
        # A real trading-session empty pool is also authoritative. Never seek the
        # last *nonempty* row: that would undo a legitimate exit from the universe.
        placeholder = exists and sessions is not None and end not in sessions and original_members == 0
        action = 'preserved'
        source = end if exists else part.index[-1]
        if not exists or placeholder:
            eligible = part if sessions is None else part.loc[part.index.isin(sessions)]
            if eligible.empty:
                raise ValueError(f'No same-month trading-session pool snapshot for {period}')
            source = eligible.index[-1]
            out.loc[end] = eligible.iloc[-1]
            action = 'replaced_nontrading_empty' if placeholder else 'appended'
        audit.append({'month': str(period), 'snapshot_date': str(source.date()),
                      'calendar_date': str(end.date()), 'action': action,
                      'original_members': original_members,
                      'aligned_members': int(out.loc[end].eq(1).sum())})
    return out.sort_index(), pd.DataFrame(audit)


def require_months(dates):
    months = pd.DatetimeIndex(pd.to_datetime(dates)).to_period('M')
    if months.has_duplicates or not months.equals(EXPECTED):
        raise ValueError(f'Expected 31 consecutive signal months; got {list(map(str, months))}')


def return_statistics(values, risk_free=.02):
    """Compute from every period return, including the first; no missing-return fill."""
    r = pd.Series(values, dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Invalid or incomplete monthly returns')
    wealth = np.r_[1.0, np.cumprod(1 + r.to_numpy())]
    vol = float(r.std(ddof=1) * np.sqrt(12))
    cagr = float(wealth[-1] ** (12 / len(r)) - 1)
    return {'Months': len(r), 'AnnualReturn': cagr,
            'Sharpe': ((r.mean()-((1+risk_free)**(1/12)-1))*12)/vol if vol > 0 else None,
            'ArithmeticSharpe': ((r.mean()-((1+risk_free)**(1/12)-1))*12)/vol if vol > 0 else None,
            'MaxDrawdown': float((1-wealth/np.maximum.accumulate(wealth)).max())}


def frame_hashes(folder):
    return {str(p): base.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def check_hashes(hashes):
    for path, digest in hashes.items():
        if not Path(path).is_file() or base.sha(path) != digest:
            raise ValueError('Changed/missing committed file: '+path)


def preflight(root, old, exp):
    identity = base.read_json(old/'01_audit/identity.json')
    if identity['start'] != START or identity['end'] != '2026-08-31' or identity['seed'] != 42:
        raise ValueError('Unexpected seed42 experiment window')
    print('Verifying original inputs and source code (large files are hashed once)...', flush=True)
    # Pin all original data and code, including pool and valuation inputs. No silent refresh.
    check_hashes(identity['files'])
    original_script = old/'00_tools/interpretable.py'
    if base.sha(Path(base.__file__)) != base.sha(original_script):
        raise ValueError('Training helper differs from the seed42 training implementation')
    generated = base.read_json(old/'01_audit/generated.json')
    check_hashes(generated['factor_hashes'])
    package = {p.name: base.sha(p) for p in Path(__file__).parent.glob('*.py')}
    signature = {'baseline_identity': identity, 'original_factors': generated['factor_hashes'],
                 'package': package, 'signal_start': START, 'signal_end': SIGNAL_END,
                 'valuation_end': VALUE_END, 'source_experiment': str(old)}
    path = exp/'01_audit/identity.json'
    if path.exists() and base.read_json(path) != signature:
        raise ValueError('Inputs/code changed. Preserve this study and use a NEW -StudyName.')
    base.save_json(path, signature)
    return identity


def stage_original(old, exp):
    jobs = []
    for family, names in FAMILIES.items():
        reference = None
        for name in names:
            src = old/'04_factors'/family/(name+'.parquet')
            dest = exp/'04_factors'/('seed42_'+family)/(name+'.parquet')
            stage_factor(src, dest, name)
            keys = base.keyed(pd.read_parquet(dest))[KEYS]
            if reference is None: reference = keys
            else: pd.testing.assert_frame_equal(reference, keys)
            jobs.append((family, 42, name, dest))
    return jobs


def stage_factor(src, dest, name):
    data = base.keyed(pd.read_parquet(src))
    data = data.loc[data.TradingDate.between(START, SIGNAL_END), KEYS+[name]].copy()
    require_months(sorted(data.TradingDate.unique()))
    if not np.isfinite(data[name]).all(): raise ValueError('Nonfinite prediction: '+str(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        pd.testing.assert_frame_equal(base.keyed(pd.read_parquet(dest)), data.reset_index(drop=True))
    else:
        temp = dest.with_name('.'+dest.name+'.tmp')
        data.to_parquet(temp, index=False); os.replace(temp, dest)


def make_processor(cfg, pool, bench):
    from backtest.factor_processor import FactorProcessor
    class CalendarProcessor(FactorProcessor):
        def _resolve_effective_period(self, df):
            # Factor clip remains July31; prices and next-day flags may extend through August.
            if df.empty: raise ValueError('Empty factors')
            return START, VALUE_END
    return CalendarProcessor(cfg, pool, bench, None, logging.getLogger('calendar_recheck'))


def load_market(root, exp):
    sources = base.backtest_sources(root)
    # Read one month beyond outcome cutoff to prove the August outcome is not partial.
    price = read_slice(sources['adj_close'], KEYS+['price'], pd.Timestamp(START), pd.Timestamp('2026-09-30'))
    if price.TradingDate.max() <= pd.Timestamp(VALUE_END):
        raise ValueError('Need prices after August31 to independently establish the August cutoff')
    limits = read_slice(sources['suspension_limit'], KEYS+['Suspension','LimitStatus'],
                        pd.Timestamp(START), pd.Timestamp('2026-09-30'))
    pools = []
    for label in ['filter_stock_pool','benchmark_stock_pool']:
        pool = pd.read_parquet(sources[label])
        pool.index = pd.to_datetime(pool.index)
        pool = pool.loc[(pool.index >= START) & (pool.index <= VALUE_END)]
        aligned, audit = complete_month_ends(pool, price.loc[price.price.gt(0) & np.isfinite(price.price),'TradingDate'])
        if not set(pd.period_range('2024-01','2026-08',freq='M')).issubset(set(aligned.index.to_period('M'))):
            raise ValueError('Pool is missing whole months: '+label)
        audit.to_csv(exp/'01_audit'/(label+'_calendar.csv'),index=False,encoding='utf-8-sig')
        pools.append(aligned)
    return price, limits, *pools


def valuation_audit(result, processed, raw_price):
    """Diagnostic only. Do not drop holdings based on future price availability."""
    pairs = adjacent_prices(raw_price.loc[raw_price.TradingDate <= VALUE_END])
    pairs['TradingDate'] = pairs['_month'].dt.to_timestamp()
    pairs['raw_adjacent_return'] = np.where(pairs.valid_next_price, pairs.price_next/pairs.price-1, np.nan)
    lookup = pairs[KEYS+['raw_adjacent_return','valid_next_price']]
    rows = result.factor_return_df.merge(lookup,on=KEYS,how='left',validate='many_to_one')
    rows['missing_adjacent_price'] = rows.valid_next_price.ne(True)
    rows['return_differs_from_raw'] = (~rows.missing_adjacent_price) & ~np.isclose(
        rows.ret, rows.raw_adjacent_return, rtol=1e-7, atol=1e-9)
    rows['needs_review'] = rows.missing_adjacent_price | rows.return_differs_from_raw
    factor = processed.factor_df.copy()
    factor['TradingDate'] = factor.TradingDate.dt.to_period('M').dt.to_timestamp()
    used = rows[KEYS].assign(used=True)
    unused = factor.merge(used,on=KEYS,how='left',validate='one_to_one')
    unused = unused.loc[unused.used.isna(),KEYS]
    return rows, unused


def run_backtests(root, exp, jobs):
    from backtest.config import BacktestConfig, BenchmarkConfig, BenchmarkItemConfig, TransactionCostConfig
    from backtest.engine import BacktestEngine
    market = None
    for family, seed, name, factor_path in jobs:
        label = f'{family}_seed{seed}_{name}'
        dest = exp/'05_backtests'/label
        marker = dest/'COMMITTED.json'
        if marker.exists():
            committed = base.read_json(marker)
            check_hashes(committed['files'])
            if committed['factor_hash'] != base.sha(factor_path):
                raise ValueError('Backtest input factor changed: '+str(factor_path))
            print('Reusing verified backtest: '+label,flush=True); continue
        if market is None: market = load_market(root, exp)
        price, limits, pool, bench = market
        cfg = BacktestConfig(START,SIGNAL_END,'monthly',10,None,.02,TransactionCostConfig())
        processor = make_processor(cfg,pool,bench)
        engine = BacktestEngine(cfg,BenchmarkConfig([BenchmarkItemConfig('pool','equal')]),None)
        processed = processor.LoadAndProcess(factor_path,price,limits)
        result = engine.Run(processed)
        coverage = pd.DataFrame(index=EXPECTED)
        coverage['eligible_factor_rows'] = processed.factor_df.groupby(processed.factor_df.TradingDate.dt.to_period('M')).size().reindex(EXPECTED,fill_value=0)
        coverage['return_present'] = EXPECTED.isin(result.period_return_df.index.to_period('M'))
        coverage.to_csv(exp/'01_audit'/(label+'_month_coverage.csv'),index_label='signal_month')
        require_months(result.period_return_df.index)
        result_rows, unused = valuation_audit(result,processed,price)
        ic = []
        for date, g in result.factor_return_df.groupby('TradingDate',sort=True):
            a,b = g[name],g.ret
            if a.nunique()<2 or b.nunique()<2: raise ValueError('Constant IC input: '+str(date))
            ic.append({'signal_month':str(date.to_period('M')),
                       'return_month':str(date.to_period('M')+1),'IC':a.corr(b),
                       'RankIC':a.corr(b,method='spearman'),'Stocks':len(g)})
        ic = pd.DataFrame(ic)
        require_months(pd.to_datetime(ic.signal_month))
        r = result.period_return_df['Group10']
        stats = return_statistics(r)
        selected = result_rows.loc[result_rows.quantile_group.astype(str).eq('Group10')]
        row = dict(family=family,seed=seed,factor=name,method=base.DEFINITIONS[name],**stats)
        for metric in ['IC','RankIC']:
            row[metric] = float(ic[metric].mean())
            sd = ic[metric].std()
            row[metric+'IR'] = float(ic[metric].mean()/sd) if sd>0 else None
            row[metric+'PositiveRatio'] = float(ic[metric].gt(0).mean())
        row.update(ValuationIssues=int(result_rows.needs_review.sum()),
                   Group10ValuationIssues=int(selected.needs_review.sum()),
                   UnusedEligibleRows=len(unused),
                   ReturnConvention='calendar-only replay; legacy execution and valuation logic',
                   SharpeConvention='Standard annualized Sharpe: sqrt(12)*mean(monthly excess)/sample std(monthly excess); effective annual rf=2%; includes first return')
        temp = dest.with_name(dest.name+'_'+uuid4().hex); temp.mkdir(parents=True)
        result.period_return_df.to_csv(temp/'period_return.csv',index_label='signal_month_start')
        ic.to_csv(temp/'monthly_ic.csv',index=False)
        result_rows.loc[result_rows.needs_review].to_csv(temp/'valuation_issues.csv',index=False)
        unused.to_csv(temp/'unused_eligible_rows.csv',index=False)
        base.save_json(temp/'metrics.json',row)
        # Store relative paths so committing the directory does not invalidate hashes.
        digests = {str(dest/p.relative_to(temp)):base.sha(p) for p in temp.iterdir()}
        base.save_json(temp/'COMMITTED.json',{'files':digests,'factor_hash':base.sha(factor_path)})
        if dest.exists(): raise ValueError('Uncommitted destination exists: '+str(dest))
        os.replace(temp,dest)
        print(f'Completed {label}: months={len(r)}, Group10 valuation issues={row["Group10ValuationIssues"]}',flush=True)


def generate_seed(root, exp, seed, baseline_identity):
    import torch
    for key, actual in {'torch':torch.__version__,'pandas':pd.__version__,'numpy':np.__version__}.items():
        if baseline_identity['runtime'][key] != actual:
            raise ValueError('Runtime differs from seed42: '+key)
    seed_exp = exp/'02_training'/f'seed{seed}'
    for sub in ['01_audit','02_windows','03_raw_factors']: (seed_exp/sub).mkdir(parents=True,exist_ok=True)
    args = SimpleNamespace(project_root=root,start=START,end='2026-08-31',seed=seed,industry_file='')
    cfg, frame, features, windows, *_ = base.prepare(args,seed_exp)
    if features != baseline_identity['features']: raise ValueError('Feature list differs from seed42')
    from models.mlp import _ResidualMlpNetwork
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    jobs = []
    for name, delta in [('M0',None),('H1',.25)]:
        torch.manual_seed(seed)
        net = _ResidualMlpNetwork(len(features),float(cfg.Section('model')['params']['dropout'])).to(device)
        parts = []
        for k,w in enumerate(windows):
            # August predictions are retained, but August signals have no complete September outcome here.
            dest = seed_exp/'02_windows'/name/w.window_id
            marker = dest/'COMMITTED.json'
            if marker.exists():
                check_hashes(base.read_json(marker)['files'])
                net.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True))
                parts.append(pd.read_parquet(dest/'factor.parquet')); continue
            tr=frame.loc[frame.TradingDate.isin(w.train_periods)].copy()
            va=frame.loc[frame.TradingDate.isin(w.valid_periods)].copy()
            te=frame.loc[frame.TradingDate.isin(w.test_periods)].copy()
            if te.empty: raise ValueError('Empty test window')
            diag=base.fit(net,tr,va,features,cfg.Section('data')['target_col'],np.ones(len(features),np.float32),
                          cfg.Section('training'),k>0,delta,seed+k,device)
            x=np.nan_to_num(te[features].to_numpy(np.float32),nan=0,posinf=0,neginf=0)
            output=te[KEYS].copy(); output[name]=base.predict(net,x,device)
            if not np.isfinite(output[name]).all(): raise ValueError('Invalid predictions')
            temp=dest.with_name(dest.name+'_'+uuid4().hex); temp.mkdir(parents=True)
            torch.save(net.state_dict(),temp/'model.pt'); output.to_parquet(temp/'factor.parquet',index=False)
            diag.update(seed=seed,model=name,window=w.window_id,device=device,
                        files={str(dest/p.name):base.sha(p) for p in temp.iterdir()})
            base.save_json(temp/'COMMITTED.json',diag); os.replace(temp,dest)
            parts.append(output); print(f'seed{seed} {name} [{k+1}/{len(windows)}] committed',flush=True)
        raw=seed_exp/'03_raw_factors'/(name+'.parquet')
        base.keyed(pd.concat(parts)).to_parquet(raw,index=False)
        staged=exp/'04_factors'/f'seed{seed}_training'/(name+'.parquet')
        stage_factor(raw,staged,name); jobs.append(('training',seed,name,staged))
    return jobs


def summarize(exp, old):
    report=exp/'07_report'; report.mkdir(parents=True,exist_ok=True)
    records=[]; monthly=[]; yearly=[]; checks=[]
    for folder in sorted((exp/'05_backtests').iterdir()):
        if not (folder/'COMMITTED.json').exists(): continue
        check_hashes(base.read_json(folder/'COMMITTED.json')['files'])
        row=base.read_json(folder/'metrics.json'); records.append(row)
        r=pd.read_csv(folder/'period_return.csv',index_col=0,parse_dates=True)['Group10']
        require_months(r.index)
        ic=pd.read_csv(folder/'monthly_ic.csv'); ic['Group10Return']=r.to_numpy()
        ic.insert(0,'factor',row['factor']); ic.insert(0,'seed',row['seed']); ic.insert(0,'family',row['family'])
        monthly.append(ic)
        for year,g in r.groupby(r.index.year):
            yearly.append({'family':row['family'],'seed':row['seed'],'factor':row['factor'],
                           'signal_year':year,'CumulativeReturn':float((1+g).prod()-1),**return_statistics(g)})
        checks.append({'experiment':folder.name,'consecutive_months':True,
                       'valuation_issues':row['ValuationIssues'],'group10_valuation_issues':row['Group10ValuationIssues']})
    if not records: raise ValueError('No completed backtests')
    results=pd.DataFrame(records); all_months=pd.concat(monthly,ignore_index=True)
    paired=[]
    for seed,g in results.loc[results.family.eq('training')].groupby('seed'):
        if not {'M0','H1'}.issubset(set(g.factor)): continue
        indexed=g.set_index('factor'); a=indexed.loc['M0']; b=indexed.loc['H1']
        m=all_months.loc[(all_months.seed==seed)&all_months.family.eq('training')]
        left=m.loc[m.factor.eq('M0')].set_index('signal_month')
        right=m.loc[m.factor.eq('H1')].set_index('signal_month')
        if not left.index.equals(right.index): raise ValueError('Unmatched seed-pair months')
        d={'seed':int(seed),'months':len(left)}
        for metric in ['Sharpe','ArithmeticSharpe','IC','RankIC','MaxDrawdown']: d['H1_minus_M0_'+metric]=float(b[metric]-a[metric])
        for metric in ['IC','RankIC','Group10Return']: d[metric+'_win_fraction']=float((right[metric]>left[metric]).mean())
        paired.append(d)
    notes=pd.DataFrame([
        ['范围','2024年1月至2026年7月信号，共31个月；收益观察至2026年8月。'],
        ['修正','股票池仅补入同月最后已知快照的日历月末键；不把未来快照回填到过去。'],
        ['指标','Sharpe=月超额收益均值/样本标准差×sqrt(12)；年化有效无风险利率2%按月复利换算；ArithmeticSharpe为同口径兼容别名；计入首月收益。'],
        ['口径','Group10最高预测分十分位组合，不是多空。ICIR未乘sqrt(12)。'],
        ['限制','保留旧引擎个股估值、停牌涨跌停及换手费用近似；价格异常逐项列出，不宣称实盘可交易。'],
        ['比较','仅同family、同种子成对比较。T0为既有三种子E0，不能和单种子M0当作纯架构差异。'],
        ['旧结果','旧20个月结果与本次31个月及首月收益口径不同；差异不代表模型进步。'],
        ['训练','seed42不重新训练；只有通过估值检查后才训练seed123与2026的M0/H1。'],
        ['判断','不自动晋级或替换生产因子。三种子一致性仍不等于独立市场样本显著性。'],
        ['待观察','2026年8月信号保留但不计收益，因现有9月日行情未覆盖整月。'],
    ],columns=['项目','说明'])
    tables={'说明':notes,'同口径结果':results,'H1对M0':pd.DataFrame(paired),
            '逐月数据':all_months,'分年度':pd.DataFrame(yearly),'检查':pd.DataFrame(checks)}
    historical = old/'07_report/comparison.csv'
    if historical.exists():
        previous = pd.read_csv(historical)
        previous.insert(0,'comparison_warning','旧20个月及旧指标口径，仅留档，不可视为本次模型提升的对照')
        tables['旧结果留档'] = previous
    if {42,123,2026}.issubset({int(x['seed']) for x in paired}):
        seed_summary=[]
        for model in ['M0','H1']:
            selected=results.loc[results.family.eq('training') & results.factor.eq(model) & results.seed.isin([42,123,2026])]
            for metric in ['Sharpe','ArithmeticSharpe','IC','RankIC','MaxDrawdown']:
                seed_summary.append({'model':model,'metric':metric,'seeds':len(selected),
                                     'mean':selected[metric].mean(),'sample_std':selected[metric].std(),
                                     'minimum':selected[metric].min(),'maximum':selected[metric].max()})
        tables['种子波动'] = pd.DataFrame(seed_summary)
    for name,frame in tables.items(): frame.to_csv(report/(name+'.csv'),index=False,encoding='utf-8-sig')
    base.save_json(report/'status.json',{'completed_experiments':len(records),
      'seed_pairs':sorted(int(x['seed']) for x in paired),'production_changed':False,
      'valuation_review_required':bool(results.ValuationIssues.gt(0).any()),
      'report_type':'calendar-only replay; research, not final tradability validation'})
    # Cloud runner retains the existing Python report backend; no bundled JS runtime is assumed there.
    # Tables above are authoritative, reproducible CSVs even if Excel is open and replacement fails.
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb=Workbook(); wb.remove(wb.active)
    for name,frame in tables.items():
        ws=wb.create_sheet(name); ws.sheet_view.showGridLines=False
        ws.append(list(frame.columns))
        for values in frame.itertuples(index=False,name=None):
            ws.append([None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for v in values])
        ws.freeze_panes='D2' if name=='逐月数据' else 'A2'; ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:
            cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
            cell.fill=PatternFill('solid',fgColor='17365D'); cell.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=36
        for col in ws.columns:
            title=str(col[0].value)
            width=70 if title in ['说明','method','ReturnConvention','SharpeConvention'] else min(32,max(14,len(title)+2))
            ws.column_dimensions[col[0].column_letter].width=width
            for c in col[1:]:
                c.font=Font(name='Arial',size=10); c.alignment=Alignment(wrap_text=True,vertical='top')
                if isinstance(c.value,float): c.number_format='0.0000'
        for i in range(2,ws.max_row+1): ws.row_dimensions[i].height=44 if name=='说明' else 32
    output=report/'Calendar_Huber_validation.xlsx'; temp=report/('.report_'+uuid4().hex+'.xlsx')
    wb.save(temp)
    verify=load_workbook(temp,read_only=True,data_only=True)
    if verify.sheetnames != list(tables): raise ValueError('Report export mismatch')
    for name,frame in tables.items():
        if verify[name].max_row!=len(frame)+1: raise ValueError('Report row count mismatch: '+name)
    verify.close(); os.replace(temp,output)
    print('Report: '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['Repair','All','Seeds','Summarize'],default='All')
    parser.add_argument('--study',default='calendar_v2')
    parser.add_argument('--baseline-study',default='screen_seed42')
    args=parser.parse_args(); root=args.project_root.resolve(); os.chdir(root)
    for name in [args.study,args.baseline_study]:
        if not name.isascii() or not name.replace('_','').replace('-','').isalnum(): raise ValueError('Invalid study name')
    sys.path.insert(0,str(root/'src'))
    old=root/'data/experiments/nn_interpretable_oct'/args.baseline_study
    exp=root/'data/experiments/nn_calendar_huber'/args.study
    for sub in ['00_tools','01_audit','02_training','04_factors','05_backtests','06_logs','07_report']:
        (exp/sub).mkdir(parents=True,exist_ok=True)
    lock=exp/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    try:
        identity=preflight(root,old,exp)
        jobs=stage_original(old,exp)
        if args.mode!='Summarize': run_backtests(root,exp,jobs)
        summarize(exp,old)
        if args.mode in ['All','Seeds']:
            issues=[base.read_json(exp/'05_backtests'/f'training_seed42_{name}'/'metrics.json')['ValuationIssues'] for name in ['M0','H1']]
            if any(issues):
                base.save_json(exp/'07_report/seed_gate.json',{'training_started':False,'reason':'Legacy returns differ from adjacent raw prices or have missing next-month prices','counts_M0_H1':issues})
                print('PAUSED: Calendar reports saved. Valuation issues require review BEFORE additional training.',flush=True)
                return
            for seed in [123,2026]:
                new_jobs=generate_seed(root,exp,seed,identity)
                # Exact key equality is required across both models and all seeds, not only month counts.
                for _,_,name,path in new_jobs:
                    reference=base.keyed(pd.read_parquet(exp/'04_factors/seed42_training'/f'{name}.parquet'))[KEYS]
                    pd.testing.assert_frame_equal(reference,base.keyed(pd.read_parquet(path))[KEYS])
                run_backtests(root,exp,new_jobs); summarize(exp,old)
            base.save_json(exp/'07_report/seed_gate.json',{'training_started':True,'completed_seeds':[42,123,2026]})
    finally:
        lock.unlink()


if __name__=='__main__': main()
