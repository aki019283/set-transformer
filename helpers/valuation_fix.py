"""Separate price-only repair and fixed causal rank-smoothing experiment."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

KEYS=['TradingDate','Stkcd']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def adjacent_returns(price):
    """Last recorded quote per stock-month; exact t+1 join, no fill or bridging."""
    p=price[KEYS+['price']].copy(); p.TradingDate=pd.to_datetime(p.TradingDate)
    if p.duplicated(KEYS).any(): raise ValueError('Duplicate daily price keys')
    p['month']=p.TradingDate.dt.to_period('M')
    p=p.sort_values('TradingDate').groupby(['month','Stkcd']).tail(1)
    nxt=p[['month','Stkcd','price','TradingDate']].copy(); nxt['month']=nxt.month-1
    pairs=p.merge(nxt,on=['month','Stkcd'],how='left',suffixes=('','_next'),validate='one_to_one')
    valid=np.isfinite(pairs.price)&np.isfinite(pairs.price_next)&pairs.price.gt(0)&pairs.price_next.gt(0)
    pairs['ret']=np.nan
    pairs.loc[valid,'ret']=pairs.loc[valid,'price_next']/pairs.loc[valid,'price']-1
    pairs['quote_date']=pairs.TradingDate
    pairs['TradingDate']=pairs.month.dt.to_timestamp()
    return pairs[KEYS+['ret','quote_date','TradingDate_next','price','price_next']]


def smooth_rank(frame,name,newname,weight=.75):
    data=frame[KEYS+[name]].copy(); data.TradingDate=pd.to_datetime(data.TradingDate)
    if data.duplicated(KEYS).any() or not np.isfinite(data[name]).all(): raise ValueError('Invalid factor input')
    data['rank_now']=data.groupby('TradingDate')[name].rank(pct=True,method='average')
    data['month']=data.TradingDate.dt.to_period('M')
    past=data[['month','Stkcd','rank_now']].rename(columns={'rank_now':'rank_previous'})
    past['month']=past.month+1
    data=data.merge(past,on=['month','Stkcd'],how='left',validate='one_to_one')
    data[newname]=weight*data.rank_now+(1-weight)*data.rank_previous.fillna(data.rank_now)
    return data[KEYS+[newname]],data[KEYS+['rank_now','rank_previous',newname]]


def make_engine(base_engine,raw,issue_dir):
    class FixedEngine(base_engine):
        def _prepare_rebalance_frame(self,processed):
            current=super()._prepare_rebalance_frame(processed)
            paired=adjacent_returns(processed.price_df)
            return current.drop(columns='ret').merge(paired[KEYS+['ret']],on=KEYS,how='left',validate='one_to_one')

        def Run(self,processed):
            full=raw.loc[raw.TradingDate.between(processed.start_date,processed.end_date)].copy()
            monthly=adjacent_returns(full)
            eligible=processed.factor_df.copy()
            eligible.TradingDate=eligible.TradingDate.dt.to_period('M').dt.to_timestamp()
            missing_current=eligible.merge(monthly[KEYS+['price']],on=KEYS,how='left',validate='one_to_one')
            missing_current=missing_current.loc[~np.isfinite(missing_current.price)|missing_current.price.le(0)]
            if len(missing_current):
                missing_current.to_csv(issue_dir/'missing_current_prices.csv',index=False)
                raise ValueError('Eligible signals lack current prices; see missing_current_prices.csv')
            result=super().Run(replace(processed,price_df=full))
            bad=result.factor_return_df.loc[~np.isfinite(result.factor_return_df.ret)]
            if len(bad):
                bad.to_csv(issue_dir/'held_positions_missing_returns.csv',index=False)
                raise ValueError('Held positions lack adjacent-month valuations; no accepted metrics saved')
            return result
    return FixedEngine


def summarize_result(result,name,previous):
    returns=result.period_return_df['Group10']
    previous.require_months(returns.index)
    stats=previous.return_statistics(returns)
    monthly=[]; last=set()
    for i,(date,g) in enumerate(result.factor_return_df.groupby('TradingDate',sort=True)):
        selected=g.loc[g.quantile_group.astype(str).eq('Group10')]
        current=set(selected.Stkcd)
        turnover=(1-len(current&last)/len(current)) if i and len(current) else np.nan
        monthly.append({'signal_month':str(date.to_period('M')),'Stocks':len(g),'Group10Stocks':len(current),
                        'IC':g[name].corr(g.ret),'RankIC':g[name].corr(g.ret,method='spearman'),
                        'Group10Return':returns.loc[date],'EntryFractionProxy':turnover})
        last=current
    monthly=pd.DataFrame(monthly)
    if not np.isfinite(monthly[['IC','RankIC','Group10Return']]).all().all(): raise ValueError('Invalid monthly metrics')
    stats.update(IC=float(monthly.IC.mean()),RankIC=float(monthly.RankIC.mean()),
                 EntryFractionProxy=float(monthly.EntryFractionProxy.mean()))
    return stats,monthly


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    root=args.project_root.resolve(); out=args.output_dir.resolve()
    source=root/'data/experiments/nn_calendar_huber/calendar_v2'
    if (source/'RUNNING.lock').exists(): raise ValueError('calendar_v2 still running')
    identity=json.loads((source/'01_audit/identity.json').read_text(encoding='utf-8'))
    for name,digest in identity['package'].items():
        if sha(source/'00_tools'/name)!=digest: raise ValueError('Source package changed: '+name)
    sys.path.insert(0,str(root/'src')); sys.path.insert(0,str(source/'00_tools'))
    import calendar_recheck as previous
    from backtest.config import BacktestConfig,BenchmarkConfig,TransactionCostConfig
    from backtest.engine import BacktestEngine
    # Verify data and source versions before mixing new and existing results.
    previous.check_hashes(identity['baseline_identity']['files'])
    for sub in ['01_audit','factors','backtests','report']: (out/sub).mkdir(parents=True,exist_ok=True)
    price,limits,pool,bench=previous.load_market(root,out)
    price=price.loc[price.TradingDate<=previous.VALUE_END].copy()
    adjacent_returns(price).to_csv(out/'01_audit/price_pairs.csv',index=False)
    jobs=[]
    for family,names in previous.FAMILIES.items():
        for name in names:
            factor=source/'04_factors'/('seed42_'+family)/(name+'.parquet')
            old_dir=source/'05_backtests'/f'{family}_seed42_{name}'
            marker=previous.base.read_json(old_dir/'COMMITTED.json')
            previous.check_hashes(marker['files'])
            if sha(factor)!=marker['factor_hash']: raise ValueError('Original factor changed')
            jobs.append((family,name,factor,name,old_dir))
            if name in ['M0','T0']:
                smoothname=name+'_SM75'
                data=pd.read_parquet(factor)
                smooth,details=smooth_rank(data,name,smoothname)
                dest=out/'factors'/(smoothname+'.parquet'); smooth.to_parquet(dest,index=False)
                details.to_csv(out/'01_audit'/(smoothname+'_inputs.csv'),index=False)
                jobs.append((family,smoothname,dest,name,None))
    summary=[]; old_comparison=[]; failures=[]
    for family,name,path,reference,old_dir in jobs:
        dest=out/'backtests'/name; dest.mkdir(exist_ok=True)
        try:
            cfg=BacktestConfig(previous.START,previous.SIGNAL_END,'monthly',10,None,.02,TransactionCostConfig())
            processor=previous.make_processor(cfg,pool,bench)
            processed=processor.LoadAndProcess(path,price,limits)
            engine=make_engine(BacktestEngine,price,dest)(cfg,BenchmarkConfig([]),None)
            result=engine.Run(processed)
            audited,unused=previous.valuation_audit(result,processed,price)
            if audited.needs_review.any():
                audited.loc[audited.needs_review].to_csv(dest/'remaining_valuation_issues.csv',index=False)
                raise ValueError('Independent raw-price cross-check failed')
            stats,monthly=summarize_result(result,name,previous)
            result.period_return_df.to_csv(dest/'period_return.csv')
            monthly.to_csv(dest/'monthly.csv',index=False,encoding='utf-8-sig')
            result.factor_return_df.to_parquet(dest/'positions.parquet',index=False)
            row=dict(family=family,factor=name,reference=reference,**stats)
            summary.append(row)
            if old_dir:
                old=previous.base.read_json(old_dir/'metrics.json')
                old_comparison.append({'factor':name,'OldSharpe':old['Sharpe'],'FixedSharpe':stats['Sharpe'],
                                       'OldIC':old['IC'],'FixedIC':stats['IC'],
                                       'OldMaxDrawdown':old['MaxDrawdown'],'FixedMaxDrawdown':stats['MaxDrawdown']})
            previous.base.save_json(dest/'metrics.json',row)
            print('Completed '+name,flush=True)
        except ValueError as exc:
            failures.append({'factor':name,'reason':str(exc)})
            print('BLOCKED '+name+': '+str(exc),flush=True)
    pd.DataFrame(summary).to_csv(out/'report/summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(old_comparison).to_csv(out/'report/valuation_before_after.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(failures,columns=['factor','reason']).to_csv(out/'report/blocked.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    byname={r['factor']:r for r in summary}
    for name in ['M0','T0']:
        if name not in byname or name+'_SM75' not in byname: continue
        a,b=byname[name],byname[name+'_SM75']
        for metric in ['Sharpe','IC','RankIC','MaxDrawdown','EntryFractionProxy']:
            comparisons.append({'reference':name,'metric':metric,'original_fixed':a[metric],
                                'smooth_fixed':b[metric],'difference':b[metric]-a[metric]})
    pd.DataFrame(comparisons).to_csv(out/'report/smoothing_comparison.csv',index=False,encoding='utf-8-sig')
    previous.base.save_json(out/'report/status.json',{'completed':len(summary),'blocked':failures,'training_performed':False,
      'production_modified':False,'all_valuation_checks_passed':len(failures)==0,
      'fixed_weight':.75,'no_hyperparameter_search':True})
    (out/'report/说明.txt').write_text('本次只改变估值价格来源及相邻月份配对。原选股、交易限制和简化手续费模型保留。\n'
      '31个信号月份为2024年1月至2026年7月。修复前后差异不是模型提升。\n'
      'SM75=75%当月截面预测排名+25%上一月排名；缺少上一月或首月则用当月排名。仅M0和T0增加平滑对照，未调权重。\n'
      'EntryFractionProxy为最高分组新增成员比例，非含权重漂移的真实换手率。旧费用模型未充分模拟冲击、容量及首期建仓费。\n'
      '本次只报告Group10，不报告尚未修复的池基准或超额收益。\n'
      '缺价不会被补零或根据未来价格是否存在删股；有受影响持仓的方案会被阻止输出指标。\n'
      '报告为CSV，可用Excel打开；blocked.csv有记录时应先处理，不能视为全部验证成功。',encoding='utf-8-sig')
    print('Report: '+str(out/'report'),flush=True)


if __name__=='__main__': main()
