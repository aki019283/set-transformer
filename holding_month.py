"""User-defined holding-month labels; explicit first-session-close execution assumption."""
import numpy as np
import pandas as pd

POLICY='Label month M uses information through M-1 month-end; daily-derived source row M-1 is relabeled M before joining; entry first session close of M, exit first session close of M+1. Execution timing is an explicit research assumption.'

def relabel_daily_derived_months(raw, target, report):
    """Move a row computed through source month S to holding-month label S+1."""
    raw=raw.copy();raw.TradingDate=pd.to_datetime(raw.TradingDate)
    source_month=raw.TradingDate.dt.to_period('M')
    holding_month=source_month+1
    audit=pd.DataFrame({'source_factor_month':source_month.astype(str),'holding_month':holding_month.astype(str)})
    if target not in raw.columns:raise ValueError('Target column missing before relabel')
    raw.TradingDate=holding_month.dt.to_timestamp('M')
    keys=pd.DataFrame({'TradingDate':raw.TradingDate,'Stkcd':raw.Stkcd.astype(str)})
    if keys.duplicated().any():raise ValueError('Duplicate keys after source-month relabel')
    report.mkdir(parents=True,exist_ok=True)
    audit.drop_duplicates().sort_values(['source_factor_month','holding_month']).to_csv(report/'feature_month_relabel.csv',index=False,encoding='utf-8-sig')
    return raw

def month_schedule(months,sessions,cutoff):
    sessions=pd.DatetimeIndex(sessions).sort_values()
    if sessions.has_duplicates or sessions.hasnans:raise ValueError('Invalid calendar sessions')
    sessions=sessions[sessions<=pd.Timestamp(cutoff)]
    rows=[]
    for month in sorted(set(months)):
        month=pd.Period(month,freq='M')
        prior=sessions[sessions<month.start_time]
        entry=sessions[sessions.to_period('M')==month]
        exit_dates=sessions[sessions.to_period('M')==month+1]
        if not len(prior) or not len(entry) or not len(exit_dates):continue
        if prior[-1].to_period('M') != month-1:raise ValueError('Missing preceding calendar month')
        rows.append(dict(holding_month=str(month),factor_label=month.to_timestamp('M'),signal_label=(month-1).to_timestamp('M'),information_cutoff=prior[-1],entry=entry[0],exit=exit_dates[0]))
    return pd.DataFrame(rows)

def align_dataset(raw,source,target,report):
    raw=raw.copy();raw.TradingDate=pd.to_datetime(raw.TradingDate);raw.Stkcd=raw.Stkcd.astype(str)
    months=raw.TradingDate.dt.to_period('M')
    if pd.DataFrame({'month':months,'stock':raw.Stkcd}).duplicated().any():raise ValueError('Multiple rows for stock/holding month')
    calendar=pd.read_parquet(source/'Calendar.parquet');calendar=calendar.loc[calendar.Markettype==1]
    if calendar.Clddt.duplicated().any():raise ValueError('Duplicate exchange calendar dates')
    sessions=pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.State.eq('O'),'Clddt'])).sort_values()
    observed=pd.read_parquet(source/'adj_close.parquet',columns=['TradingDate']).TradingDate
    cutoff=pd.to_datetime(observed).max()
    schedule=month_schedule(months,sessions,cutoff)
    if schedule.empty:raise ValueError('No mature holding periods')
    endpoints=pd.DatetimeIndex(pd.concat([schedule.entry,schedule.exit]).unique()).to_pydatetime().tolist()
    prices=pd.read_parquet(source/'adj_close.parquet',filters=[('TradingDate','in',endpoints)])
    prices.TradingDate=pd.to_datetime(prices.TradingDate);prices.Stkcd=prices.Stkcd.astype(str)
    if prices.duplicated(['TradingDate','Stkcd']).any():raise ValueError('Duplicate price keys')
    lookup=prices.set_index(['TradingDate','Stkcd']).price
    raw['holding_month']=months.astype(str)
    work=raw.merge(schedule[['holding_month','signal_label','entry','exit']],on='holding_month',how='left',validate='many_to_one')
    left=lookup.reindex(pd.MultiIndex.from_arrays([work.entry,work.Stkcd])).to_numpy()
    right=lookup.reindex(pd.MultiIndex.from_arrays([work.exit,work.Stkcd])).to_numpy()
    valid=np.isfinite(left)&np.isfinite(right)&(left>0)&(right>0)
    rebuilt=np.full(len(work),np.nan);rebuilt[valid]=right[valid]/left[valid]-1
    work[target]=rebuilt
    work['label_observed']=valid
    report.mkdir(parents=True,exist_ok=True)
    work.groupby('holding_month').agg(stocks=('Stkcd','size'),observed_labels=('label_observed','sum')).to_csv(report/'label_coverage.csv',encoding='utf-8-sig')
    schedule.to_csv(report/'holding_month_schedule.csv',index=False,encoding='utf-8-sig')
    work.loc[work.signal_label.isna(),['TradingDate','Stkcd']].to_csv(report/'unmatured_rows.csv',index=False,encoding='utf-8-sig')
    work=work.loc[work.signal_label.notna()].copy()
    work.TradingDate=work.signal_label
    work=work.drop(columns=['holding_month','signal_label','entry','exit','label_observed'])
    return work,schedule,cutoff


def check_label_availability(windows, schedule):
    """Labels used to fit/select must have realized before the next split's information cutoff."""
    by_signal=schedule.set_index('signal_label')
    records=[]
    for w in windows:
        for earlier,later in [('train','valid'),('valid','test')]:
            available=by_signal.loc[list(getattr(w,earlier+'_periods')),'exit'].max()
            cutoff=by_signal.loc[min(getattr(w,later+'_periods')),'information_cutoff']
            if available>cutoff:raise ValueError(f'Future label in {w.window_id}: {earlier} -> {later}')
            records.append(dict(window=w.window_id,split=earlier,label_realized=available,next_information_cutoff=cutoff))
    return pd.DataFrame(records)


def audit_feature_timing(raw,features,target,start,end,report):
    """Descriptive timing diagnostic, never a test-set-driven feature selector."""
    lookup=raw.set_index(['TradingDate','Stkcd'])[target]
    rows=[]
    for date,g in raw.loc[raw.TradingDate.between(start,end)].groupby('TradingDate'):
        x=g[features].replace([np.inf,-np.inf],np.nan)
        for offset in [-1,0,1]:
            label_date=(date.to_period('M')+offset).to_timestamp('M')
            y=lookup.reindex(pd.MultiIndex.from_arrays([[label_date]*len(g),g.Stkcd])).to_numpy()
            valid=np.isfinite(y)
            if valid.sum()<3:continue
            xx=x.loc[valid].reset_index(drop=True);yy=pd.Series(y[valid])
            if yy.nunique()<2:continue
            varying=xx.columns[xx.nunique()>1]
            cor=xx[varying].corrwith(yy,method='spearman')
            rows.extend(dict(holding_month=str(date.to_period('M')+1),return_month_offset=offset,feature=k,rank_ic=v,n_observed=int(valid.sum())) for k,v in cor.items())
    table=pd.DataFrame(rows)
    table.to_csv(report/'feature_timing_monthly.csv',index=False,encoding='utf-8-sig')
    summary=table.groupby(['feature','return_month_offset']).rank_ic.agg(['mean','std','count']).reset_index()
    summary.to_csv(report/'feature_timing_summary.csv',index=False,encoding='utf-8-sig')
    flags=summary.loc[summary.return_month_offset.eq(0)&summary['mean'].abs().ge(.5)&summary['count'].ge(2)]
    return dict(review_required=not flags.empty,flags=flags.to_dict('records'),
        note='Absolute mean single-feature holding-month Rank IC >= 0.5 is a diagnostic flag, not proof of leakage or a significance test. Verify source formula and information timestamps; do not select/drop features based on these test-period results.')
