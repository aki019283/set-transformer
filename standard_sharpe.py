"""Standard periodic-excess-return Sharpe. No CAGR-based Sharpe."""
from pathlib import Path
import numpy as np
import pandas as pd

CONVENTION = 'sqrt(P)*mean(r-rf_period)/std(r-rf_period,ddof=1); rf_period=(1+rf_annual)**(1/P)-1'

def standard_sharpe(returns, periods=12, annual_rf=.02):
    values=np.asarray(returns,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Sharpe requires complete finite periodic returns; no missing-value fill/drop')
    if periods<=0 or annual_rf<=-1: raise ValueError('Invalid annualization or risk-free rate')
    if len(values)<2: return float('nan')
    excess=values-((1+annual_rf)**(1/periods)-1)
    sd=excess.std(ddof=1)
    if sd<=np.finfo(float).eps: return float('nan')
    return float(np.sqrt(periods)*excess.mean()/sd)

def checked_monthly(path):
    frame=pd.read_csv(path,index_col=0)
    dates=pd.DatetimeIndex(pd.to_datetime(frame.index))
    periods=dates.to_period('M')
    if dates.has_duplicates or not dates.is_monotonic_increasing or periods.has_duplicates:
        raise ValueError(f'Duplicate/unsorted monthly returns: {path}')
    if len(periods)>1 and not np.all(np.diff(periods.asi8)==1):
        raise ValueError(f'Missing return months: {path}')
    return frame

def read_standard_performance(path,*args,**kwargs):
    """Recompute old cached performance Sharpe from its actual return file when read."""
    frame=pd.read_excel(path,*args,**kwargs)
    if not isinstance(frame,pd.DataFrame): raise ValueError('Select a single performance sheet')
    path=Path(path)
    returns=checked_monthly(path.parent.parent/'period_return/period_return.csv')
    label=frame.columns[0]
    for i,row in frame.iterrows():
        name=str(row[label])
        if name not in returns: continue
        # Excess_* is an information ratio, stored in the compatibility column.
        frame.loc[i,'Sharpe']=standard_sharpe(returns[name],annual_rf=0. if name.startswith('Excess') else .02)
    frame.attrs['SharpeConvention']=CONVENTION
    return frame

def refreshed_baseline(exp):
    """Never compare new Sharpe with the former hard-coded CAGR ratio."""
    exp=Path(exp).resolve()
    roots=[p for p in [exp,*exp.parents] if (p/'src').is_dir() and (p/'data').is_dir()]
    if not roots: raise ValueError('Cannot locate project for original E0 baseline')
    folder=roots[0]/'data/experiments/nn_followup_validation/05_backtests/E0_warm_mse_ensemble3'
    paths=sorted(folder.glob('*/period_return/period_return.csv'))
    if not paths: raise ValueError('Original E0 monthly returns unavailable; cannot reuse legacy Sharpe threshold')
    source=paths[-1];r=checked_monthly(source)
    for candidate in exp.rglob('period_return.csv'):
        other=checked_monthly(candidate)
        if not other.index.equals(r.index): raise ValueError('Baseline/candidate dates differ; rerun matched baseline')
    return {'Sharpe':standard_sharpe(r['Group10'])}
