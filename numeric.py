import json
import numpy as np

def audit_raw_features(frame, features, destination, allow_rank64=False):
    """Audit before project sanitize can cast overflow to inf and erase it."""
    records=[]
    for column in features:
        values=frame[column].to_numpy(dtype=np.float64)
        bad=np.isinf(values) | (np.isfinite(values) & (np.abs(values)>np.finfo(np.float32).max))
        ix=np.flatnonzero(bad)
        samples=[]
        for i in ix[:5]:
            row=frame.iloc[int(i)]
            samples.append(dict(TradingDate=str(row.get('TradingDate','')),Stkcd=str(row.get('Stkcd','')),value=str(values[i])))
        records.append(dict(feature=column,rows=len(values),missing=int(np.isnan(values).sum()),
            infinite=int(np.isinf(values).sum()),finite_float32_overflow=int((np.isfinite(values)&bad).sum()),samples=samples))
    failed=any(r['infinite'] or (r['finite_float32_overflow'] and not allow_rank64) for r in records)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(dict(state='BLOCKED_RAW_NUMERIC' if failed else 'PASS',finite_large_value_policy='Rank in float64 before float32 conversion' if allow_rank64 else 'Block',columns=records),ensure_ascii=False,indent=2),encoding='utf8')
    if failed:raise ValueError('Raw factor infinity/float32 overflow BEFORE sanitize. No values replaced. Read '+str(destination))

def rank64_frame(frame, features, target, time_col, transform):
    expected=[{'kind':'sanitize'},{'kind':'cross_section_rank','params':{'output_range':[-1.0,1.0],'rank_target':True}}]
    if transform.get('dataset')!=expected:
        raise ValueError('rank64 adapter requires the exact registered sanitize/rank config')
    result=frame.copy()
    for column in [*features,target]:
        values=frame[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():raise ValueError('Infinite raw value in '+column)
    # Rank finite raw magnitudes without clipping or narrowing to float32 first.
    # Preserve original average-tie ranks, missing values, target ranks, and scaling order.
    ranks=frame.groupby(time_col,sort=False)[features].rank(pct=True,method='average')
    values=ranks.to_numpy(dtype=np.float32)
    result[features]=-1.0+values*2.0
    targets=frame.groupby(time_col,sort=False)[target].rank(pct=True,method='average')
    result[target]=-1.0+targets.to_numpy(dtype=np.float32)*2.0
    return result

def safe_features(frame):
    a=frame.to_numpy(dtype=np.float64)
    if np.isinf(a).any() or (np.abs(a[np.isfinite(a)])>np.finfo(np.float32).max).any():
        raise ValueError('Infinite/float32-overflow features: audit source and transforms before training')
    # Original pipeline explicitly specifies constant-zero missing-value imputation.
    return np.nan_to_num(a,nan=0.).astype(np.float32)
