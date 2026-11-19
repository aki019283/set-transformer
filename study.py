"""Real-data rolling comparison. Never substitutes simulated performance for failures."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from model import Hybrid, canonical_name
from numeric import safe_features, audit_raw_features, rank64_frame
from registration import register
from holding_month import align_dataset, check_label_availability, audit_feature_timing, relabel_daily_derived_months, POLICY

KEYS = ['TradingDate', 'Stkcd']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, p)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def predict(net, x, device, batch):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.as_tensor(x[i:i+batch], device=device)).cpu().numpy()
                               for i in range(0, len(x), batch)])


def fit(net, tr, va, features, target, cfg, update, seed, device, batch):
    from training.batching import BuildPeriodBatches
    seed_all(seed)
    excluded_train=int((~np.isfinite(tr[target])).sum());excluded_valid=int((~np.isfinite(va[target])).sum())
    tr=tr.loc[np.isfinite(tr[target])].reset_index(drop=True)
    va=va.loc[np.isfinite(va[target])].reset_index(drop=True)
    def arrays(d):
        y = d[target].to_numpy(np.float32)
        if not np.isfinite(y).all(): raise ValueError('Nonfinite train/validation labels')
        return safe_features(d[features]), y
    x,y = arrays(tr); vx,vy = arrays(va)
    if not len(y) or not len(vy): raise ValueError('Empty train/validation set')
    lr = cfg['optimizer']['learning_rate'] * (cfg.get('update_learning_rate_scale', .25) if update else 1)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg['optimizer']['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg.get('lr_scheduler_patience',4),
        factor=cfg.get('lr_scheduler_factor',.5), min_lr=cfg.get('min_learning_rate',1e-5))
    best, stale, history, state = math.inf, 0, [], None
    for epoch in range(cfg['update_epochs'] if update else cfg['full_epochs']):
        net.train(); losses=[]
        for ix in BuildPeriodBatches(tr.TradingDate.to_numpy(), cfg.get('periods_per_batch',8)):
            opt.zero_grad(set_to_none=True); total=0.
            # Preserve an optimizer step per original period-batch. Microbatches avoid attention OOM.
            for j in range(0,len(ix),batch):
                sub=ix[j:j+batch]
                pred=net(torch.as_tensor(x[sub],device=device))
                loss=torch.nn.functional.mse_loss(pred,torch.as_tensor(y[sub],device=device))
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                # 按微批实际行数加权，使累积梯度对应整个时期批次的平均损失。
                (loss * (len(sub)/len(ix))).backward(); total+=float(loss.detach())*len(sub)/len(ix)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); losses.append(total)
        score=float(np.mean((predict(net,vx,device,batch)-vy)**2)); sched.step(score)
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),valid_mse=score))
        if np.isfinite(score) and score < best-cfg.get('early_stopping_min_delta',1e-6):
            best=score; state=copy.deepcopy(net.state_dict()); best_epoch=epoch+1; stale=0
        else: stale+=1
        if epoch == 0 or (epoch+1)%5 == 0:
            print(f'  epoch {epoch+1}: validation MSE={score:.6f}',flush=True)
        if stale>=cfg.get('early_stopping_patience',12): break
    if state is None: raise ValueError('No finite validation checkpoint')
    net.load_state_dict(state)
    return dict(best_epoch=best_epoch,validation_mse=best,history=history,train_rows=len(y),valid_rows=len(vy),excluded_missing_train_labels=excluded_train,excluded_missing_valid_labels=excluded_valid)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='2021-01-01',help='First holding month');p.add_argument('--end',default='2026-08-31',help='Last holding month; default latest mature month')
    p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    register(args, Path(__file__).parent)
    alignment_path=Path(__file__).parent/'date_alignment_status.json'
    provenance_path=Path(__file__).parent/'factor_provenance_status.json'
    if not alignment_path.exists() or json.loads(alignment_path.read_text(encoding='utf-8'))['state']!='REBUILD_FROM_PRICES_V3_SOURCE_MONTH_PLUS_ONE':
        raise ValueError('Expected independently tested holding-month label rebuild package V3')
    if not provenance_path.exists() or json.loads(provenance_path.read_text(encoding='utf-8'))['state']!='DAILY_LAST_SAME_MONTH_RELABELED_PLUS_ONE_V1':
        raise ValueError('Expected verified daily-to-monthly factor provenance package')
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for sub in ['report','factors','windows']: (out/sub).mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'src'));os.chdir(root)
    from config import MlpFactorConfig
    from data import PanelDataset
    from windows import WindowBuilder
    cp=root/'config/mlp_warm/old(58)+history-rcykc-multi(45).yaml';cfg=MlpFactorConfig.Load(cp)
    dc=cfg.Section('data');tc=cfg.Section('training');transform=cfg.Section('transforms')
    if transform.get('window') != [{'kind':'impute','params':{'strategy':'constant','value':0}}]: raise ValueError('Unsupported window transform')
    if tc['policy']!='warm_start' or tc['loss']['kind']!='mse': raise ValueError('Expected original warm-start MSE baseline')
    ds=PanelDataset.Load(cfg.DataPath(),time_col=dc['time_col'],asset_col=dc['asset_col'],target_col=dc['target_col'],feature_cols=dc.get('feature_cols'))
    features=list(ds.feature_cols)
    feature_reference = json.loads((Path(__file__).parent/'reference/features.json').read_text(encoding='utf8'))
    if set(map(canonical_name, features)) != set(feature_reference) or len(features) != len(feature_reference):
        raise ValueError('The 103 input factors do not match the FT reference')
    audit_raw_features(ds.ToFrame(),features,out/'report/raw_numeric_audit.json',allow_rank64=True)
    alignment_inputs=[Path(cfg.DataPath()),root/'data/source_data/adj_close.parquet',root/'data/source_data/Calendar.parquet']
    alignment_hashes={str(f):sha(f) for f in alignment_inputs}
    holding_start=pd.Timestamp(args.start).to_period('M')
    source_frame=relabel_daily_derived_months(ds.ToFrame(),dc['target_col'],out/'report')
    raw,schedule,price_cutoff=align_dataset(source_frame,root/'data/source_data',dc['target_col'],out/'report')
    if any(sha(Path(f))!=h for f,h in alignment_hashes.items()):raise ValueError('Inputs changed during label reconstruction')
    holding_end=pd.Timestamp(args.end).to_period('M') if args.end else pd.Period(schedule.holding_month.max(),freq='M')
    if args.smoke: holding_end=holding_start+1
    if str(holding_end) not in set(schedule.holding_month):raise ValueError('Requested holding month has no observed exit prices')
    args.start=str((holding_start-1).start_time.date());args.end=str((holding_end-1).end_time.date())
    save(out/'report/date_policy.json',dict(policy=POLICY,holding_start=str(holding_start),holding_end=str(holding_end),price_cutoff=str(price_cutoff),original_target_used=False,source_month_relabel='+1 month for all 103 daily-derived Ashare_new and rcykc_multi features',missing_training_labels='Excluded explicitly with counts; test predictions retain all stocks, missing held prices still block trading',historical_point_in_time_availability='Daily-to-monthly generator inspected: source month used its last daily observation, so source rows are relabeled to the following holding month. Individual vendor release lags beyond this price-data rule remain unaudited.'))
    timing_review=audit_feature_timing(raw,features,dc['target_col'],args.start,args.end,out/'report')
    save(out/'report/feature_timing_review.json',timing_review)
    if timing_review['review_required']:print('WARNING: feature timing requires source-level review; do not treat high IC/Sharpe as validated alpha.',flush=True)
    ds=PanelDataset.FromFrame(raw,dc['time_col'],dc['asset_col'],dc['target_col'],features)
    raw=ds.ToFrame().copy();frame=rank64_frame(raw,features,dc['target_col'],dc['time_col'],transform)
    for d in [raw,frame]:
        d.TradingDate=pd.to_datetime(d.TradingDate)
        if d.duplicated(KEYS).any():raise ValueError('Duplicate raw keys')
    wc=cfg.Section('window');wc.update(test_start=args.start,test_end=args.end)
    windows=WindowBuilder(wc).Build(PanelDataset.FromFrame(frame,dc['time_col'],dc['asset_col'],dc['target_col'],features))
    if args.smoke: windows=windows[:2];tc=dict(tc,full_epochs=2,update_epochs=2);args.seeds=[42]
    if not windows:raise ValueError('No windows')
    expected=pd.period_range(args.start,args.end,freq='M')
    if not args.smoke and [pd.Timestamp(w.test_periods[0]).to_period('M') for w in windows]!=list(expected):raise ValueError('Missing expected test months')
    for w in windows:
        if pd.Timestamp(max(w.valid_periods))+pd.offsets.MonthEnd(1)>=pd.Timestamp(min(w.test_periods)):raise ValueError('Insufficient validation/test purge')
        if pd.Timestamp(max(w.train_periods))+pd.offsets.MonthEnd(1)>=pd.Timestamp(min(w.valid_periods)):raise ValueError('Insufficient training/validation purge')
    check_label_availability(windows,schedule).to_csv(out/'report/label_availability.csv',index=False)
    price_path=root/'data/source_data/adj_close.parquet'
    dates=pd.read_parquet(price_path,columns=['TradingDate']).TradingDate
    if schedule.loc[schedule.holding_month.eq(str(holding_end)),'exit'].iloc[0]>dates.max():raise ValueError('Requested holding-period exit not observed')
    market = [root/'data/source_data'/f for f in ['Calendar.parquet','Suspension_Limit.parquet','history-new.pq','history-1800.pq']]
    package = Path(__file__).resolve().parent
    hashes = {str(f):sha(f) for f in [cp, Path(cfg.DataPath()), price_path, *market,
        *sorted((root/'src').rglob('*.py')), *sorted(package.rglob('*.py')), *sorted((package/'reference').glob('*'))]}
    archived_panel_sha = json.loads((package/"protocol.json").read_text(encoding="utf8"))["ft_panel_sha256"]
    save(out/"report/panel_comparison.json", dict(current=hashes[str(Path(cfg.DataPath()))],
         archived_ft=archived_panel_sha, identical=hashes[str(Path(cfg.DataPath()))]==archived_panel_sha))
    ident=dict(hashes=hashes,args=vars(args),features=features,training=tc,runtime={'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},windows=[dict(train=list(w.train_periods),valid=list(w.valid_periods),test=list(w.test_periods)) for w in windows])
    # Normalize Paths to strings before equality checks.
    ident=json.loads(json.dumps(ident,default=str))
    identity=out/'identity.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))!=ident:raise ValueError('Resume identity mismatch; new output directory required')
    save(identity,ident);save(out/'report/status.json',dict(state='TRAINING',smoke=args.smoke))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    seed = args.seeds[0]
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Hybrid(features).to(device)
    name = f"dcn_set_s{seed}"
    predictions = []
    seen_gap = False
    print(f"{name}: {device}, {len(windows)} windows, 8 factor-family tokens", flush=True)
    for k, w in enumerate(windows):
        dest = out/"windows"/name/w.window_id
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest/"COMMITTED.json"
        files = ["model.pt", "prediction.parquet", "diagnostics.json"]
        if marker.exists():
            if seen_gap:
                raise ValueError("Committed window after a gap in the warm-start chain")
            record = json.loads(marker.read_text())
            if record != {f: sha(dest/f) for f in files}:
                raise ValueError(f"Invalid checkpoint in {dest}")
            net.load_state_dict(torch.load(dest/"model.pt", map_location=device, weights_only=True))
            pr = pd.read_parquet(dest/"prediction.parquet")
        else:
            if any((out/"windows"/name/later.window_id/"COMMITTED.json").exists() for later in windows[k+1:]):
                raise ValueError("Broken warm-start checkpoint sequence")
            seen_gap = True
            print(f"Training {name} {k+1}/{len(windows)} ({w.test_periods[0]})", flush=True)
            tr = frame.loc[frame.TradingDate.isin(w.train_periods)]
            va = frame.loc[frame.TradingDate.isin(w.valid_periods)]
            te = frame.loc[frame.TradingDate.isin(w.test_periods)]
            start = time.monotonic()
            diag = fit(net, tr, va, features, dc["target_col"], tc, k>0, seed+k, device, args.batch)
            pr = te[KEYS].copy()
            pr[name] = predict(net, safe_features(te[features]), device, args.batch)
            if not np.isfinite(pr[name]).all():
                raise ValueError("Nonfinite predictions")
            diag.update(seconds=time.monotonic()-start, device=device)
            save(dest/"diagnostics.json", diag)
            torch.save(net.state_dict(), dest/"model.pt")
            pr.to_parquet(dest/"prediction.parquet", index=False)
            save(marker, {f: sha(dest/f) for f in files})
        predictions.append(pr)
        print(f"{name}: {k+1}/{len(windows)} windows complete", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    expected_keys = raw.loc[raw.TradingDate.isin([w.test_periods[0] for w in windows]), KEYS]
    if result.duplicated(KEYS).any() or set(map(tuple,result[KEYS].values)) != set(map(tuple,expected_keys.values)):
        raise ValueError("Prediction coverage differs from test universe")
    # 这里导出的是实验信号日期；对应持有月需按 holding_month_schedule.csv 解释。
    result.to_parquet(out/"factors"/(name+".parquet"), index=False)
    labeled = result.merge(raw[KEYS+[dc["target_col"]]], on=KEYS, validate="one_to_one")
    rows = []
    for date, group in labeled.groupby("TradingDate"):
        y = group[dc["target_col"]]
        observed = np.isfinite(y)
        scores = group.loc[observed,name]
        rows.append(dict(signal_date=date, holding_month=str(date.to_period("M")+1),
                         observed=int(observed.sum()), missing=int((~observed).sum()),
                         IC=scores.corr(y[observed]), RankIC=scores.corr(y[observed], method="spearman")))
    pd.DataFrame(rows).to_csv(out/"report/monthly_ic.csv", index=False)
    if any(sha(Path(f)) != digest for f,digest in hashes.items()):
        raise ValueError("Inputs changed during training")
    save(out/"report/status.json",dict(state="SMOKE_ONLY" if args.smoke else "PREDICTIONS_COMPLETE",
                                       windows=len(windows), timing_review=timing_review["review_required"]))


if __name__ == "__main__":
    main()
