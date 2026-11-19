"""Isolated, paired exploratory experiments. Never writes production artifacts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

KEYS = ['TradingDate', 'Stkcd']
DEFINITIONS = {
    'M0': '新训练公平基准：原残差MLP，Rank收益标签，MSE',
    'H1': '同M0，仅改为Huber delta=0.25（乘2保持局部曲率）',
    'H2': '同M0，仅改为Huber delta=0.50（乘2保持局部曲率）',
    'F50': '同M0，仅保留每个训练窗口稳定性前50%特征，其余置零',
    'F70': '同M0，仅保留每个训练窗口稳定性前70%特征，其余置零',
    'L0': '标签实验对照：MSE原Rank标签，仅完整风险暴露样本',
    'L1': '同L0，仅将Rank标签对行业/对数市值/Beta回归取残差',
    'T0': '已有E0三种子集成，未经中性化（不重新训练）',
    'T1': 'E0预测值按月行业中性化',
    'T2': 'E0预测值按月对数市值中性化',
    'T3': 'E0预测值按月行业+对数市值联合中性化',
    'T4': 'E0预测值按月行业+对数市值+Beta联合中性化',
}

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def keyed(frame):
    out = frame.reset_index() if not set(KEYS).issubset(frame.columns) else frame.copy()
    out['TradingDate'] = pd.to_datetime(out['TradingDate']).dt.to_period('M').dt.to_timestamp('M')
    out['Stkcd'] = pd.to_numeric(out['Stkcd'], errors='raise').astype('int64')
    if out[KEYS].isna().any().any() or out.duplicated(KEYS).any():
        raise ValueError('Null or duplicate stock-month keys')
    return out.sort_values(KEYS).reset_index(drop=True)

def residual(frame, value, controls):
    """Contemporaneous joint OLS; unknown exposures stay missing, never zero-filled."""
    result = pd.Series(np.nan, index=frame.index)
    for _, part in frame.groupby('TradingDate', sort=True):
        g = part.replace([np.inf, -np.inf], np.nan).dropna(subset=[value] + controls)
        blocks = [np.ones((len(g), 1))]
        for c in controls:
            if c == 'industry':
                if g[c].nunique() < 2: break
                blocks.append(pd.get_dummies(g[c].astype(str), drop_first=True, dtype=float).to_numpy())
            else:
                z = g[c].to_numpy(float); sd = z.std()
                if sd > 1e-12: blocks.append(((z - z.mean()) / sd)[:, None])
        else:
            x = np.concatenate(blocks, axis=1)
            if len(g) >= max(30, 2 * x.shape[1]):
                y = g[value].to_numpy(float)
                result.loc[g.index] = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return result

def select_features(frame, features, target, fraction):
    """Training-only mean IC / volatility and sign consistency, stable tie-breaks."""
    values = []
    for _, g in frame.groupby('TradingDate'):
        with np.errstate(invalid='ignore', divide='ignore'):
            values.append(g[features].corrwith(g[target]).fillna(0).to_numpy())
    a = np.asarray(values); mean = a.mean(axis=0)
    score = np.abs(mean) / (a.std(axis=0) + .02) * np.sqrt((np.sign(a) == np.sign(mean)).mean(axis=0))
    count = max(1, math.ceil(len(features) * fraction))
    indices = np.argsort(-score, kind='stable')[:count]
    mask = np.zeros(len(features), dtype=np.float32); mask[indices] = 1
    return mask, score

def backtest_sources(root):
    return {
        'adj_close': root / 'data/source_data/adj_close.parquet',
        'suspension_limit': root / 'data/source_data/Suspension_Limit.parquet',
        'filter_stock_pool': root / 'data/experiments/nn_followup_validation/02_aligned_pools/history-new-month-aligned.parquet',
        'benchmark_stock_pool': root / 'data/experiments/nn_followup_validation/02_aligned_pools/history-1800-month-aligned.parquet',
        'market_value': root / 'data/source_data/mv_turn.parquet',
        'index_quotes': root / 'data/source_data/指数日行情.parquet',
    }

def prepare(args, exp):
    import torch
    import legacy_metrics  # Fail early if the cloud reporting dependencies are absent.
    root = args.project_root
    sys.path.insert(0, str(root / 'src'))
    from config import MlpFactorConfig
    from data import PanelDataset
    from transforms import BuildDatasetPipeline
    from windows import WindowBuilder
    config_path = root / 'config/mlp_warm/old(58)+history-rcykc-multi(45).yaml'
    cfg = MlpFactorConfig.Load(config_path)
    d = cfg.Section('data')
    if [d['time_col'], d['asset_col']] != KEYS: raise ValueError('Unexpected key names')
    if cfg.Section('transforms').get('window') != [{'kind': 'impute', 'params': {'strategy': 'constant', 'value': 0}}]:
        raise ValueError('This runner supports only the verified constant-zero window transform')
    dataset = PanelDataset.Load(cfg.DataPath(), time_col=d['time_col'], asset_col=d['asset_col'], target_col=d['target_col'], feature_cols=d.get('feature_cols'))
    raw = keyed(dataset.ToFrame()); features = list(dataset.feature_cols)
    ranked = BuildDatasetPipeline(cfg.Section('transforms')).ApplyDataset(dataset)
    frame = keyed(ranked.ToFrame())
    wc = cfg.Section('window'); wc.update(test_start=args.start, test_end=args.end)
    windows = WindowBuilder(wc).Build(ranked)
    if not windows: raise ValueError('No test windows in requested range')
    for w in windows:
        if max(w.valid_periods) + pd.offsets.MonthEnd(1) >= min(w.test_periods):
            raise ValueError('Insufficient forward-label purge')
    e0_path = root / 'data/experiments/nn_followup_validation/04_factors/E0_warm_mse_ensemble3.parquet'
    sources = backtest_sources(root)
    required = [config_path, cfg.DataPath(), e0_path, *sources.values(), root / 'run/run_backtest.py']
    for path in required:
        if not path.is_file(): raise FileNotFoundError(f'Required input missing: {path}')
    mv = pd.read_parquet(sources['market_value'])
    mv = mv.reset_index() if not set(KEYS).issubset(mv.columns) else mv
    mv['TradingDate'] = pd.to_datetime(mv['TradingDate'])
    mv = mv.sort_values('TradingDate'); mv['TradingDate'] = mv['TradingDate'].dt.to_period('M').dt.to_timestamp('M')
    mv = keyed(mv.drop_duplicates(KEYS, keep='last'))
    mv['logsize'] = np.log(pd.to_numeric(mv['MarketValue'], errors='coerce').where(lambda x: x > 0))
    risk = raw[KEYS].copy()
    risk['risk_beta'] = pd.to_numeric(raw['Beta'], errors='coerce') if 'Beta' in raw else np.nan
    risk = risk.merge(mv[KEYS + ['logsize']], on=KEYS, how='left', validate='one_to_one')
    aliases = ['Industry','industry','IndustryCode','industry_code','SW1','sw1','行业','行业代码']
    candidates = [Path(args.industry_file)] if args.industry_file else [cfg.DataPath(), root/'data/source_data/industry_monthly.parquet', root/'data/source_data/industry.parquet', root/'data/source_data/merge_daily_info.parquet']
    industry_path = None; warnings = []
    for path in candidates:
        if not path.exists(): continue
        try:
            import pyarrow.parquet as pq
            columns = pq.read_schema(path).names
            col = next((c for c in aliases if c in columns), None)
            if not col: continue
            ind = keyed(pd.read_parquet(path))[KEYS + [col]].rename(columns={col:'industry'})
            risk = risk.merge(ind, on=KEYS, how='left', validate='one_to_one'); industry_path = path; break
        except (ValueError, KeyError) as exc: warnings.append(f'{path}: {exc}')
    if 'industry' not in risk: risk['industry'] = np.nan
    risk = risk.replace([np.inf,-np.inf], np.nan)
    coverage = risk.assign(complete=risk[['industry','logsize','risk_beta']].notna().all(axis=1)).groupby('TradingDate')['complete'].mean()
    relevant = sorted(set(p for w in windows for p in (*w.train_periods,*w.valid_periods,*w.test_periods)))
    industry_ok = bool(coverage.reindex(relevant).fillna(0).ge(.95).all())
    blocked = [] if industry_ok else ['T1/T3/T4/L0/L1: dated industry+size+Beta completeness below 95% in at least one required month']
    # Alias exposure columns to avoid overwriting ranked input features.
    # Industry can already exist in the input; risk columns must not alter feature columns.
    if set(['industry','logsize','risk_beta']).intersection(features):
        raise ValueError('Risk helper names collide with model feature names')
    frame = frame.drop(columns=['industry','logsize','risk_beta'],errors='ignore').merge(risk, on=KEYS, how='left', validate='one_to_one')
    if industry_ok: frame['residual_y'] = residual(frame, d['target_col'], ['industry','logsize','risk_beta'])
    code_paths = [Path(__file__), Path(__file__).with_name('legacy_metrics.py')]
    code_paths += sorted((root/'src').rglob('*.py'))
    if industry_path: required.append(industry_path)
    print('Fingerprinting input files for safe resume...',flush=True)
    identity = {'start':args.start, 'end':args.end, 'seed':args.seed,
                'runtime':{'torch':torch.__version__,'pandas':pd.__version__,'numpy':np.__version__},
                'files':{str(p):sha(p) for p in dict.fromkeys(required + code_paths)}, 'features':features}
    identity_path = exp/'01_audit/identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Inputs/code changed. Use a NEW -StudyName; old results are preserved.')
    save_json(identity_path, identity)
    plan = {'windows':len(windows), 'test_months':[str(w.test_periods[0]) for w in windows],
            'features':len(features), 'industry_available':industry_ok, 'blocked':blocked,
            'warnings':warnings, 'seed':args.seed, 'training':cfg.Section('training'),
            'tuning_enabled':False, 'exploratory_not_untouched_holdout':True}
    save_json(exp/'01_audit/plan.json', plan)
    coverage.to_csv(exp/'01_audit/exposure_coverage.csv')
    print(json.dumps(plan, ensure_ascii=False, default=str), flush=True)
    return cfg, frame, features, windows, risk, e0_path, plan

def predict(net, x, device):
    import torch
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.as_tensor(x[i:i+8192], device=device)).cpu().numpy() for i in range(0,len(x),8192)])

def fit(net, train, valid, features, target, mask, config, update, delta, seed, device):
    import torch
    from training.batching import BuildPeriodBatches
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    def arrays(g):
        g = g.loc[np.isfinite(pd.to_numeric(g[target], errors='coerce'))]
        x = g[features].to_numpy(np.float32); x = np.nan_to_num(x,nan=0,posinf=0,neginf=0) * mask
        return g, x, g[target].to_numpy(np.float32)
    tr, x, y = arrays(train); va, vx, vy = arrays(valid)
    if not len(tr) or not len(va): raise ValueError('Empty training/validation labels')
    optimizer = config['optimizer']
    lr = float(optimizer['learning_rate']) * (float(config.get('update_learning_rate_scale',.25)) if update else 1)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(optimizer['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=int(config.get('lr_scheduler_patience',4)), factor=float(config.get('lr_scheduler_factor',.5)), min_lr=float(config.get('min_learning_rate',1e-5)))
    best = math.inf; state = None; stale = 0
    epochs = int(config['update_epochs'] if update else config['full_epochs'])
    for epoch in range(epochs):
        net.train()
        for ix in BuildPeriodBatches(tr['TradingDate'].to_numpy(), int(config.get('periods_per_batch',8))):
            tx = torch.as_tensor(x[ix],device=device); ty = torch.as_tensor(y[ix],device=device)
            opt.zero_grad(set_to_none=True); pred = net(tx)
            loss = 2 * torch.nn.functional.huber_loss(pred,ty,delta=delta) if delta else torch.nn.functional.mse_loss(pred,ty)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        vp = predict(net,vx,device); score = float(np.mean((vp-vy)**2))
        scheduler.step(score)
        if score < best - float(config.get('early_stopping_min_delta',1e-6)):
            best = score; state = copy.deepcopy(net.state_dict()); stale = 0
        else: stale += 1
        if stale >= int(config.get('early_stopping_patience',12)): break
    if state is None: raise ValueError('No finite validation checkpoint')
    net.load_state_dict(state)
    errors = predict(net,vx,device)-vy
    return {'epochs':epoch+1,'validation_mse':best,'train_rows':len(tr),'valid_rows':len(va),
            'huber_linear_fraction':float((np.abs(errors)>delta).mean()) if delta else None}

def generate(args, exp, context):
    import torch
    from models.mlp import _ResidualMlpNetwork
    completed = exp/'01_audit/generated.json'
    if completed.exists():
        for path,digest in read_json(completed)['factor_hashes'].items():
            if sha(path)!=digest: raise ValueError('Generated factor was modified: '+path)
        print('Generation already committed; skip.',flush=True)
        return
    cfg,frame,features,windows,risk,e0_path,plan = context
    names = ['M0','H1','H2','F50','F70'] + (['L0','L1'] if plan['industry_available'] else [])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for name in names:
        torch.manual_seed(args.seed)
        net = _ResidualMlpNetwork(len(features), float(cfg.Section('model')['params']['dropout'])).to(device)
        parts = []
        for k,w in enumerate(windows):
            dest = exp/'02_windows'/name/w.window_id
            if (dest/'COMMITTED.json').exists():
                net.load_state_dict(torch.load(dest/'model.pt', map_location=device, weights_only=True))
                parts.append(pd.read_parquet(dest/'factor.parquet')); continue
            train = frame.loc[frame.TradingDate.isin(w.train_periods)].copy()
            valid = frame.loc[frame.TradingDate.isin(w.valid_periods)].copy()
            test = frame.loc[frame.TradingDate.isin(w.test_periods)].copy()
            target = cfg.Section('data')['target_col']
            if name.startswith('L'):
                train = train.loc[train.residual_y.notna()]; valid = valid.loc[valid.residual_y.notna()]
                test = test.dropna(subset=['industry','logsize','risk_beta'])
                if name == 'L1': target = 'residual_y'
            mask = np.ones(len(features),np.float32)
            if name.startswith('F'): mask,_ = select_features(train,features,target,int(name[1:])/100)
            diag = fit(net,train,valid,features,target,mask,cfg.Section('training'),k>0,
                       {'H1':.25,'H2':.5}.get(name),args.seed+k,device)
            if test.empty: raise ValueError(f'No inference rows: {name}/{w.window_id}')
            x = np.nan_to_num(test[features].to_numpy(np.float32),nan=0,posinf=0,neginf=0)*mask
            out = test[KEYS].copy(); out[name] = predict(net,x,device)
            if not np.isfinite(out[name]).all(): raise ValueError('Invalid predictions')
            diag.update(name=name,window=w.window_id,test_month=str(w.test_periods[0]),
                        selected_features=[f for f,m in zip(features,mask) if m], device=device)
            temp = dest.with_name(dest.name+'_'+uuid4().hex); temp.mkdir(parents=True)
            torch.save(net.state_dict(),temp/'model.pt'); out.to_parquet(temp/'factor.parquet',index=False)
            save_json(temp/'COMMITTED.json',diag); os.replace(temp,dest)
            parts.append(out); print(f'{name} [{k+1}/{len(windows)}] committed {w.window_id}',flush=True)
        keyed(pd.concat(parts)).to_parquet(exp/'03_raw_factors'/f'{name}.parquet',index=False)
    e0 = keyed(pd.read_parquet(e0_path)); cols = [c for c in e0 if c not in KEYS]
    if len(cols)!=1: raise ValueError('E0 must have exactly one prediction column')
    e0 = e0.rename(columns={cols[0]:'T0'})
    e0 = e0.loc[e0.TradingDate.between(pd.Timestamp(args.start),pd.Timestamp(args.end))]
    e0 = e0.merge(risk,on=KEYS,how='left',validate='one_to_one')
    variants = {'T2':['logsize']}
    if plan['industry_available']: variants.update(T1=['industry'],T3=['industry','logsize'],T4=['industry','logsize','risk_beta'])
    for name,controls in variants.items(): e0[name] = residual(e0,'T0',controls)
    for name in ['T0',*variants]: e0[KEYS+[name]].to_parquet(exp/'03_raw_factors'/f'{name}.parquet',index=False)
    families = {'training':['M0','H1','H2','F50','F70'],'size':['T0','T2']}
    if plan['industry_available']: families.update(neutralization=['T0','T1','T2','T3','T4'],label=['L0','L1'])
    coverage = []
    for family,members in families.items():
        merged = None; counts = {}
        for name in members:
            f = keyed(pd.read_parquet(exp/'03_raw_factors'/f'{name}.parquet')); counts[name]=len(f)
            merged = f if merged is None else merged.merge(f,on=KEYS,how='inner',validate='one_to_one')
        merged = merged.replace([np.inf,-np.inf],np.nan).dropna(subset=members)
        if merged.empty: raise ValueError(f'Empty common coverage: {family}')
        expected = {p for w in windows for p in w.test_periods}
        if set(pd.to_datetime(merged.TradingDate.unique())) != expected: raise ValueError(f'Missing common test months: {family}')
        folder = exp/'04_factors'/family; folder.mkdir(parents=True,exist_ok=True)
        for name in members:
            merged[KEYS+[name]].to_parquet(folder/f'{name}.parquet',index=False)
            coverage.append({'family':family,'name':name,'raw_rows':counts[name],'paired_rows':len(merged),'months':merged.TradingDate.nunique()})
    pd.DataFrame(coverage).to_csv(exp/'01_audit/factor_coverage.csv',index=False)
    save_json(exp/'01_audit/generated.json',{'families':families,'definitions':DEFINITIONS,
              'factor_hashes':{str(p):sha(p) for p in (exp/'04_factors').rglob('*.parquet')}})

def backtest(args, exp):
    from legacy_metrics import run, metrics
    audit = read_json(exp/'01_audit/generated.json')
    for path,digest in audit['factor_hashes'].items():
        if sha(path)!=digest: raise ValueError('Generated factors modified: '+path)
    for family,names in audit['families'].items():
        for name in names:
            folder = exp/'05_backtests'/family
            marker = folder/(name+'.done.json')
            if marker.exists(): metrics(folder/name); continue
            config = {'data':{'source_data':{k:str(v) for k,v in backtest_sources(args.project_root).items()}},
              'factors':[str(exp/'04_factors'/family/(name+'.parquet'))], 'output':{'output_dir':str(folder)},
              'backtest':{'period':{'start_date':args.start[:7],'end_date':args.end[:7]},'frequency':'monthly','groups':10,
                'neutralize':None,'risk_free_rate':.02,'transaction_cost':{'stamp_tax':.001,'commission_buy':.0001,'commission_sell':.0001,'transfer_buy':.00001,'transfer_sell':.00001}},
              'benchmark':[{'mode':'pool','weights':'equal'}], 'log':{'log_file':str(exp/'06_logs'/f'{family}_{name}.log'),'level':'INFO'}}
            path = exp/'00_tools'/f'backtest_{family}_{name}.yaml'; save_json(path,config)
            run([sys.executable,'-u',str(args.project_root/'run/run_backtest.py'),'--config',str(path)],args.project_root,exp/'06_logs'/f'{family}_{name}_console.log')
            row,_ = metrics(folder/name); save_json(marker,row)

def summarize(exp):
    from legacy_metrics import metrics
    from openpyxl.styles import Font, PatternFill, Alignment
    audit = read_json(exp/'01_audit/generated.json'); plan = read_json(exp/'01_audit/plan.json')
    rows=[]; series=[]; checks=[]; yearly=[]
    for family,names in audit['families'].items():
        group=[]; dates=None
        for name in names:
            row,s = metrics(exp/'05_backtests'/family/name); row.update(family=family,method=DEFINITIONS[name]); group.append(row)
            s.insert(0,'family',family); series.append(s)
            r = pd.read_csv(Path(row['ResultDirectory'])/'period_return/period_return.csv',index_col=0,parse_dates=True)['Group10'].dropna()
            if dates is None: dates=r.index
            if not r.index.equals(dates): raise ValueError('Unmatched backtest dates: '+family)
            for year,g in r.groupby(r.index.year):
                yearly.append({'family':family,'factor':name,'year':year,'months':len(g),'compound_return':float((1+g).prod()-1),
                   'annualized_sharpe':float((g.mean()*12-(12 * ((1.02)**(1/12)-1)))/(g.std()*np.sqrt(12))) if len(g)>1 and g.std()>0 else None})
        base=group[0]
        for row in group:
            for metric in ['Sharpe','IC','RankIC','MaxDrawdown']: row['Delta_'+metric]=row[metric]-base[metric]
            row['reference']=names[0]
            row['status']='BASE' if row['factor']==names[0] else 'Exploratory; no automatic promotion'
        rows.extend(group); checks.append({'check':family+' same return months','passed':True})
    results=pd.DataFrame(rows)
    checks.append({'check':'finite core metrics','passed':bool(np.isfinite(results[['Sharpe','IC','RankIC','MaxDrawdown']].to_numpy(float)).all())})
    diagnostics=[read_json(p) for p in (exp/'02_windows').rglob('COMMITTED.json')]
    for d in diagnostics: d['selected_features']=', '.join(d['selected_features'])
    summary=pd.DataFrame([
        ['实验性质','独立探索性筛选；已经反复观察过该区间，不是未触碰的最终样本外验证。'],
        ['比较口径','仅比较同family：训练组对M0，标签组对L0，中性化组对T0。不跨组排名。'],
        ['收益口径','原回测程序Group10最高分十分位组合；不是多空Sharpe。采用固定手续费；未充分建模冲击与容量。'],
        ['未运行组','; '.join(plan['blocked']) or '无'],
        ['标签解释','L1预测的是Rank收益的风险残差，不是原始收益率百分比。'],
        ['特征选择','每个窗口仅使用训练期IC稳定性，保持输入维度与顺序，未选列置零。'],
        ['下一步','先检查覆盖率/年度表现/风险暴露，再对稳定候选做三种子与新增月份前瞻验证；不自动替换E0。'],
        ['正式文件','未更改生产模型、风格轮动和策略汇总文件。'],
    ],columns=['项目','说明'])
    output=exp/'07_report/NN_interpretable_screen.xlsx'
    temp=output.with_name('NN_interpretable_screen.tmp.xlsx')
    with pd.ExcelWriter(temp,engine='openpyxl') as writer:
        sheets={'汇报说明':summary,'方案定义':pd.DataFrame(DEFINITIONS.items(),columns=['代号','方案']),
          '同口径对比':results,'分年度表现':pd.DataFrame(yearly),'Monthly IC':pd.concat(series,ignore_index=True),
          '样本覆盖':pd.read_csv(exp/'01_audit/factor_coverage.csv'),'窗口诊断':pd.DataFrame(diagnostics),'交叉检查':pd.DataFrame(checks)}
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]: cell.font=Font(color='FFFFFF',bold=True); cell.fill=PatternFill('solid',fgColor='17365D')
            for col in ws.columns:
                letter=col[0].column_letter; ws.column_dimensions[letter].width=min(65,max(14,max(len(str(c.value or '')) for c in col[:100])+2))
                for c in col[1:]:
                    if isinstance(c.value,float): c.number_format='0.0000'
                    c.alignment=Alignment(vertical='top',wrap_text=True)
    os.replace(temp,output)
    results.to_csv(exp/'07_report/comparison.csv',index=False,encoding='utf-8-sig')
    save_json(exp/'07_report/cross_checks.json',checks)
    print('Completed. Report: '+str(output),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--mode',choices=['Plan','All','Generate','Backtest','Summarize'],default='All')
    p.add_argument('--study',default='screen_seed42'); p.add_argument('--start',default='2024-01-01'); p.add_argument('--end',default='2026-08-31')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--industry-file',default='')
    args=p.parse_args(); args.project_root=args.project_root.resolve()
    if not args.study.replace('_','').replace('-','').isascii() or not args.study.replace('_','').replace('-','').isalnum(): raise ValueError('Invalid study name')
    exp=args.project_root/'data/experiments/nn_interpretable_oct'/args.study
    for directory in ['00_tools','01_audit','02_windows','03_raw_factors','04_factors','05_backtests','06_logs','07_report']:
        (exp/directory).mkdir(parents=True,exist_ok=True)
    lock=exp/'RUNNING.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Another run or interrupted run lock: {lock}. Verify no process remains before removing this single lock.')
    os.close(fd)
    try:
        context=prepare(args,exp)
        if args.mode in ['All','Generate']: generate(args,exp,context)
        if args.mode in ['All','Backtest']: backtest(args,exp)
        if args.mode in ['All','Summarize']: summarize(exp)
    finally: lock.unlink()

if __name__=='__main__': main()
