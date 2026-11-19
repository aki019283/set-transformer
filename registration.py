import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

def register(args, package):
    if args.start!='2021-01-01' or args.end!='2026-08-31' or args.seeds!=[42] or args.batch!=256:
        raise ValueError('Frozen protocol arguments changed; register a separate experiment')
    manifest={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
        and p.suffix in {'.py','.json','.yaml','.ps1'}}
    root=args.project_root
    config=root/'config/mlp_warm/old(58)+history-rcykc-multi(45).yaml'
    manifest['PROJECT_CONFIG']=hashlib.sha256(config.read_bytes()).hexdigest()
    for p in sorted((root/'src').rglob('*.py')):
        manifest['PROJECT_SRC/'+str(p.relative_to(root/'src'))]=hashlib.sha256(p.read_bytes()).hexdigest()
    payload=dict(manifest=manifest,smoke=args.smoke,protocol=json.loads((package/'protocol.json').read_text(encoding='utf8')))
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/'preregistration.json'
    if dest.exists():
        previous=json.loads(dest.read_text(encoding='utf8'));previous.pop('registered_utc')
        if previous!=payload: raise ValueError('Preregistration changed; refusing resume')
    else:
        payload['registered_utc']=datetime.now(timezone.utc).isoformat()
        with dest.open('x',encoding='utf8') as f: json.dump(payload,f,ensure_ascii=False,indent=2)
