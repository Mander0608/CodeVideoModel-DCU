import json, time
from pathlib import Path
root=Path(__file__).resolve().parent
model=root/'models/MiniMax-H3-ref2va'
manifest=model/'download-manifest.json'
rows=json.loads(manifest.read_text()) if manifest.exists() else []
received=verified=total=0; by={}
for row in rows:
    name=row['rfilename']; n=row['size']; total+=n; p=model/name
    marker=p.with_name(p.name+'.verified.json')
    if name=='modular_model_index.json' and (model/'modular_model_index.upstream.json').exists():
        p=model/'modular_model_index.upstream.json'
    if p.exists() and p.stat().st_size==n:
        amount=n
        if marker.exists(): verified+=n
    else:
        ranges=p.with_name(p.name+'.ranges'); part=p.with_name(p.name+'.part')
        if (ranges/'plan.json').exists():
            amount=json.loads((ranges/'plan.json').read_text())['start']
            amount+=sum(x.stat().st_size for x in ranges.iterdir() if x.name!='plan.json')
        else: amount=part.stat().st_size if part.exists() else 0
    amount=min(amount,n); received+=amount
    component=name.split('/')[0]; b=by.setdefault(component,{'received_GB':0,'total_GB':0})
    b['received_GB']+=amount/1e9; b['total_GB']+=n/1e9
print(json.dumps({'epoch':time.time(),'received_GB':received/1e9,'verified_GB':verified/1e9,
    'total_GB':total/1e9,'percent':received/total*100 if total else 0,
    'components':{k:{a:round(b,2) for a,b in v.items()} for k,v in by.items() if v['total_GB']>0.1}},indent=2))
