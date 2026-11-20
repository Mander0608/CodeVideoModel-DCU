"""Resume pinned weights using validated byte ranges; bounded 128 connections."""
import concurrent.futures as cf
import hashlib, json, time, urllib.request as u
from download_cvm import ROOT, OUT, SPEC, PROXY, fetch

def finish(row):
    path=OUT/row['rfilename']; size=row['size']
    marker=path.with_name(path.name+'.verified.json')
    if path.exists() and path.stat().st_size==size and marker.exists(): return
    if size<32*1024**2: return fetch(row)
    part=path.with_name(path.name+'.part'); path.parent.mkdir(parents=True,exist_ok=True)
    chunks=path.with_name(path.name+'.ranges'); chunks.mkdir(exist_ok=True)
    plan=chunks/'plan.json'
    if plan.exists(): start=json.loads(plan.read_text())['start']
    else:
        start=part.stat().st_size if part.exists() else 0
        plan.write_text(json.dumps({'start':start,'size':size}))
    url=f'https://huggingface.co/{SPEC["repo"]}/resolve/{SPEC["revision"]}/{row["rfilename"]}'
    with PROXY.open(u.Request(url,method='HEAD'),timeout=90) as r: target=r.url
    block=32*1024**2
    print('RANGES',row['rfilename'],'remaining_GB',(size-start)/1e9,flush=True)
    def get(offset):
        end=min(offset+block,size)-1
        dst=chunks/f'{offset}-{end}'; tmp=dst.with_suffix('.partial')
        if dst.exists() and dst.stat().st_size==end-offset+1: return dst
        for attempt in range(10):
            try:
                pos=offset+(tmp.stat().st_size if tmp.exists() else 0)
                if pos<=end:
                    opener=u.build_opener(u.ProxyHandler({})) if attempt<4 else PROXY
                    req=u.Request(target,headers={'Range':f'bytes={pos}-{end}'})
                    with opener.open(req,timeout=40) as r:
                        if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {pos}-{end}/'):
                            raise RuntimeError('Bad byte range response')
                        with tmp.open('ab') as f:
                            while b:=r.read(1024**2): f.write(b)
                if tmp.stat().st_size!=end-offset+1: raise RuntimeError('Bad range size')
                tmp.replace(dst); return dst
            except Exception as e:
                print('RANGE_RETRY',row['rfilename'],offset,attempt+1,type(e).__name__,flush=True)
                time.sleep(2)
        raise RuntimeError(f'Range failed {row["rfilename"]} {offset}')
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        completed=list(pool.map(get,range(start,size,block)))
    with part.open('r+b' if part.exists() else 'wb') as f:
        f.truncate(start); f.seek(start)
        for chunk in completed:
            with chunk.open('rb') as src:
                while b:=src.read(8*1024**2): f.write(b)
    if part.stat().st_size!=size: raise RuntimeError('Bad total size')
    h=hashlib.sha256()
    with part.open('rb') as f:
        while b:=f.read(16*1024**2): h.update(b)
    if h.hexdigest()!=SPEC['files_sha256'][row['rfilename']]: raise RuntimeError('SHA256 mismatch')
    part.replace(path)
    marker.write_text(json.dumps({'revision':SPEC['revision'],'bytes':size,'sha256':h.hexdigest()}))
    for chunk in completed: chunk.unlink()
    plan.unlink(); chunks.rmdir()
    print('VERIFIED',row['rfilename'],flush=True)

rows=json.loads((OUT/'download-manifest.json').read_text())
with cf.ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(finish,rows))
from download_h3 import localize_index, verify_index
original=OUT/'modular_model_index.upstream.json'
if not original.exists(): original.write_bytes((OUT/'modular_model_index.json').read_bytes())
localize_index(original,OUT,SPEC); verify_index(OUT,SPEC)
print('REF2VA_DOWNLOAD_COMPLETE_SHA256_VERIFIED',flush=True)
