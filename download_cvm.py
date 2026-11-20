"""Pinned minimal ref2va download. Omits unused text-to-video transformer weights."""
import concurrent.futures as cf
import hashlib, json, time, urllib.request as u
from pathlib import Path
ROOT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
OUT=ROOT/'models/MiniMax-H3-ref2va'
SPEC=json.loads((ROOT/'configs/h3_checkpoint.json').read_text())
PROXY=u.build_opener(u.ProxyHandler({'https':'http://127.0.0.1:17997'}))
DIRECT=u.build_opener(u.ProxyHandler({}))

def fetch(row):
    name=row['rfilename']; size=row['size']; expected=SPEC['files_sha256'].get(name)
    path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    marker=path.with_name(path.name+'.verified.json')
    if path.exists() and path.stat().st_size==size and marker.exists(): return
    part=path.with_name(path.name+'.part')
    url=f'https://huggingface.co/{SPEC["repo"]}/resolve/{SPEC["revision"]}/{name}'
    for attempt in range(8):
        try:
            start=part.stat().st_size if part.exists() else 0
            if start<size:
                target=url
                if size>1024*1024:
                    with PROXY.open(u.Request(url,method='HEAD'),timeout=90) as r: target=r.url
                opener=DIRECT if size>1024*1024 else PROXY
                # After two failed direct attempts, fall back to the SSH proxy.
                if attempt>=2: opener=PROXY
                headers={'Range':f'bytes={start}-'} if start else {}
                with opener.open(u.Request(target,headers=headers),timeout=90) as r:
                    if start and r.status!=206: raise RuntimeError('Server ignored resume range')
                    with part.open('ab' if start else 'wb') as f:
                        while chunk:=r.read(4*1024*1024): f.write(chunk)
            if part.stat().st_size!=size: raise RuntimeError('Incomplete size')
            digest=hashlib.sha256()
            with part.open('rb') as f:
                while chunk:=f.read(16*1024*1024): digest.update(chunk)
            if expected and digest.hexdigest()!=expected:
                part.rename(part.with_name(part.name+f'.bad-{int(time.time())}'))
                raise RuntimeError('SHA256 mismatch')
            part.replace(path)
            marker.write_text(json.dumps({'revision':SPEC['revision'],'bytes':size,'sha256':digest.hexdigest()}))
            print('VERIFIED',name,size,flush=True); return
        except Exception as e:
            print('RETRY',name,attempt+1,type(e).__name__,str(e)[:180],flush=True)
            time.sleep(min(5*(attempt+1),30))
    raise RuntimeError(f'Cannot download {name}')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/models/{SPEC["repo"]}/revision/{SPEC["revision"]}?blobs=true'
    with PROXY.open(api,timeout=90) as r: metadata=json.load(r)
    (OUT/'upstream-metadata.json').write_text(json.dumps(metadata,indent=2))
    names={n for n in SPEC['files_sha256'] if not n.startswith('transformer/')}
    names.update(['modular_model_index.json','LICENSE'])
    rows=[r for r in metadata['siblings'] if r['rfilename'] in names]
    missing=names-{r['rfilename'] for r in rows}
    if missing: raise RuntimeError(f'Missing upstream files: {missing}')
    (OUT/'download-manifest.json').write_text(json.dumps(rows,indent=2))
    print('TOTAL',len(rows),'files',sum(r['size'] for r in rows)/1e9,'GB',flush=True)
    rows.sort(key=lambda r:r['size'])
    with cf.ThreadPoolExecutor(max_workers=16) as pool: list(pool.map(fetch,rows))
    import sys
    sys.path.insert(0,str(ROOT))
    from download_h3 import localize_index, verify_index
    index=OUT/'modular_model_index.json'
    original=OUT/'modular_model_index.upstream.json'
    if not original.exists(): original.write_bytes(index.read_bytes())
    localize_index(original,OUT,SPEC); verify_index(OUT,SPEC)
    print('REF2VA_DOWNLOAD_COMPLETE_SHA256_VERIFIED',flush=True)

if __name__=='__main__': main()
