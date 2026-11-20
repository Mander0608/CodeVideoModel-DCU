"""Read-only runtime/model metadata receipt; does not import torch or hash large weights."""
from pathlib import Path
import sys,json,hashlib,platform,importlib.metadata as metadata,subprocess,datetime
case=Path(__file__).resolve().parent
project=Path('/public/home/lixin1/chenxy/CodeVideoModel');model=project/'models/MiniMax-H3-ref2va'
packages={}
for name in ['torch','diffusers','transformers','accelerate','numpy','safetensors','pillow','peft']:
 try:packages[name]=metadata.version(name)
 except metadata.PackageNotFoundError:packages[name]=None
files=[]
for p in sorted(model.rglob('*')):
 if not p.is_file():continue
 st=p.stat();row=dict(path=p.relative_to(model).as_posix(),bytes=st.st_size,mtime_ns=st.st_mtime_ns,is_symlink=p.is_symlink())
 if p.is_symlink():row['resolved_path']=str(p.resolve())
 if p.suffix in ['.json','.yaml','.yml','.txt'] and st.st_size<10*1024*1024:row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 files.append(row)
result=dict(captured_at=datetime.datetime.now().astimezone().isoformat(),python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
 model_root=str(model),model_files=files,large_weight_content_hashes_computed=False,
 note='Large model weights are referenced by path/size/mtime and symlink target where present; no claim of full weight checksum verification. Configuration files are SHA256 recorded.')
(case/'outputs').mkdir(exist_ok=True)
(case/'outputs/environment_snapshot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('ENVIRONMENT_RECEIPT_SAVED',len(files))
