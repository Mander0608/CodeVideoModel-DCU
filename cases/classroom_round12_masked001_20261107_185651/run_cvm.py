from pathlib import Path
import argparse,sys,runpy,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--project-root');p.add_argument('--execute',action='store_true');a=p.parse_args()
assert a.execute
cfg=json.loads((ROOT/'cvm_config.json').read_text(encoding='utf-8'));out=ROOT/'outputs';out.mkdir(exist_ok=True)
prompt=(ROOT/'inputs/prompt.txt').read_text(encoding='utf-8').strip()
assert prompt==(ROOT/'prompt.txt').read_text(encoding='utf-8').strip()
(out/'effective_prompt.txt').write_text(prompt,encoding='utf-8')
shutil.copy2(ROOT/'masked_inpaint.py',out/'generate_effective.py')
argv=[str(ROOT/'masked_inpaint.py'),'--case',str(ROOT),'--output',str(ROOT/cfg['output']),'--steps','20','--seed','42','--no-turbo']
(out/'launch_record.json').write_text(json.dumps({'argv':argv,'config':cfg,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'protocol':'explicit_masked_completion_not_original_cvm'},indent=2),encoding='utf-8')
sys.argv=argv;runpy.run_path(str(ROOT/'masked_inpaint.py'),run_name='__main__')
