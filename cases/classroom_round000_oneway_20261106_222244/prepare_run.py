"""Stage the rendered single-direction input; preserve all prior runs."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parent
VARIANT=ROOT.parents[1]/'round_000_oneway_20261106_222244'
OLD=ROOT.parent/'round000_shortprompt_seed42_20261106_162713'
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round000_oneway_20261106_222244'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((VARIANT/'validation.json').read_text())['status']=='passed'
assert not (ROOT/'inputs').exists()
(ROOT/'inputs').mkdir()
for name in ['partial_proxy.mp4','reference.png','prompt.txt','camera_trajectory.json']:
    shutil.copy2(VARIANT/'inputs'/name,ROOT/'inputs'/name)
assert sha(ROOT/'inputs/reference.png')==sha(OLD/'inputs/reference.png')
assert sha(ROOT/'inputs/partial_proxy.mp4')!=sha(OLD/'inputs/partial_proxy.mp4')
for name in ['cvm_config.json','run_cvm.py','preflight.py','supervise.py']:
    shutil.copy2(OLD/name,ROOT/name)
old_prompt=(OLD/'inputs/prompt.txt').read_text();new_prompt=(ROOT/'inputs/prompt.txt').read_text()
a='The camera smoothly turns left from a fixed position, briefly pauses, then returns to the initial front view, following <Video 1>.'
b='The camera smoothly turns left from a fixed position through <Subject 1>, following <Video 1>.'
assert old_prompt.replace(a,b)==new_prompt
watch=(OLD/'watch_and_collect.py').read_text().replace('/cases/classroom_round000_shortprompt_20261106_162713','/cases/classroom_round000_oneway_20261106_222244')
(ROOT/'watch_and_collect.py').write_text(watch,encoding='utf-8',newline='\n')
(ROOT/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nCASE='+REMOTE+'\ncd "$CASE"\nexec >> "$CASE/run.log" 2>&1\nexec python3 -u "$CASE/supervise.py"\n',encoding='utf-8',newline='\n')
manifest={'comparison_run':OLD.name,'scene_snapshot':VARIANT.name,'changes':['camera trajectory and resulting proxy video','one prompt camera sentence'],
 'unchanged_reference_sha256':sha(ROOT/'inputs/reference.png'),
 'sampling':{'seed':42,'steps':20,'keep_steps':5,'frames':124},'runner_identical':sha(ROOT/'run_cvm.py')==sha(OLD/'run_cvm.py'),
 'automatic_style_guidance':False,'variant':json.loads((VARIANT/'variant_manifest.json').read_text())}
(ROOT/'paired_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='prepare_run.py']
(ROOT/'input_checksums.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files},indent=2),encoding='utf-8')
(ROOT/'deployment.json').write_text(json.dumps({'remote_host':'scnet2','remote_case':REMOTE,'screen_session':'cvm_oneway_222244','status':'prepared_to_launch'},indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
