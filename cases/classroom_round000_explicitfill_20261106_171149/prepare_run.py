"""Paired run changing only detailed_description, preserving all other conditions."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'round000_shortprompt_seed42_20261106_162713'
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round000_explicitfill_20261106_171149'
DETAIL='''Photorealistic classroom interior with natural daylight.
[Shot 1] The camera smoothly turns left from a fixed position, briefly pauses, then returns to the initial front view, following <Video 1>. As the camera turns left, the uniform gray regions mark missing, unobserved parts of the classroom, not gray walls or objects. Replace these regions completely with a plausible continuation of the same room: cream-painted walls, continuous flooring, and classroom details consistent with the visible windows and desks. Preserve the existing classroom layout and connect the completed areas naturally to the visible scene, with consistent perspective, lighting and occlusion. The completed content remains stationary and consistent as the camera returns. The entire frame depicts one complete classroom, with no flat gray placeholders or cutout boundaries.'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fields(text):
    pieces=re.split(r'^([a-z_]+):\s*\n',text,flags=re.M)
    return {pieces[i]:pieces[i+1].strip() for i in range(1,len(pieces),2)}
assert not (ROOT/'inputs').exists()
(ROOT/'inputs').mkdir()
same={}
for name in ['partial_proxy.mp4','reference.png','camera_trajectory.json']:
    shutil.copy2(OLD/'inputs'/name,ROOT/'inputs'/name)
    assert sha(OLD/'inputs'/name)==sha(ROOT/'inputs'/name)
    same[name]=sha(ROOT/'inputs'/name)
old_prompt=(OLD/'inputs/prompt.txt').read_text(encoding='utf-8')
start=old_prompt.index('detailed_description:\n')
end=old_prompt.index('overall_soundscape:\n')
prompt=old_prompt[:start]+'detailed_description:\n'+DETAIL+'\n\n'+old_prompt[end:]
before,after=fields(old_prompt),fields(prompt)
assert len(before)==len(after)==6
changed=[k for k in before if before[k]!=after[k]]
assert changed==['detailed_description'],changed
(ROOT/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
for name in ['cvm_config.json','run_cvm.py','preflight.py','supervise.py']:
    shutil.copy2(OLD/name,ROOT/name)
watch=(OLD/'watch_and_collect.py').read_text().replace('/cases/classroom_round000_shortprompt_20261106_162713','/cases/classroom_round000_explicitfill_20261106_171149')
(ROOT/'watch_and_collect.py').write_text(watch,encoding='utf-8',newline='\n')
(ROOT/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nCASE='+REMOTE+'\ncd "$CASE"\nexec >> "$CASE/run.log" 2>&1\nexec python3 -u "$CASE/supervise.py"\n',encoding='utf-8',newline='\n')
analysis=(OLD/'analyze_pair.py').read_text().replace("OLD=ROOT.parent/'round000_seed42_20261106_151749'", "OLD=ROOT.parent/'round000_shortprompt_seed42_20261106_162713'")
analysis=analysis.replace("'Short prompt'","'Explicit fill'").replace("'Previous prompt'","'Short prompt'").replace('old_vs_short_prompt.mp4','short_vs_explicit_fill.mp4')
(ROOT/'analyze_pair.py').write_text(analysis,encoding='utf-8',newline='\n')
paired={'comparison_run':OLD.name,'changed_variable':'detailed_description only','changed_prompt_fields':changed,
 'unchanged_prompt_fields':[k for k in before if k not in changed],
 'identical_inputs_sha256':same,'identical_sampling':{'seed':42,'steps':20,'keep_steps':5,'frames':124},
 'no_appended_style_reference_guidance_in_either_run':True,'runner_identical':sha(ROOT/'run_cvm.py')==sha(OLD/'run_cvm.py'),
 'old_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'new_prompt_sha256':sha(ROOT/'inputs/prompt.txt'),
 'scope':'Single paired seed. No change to mask interface, video, scene, reference, model, sampler or other five prompt sections.'}
(ROOT/'paired_manifest.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='prepare_run.py']
(ROOT/'input_checksums.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files},indent=2),encoding='utf-8')
(ROOT/'deployment.json').write_text(json.dumps({'remote_host':'scnet2','remote_case':REMOTE,'screen_session':'cvm_explicitfill_171149','status':'prepared_to_launch'},indent=2),encoding='utf-8')
print(json.dumps(paired,indent=2))
