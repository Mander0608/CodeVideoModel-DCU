"""Create a prompt-only paired run; preserve the original scene and failed run."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'round000_seed42_20261106_151749'
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round000_shortprompt_20261106_162713'
PROMPT='''subject_definitions:
<Subject 1> is the classroom in <Picture 1>, with wooden desks, a green chalkboard and soft daylight.
<Video 1> is the partial classroom preview; uniform gray regions indicate missing content.

summary:
[video editing + reference generation] The target video is an edited version of <Video 1>, completing its missing regions and matching the realistic appearance of <Subject 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - classroom identity, materials and lighting.
<Video 1>: partially_preserved - retain camera motion and visible object layout; replace gray regions and their boundaries with coherent room content.

detailed_description:
Photorealistic interior with natural daylight.
[Shot 1] The camera smoothly turns left from a fixed position, briefly pauses, then returns to the initial front view, following <Video 1>. The chalkboard and desks remain stationary. Newly revealed areas contain continuous cream walls, flooring and plausible classroom details, filling the entire image. Surfaces connect seamlessly, with consistent perspective, lighting and occlusion. The completed room remains stable throughout the unbroken shot.

overall_soundscape:
Quiet classroom ambience.

non_diegetic_music:
N/A
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'inputs').exists(),'Refuse to replace an existing run'
(ROOT/'inputs').mkdir()
same={}
for name in ['partial_proxy.mp4','reference.png','camera_trajectory.json']:
    shutil.copy2(OLD/'inputs'/name,ROOT/'inputs'/name)
    assert sha(OLD/'inputs'/name)==sha(ROOT/'inputs'/name)
    same[name]=sha(ROOT/'inputs'/name)
(ROOT/'inputs/prompt.txt').write_text(PROMPT,encoding='utf-8')
for name in ['cvm_config.json','run_cvm.py','preflight.py','supervise.py']:
    shutil.copy2(OLD/name,ROOT/name)
old_preflight=json.loads((OLD/'preflight.json').read_text())
with (ROOT/'preflight.py').open('a',encoding='utf-8') as f:
    f.write('\nassert report["shared_source_sha256"] == '+repr(old_preflight['shared_source_sha256'])+'\n')
watch=(OLD/'watch_and_collect.py').read_text().replace('/cases/classroom_round000_20261106_151749','/cases/classroom_round000_shortprompt_20261106_162713')
(ROOT/'watch_and_collect.py').write_text(watch,encoding='utf-8')
(ROOT/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nCASE='+REMOTE+'\ncd "$CASE"\nexec >> "$CASE/run.log" 2>&1\nexec python3 -u "$CASE/supervise.py"\n',encoding='utf-8')
paired={'comparison_run':str(OLD.name),'changed_variable':'prompt text only',
 'user_prompt_adjustment':'Only the camera sentence is changed to left turn, brief pause, return front, matching the unchanged input video.',
 'identical_inputs_sha256':same,'identical_sampling':{'seed':42,'steps':20,'keep_steps':5,'frames':124},
 'no_appended_style_reference_guidance_in_either_run':True,
 'runner_identical':sha(ROOT/'run_cvm.py')==sha(OLD/'run_cvm.py'),
 'old_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'new_prompt_sha256':sha(ROOT/'inputs/prompt.txt'),
 'previous_shared_source_sha256':old_preflight['shared_source_sha256'],
 'scope':'One paired seed. Can compare the text change in this example, not establish general reliability or exact geometry.'}
(ROOT/'paired_manifest.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='prepare_run.py']
(ROOT/'input_checksums.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files},indent=2),encoding='utf-8')
(ROOT/'deployment.json').write_text(json.dumps({'remote_host':'scnet2','remote_case':REMOTE,
 'screen_session':'cvm_round000_short_162713','status':'prepared_to_launch'},indent=2),encoding='utf-8')
print(json.dumps(paired,indent=2))
