"""User-requested prompt-only control of original round9; prepare without launching."""
from pathlib import Path
import json,hashlib,shutil,datetime,difflib
B=Path(__file__).resolve().parent;R=B/'round9';OLD=R/'05_cvm/attempt_001';NEW=R/'05_cvm/attempt_005_prompt_only'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OLD/'result_status.json')['accepted_for_next_round'] is False
assert all(sha(OLD/p)==h for p,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items())
shutil.copytree(OLD/'inputs',NEW/'inputs');(NEW/'baseline').mkdir()
for name in ['prompt.txt','prompt_adaptation.json','input_manifest.json']:shutil.copy2(OLD/'inputs'/name,NEW/'baseline'/name)
prompt='''subject_definitions:
<Subject 1> is the same classroom throughout <Picture 1> and <Video 1>, with cream walls, wooden desks, a navy wall board, a continuous floor and ceiling.
<Picture 1> supplies the classroom appearance and starting view.
<Video 1> is a partially observed render of one continuous camera turn. Every flat neutral-gray area is missing scene content. All non-gray regions, including a narrow strip that reappears at the far left later in the clip, show existing parts of this same room. The gray cutout edges are coverage boundaries, not physical wall edges or separate pictures.

summary:
[video editing + reference generation] Complete all gray holes of <Video 1>, including the interior gap between the returning left strip and the larger known region on the right, to produce one coherent full-frame classroom video.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the existing classroom identity, number and placement of boards and furnishings.
<Video 1>: partially_preserved - preserve the camera framing, chronological motion, visible room layout and every non-gray region on both sides of each hole. Replace only the missing gray content and its artificial cutout boundary with connected room surfaces.

detailed_description:
[Shot 1] Follow the camera path in <Video 1> frame by frame: the initial pause, the smooth in-place LEFT turn totaling 30 degrees, and the final pause. Match the displacement and perspective of the existing navy board and desks throughout the clip. Stop turning exactly when the source stops. Keep the optical center fixed, with no translation, zoom, extra rotation or return turn.
At first, complete the newly exposed gaps at the left, top and bottom. Later, when an already known strip of the classroom re-enters at the far left, retain that strip and the larger known view on the right simultaneously. Fill the gray gap BETWEEN them by connecting their existing cream wall, baseboard, ceiling, floor and any genuinely clipped furniture. Use the surrounding visible edges and perspective to determine the connection. Preserve the room corner indicated by the visible geometry; do not turn a gray boundary into a new corner, pillar, recess, doorway or partition. The two known regions are parts of one room, not independent scenes.
Keep the existing navy board as one stationary object at its source-video position. Leave the newly connected wall bare where no existing fixture is indicated. Do not invent or duplicate a board, clock, window, door or row of desks. Continue a clipped object only where its visible shape requires continuation. Apply the appearance of <Picture 1> consistently while retaining the source layout. Fill the upper holes with the connected ceiling and the lower holes with the continuous floor. Maintain stable surfaces, occlusion and lighting over time. Every frame is a single full-frame classroom view without gray holes, inset images, cutout outlines, cuts or scene transitions.

overall_soundscape:
Quiet classroom ambience.

non_diegetic_music:
N/A
'''
(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
diff=''.join(difflib.unified_diff((OLD/'inputs/prompt.txt').read_text(encoding='utf-8').splitlines(True),prompt.splitlines(True),fromfile='attempt_001/prompt.txt',tofile='attempt_005_prompt_only/prompt.txt'))
(NEW/'prompt.diff').write_text(diff,encoding='utf-8')
save(NEW/'inputs/prompt_adaptation.json',{'baseline_attempt':'attempt_001','changed_model_conditions':['prompt.txt'],'user_requested':True,'purpose':'Test explicit internal-hole and returned-known-strip semantics plus source-matched motion; no mask/reference/sampling change.','not_a_guarantee':'Text cannot enforce known-pixel preservation in original RGB CVM.'})
m=read(NEW/'inputs/input_manifest.json');m['status']='prompt_only_control_prepared_not_run'
for name,rec in m['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',m)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round9_prompt005_{stamp}'
olddep=read(OLD/'deployment.json')
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:
 (NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(olddep['remote_case'],remote),encoding='utf-8',newline='\n')
shutil.copy2(B/'round10/05_cvm/attempt_001/check_prelaunch_resources.py',NEW/'check_prelaunch_resources.py')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='User-requested round9 original-RGB prompt-only control against attempt001. Explicit internal-gap/return-strip retention and source-matched30degree motion; one i8 image, identical partial video/seed42/20steps/keep5. No explicit mask or pasteback. Independent retrospective branch; do not overwrite accepted chronological chain.');save(NEW/'cvm_config.json',cfg)
fixed={p.name:sha(p) for p in (OLD/'inputs').iterdir() if p.is_file() and p.name not in ['prompt.txt','prompt_adaptation.json','input_manifest.json']}
assert all(sha(NEW/'inputs'/p)==h for p,h in fixed.items())
assert sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py')
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_001','changed_model_conditions':['prompt.txt'],'all_other_input_files_identical':fixed,'runner_identical':True,'runner_sha256':sha(NEW/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'independent_reference_images':1,'reference':'i8 from round8','mask_input':False,'camera_parameter_input':False,'pasteback':False,'baseline_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'candidate_prompt_sha256':sha(NEW/'inputs/prompt.txt'),'limits':'One seed, multiple wording changes within the sole changed input prompt. Does not isolate individual sentences or establish general capability.'})
(NEW/'README.md').write_text('''# User-requested round9 prompt-only control

Compare with original attempt001, not masked attempts002-004. Same raw partial Blender video, one i8 image, seed42,20steps,keep5 and runner. Only effective model prompt changes. Both prompts and unified diff retained. No new image, explicit mask, pasteback or scene correction.

New wording describes the returning left known strip, interior gray gap, nonphysical gray boundaries, stationary existing board/furniture, and exact source timing. The requested known-pixel retention is textual and is not guaranteed by the original sampler.

Serial priority after active round10 masked002 ends, before launching round11. Check resources and deploy with deploy_round9_prompt005.py once; that script leaves the main chain's current run unchanged and tracks this side experiment separately. Preserve/audit round10 normally. Never rerun prepare_round9_prompt005.py.

After completion: run audit_cvm_attempt.py --round 9 --attempt attempt_005_prompt_only. Compute local motion-chain diagnostic when direct endpoint correspondence is unreliable; inspect input/output and compare with attempt001. Archive actual prompt/config/environment/timing/logs/output/hash. Report whether completion, motion and structural retention improved separately. Do not automatically replace corrected i9 or reconstruct already-running downstream rounds from this retrospective test.
''',encoding='utf-8')
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
for p in NEW.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
now=datetime.datetime.now().astimezone().isoformat();dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r9prompt005_{stamp}','status':'prepared_to_launch','prepared_at':now};save(NEW/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s['pending_user_experiment']={'path':NEW.relative_to(B).as_posix(),'status':'prepared_waiting_for_round10_resources','round':9,'baseline':'round9/05_cvm/attempt_001','priority':'Run after current round10 completes, before next new chain inference. Do not switch main current_run.','launch_script':'deploy_round9_prompt005.py','requested_at':now,'replace_chain_reference_automatically':False};save(B/'CONTINUOUS_LEFT_STATE.json',s)
p=B/'CONTINUOUS_LEFT_PLAN.md';p.write_text(f'LATEST {now}: USER REQUEST: round9 prompt-only control prepared at round9/05_cvm/attempt_005_prompt_only, NOT submitted. Originalattempt001 video/i8/runner/seed42/20steps/keep5 identical; only prompt changed for interior hole, returning known strip and source-timed motion. Existing round10 keeps running. After it exits, check resources and deploy_round9_prompt005.py ONCE before new chain inference; main current_run must remain round10 until its own audit/transition. Side experiment tracked in pending_user_experiment. Read candidate README. Retrospective result does not automatically replace i9 or downstream chain.\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'prepared':str(NEW),'same_input_files':len(fixed),'prompt_sha256':sha(NEW/'inputs/prompt.txt'),'deployment':dep},indent=2))
