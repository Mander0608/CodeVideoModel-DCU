"""Prepare a frozen, independent R2/R3 paired pilot. Never submit inference."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OLD = WORK / 'Classroom-Revisit8-20261107'
sys.path.insert(0, str(WORK / 'Blender-Classroom-Reconstruction/runtime'))
import cv2

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf-8')

def frame(n, f):
    p = OLD / f'round{n}/03_result/round{n}_cvm_output.mp4'
    cap = cv2.VideoCapture(str(p))
    # Sequential decode avoids seeking ambiguity for the three adjacent frames.
    selected = None
    for i in range(f + 1):
        ok, bgr = cap.read()
        assert ok, (p, i)
        if i == f: selected = Image.fromarray(bgr[:, :, ::-1])
    cap.release()
    traj = json.loads((OLD / f'round{n}/02_cvm/inputs/camera_trajectory.json').read_text(encoding='utf-8-sig'))
    return selected, {'source_video': str(p), 'source_sha256': sha(p), 'frame': f,
                      'nominal_proxy_yaw': traj['frames'][f]['yaw_degrees'],
                      'actual_generated_pose': 'not_calibrated', 'kind': 'generated_historical_prior'}

assert not (ROOT / 'pilot_001').exists(), 'Do not overwrite a prepared pilot'
P = ROOT / 'pilot_001'
P.mkdir()
jobs = []
for n in (2, 3):
    old = OLD / f'round{n}/02_cvm'
    original_prompt = (old / 'outputs/effective_prompt.txt').read_text(encoding='utf-8').strip()
    for variant in ('single', 'three'):
        job = f'r{n:02d}_{variant}'
        a = P / job
        (a / 'inputs').mkdir(parents=True)
        (a / 'review').mkdir()
        shutil.copy2(old / 'inputs/partial_proxy.mp4', a / 'inputs/proxy.mp4')
        refs = []
        if n == 2:
            for f in ([123] if variant == 'single' else [121, 122, 123]):
                im, meta = frame(1, f)
                rel = f'inputs/reference_{len(refs) + 1}.png'
                im.save(a / rel)
                meta.update(path=rel, role='immediate_start' if f == 123 else 'preceding_terminal_hold',
                            noise_stream='target_first_frame_condition_noise' if f == 123 else f'historical_r1_f{f}',
                            noise_step=-20)
                refs.append(meta)
            prompt = original_prompt
            if variant == 'three':
                prompt = prompt.replace('<Subject 1> is the classroom in <Picture 1>.',
                    '<Subject 1> is the same classroom in <Picture 1>, <Picture 2> and <Picture 3>.\n'
                    '<Picture 1>, <Picture 2> and <Picture 3> are the final three consecutive frames of the preceding shot, in chronological order. <Picture 3> is the immediate starting-view reference for this shot. They are not three camera destinations.')
                prompt = prompt.replace('Begin at the view in <Picture 1>.', 'Begin at the view in <Picture 3>.')
        else:
            rel = 'inputs/reference_1.png'
            shutil.copy2(old / 'inputs/reference.png', a / rel)
            refs.append({'path': rel, 'role': 'historical_destination', 'source': str(old / 'inputs/reference.png'),
                         'source_sha256': sha(old / 'inputs/reference.png'), 'source_round': 1, 'frame': 122,
                         'nominal_proxy_yaw': -30, 'actual_generated_pose': 'not_calibrated',
                         'kind': 'generated_historical_prior', 'noise_stream': 'target_first_frame_condition_noise', 'noise_step': -20})
            if variant == 'three':
                for f, role in [(123, 'current_start'), (61, 'historical_intermediate')]:
                    im, meta = frame(2, f)
                    rel = f'inputs/reference_{len(refs) + 1}.png'
                    im.save(a / rel)
                    meta.update(path=rel, role=role, noise_stream=f'historical_r2_f{f}', noise_step=-20)
                    refs.append(meta)
            prompt = original_prompt
            if variant == 'three':
                prompt = prompt.replace('<Subject 1> is the classroom in <Picture 1>.',
                    '<Subject 1> is the same classroom in <Picture 1>, <Picture 2> and <Picture 3>.\n'
                    '<Picture 1> shows the historical destination view. <Picture 2> shows the current starting view. <Picture 3> shows a previously observed intermediate view. These are observations of one stationary room; their numbering is not the playback order.')
                prompt = prompt.replace('From the fixed starting position shown in <Video 1>,',
                    'Start from the view in <Picture 2>, corresponding to the first frame of <Video 1>. From this fixed starting position,')
        for ref in refs: ref['sha256'] = sha(a / ref['path'])
        (a / 'inputs/prompt.txt').write_text(prompt + '\n', encoding='utf-8')
        config = {'job_id': job, 'round': n, 'variant': variant, 'seed': 42, 'steps': 20,
                  'scheduler_steps_requested': 21, 'keep_steps': 5, 'frames': 124, 'fps': 24,
                  'width': 960, 'height': 544, 'turbo': False, 'mask_input': False,
                  'camera_parameter_input': False, 'hard_anchor': False, 'pasteback': False,
                  'reference_short_edge_override': None, 'proxy': 'inputs/proxy.mp4',
                  'prompt': 'inputs/prompt.txt', 'output': 'outputs/result.mp4',
                  'references': refs, 'source_proxy_sha256': sha(old / 'inputs/partial_proxy.mp4'),
                  'original_prompt_sha256': sha(old / 'outputs/effective_prompt.txt'),
                  'paired_scope': 'Frozen historical proxy and parent output; no new outputs fed into other cases.'}
        save(a / 'config.json', config)
        sheet = Image.new('RGB', (480 * 3, 296 * 2), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, ref in enumerate(refs):
            im = Image.open(a / ref['path'])
            sheet.paste(im.resize((480, 272)), (480 * k, 24))
            draw.text((480 * k + 4, 4), f'Picture {k+1}: {ref["role"]}', fill='black')
        cap = cv2.VideoCapture(str(a / 'inputs/proxy.mp4'))
        picked = {}
        for f in range(124):
            ok, bgr = cap.read()
            assert ok
            if f in [0, 31, 62, 92, 123]:
                im = Image.fromarray(bgr[:, :, ::-1]); im.save(a / f'review/proxy_F{f:03d}.png'); picked[f] = im
        assert not cap.read()[0]
        cap.release()
        for k, f in enumerate([0, 62, 123]):
            sheet.paste(picked[f].resize((480, 272)), (480 * k, 320))
            draw.text((480 * k + 4, 300), f'Proxy F{f}', fill='black')
        sheet.save(a / 'review/reference_and_proxy.jpg', quality=95)
        jobs.append(job)
save(P / 'batch_config.json', {'jobs': jobs, 'created_at': datetime.datetime.now().astimezone().isoformat(),
                             'status': 'prepared_pending_code_and_visual_review',
                             'devices': [0, 1], 'queue_max_seconds': 14400,
                             'max_free_check_age_seconds': 600, 'min_free_GiB_per_device': 60,
                             'serial': True, 'stop_on_first_technical_failure': True})
print(json.dumps({'jobs': jobs, 'root': str(P)}, indent=2))
