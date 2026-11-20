"""Re-render an upstream editable Three.js case, without paid APIs."""
import json, shutil, subprocess, time
from pathlib import Path
from inference.render import render
from cvm import check_video
scene=Path('outputs/chair_render')
if not scene.exists(): shutil.copytree('examples/chair_placement/original',scene)
start=time.monotonic()
render(scene,'video')
path=scene/'case/reconstruction.mp4'
check_video(path)
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))
record={'status':'passed','elapsed_seconds':time.monotonic()-start,'media':info,
        'meaning':'Three.js proxy re-render only, not H3 generated final video'}
(scene/'render_report.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
print('THREEJS_RENDER_PASSED',flush=True)
