"""Decode every frame and create matched proxy/output contact sheets."""
import hashlib
import io
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
ffmpeg = str(root / 'tools/ffmpeg/ffmpeg')
ffprobe = str(root / 'tools/ffmpeg/ffprobe')
output = root / 'outputs/robot_navigation_dcu.mp4'
proxy = root / 'examples/robot_navigation/code_video.mp4'
media = json.loads(subprocess.check_output([
    ffprobe, '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(output)]))
video = next(s for s in media['streams'] if s['codec_type'] == 'video')
assert (video['width'], video['height'], int(video['nb_read_frames'])) == (960, 544, 124)
assert video['avg_frame_rate'] == '24/1'
assert any(s['codec_type'] == 'audio' for s in media['streams'])
subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], check=True)

indices = [0, 31, 62, 93, 123]
board = Image.new('RGB', (1600, 450), 'white')
draw = ImageDraw.Draw(board)
for row, (label, path) in enumerate([('Three.js proxy', proxy), ('CVM on scnet2: 20 updates, proxy in first 5', output)]):
    y = row * 225
    draw.text((8, y + 5), label, fill='black')
    for col, index in enumerate(indices):
        data = subprocess.check_output([
            ffmpeg, '-v', 'error', '-i', str(path), '-vf', f'select=eq(n\\,{index})',
            '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
        frame = Image.open(io.BytesIO(data)).convert('RGB').resize((320, 181))
        board.paste(frame, (col * 320, y + 23))
        draw.text((col * 320 + 8, y + 207), f'Frame {index} / {index/24:.2f} s', fill='black')
board.save(root / 'outputs/robot_navigation_comparison.png')
record = {'status': 'media_validation_passed', 'full_decode': True, 'media': media,
          'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
          'visual_review': 'Contact sheet exported; interpretation requires human or model review.'}
(root / 'outputs/robot_navigation_media_validation.json').write_text(json.dumps(record, indent=2))
print('MEDIA_VALIDATION_PASSED', flush=True)
