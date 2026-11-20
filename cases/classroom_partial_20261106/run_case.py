"""Isolated text-role override using the existing DCU H3 runtime and weights."""
from pathlib import Path
import hashlib
import json
import runpy
import sys

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[1]
sys.path.insert(0, str(PROJECT))
from cvm import video

GUIDANCE = (
    'The provided image is an appearance reference for the same classroom. '
    'Preserve its scene identity, material palette, lighting character, and the appearance of recognizable objects. '
    'Use the partial reference video for camera motion and the coarse arrangement of the visible reconstructed objects. '
    'Its uniform neutral-gray regions are unspecified content, not physical gray surfaces. '
    'Complete those regions plausibly and maintain temporal consistency. '
    'The reference image is not a fixed camera view to repeat throughout the clip.'
)
old_guidance = video.STYLE_REFERENCE
video.STYLE_REFERENCE = GUIDANCE
prompt = (CASE / 'prompt.txt').read_text(encoding='utf-8').strip()
(CASE / 'effective_prompt.txt').write_text(prompt.rstrip() + '\n\nstyle_reference_guidance: ' + GUIDANCE, encoding='utf-8')
record = {'seed': 42, 'keep_steps': 5, 'steps': 20, 'frames': 124,
          'modification': 'Only STYLE_REFERENCE text is overridden; no weights, conditioning layouts, masks, or sampling algorithms changed.',
          'original_style_guidance': old_guidance, 'new_style_guidance': GUIDANCE,
          'source_sha256': {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                            for name in ['cvm/video.py', 'run_dcu.py']}}
(CASE / 'runtime_config.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
if __name__ == '__main__':
    runpy.run_path(str(PROJECT / 'run_dcu.py'), run_name='__main__')
