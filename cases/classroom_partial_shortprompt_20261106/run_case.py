"""Run the short official-format prompt, with no appended style guidance."""
from pathlib import Path
import difflib
import hashlib
import inspect
import json
import runpy
import sys

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[1]
sys.path.insert(0, str(PROJECT))
from cvm import video

original = inspect.getsource(video.generate)
old_line = 'prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE'
assert original.count(old_line) == 1
patched = original.replace(old_line, 'prompt = prompt.rstrip()')
assert 'STYLE_REFERENCE' not in patched
patched_path = CASE / 'generate_no_appended_guidance.py'
patched_path.write_text(patched, encoding='utf-8')
(CASE / 'guidance_removal.diff').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), patched.splitlines(keepends=True),
    fromfile='original_generate', tofile='case_generate')), encoding='utf-8')
# Replace only this function in this process. Shared files and all sampling code stay intact.
exec(compile(patched, str(patched_path), 'exec'), video.__dict__)
prompt = (CASE / 'prompt.txt').read_text(encoding='utf-8').strip()
(CASE / 'effective_prompt.txt').write_text(prompt, encoding='utf-8')
record = {'seed': 42, 'keep_steps': 5, 'steps': 20, 'frames': 124,
          'modification': 'Use official-format short prompt and remove automatic STYLE_REFERENCE concatenation in this process only.',
          'style_reference_guidance_appended': False,
          'weights_architecture_sampling_inputs_unchanged': True,
          'effective_prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
          'patched_generate_sha256': hashlib.sha256(patched.encode('utf-8')).hexdigest(),
          'source_sha256': {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                            for name in ['cvm/video.py', 'run_dcu.py']}}
(CASE / 'runtime_config.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
if __name__ == '__main__':
    runpy.run_path(str(PROJECT / 'run_dcu.py'), run_name='__main__')
