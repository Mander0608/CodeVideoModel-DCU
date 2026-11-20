"""Read-only progress watcher for one case. Does not launch or restart jobs."""
from pathlib import Path
import json
import re
import time

root = Path(__file__).resolve().parent
previous = None
start = time.monotonic()
while time.monotonic() - start < 3600:
    log = (root / 'run.log').read_text(errors='replace') if (root / 'run.log').exists() else ''
    lines = [line for line in log.splitlines() if re.match(r'^(CASE_|PIN_|MODEL_READY|ENCODE_|DENOISE|DECODE_|RUN_COMPLETE|Traceback)', line)]
    steps = re.findall(r'^DENOISE (\d+) seconds ([0-9.]+)', log, re.M)
    phase = lines[-1] if lines else 'waiting_for_log'
    exit_code = (root / 'exit.code').read_text().strip() if (root / 'exit.code').exists() else None
    state = {'phase': phase, 'steps': len(steps), 'latest_step_seconds': float(steps[-1][1]) if steps else None,
             'exit_code': exit_code, 'output_exists': (root / 'outputs/partial_seed42.mp4').exists()}
    if state != previous:
        print(json.dumps(state), flush=True)
        previous = state
    if exit_code is not None:
        break
    time.sleep(5)
