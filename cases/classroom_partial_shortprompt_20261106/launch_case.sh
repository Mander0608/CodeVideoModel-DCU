#!/usr/bin/env bash
set -euo pipefail
BASE=/public/home/lixin1/chenxy
CASE="$BASE/CodeVideoModel/cases/classroom_partial_shortprompt_20261106"
cd "$BASE/CodeVideoModel"
exec > >(tee -a "$CASE/run.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$CASE/exit.code"; date -Is > "$CASE/finished_at.txt"' EXIT
test ! -e "$CASE/outputs/partial_seed42.mp4"
test ! -e "$CASE/started_at.txt"
exec 9>"$BASE/worldcrafter_dcu0.lock"
exec 8>"$BASE/worldcrafter_dcu1.lock"
flock -n 9
flock -n 8
date -Is | tee "$CASE/started_at.txt"
echo "CASE_START classroom_partial_shortprompt_20261106 seed=42 keep_steps=5 devices=0,1"
bash "$BASE/run_cvm_dcu.sh" "$CASE/preflight_case.py"
bash "$BASE/run_cvm_dcu.sh" "$CASE/run_case.py" \
  --code-video "$CASE/inputs/partial_proxy.mp4" \
  --reference-image "$CASE/inputs/reference_left30.png" \
  --prompt-file "$CASE/prompt.txt" \
  --model-root "$BASE/CodeVideoModel/models/MiniMax-H3-ref2va" \
  --keep-steps 5 --seed 42 \
  --output "$CASE/outputs/partial_seed42.mp4"
echo "CASE_COMPLETE $(date -Is)"
