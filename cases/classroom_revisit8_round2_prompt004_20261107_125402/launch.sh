#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=4,5
CASE=/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit8_round2_prompt004_20261107_125402
cd "$CASE"
exec >> "$CASE/run.log" 2>&1
exec python3 -u "$CASE/supervise.py"
