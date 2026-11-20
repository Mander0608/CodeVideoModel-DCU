#!/usr/bin/env bash
set -euo pipefail
CASE=/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round000_shortprompt_20261106_162713
cd "$CASE"
exec >> "$CASE/run.log" 2>&1
exec python3 -u "$CASE/supervise.py"
