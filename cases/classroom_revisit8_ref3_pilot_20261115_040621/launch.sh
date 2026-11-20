#!/usr/bin/env bash
set -euo pipefail
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit8_ref3_pilot_20261115_040621
exec >> batch.log 2>&1
exec python3 -u code/batch_supervise.py
