#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=3,4
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom360_seam_arm02_20261117_195140
exec >> run.log 2>&1
exec python3 -u supervise.py
