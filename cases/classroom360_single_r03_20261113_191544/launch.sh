#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=6,7
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom360_single_r03_20261113_191544
exec >> run.log 2>&1
exec python3 -u supervise.py
