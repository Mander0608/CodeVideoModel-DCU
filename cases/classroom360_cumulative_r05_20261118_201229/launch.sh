#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=5,6
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom360_cumulative_r05_20261118_201229
exec >> run.log 2>&1
exec python3 -u supervise.py
