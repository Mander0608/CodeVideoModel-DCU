#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=6,7
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/cabazon_rebuilt_r1_trial001_20261108_202645
exec >> run.log 2>&1
exec python3 -u supervise.py
