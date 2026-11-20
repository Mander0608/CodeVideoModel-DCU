#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=0
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_ref3_single_gpu_20261115_204520
exec >> batch.log 2>&1
exec python3 -u code/batch_supervise.py
