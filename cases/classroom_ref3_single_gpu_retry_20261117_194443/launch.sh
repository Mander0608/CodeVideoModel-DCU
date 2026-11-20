#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=1
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_ref3_single_gpu_retry_20261117_194443
exec >> batch.log 2>&1
exec python3 -u code/batch_supervise.py
