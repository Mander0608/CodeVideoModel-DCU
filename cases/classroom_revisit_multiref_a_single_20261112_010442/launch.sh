#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES=4,5
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit_multiref_a_single_20261112_010442
exec >> run.log 2>&1
exec python3 -u supervise.py
