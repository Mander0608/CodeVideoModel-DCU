#!/usr/bin/env bash
set -euo pipefail
cd /public/home/lixin1/chenxy/CodeVideoModel/cases/infinigen_124_20261117_194112
exec >> run.log 2>&1
exec python3 -u supervise.py
