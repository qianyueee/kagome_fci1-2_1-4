#!/usr/bin/env bash
# v4 runner: coulomb ext 2.2-4 + 0-4 combine plot.
set -uo pipefail
cd /home/qianyue/fci

echo "[runner v4] === coulomb ext: vnn 2.2..4.0 (10 pts), vnnn=vnn/2 ==="
python -u scripts/ed_nb5_coulomb_scan_2to4.py >> nb5_coulomb_scan_2to4.log 2>&1

echo "[runner v4] === combine coulomb 0..2 + 2.2..4 plot ==="
python -u scripts/plot_nb5_coulomb_scan_0to4.py >> nb5_coulomb_scan_0to4_combine.log 2>&1

echo "[runner v4] all done."
