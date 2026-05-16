#!/usr/bin/env bash
# v5 runner: sqrt3 coulomb ext 4.2-6 + 0-6 combine plot.
set -uo pipefail
cd /home/qianyue/fci

echo "[runner v5] === sqrt3 coulomb ext: vnn 4.2..6.0 (10 pts), vnnn=vnn/sqrt(3) ==="
python -u scripts/ed_nb5_coulomb_sqrt3_scan_4to6.py >> nb5_coulomb_sqrt3_scan_4to6.log 2>&1

echo "[runner v5] === combine sqrt3 coulomb 0..4 + 4.2..6 plot ==="
python -u scripts/plot_nb5_coulomb_sqrt3_scan_0to6.py >> nb5_coulomb_sqrt3_scan_0to6_combine.log 2>&1

echo "[runner v5] all done."
