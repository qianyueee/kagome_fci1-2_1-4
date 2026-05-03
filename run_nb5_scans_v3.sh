#!/usr/bin/env bash
# v3 runner: vnn 4.2-6 ext + 0-6 combine + vnn=2 vnnn 0-2 scan.
set -uo pipefail
cd /home/qianyue/fci

echo "[runner v3] === ext: vnn 4.2..6.0 (10 pts) ==="
python -u scripts/ed_nb5_vnn_scan_4to6.py >> nb5_vnn_scan_4to6.log 2>&1

echo "[runner v3] === combine 0..2 + 2.2..4 + 4.2..6 plot ==="
python -u scripts/plot_nb5_vnn_scan_0to6.py >> nb5_vnn_scan_0to6_combine.log 2>&1

echo "[runner v3] === vnn=2 vnnn 0..2 scan (11 pts) ==="
python -u scripts/ed_nb5_vnn2_vnnn_scan.py >> nb5_vnn2_vnnn_scan.log 2>&1

echo "[runner v3] all done."
