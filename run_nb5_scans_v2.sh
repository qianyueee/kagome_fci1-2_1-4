#!/usr/bin/env bash
# Replacement runner: after scan2 (vnn=0.8 vnnn) finishes, insert
# vnn 2-4 extension scan + combine plot, then continue with scan3 / scan4.
set -uo pipefail

cd /home/qianyue/fci

PID_SCAN2="${1:-}"
if [[ -z "$PID_SCAN2" ]]; then
    PID_SCAN2=$(pgrep -f ed_nb5_vnn0p8_vnnn_scan.py | head -n 1 || true)
fi

if [[ -n "$PID_SCAN2" ]]; then
    echo "[runner v2] waiting for scan2 PID=$PID_SCAN2 ..."
    while kill -0 "$PID_SCAN2" 2>/dev/null; do
        sleep 30
    done
    echo "[runner v2] scan2 finished."
fi

echo "[runner v2] === extension: vnn 2.2..4.0 (10 pts) ==="
python -u scripts/ed_nb5_vnn_scan_2to4.py >> nb5_vnn_scan_2to4.log 2>&1

echo "[runner v2] === combine 0..2 + 2.2..4 plot ==="
python -u scripts/plot_nb5_vnn_scan_0to4.py >> nb5_vnn_scan_combine.log 2>&1

echo "[runner v2] === scan3: vnn=1, vnnn 0..1 (11 pts) ==="
python -u scripts/ed_nb5_vnn1_vnnn_scan.py >> nb5_vnn1_vnnn_scan.log 2>&1

echo "[runner v2] === scan4: coulomb vnnn=vnn/2, vnn 0..2 (11 pts) ==="
python -u scripts/ed_nb5_coulomb_scan.py >> nb5_coulomb_scan.log 2>&1

echo "[runner v2] all done."
