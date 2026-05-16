#!/usr/bin/env bash
# v6 runner: wait for v5 runner (PID 30641) to finish, then run V_nnn=V_nn/1.5 scan, vnn 2-6.
set -uo pipefail
cd /home/qianyue/fci

V5_PID=30641

echo "[runner v6] waiting for v5 runner PID=$V5_PID to finish ..."
while kill -0 $V5_PID 2>/dev/null; do
    sleep 60
done
echo "[runner v6] v5 done; starting V_nnn=V_nn/1.5 scan."

echo "[runner v6] === V_nnn=V_nn/1.5: vnn 2..6 (21 pts), n_show=10 ==="
python -u scripts/ed_nb5_vnnn_1p5_scan_2to6.py >> nb5_vnnn_1p5_scan_2to6.log 2>&1

echo "[runner v6] all done."
