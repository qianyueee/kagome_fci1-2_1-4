#!/usr/bin/env bash
# Run all Nb=5 scans serially. Scan 1 already started (PID supplied via $1
# or auto-detected). Waits for scan 1 to finish, then runs scan 2, 3, 4 in
# sequence. Each script supports resume from npz.
set -uo pipefail

cd /home/qianyue/fci

PID1="${1:-}"
if [[ -z "$PID1" ]]; then
    PID1=$(pgrep -f ed_nb5_vnn_scan.py | head -n 1 || true)
fi

if [[ -n "$PID1" ]]; then
    echo "[runner] waiting for scan1 PID=$PID1 ..."
    while kill -0 "$PID1" 2>/dev/null; do
        sleep 30
    done
    echo "[runner] scan1 finished."
fi

echo "[runner] === scan2: vnn=0.8, vnnn 0..0.8 (9 pts) ==="
python -u scripts/ed_nb5_vnn0p8_vnnn_scan.py >> nb5_vnn0p8_vnnn_scan.log 2>&1

echo "[runner] === scan3: vnn=1, vnnn 0..1 (11 pts) ==="
python -u scripts/ed_nb5_vnn1_vnnn_scan.py >> nb5_vnn1_vnnn_scan.log 2>&1

echo "[runner] === scan4: coulomb vnnn=vnn/2, vnn 0..2 (11 pts) ==="
python -u scripts/ed_nb5_coulomb_scan.py >> nb5_coulomb_scan.log 2>&1

echo "[runner] all 4 nb=5 scans done."
