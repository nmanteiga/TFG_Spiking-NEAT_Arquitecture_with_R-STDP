#!/bin/bash

# R-STDP groups (the SNN/ANN batch is in run_batch.sh on main). leave a list empty ("") to skip that group
# pilot (alice, Oct 2026): rstdp 301-305, lamarck 401-405, control 501-505
# batch 2: stronger learning-rate range, Lamarckian fix, win both games to solve. the pilot controls are reused
RSTDP_SEEDS="601 602 603 604 605 606 607 608 609 610"   # main design:  train_rstdp.py <seed>
LAMARCK_SEEDS="701 702 703 704 705"                     # ablation:     train_rstdp.py <seed> --lamarckian
CONTROL_SEEDS=""                                        # control:      train_rstdp.py <seed> --no-plasticity

cd "$(dirname "$0")" || exit 1
mkdir -p logs

# safe to launch again after a crash/reboot: finished runs (winner.pkl) are skipped, interrupted ones are resumed
# from their last checkpoint in the same folder, and the log is appended to
run_one() {
    local name=$1 seed=$2 flag=$3
    local log="logs/${name}_seed_${seed}.txt"
    local old resume=""
    old=$(ls -d logs/seed_${seed}_* 2>/dev/null | head -1)
    if [ -n "$old" ] && [ -f "$old/winner.pkl" ]; then
        echo "$(date '+%F %T')  skip   $name seed $seed (already finished)"
        return
    fi
    [ -n "$old" ] && resume="--resume $old"
    echo "$(date '+%F %T')  start  $name seed $seed $resume"
    if python3 -u train_rstdp.py "$seed" $flag $resume >> "$log" 2>&1; then
        echo "$(date '+%F %T')  done   $name seed $seed"
    else
        echo "$(date '+%F %T')  FAILED $name seed $seed (see $log)"
    fi
}

echo "$(date '+%F %T')  batch started on $(hostname) with $(python3 --version)"

for seed in $RSTDP_SEEDS; do
    run_one rstdp "$seed" "" &
    sleep 2
done
for seed in $LAMARCK_SEEDS; do
    run_one rstdp_lamarck "$seed" --lamarckian &
    sleep 2
done
for seed in $CONTROL_SEEDS; do
    run_one snn_potential "$seed" --no-plasticity &
    sleep 2
done

wait
