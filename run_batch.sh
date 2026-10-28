#!/bin/bash

# R-STDP groups (the SNN/ANN batch is in run_batch.sh on main). leave a list empty ("") to skip that group
RSTDP_SEEDS="301 302 303 304 305"      # main design:  train_rstdp.py <seed>
LAMARCK_SEEDS="401 402 403 404 405"    # ablation:     train_rstdp.py <seed> --lamarckian
CONTROL_SEEDS="501 502 503 504 505"    # control:      train_rstdp.py <seed> --no-plasticity

cd "$(dirname "$0")" || exit 1
mkdir -p logs

run_one() {
    local name=$1 seed=$2 flag=$3
    local log="logs/${name}_seed_${seed}.txt"
    echo "$(date '+%F %T')  start  $name seed $seed"
    if python3 -u train_rstdp.py "$seed" $flag > "$log" 2>&1; then
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
