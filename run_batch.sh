#!/bin/bash

SNN_SEEDS="101 102 103 104 105"
ANN_SEEDS="201 202 203 204 205"

cd "$(dirname "$0")" || exit 1
mkdir -p logs

run_one() {
    local script=$1 name=$2 seed=$3
    local log="logs/${name}_seed_${seed}.txt"
    echo "$(date '+%F %T')  start  $name seed $seed"
    if python3 -u "$script" "$seed" > "$log" 2>&1; then
        echo "$(date '+%F %T')  done   $name seed $seed"
    else
        echo "$(date '+%F %T')  FAILED $name seed $seed (see $log)"
    fi
}

echo "$(date '+%F %T')  batch started on $(hostname) with $(python3 --version)"

for seed in $SNN_SEEDS; do
    run_one train_spiking.py spiking "$seed" &
    sleep 2
done
for seed in $ANN_SEEDS; do
    run_one train_baseline.py baseline "$seed" &
    sleep 2
done

wait