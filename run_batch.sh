#!/bin/bash

echo "Starting Spiking Batch 1..."
python3 -u train_spiking.py 101 > logs/spiking_seed_101.txt 2>&1 &
sleep 2 
python3 -u train_spiking.py 102 > logs/spiking_seed_102.txt 2>&1 &
sleep 2
python3 -u train_spiking.py 103 > logs/spiking_seed_103.txt 2>&1 &
sleep 2
python3 -u train_spiking.py 104 > logs/spiking_seed_104.txt 2>&1 &
sleep 2
python3 -u train_spiking.py 105 > logs/spiking_seed_105.txt 2>&1 &

echo "Starting Baseline Batch 1..."
python3 -u train_baseline.py 201 > logs/baseline_seed_201.txt 2>&1 &
sleep 2
python3 -u train_baseline.py 202 > logs/baseline_seed_202.txt 2>&1 &
sleep 2
python3 -u train_baseline.py 203 > logs/baseline_seed_203.txt 2>&1 &
sleep 2
python3 -u train_baseline.py 204 > logs/baseline_seed_204.txt 2>&1 &
sleep 2
python3 -u train_baseline.py 205 > logs/baseline_seed_205.txt 2>&1 &

echo "All 10 tests are running in the background"