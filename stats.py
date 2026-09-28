import os
import re
import numpy as np
from scipy.stats import mannwhitneyu

snn_logs = [
    "logs/spiking_seed_101.txt",
    "logs/spiking_seed_102.txt",
    "logs/spiking_seed_103.txt",  # this one is not done yet
    "logs/spiking_seed_104.txt",
    "logs/spiking_seed_105.txt"
]

ann_logs = [
    "logs/baseline_seed_201.txt",
    "logs/baseline_seed_202.txt",
    "logs/baseline_seed_203.txt",
    "logs/baseline_seed_204.txt",
    "logs/baseline_seed_205.txt"
]

def parse_log_file(filepath):
    ''' reads a log file and extracts the final metrics '''
    
    metrics = { # default values for failed/incomplete runs
        "solved": False,
        "generations": 1000, 
        "best_run_energy_pj": 0.0,
        "total_training_energy_uj": 0.0
    }
    
    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        return metrics

    with open(filepath, 'r') as f:
        content = f.read()

    # did it reach the flag?
    if "FLAG REACHED" in content:
        metrics["solved"] = True

    # extract the last generation run
    gen_matches = re.findall(r"\*+ Running generation (\d+) \*+", content)
    if gen_matches:
        metrics["generations"] = int(gen_matches[-1])

    # extract energy metrics from the last line
    # regex handles both ANN (with frames) and SNN (without frames) log formats
    best_energy_matches = re.findall(r"Best Run Energy:\s*([\d.]+)\s*pJ", content)
    if best_energy_matches:
        metrics["best_run_energy_pj"] = float(best_energy_matches[-1])
        
    total_energy_matches = re.findall(r"Total Training Energy So Far:\s*([\d.]+)\s*µJ", content)
    if total_energy_matches:
        metrics["total_training_energy_uj"] = float(total_energy_matches[-1])

    return metrics

def calculate_and_print_stats(name, logs):
    """Calculates mean, median, std for a group of logs"""
    data = [parse_log_file(log) for log in logs]
    
    gens = [d["generations"] for d in data]
    best_energy = [d["best_run_energy_pj"] for d in data]
    total_energy = [d["total_training_energy_uj"] for d in data]
    success_rate = sum([1 for d in data if d["solved"]]) / len(data) * 100

    print(f"\n{'='*40}")
    print(f" {name} RESULTS (N={len(logs)})")
    print(f"{'='*40}")
    print(f"Success Rate:             {success_rate:.1f}%")
    print(f"Generations to Solve:     Mean: {np.mean(gens):.1f} | Median: {np.median(gens):.1f} | Std: {np.std(gens):.1f}")
    print(f"Best Run Energy (pJ):     Mean: {np.mean(best_energy):.1f} | Median: {np.median(best_energy):.1f} | Std: {np.std(best_energy):.1f}")
    print(f"Total Train Energy (µJ):  Mean: {np.mean(total_energy):.1f} | Median: {np.median(total_energy):.1f} | Std: {np.std(total_energy):.1f}")
    
    return gens, best_energy, total_energy

# extract data
print("Extracting data from logs...")
snn_gens, snn_best_e, snn_total_e = calculate_and_print_stats("SPIKING NEAT (SNN)", snn_logs)
ann_gens, ann_best_e, ann_total_e = calculate_and_print_stats("STANDARD NEAT (ANN)", ann_logs)

# MANN-WHITNEY U TESTS (experimental, need to do more research on this)
def run_mwu(metric_name, snn_data, ann_data, alternative='less'):
    stat, p_value = mannwhitneyu(snn_data, ann_data, alternative=alternative)
    sig = "YES" if p_value < 0.05 else "NO"
    print(f"{metric_name:<25} | P-Value: {p_value:.5f} | Significant (p<0.05): {sig}")

print(f"\n{'='*40}")
print(" MANN-WHITNEY U TEST RESULTS")
print(f"{'='*40}")
print("Testing hypothesis: SNN requires LESS generations/energy than ANN\n")

run_mwu("Generations to Solve", snn_gens, ann_gens)
run_mwu("Best Run Energy (Inference)", snn_best_e, ann_best_e)
run_mwu("Total Training Energy", snn_total_e, ann_total_e)
print(f"{'='*40}\n")