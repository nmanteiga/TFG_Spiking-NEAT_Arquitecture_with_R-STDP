import csv, glob, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
# python3 stats.py -> level 1-1 (results/);  python3 stats.py 1-2 -> transfer results (results/level_1-2/)
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "1-1"
RES = os.path.join(HERE, "results") if LEVEL == "1-1" else os.path.join(HERE, "results", f"level_{LEVEL}")
GROUPS = {"snn": "Spiking NEAT (SNN)", "ann": "Standard NEAT (ANN)"}
ENERGY_VARIANT = "default"
ALPHA = 0.05


def load(kind):
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, f"{kind}_seed_*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def col(runs, getter):
    out = []
    for r in runs:
        try:
            v = getter(r)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(float(v))
    return np.array(out)


def describe(x):
    if len(x) == 0:
        return "n/a"
    sd = np.std(x, ddof=1) if len(x) > 1 else float("nan")        # sample std (n is small)
    return f"mean {np.mean(x):.4g} | median {np.median(x):.4g} | sd {sd:.3g} (n={len(x)})"


def cliffs_delta(x, y):
    """>0: x tends to be larger than y. |d|<0.147 negligible, <0.33 small, <0.474 medium, else large."""
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def compare(name, x, y, alternative="two-sided"):
    if len(x) < 2 or len(y) < 2:
        return None
    p = mannwhitneyu(x, y, alternative=alternative).pvalue
    return {"metric": name, "p": p, "delta": cliffs_delta(x, y), "alt": alternative,
            "sig": p < ALPHA}


METRICS = {   # name: (getter, unit)
    "completed (0/1)":           (lambda r: float(r["completed"]), ""),
    "distance max_x":            (lambda r: r["max_x"], "px"),
    "frames":                    (lambda r: r["frames"], ""),
    "energy / frame":            (lambda r: r["energy_per_frame_pj"][ENERGY_VARIANT], "pJ"),
    "energy / episode":          (lambda r: r["energy"][ENERGY_VARIANT]["total_pj"], "pJ"),
    "connections":               (lambda r: r["n_connections"], ""),
    "training energy (estimated)": (lambda r: r.get("training_kwh"), "kWh"),
}


def main():
    data = {k: load(k) for k in GROUPS}
    for k, runs in data.items():
        print(f"\n{'=' * 70}\n {GROUPS[k]}  (N={len(runs)})\n{'=' * 70}")
        if not runs:
            print(" no results found"); continue
        for name, (g, unit) in METRICS.items():
            print(f" {name:<27} {describe(col(runs, g))} {unit}")
        solved = [r.get("solved") for r in runs if "solved" in r]
        gens = col([r for r in runs if r.get("solved")], lambda r: r["generations"])
        if solved:
            print(f" {'success (reached threshold)':<27} {sum(bool(s) for s in solved)}/{len(solved)}")
            print(f" {'generations to solve':<27} {describe(gens)}   <- solved runs only (unsolved are censored)")
        if k == "snn":
            print(f" {'input density (spikes/pixel)':<27} {describe(col(runs, lambda r: r['input_density']))}")
            for part in ("synop_pj", "update_pj", "encoding_pj", "plasticity_pj"):
                print(f"   energy part {part:<14} {describe(col(runs, lambda r, p=part: r['energy'][ENERGY_VARIANT][p]))}")
        else:
            for part in ("mac_pj", "bias_pj", "act_pj"):
                print(f"   energy part {part:<14} {describe(col(runs, lambda r, p=part: r['energy'][ENERGY_VARIANT][p]))}")

    s, a = data["snn"], data["ann"]
    if not s or not a:
        return

    print(f"\n{'=' * 70}\n SUCCESS CRITERIA OF THE MEMORIA (SNN vs ANN means)\n{'=' * 70}")
    def ratio(g):
        xs, ya = col(s, g), col(a, g)
        return np.mean(xs) / np.mean(ya) if len(xs) and len(ya) and np.mean(ya) else float("nan")
    crit = [("completion rate >= 90%", ratio(METRICS["completed (0/1)"][0]), lambda v: v >= 0.90),
            ("distance        >= 90%", ratio(METRICS["distance max_x"][0]), lambda v: v >= 0.90),
            ("frames          <= 110%", ratio(METRICS["frames"][0]), lambda v: v <= 1.10)]
    for label, v, ok in crit:
        print(f" {label:<26} ratio = {v:6.3f}   {'PASS' if ok(v) else 'FAIL'}")

    print(f"\n{'=' * 70}\n MANN-WHITNEY U (two-sided) + Cliff's delta   (n={len(s)} vs {len(a)}: low power)\n{'=' * 70}")
    rows = []
    for name in ("distance max_x", "frames", "energy / frame", "energy / episode"):
        c = compare(name, col(s, METRICS[name][0]), col(a, METRICS[name][0]))
        if c:
            rows.append(c)
            print(f" {name:<20} p = {c['p']:.4f}  delta = {c['delta']:+.2f}  {'significant' if c['sig'] else 'not significant'}")
    print(" (delta < 0: SNN lower than ANN.  With 5 vs 5 runs the smallest possible two-sided p is ~0.008.)")

    sens = sorted({v for r in s + a for v in r["energy"]})
    print(f"\n{'=' * 70}\n ENERGY / FRAME SENSITIVITY TO THE MODEL ASSUMPTIONS (mean pJ/frame)\n{'=' * 70}")
    print(f" {'assumption set':<22}{'SNN':>12}{'ANN':>12}{'SNN/ANN':>10}")
    for v in sens:
        gs = lambda r, v=v: r["energy_per_frame_pj"][v]
        xs, ya = col(s, gs), col(a, gs)
        if len(xs) and len(ya):
            print(f" {v:<22}{np.mean(xs):12.1f}{np.mean(ya):12.1f}{np.mean(xs) / np.mean(ya):10.3f}")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "metric", "n", "mean", "median", "sd"])
        for k, runs in data.items():
            for name, (g, _) in METRICS.items():
                x = col(runs, g)
                if len(x):
                    w.writerow([k, name, len(x), np.mean(x), np.median(x), np.std(x, ddof=1) if len(x) > 1 else ""])
    with open(os.path.join(RES, "summary_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrr}\n\\toprule\nMetric & SNN & ANN \\\\\n\\midrule\n")
        for name in ("completed (0/1)", "distance max_x", "frames", "energy / frame", "energy / episode"):
            cells = []
            for runs in (s, a):
                x = col(runs, METRICS[name][0])
                cells.append(f"${np.mean(x):.3g} \\pm {np.std(x, ddof=1):.2g}$" if len(x) > 1 else "--")
            f.write(f"{name.replace('_', ' ').replace('/', '/')} & {cells[0]} & {cells[1]} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nWritten: {RES}/summary.csv and summary_table.tex")


if __name__ == "__main__":
    main()