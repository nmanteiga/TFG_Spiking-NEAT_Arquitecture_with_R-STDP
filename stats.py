import csv, glob, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
# python3 stats.py -> level 1-1 (results/);  python3 stats.py 1-2 -> transfer results (results/level_1-2/)
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "1-1"
RES = os.path.join(HERE, "results") if LEVEL == "1-1" else os.path.join(HERE, "results", f"level_{LEVEL}")
GROUPS = {"snn": "Spiking NEAT (SNN)",
          "ann": "Standard NEAT (ANN)",
          "snn_potential": "SNN, potential-based action, no learning (control)",
          "rstdp_pilot": "SNN + R-STDP, pilot (weak learning-rate range)",
          "rstdp_lamarck_pilot": "SNN + R-STDP, Lamarckian, pilot (saved the learned weights)",
          "rstdp": "SNN + R-STDP",
          "rstdp_lamarck": "SNN + R-STDP, Lamarckian (ablation)"}
SHORT = {"snn": "SNN", "ann": "ANN", "snn_potential": "Control", "rstdp_pilot": "R-STDP pilot",
         "rstdp_lamarck_pilot": "Lamarck pilot", "rstdp": "R-STDP", "rstdp_lamarck": "Lamarckian"}
# groups that share the eval.py kind (file name) and are told apart by seed: pilot on alice (Oct 2026) vs batch 2
SEEDS = {"rstdp_pilot": ("rstdp", range(301, 306)), "rstdp_lamarck_pilot": ("rstdp_lamarck", range(401, 406)),
         "rstdp": ("rstdp", range(601, 611)), "rstdp_lamarck": ("rstdp_lamarck", range(701, 706))}
SPIKING = {"snn", "snn_potential", "rstdp_pilot", "rstdp_lamarck_pilot", "rstdp", "rstdp_lamarck"}
PLASTIC = {"rstdp_pilot", "rstdp_lamarck_pilot", "rstdp", "rstdp_lamarck"}
BASELINE = "ann"
# pairs compared with Mann-Whitney (only when both groups have results), and the question each one answers
PAIRS = [("snn", "ann", "spiking vs conventional"),
         ("rstdp", "ann", "R-STDP vs conventional"),
         ("rstdp", "snn_potential", "effect of learning (same action choice, with vs without R-STDP)"),
         ("snn_potential", "snn", "effect of the action choice (potential vs argmax of spikes)"),
         ("rstdp_lamarck", "rstdp", "Lamarckian vs Baldwinian"),
         ("rstdp", "rstdp_pilot", "stronger learning-rate range vs pilot"),
         ("rstdp_pilot", "ann", "R-STDP pilot vs conventional"),
         ("rstdp_pilot", "snn_potential", "R-STDP pilot vs control")]
ENERGY_VARIANT = "default"
ALPHA = 0.05


def load(group):
    kind, seeds = SEEDS.get(group, (group, None))
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, f"{kind}_seed_*.json"))):
        with open(p) as f:
            r = json.load(f)
        if seeds is None or r["seed"] in seeds:
            runs.append(r)
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
    "completion, 31 starts":     (lambda r: r["starts_completion_rate"], ""),     # robustness (eval.py STARTS)
    "distance, 31 starts":       (lambda r: r["starts_mean_max_x"], "px"),
    "frames":                    (lambda r: r["frames"], ""),
    "energy / frame":            (lambda r: r["energy_per_frame_pj"][ENERGY_VARIANT], "pJ"),
    "energy / episode":          (lambda r: r["energy"][ENERGY_VARIANT]["total_pj"], "pJ"),
    "connections":               (lambda r: r["n_connections"], ""),
    "training energy (estimated)": (lambda r: r.get("training_kwh"), "kWh"),
}
RSTDP_METRICS = {   # only for the groups that learn while playing
    "evolved learning rate":     (lambda r: r["learning_rate"], ""),
    "plasticity ops / frame":    (lambda r: r["plasticity_updates"] / r["frames"], ""),
    "trace ops / frame":         (lambda r: r["trace_updates"] / r["frames"], ""),
    "synapses changed (game)":   (lambda r: r["weights_changed"], ""),
    "mean |weight change|":      (lambda r: r["mean_abs_weight_change"], ""),
    "completion, learning off":  (lambda r: r["starts_no_learning_completion_rate"], ""),   # learning ablation
    "distance, learning off":    (lambda r: r["starts_no_learning_mean_max_x"], "px"),
}
# learning ablation: the same winner with vs without R-STDP (paired, Wilcoxon signed-rank)
ABLATION = [("completion, 31 starts", "completion, learning off"),
            ("distance, 31 starts", "distance, learning off")]


def main():
    data = {k: load(k) for k in GROUPS}
    for k, runs in data.items():
        if not runs and k not in ("snn", "ann"):
            continue                                    # R-STDP groups not run yet: stay quiet
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
        if k in PLASTIC:
            for name, (g, unit) in RSTDP_METRICS.items():
                print(f" {name:<27} {describe(col(runs, g))} {unit}")
        if k in SPIKING:
            print(f" {'input density (spikes/pixel)':<27} {describe(col(runs, lambda r: r['input_density']))}")
            for part in ("synop_pj", "update_pj", "encoding_pj", "plasticity_pj", "trace_pj"):
                print(f"   energy part {part:<14} {describe(col(runs, lambda r, p=part: r['energy'][ENERGY_VARIANT][p]))}")
        else:
            for part in ("mac_pj", "bias_pj", "act_pj"):
                print(f"   energy part {part:<14} {describe(col(runs, lambda r, p=part: r['energy'][ENERGY_VARIANT][p]))}")

    present = [k for k in GROUPS if data[k]]
    a = data[BASELINE]
    if not a or len(present) < 2:
        return

    def ratio(x_runs, g):
        xs, ya = col(x_runs, g), col(a, g)
        return np.mean(xs) / np.mean(ya) if len(xs) and len(ya) and np.mean(ya) else float("nan")

    for k in present:
        if k not in SPIKING:
            continue
        print(f"\n{'=' * 70}\n SUCCESS CRITERIA OF THE MEMORIA ({SHORT[k]} vs ANN means)\n{'=' * 70}")
        crit = [("completion rate >= 90%", ratio(data[k], METRICS["completed (0/1)"][0]), lambda v: v >= 0.90),
                ("distance        >= 90%", ratio(data[k], METRICS["distance max_x"][0]), lambda v: v >= 0.90),
                ("frames          <= 110%", ratio(data[k], METRICS["frames"][0]), lambda v: v <= 1.10)]
        for label, v, ok in crit:
            print(f" {label:<26} ratio = {v:6.3f}   {'PASS' if ok(v) else 'FAIL'}")

    for x_kind, y_kind, question in PAIRS:
        x_runs, y_runs = data[x_kind], data[y_kind]
        if len(x_runs) < 2 or len(y_runs) < 2:
            continue
        print(f"\n{'=' * 70}\n MANN-WHITNEY U + Cliff's delta: {SHORT[x_kind]} vs {SHORT[y_kind]}   (n={len(x_runs)} vs {len(y_runs)}: low power)"
              f"\n {question}\n{'=' * 70}")
        for name in ("distance max_x", "completion, 31 starts", "distance, 31 starts", "frames", "energy / frame",
                     "energy / episode"):
            c = compare(name, col(x_runs, METRICS[name][0]), col(y_runs, METRICS[name][0]))
            if c:
                print(f" {name:<22} p = {c['p']:.4f}  delta = {c['delta']:+.2f}  {'significant' if c['sig'] else 'not significant'}")
        print(f" (delta < 0: {SHORT[x_kind]} lower than {SHORT[y_kind]}.  With 5 vs 5 runs the smallest possible two-sided p is ~0.008.)")

    for k in present:
        if k not in PLASTIC:
            continue
        runs = [r for r in data[k] if "starts_no_learning" in r]
        if len(runs) < 2:
            continue
        print(f"\n{'=' * 70}\n LEARNING ABLATION: {SHORT[k]}, same winner with vs without R-STDP   (n={len(runs)}, paired)"
              f"\n{'=' * 70}")
        for on_name, off_name in ABLATION:
            on = col(runs, METRICS[on_name][0])
            off = col(runs, RSTDP_METRICS[off_name][0])
            diff = on - off
            p = wilcoxon(on, off).pvalue if np.any(diff != 0) else 1.0
            print(f" {on_name:<22} on {np.mean(on):8.4g} | off {np.mean(off):8.4g} | better with learning in "
                  f"{int(np.sum(diff > 0))}/{len(runs)} runs, worse in {int(np.sum(diff < 0))}  "
                  f"p = {p:.4f}  {'significant' if p < ALPHA else 'not significant'}")
        print(f" (Wilcoxon signed-rank, two-sided. with n=5 the smallest possible p is 0.0625, so n=10 is needed)")

    sens = sorted({v for k in present for r in data[k] for v in r["energy"]})
    print(f"\n{'=' * 70}\n ENERGY / FRAME SENSITIVITY TO THE MODEL ASSUMPTIONS (mean pJ/frame, ratio to ANN)\n{'=' * 70}")
    print(f" {'assumption set':<22}" + "".join(f"{SHORT[k]:>18}" for k in present))
    for v in sens:
        gs = lambda r, v=v: r["energy_per_frame_pj"][v]
        ya = col(a, gs)
        cells = []
        for k in present:
            xs = col(data[k], gs)
            if not len(xs):
                cells.append(f"{'--':>18}")
            elif k == BASELINE or not len(ya):
                cells.append(f"{np.mean(xs):18.1f}")
            else:
                cells.append(f"{np.mean(xs):10.1f} ({np.mean(xs) / np.mean(ya):5.2f})")
        print(f" {v:<22}" + "".join(cells))

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "metric", "n", "mean", "median", "sd"])
        for k, runs in data.items():
            metrics = {**METRICS, **(RSTDP_METRICS if k in PLASTIC else {})}
            for name, (g, _) in metrics.items():
                x = col(runs, g)
                if len(x):
                    w.writerow([k, name, len(x), np.mean(x), np.median(x), np.std(x, ddof=1) if len(x) > 1 else ""])
    with open(os.path.join(RES, "summary_table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(present) + "}\n\\toprule\nMetric & "
                + " & ".join(SHORT[k] for k in present) + " \\\\\n\\midrule\n")
        for name in ("completed (0/1)", "distance max_x", "frames", "energy / frame", "energy / episode"):
            cells = []
            for k in present:
                x = col(data[k], METRICS[name][0])
                cells.append(f"${np.mean(x):.3g} \\pm {np.std(x, ddof=1):.2g}$" if len(x) > 1 else "--")
            f.write(f"{name.replace('_', ' ')} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nWritten: {RES}/summary.csv and summary_table.tex")


if __name__ == "__main__":
    main()
