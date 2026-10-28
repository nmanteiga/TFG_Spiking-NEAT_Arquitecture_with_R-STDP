"""
eval.py -- replays the winner of one training run and writes results/<kind>_seed_<seed>.json

    python3 eval.py <kind> <seed> <run folder> [level]
    python3 eval.py snn 101 logs/seed_101_2026-10-20_12-00-00
    python3 eval.py snn 101 logs/seed_101_2026-10-20_12-00-00 1-2     # transfer -> results/level_1-2/

kinds: ann, snn (train_baseline.py / train_spiking.py)
       rstdp, rstdp_lamarck, snn_potential (train_rstdp.py: plain, --lamarckian, --no-plasticity)

the evaluation is one deterministic game (no random do-nothing start), the same for every group. R-STDP groups keep
learning during it, starting from the winner's genome weights, exactly as they did in training.
"""
import json, os, pickle, sys
import neat
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import energy_model as em
from preprocessing_module import preprocess_state

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {"snn": "config-spiking.txt", "ann": "config-MarI-O_NEAT.txt",
           "rstdp": "config-rstdp.txt", "rstdp_lamarck": "config-rstdp.txt", "snn_potential": "config-rstdp.txt"}
SPIKING = {"snn", "rstdp", "rstdp_lamarck", "snn_potential"}
RSTDP_GENOME = {"rstdp", "rstdp_lamarck", "snn_potential"}     # trained with models/rstdp_genome.py
PLASTIC = {"rstdp", "rstdp_lamarck"}                           # learn while playing
BY_POTENTIAL = {"rstdp", "rstdp_lamarck", "snn_potential"}     # button = most charged output (not argmax of spikes)


def run_episode(net, level, by_potential=False, learn=False, max_stuck=250):
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"), COMPLEX_MOVEMENT)
    state, info = env.reset()
    frames, max_x, stuck, done, completed = 0, 0, 0, False, False
    while not done:
        output = net.activate(preprocess_state(state))
        action = int(np.argmax(net.potential_outputs if by_potential else output))
        state, reward, terminated, truncated, info = env.step(action)
        if learn:
            net.reward(reward, action)
        frames += 1
        done = terminated or truncated
        if info["flag_get"]:
            completed, done = True, True
        if info["x_pos"] > max_x:
            max_x, stuck = info["x_pos"], 0
        else:
            stuck += 1
        if stuck > max_stuck:
            done = True
    env.close()
    return completed, int(max_x), frames


def main(kind, seed, log_dir, level="1-1"):
    if kind in RSTDP_GENOME:
        from models.rstdp_genome import RSTDPGenome as Genome
    else:
        Genome = neat.DefaultGenome
    config = neat.config.Config(Genome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, os.path.join(HERE, CONFIGS[kind]))
    with open(os.path.join(log_dir, "winner.pkl"), "rb") as f:
        genome = pickle.load(f)

    n_nodes = len(genome.nodes)
    enabled = [k for k, c in genome.connections.items() if c.enabled]
    n_conn, n_conn_in = len(enabled), sum(1 for (i, _) in enabled if i < 0)

    if kind in SPIKING:
        from models.spiking_network import SpikingNetwork
        if kind in RSTDP_GENOME:
            net = SpikingNetwork(genome, config, plastic=kind in PLASTIC, learning_rate=genome.learning_rate)
        else:
            net = SpikingNetwork(genome, config)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)

    completed, max_x, frames = run_episode(net, level, by_potential=kind in BY_POTENTIAL, learn=kind in PLASTIC)

    res = {"kind": kind, "seed": seed, "level": level, "completed": completed, "max_x": max_x, "frames": frames,
           "n_nodes": n_nodes, "n_connections": n_conn, "fitness_in_training": genome.fitness}

    variants = {"default": em.DEFAULT,
                "act5": em.Assumptions(act_cost_in_macs=5),
                "act10": em.Assumptions(act_cost_in_macs=10),
                "ann_binary_aware": em.Assumptions(ann_binary_inputs_as_adds=True),
                "encoding_not_shared": em.Assumptions(binarisation_shared=False),
                "plasticity_2mac": em.Assumptions(e_plasticity=2 * em.E_MAC_PJ)}
    if kind in SPIKING:
        res.update(synops=net.synaptic_operations, spikes=net.total_spikes, neuron_updates=net.neuron_updates,
                   input_spikes=net.input_spikes, plasticity_updates=net.plasticity_updates,
                   trace_updates=net.trace_updates,
                   input_density=net.input_spikes / max(1, frames * net.n_inputs))
        res["energy"] = {n: em.snn_energy(frames, net.synaptic_operations, net.neuron_updates, net.n_inputs,
                                          plasticity_updates=net.plasticity_updates, a=a,
                                          trace_updates=net.trace_updates) for n, a in variants.items()}
    else:
        res["energy"] = {n: em.ann_energy(frames, n_conn, n_nodes, n_conn_in, a) for n, a in variants.items()}
    res["energy_per_frame_pj"] = {n: e["total_pj"] / max(1, frames) for n, e in res["energy"].items()}

    if kind in RSTDP_GENOME:
        res["learning_rate"] = genome.learning_rate
    if kind in PLASTIC:
        # how much the weights moved during this game (genome weights -> weights at the end of the game)
        changes = [abs(w - genome.connections[(i, o)].weight) for i, o, w in net.connections]
        res.update(weights_changed=sum(c > 0 for c in changes),
                   mean_abs_weight_change=float(np.mean(changes)) if changes else 0.0,
                   max_abs_weight_change=float(max(changes, default=0.0)))

    summary = os.path.join(log_dir, "run_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            res.update({k: v for k, v in json.load(f).items() if k not in res})

    # 1-1 (training level) -> results/, any other level (transfer) -> results/level_<level>/
    res_dir = os.path.join(HERE, "results") if level == "1-1" else os.path.join(HERE, "results", f"level_{level}")
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"{kind}_seed_{seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"{kind} seed {seed}: completed={completed} max_x={max_x} frames={frames} "
          f"E={res['energy']['default']['total_pj']:.0f} pJ ({res['energy_per_frame_pj']['default']:.1f} pJ/frame) -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] not in CONFIGS:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], *(sys.argv[4:5] or ["1-1"]))
