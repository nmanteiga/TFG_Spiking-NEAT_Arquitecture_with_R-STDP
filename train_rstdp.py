import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import os
import neat
import numpy as np
import pickle
import datetime
import sys
import random
import csv, json, time
from codecarbon import OfflineEmissionsTracker

# import the preprocessing module
from preprocessing_module import preprocess_state

# import the spiking neural network and the genome with the evolved learning rate
from models.spiking_network import SpikingNetwork
from models.rstdp_genome import RSTDPGenome

'''
same as train_spiking.py, plus R-STDP. the differences:
    - the network learns while it plays: after every frame it gets the game reward (net.reward)
    - the button is the output with the highest membrane potential (no ties, always an action)
    - each genome plays EPISODES games, each one after a random number of do-nothing frames (so the games differ),
      and its fitness is the average. the weights go back to the genome's values at the start of every game
    - the learning rate comes from the genome (evolved), see models/rstdp_genome.py and config-rstdp.txt

    python3 train_rstdp.py 301                    -> "rstdp"          main design (adapted weights are discarded)
    python3 train_rstdp.py 401 --lamarckian       -> "rstdp_lamarck"  ablation: adapted weights go back into the genome
    python3 train_rstdp.py 501 --no-plasticity    -> "snn_potential"  control: new action choice, no learning
'''

MAX_GENERATIONS = 1500
EPISODES = 2              # games per genome per generation (fitness = average)
MAX_NOOP_START = 30       # each game starts after 0..30 do-nothing frames (enemies in slightly different places)
NOOP = 0                  # COMPLEX_MOVEMENT[0] = ['NOOP']

LAMARCKIAN = "--lamarckian" in sys.argv
PLASTIC = "--no-plasticity" not in sys.argv
KIND = "snn_potential" if not PLASTIC else ("rstdp_lamarck" if LAMARCKIAN else "rstdp")

# this is only so mac does not crash
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)


def play_episode(genome, config, genome_id):
    ''' one game: returns (fitness, frames, final network) '''
    # a NEW network every game: the weights start again from the genome's values (Baldwinian reset)
    net = SpikingNetwork(genome, config, plastic=PLASTIC, learning_rate=genome.learning_rate)

    state, info = env.reset()
    # random do-nothing start (the network is not running yet, mario just stands still)
    for _ in range(random.randint(0, MAX_NOOP_START)):
        state, reward, terminated, truncated, info = env.step(NOOP)

    fitness = 0
    done = False

    # to track if mario is stuck against a pipe
    current_max_x = 0
    stuck_count = 0

    # to track energy efficiency
    frame_counter = 0

    while not done:
        inputs = preprocess_state(state)
        net.activate(inputs)
        action = int(np.argmax(net.potential_outputs))         # the most charged output (button)
        state, reward, terminated, truncated, info = env.step(action)
        net.reward(reward, action)                              # R-STDP: learn from this frame's reward
        done = terminated or truncated
        fitness += reward
        frame_counter += 1

        if info['flag_get']:
            done = True
            fitness += 100000  # beats the 50k threshold (as the average of EPISODES games: if reached in any of them)
            print(f"\n>>> FLAG REACHED by Genome {genome_id}! Ending training after this generation. <<<\n")

        # anti-stuck conditions
        if info['x_pos'] > current_max_x:
            current_max_x = info['x_pos']
            stuck_count = 0
        else:
            stuck_count += 1

        if stuck_count > 250:
            done = True

    return fitness, frame_counter, net


def eval_genomes(genomes, config):
    # loop through every genome in the population
    for genome_id, genome in genomes:
        results = [play_episode(genome, config, genome_id) for _ in range(EPISODES)]

        genome.fitness = sum(f for f, _, _ in results) / EPISODES
        genome.frames = sum(n for _, n, _ in results)

        if LAMARCKIAN:
            # ablation: the weights learned in the last game are written back into the genome, so the children
            # inherit what this genome learned while playing
            last_net = results[-1][2]
            for in_node, out_node, weight in last_net.connections:
                genome.connections[(in_node, out_node)].weight = weight

    best = max(genomes, key=lambda x: x[1].fitness)[1]
    rates = [g.learning_rate for _, g in genomes]
    print(f"Gen Best -> Fitness: {best.fitness:.1f} | Frames: {best.frames} | Learning rate: {best.learning_rate:.4f}"
          f" | Population learning rate: mean {np.mean(rates):.4f}, zero in {sum(r == 0 for r in rates)}/{len(rates)}")


def read_codecarbon(log_dir):
    """Last row of the emissions.csv that CodeCarbon writes into log_dir -> (kWh, kg CO2eq)."""
    path = os.path.join(log_dir, "emissions.csv")
    if not os.path.exists(path):
        return None, None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    return float(rows[-1]["energy_consumed"]), float(rows[-1]["emissions"])


def run(config_path, seed_value):
    config = neat.config.Config(RSTDPGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(local_dir, "logs", f"seed_{seed_value}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(log_dir, "neat-checkpoint-")))

    # CodeCarbon footprint of the TRAINING run: measured via RAPL if readable, otherwise estimated from CPU load.
    # tracking_mode="process" only counts this run in load mode; RAPL always reports the whole CPU package
    tracker = OfflineEmissionsTracker(country_iso_code="ESP", project_name=f"{KIND}_seed_{seed_value}",
                                      output_dir=log_dir, measure_power_secs=5, log_level="warning",
                                      tracking_mode="process")
    t0, cpu0 = time.time(), time.process_time()
    tracker.start()
    try:
        winner = population.run(eval_genomes, MAX_GENERATIONS)
    finally:
        tracker.stop()
    train_seconds = time.time() - t0
    cpu_seconds = time.process_time() - cpu0      # to split a shared RAPL reading between parallel runs

    with open(os.path.join(log_dir, "winner.pkl"), "wb") as f:
        pickle.dump(winner, f)

    kwh, kg = read_codecarbon(log_dir)
    summary = {"kind": KIND, "seed": seed_value, "generations": population.generation,
               "solved": bool(winner.fitness >= config.fitness_threshold),
               "train_seconds": train_seconds, "cpu_seconds": cpu_seconds,
               "training_kwh": kwh, "training_kg_co2eq": kg,
               "plastic": PLASTIC, "lamarckian": LAMARCKIAN, "episodes": EPISODES, "max_noop_start": MAX_NOOP_START,
               "winner_learning_rate": winner.learning_rate}
    with open(os.path.join(log_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Run summary:", summary)
    env.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)

    ''' used for testing and comparison purpouses '''
    # grab the seed from the terminal command, default to 42
    seed_value = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 42
    # forces python and numpy to use this exact seed (also fixes the random do-nothing starts)
    random.seed(seed_value)
    np.random.seed(seed_value)

    print(f"--- STARTING {KIND} RUN WITH RANDOM SEED: {seed_value} ---")

    config_path = os.path.join(local_dir, "config-rstdp.txt")
    run(config_path, seed_value)
