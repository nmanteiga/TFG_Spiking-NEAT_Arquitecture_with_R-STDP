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

MAX_GENERATIONS = 300     
KIND = "ann"

# this is only so mac does not crash 
# env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human') # so it renders the machine playing
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)


def eval_genomes(genomes, config):
    # loop through every genome in the population 
    for genome_id, genome in genomes:    
        # create the Recurrent Neural Network for this specific genome
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        # reset the game and the fitness for this Mario
        state, info = env.reset()
        genome.fitness = 0
        done = False
        
        # to track if mario is stuck against a pipe
        current_max_x = 0
        stuck_count = 0

        # to track energy efficiency
        frame_counter = 0

        while not done:
            inputs = preprocess_state(state)
            output = net.activate(inputs)
            action = np.argmax(output) 
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            genome.fitness += reward
            frame_counter += 1

            if info['flag_get']:
                done = True
                genome.fitness += 100000  # beats the 50,000 threshold immediately
                print(f"\n>>> FLAG REACHED by Genome {genome_id}! Ending training after this generation. <<<\n")

            # anti-stuck conditions
            if info['x_pos'] > current_max_x:
                current_max_x = info['x_pos']
                stuck_count = 0
            else:
                stuck_count += 1
                
            if stuck_count > 250:
                done = True 
            # env.render()
    
        genome.frames = frame_counter
    
    best = max(genomes, key=lambda x: x[1].fitness)[1]
    print(f"Gen Best -> Fitness: {best.fitness:.1f} | Frames: {best.frames}")
    


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
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
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
               "training_kwh": kwh, "training_kg_co2eq": kg}
    with open(os.path.join(log_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Run summary:", summary)
    env.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    #config_path = os.path.join(local_dir, "config-MarI-O_NEAT.txt")
    #run(config_path)

    ''' used for testing and comparison purpouses '''
    # grab the seed from the terminal command, default to 42
    seed_value = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    # forces python and numpy to use this exact seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    print(f"--- STARTING RUN WITH RANDOM SEED: {seed_value} ---")
    
    config_path = os.path.join(local_dir, "config-MarI-O_NEAT.txt") 
    run(config_path, seed_value)