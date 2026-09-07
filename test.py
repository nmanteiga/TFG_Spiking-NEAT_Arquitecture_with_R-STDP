import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import os
import neat
import numpy as np
import cv2          # to shrink the screen
import pickle
import datetime  

# this is only so mac does not crash 
# env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human') # so it renders the machine playing
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def preprocess_state(state):
    # grayscale
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # 13x13 grid
    resized = cv2.resize(gray, (13, 13))
    # flatten it into a 1D array of 169 numbers and normalize (0 to 1)
    return resized.flatten() / 255.0


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

        while not done:
            inputs = preprocess_state(state)
            output = net.activate(inputs)
            action = np.argmax(output) 
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            genome.fitness += reward

            if info['flag_get']:
                done = True
                print("Mario hit the flagpole!")

            # anti-stuck conditions
            if info['x_pos'] > current_max_x:
                current_max_x = info['x_pos']
                stuck_count = 0
            else:
                stuck_count += 1
                
            if stuck_count > 250:
                done = True 
            # env.render()



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)
    population = neat.Population(config)

    # to save the results on logs
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(local_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, ".keep"), "w") as f:
        f.write("do not delete")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    checkpoint_prefix = os.path.join(log_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix))
    print(f"Training started. Logs and backups saving to: {log_dir}")
    winner = population.run(eval_genomes, 150)

    winner_path = os.path.join(log_dir, "winner.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
        
    print(f"Training complete. Best genome saved to {winner_path}.")
    env.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-MarI-O_NEAT.txt")
    run(config_path)