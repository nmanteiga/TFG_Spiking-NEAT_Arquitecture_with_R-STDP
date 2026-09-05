import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import os
import neat
import numpy as np
import cv2
import pickle

env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (13, 13))
    return resized.flatten() / 255.0

def replay_genome(config_path, genome_path):
    # load the NEAT configuration
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # load the winning brain from the .pkl file
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print(f"Loaded Genome with Fitness: {genome.fitness}")

    net = neat.nn.RecurrentNetwork.create(genome, config)

    # start the game
    state, info = env.reset()
    done = False

    while not done:
        # mario looks at the screen
        inputs = preprocess_state(state)
        
        # the brain decides what buttons to press
        output = net.activate(inputs)
        action = np.argmax(output)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()

    env.close()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-MarI-O_NEAT.txt")
    
    # paste the exact folder name from your terminal
    # f.e.: "logs/2026-09-04_16-55-50/winner.pkl"
    winner_path = os.path.join(local_dir, "logs", "2026-09-04_16-59-02", "winner.pkl")

    replay_genome(config_path, winner_path)