import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import os
import neat
import numpy as np
import cv2
import pickle
import random

# config
SEED_TO_REPLAY = 202
IS_SPIKING = True  # True if replaying a SNN
TIMESTAMP_FOLDER = "2026-09-19_11-03-24" # folder name here
# ----------------------------------

# force deterministic random state
random.seed(SEED_TO_REPLAY)
np.random.seed(SEED_TO_REPLAY)

# rgb_array to prevent macos window frame drops
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def preprocess_state(state):
    # MUST perfectly match the binary logic from training
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (13, 13))
    flattened = resized.flatten() / 255.0
    return np.where(flattened > 0.5, 1.0, 0.0)

def replay_winner(config_path, winner_path):
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)

    if IS_SPIKING:
        from models.spiking_network import SpikingNetwork
        net = SpikingNetwork(winner, config)
    else:
        net = neat.nn.RecurrentNetwork.create(winner, config)

    state, info = env.reset()
    done = False
    
    print(f"Replaying Champion for Seed {SEED_TO_REPLAY}...")

    while not done:
        # cv2 to show the frame safely without desyncing the emulator
        cv2.imshow('Mario Replay', cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        cv2.waitKey(8) # replay speed 

        inputs = preprocess_state(state)
        output = net.activate(inputs)
        action = np.argmax(output) 
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info['flag_get']:
            print(">>> Mario hit the flagpole in replay<<<")
            cv2.waitKey(2000)
            done = True

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = "config-spiking.txt" if IS_SPIKING else "config-MarI-O_NEAT.txt"
    config_path = os.path.join(local_dir, config_file)
    winner_path = os.path.join(local_dir, "logs", TIMESTAMP_FOLDER, "winner.pkl") 
    
    replay_winner(config_path, winner_path)