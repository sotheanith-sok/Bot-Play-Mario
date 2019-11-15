#Code for program goes here
import retro
import tensorflow as tf
import numpy as np

def get_initial_data():
    env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland1")
    observations = env.reset()
    while True:
        observations, rewards, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            print("Complete")
            obs = env.reset()
    env.close()

get_initial_data()