"""
Space Invaders bot - Fully featured deep q-learning network
"""

import cv2
import gym
import numpy as np
import random
import tensorflow
from space_invaders.space_invaders_bot import a3c_model

tf = tensorflow.compat.v1

np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)

num_games = 10
report_interval = 1


def resize_image(state, dsize=(84, 84)):
    """Model was trained to handle 84x84 images, therefore resizing is needed"""

    return cv2.resize(state, dsize=dsize, interpolation=cv2.INTER_LINEAR)[None]


def display_results(rewards):
    """Measure the results of our model"""

    print("Average reward: ", sum(rewards) / len(rewards))


def main():
    env = gym.make("SpaceInvaders-v0")
    env.seed(42)

    rewards = []
    model = a3c_model(load='../models/SpaceInvaders-v0.tfmodel')

    for _ in range(num_games):
        game_reward = 0
        states = [resize_image(env.reset())]
        game_over = False

        while not game_over:
            if len(states) < 4:
                action = env.action_space.sample()
            else:
                frames = np.concatenate(states[-4:], axis=3)
                action = np.argmax(model([frames]))

            state, step_reward, game_over, _ = env.step(action=action)
            env.render()
            states.append(resize_image(state))
            game_reward += step_reward

        rewards.append(game_reward)

        if len(rewards) % report_interval == 0:
            display_results(rewards)


if __name__ == '__main__':
    main()
