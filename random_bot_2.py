import gym
import random

random.seed(0)


def main():

    # Create gym

    env = gym.make(id='SpaceInvaders-v0')

    # Make the gym determinstic

    env.seed(0)
    env.action_space.seed(0)
    env.reset()

    # Initialize the game

    game_over = False
    total_reward = 0

    # Play until bot is disqualified

    while not game_over:
        action = env.action_space.sample()
        _, reward, game_over, _ = env.step(action)
        total_reward += reward

    # Print out total score of bot

    print('Random Bot 2 achieved score of: ', total_reward)


if __name__ == '__main__':
    main()
