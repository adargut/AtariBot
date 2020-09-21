import gym
import random
import numpy as np
from typing import List

random.seed(0)
np.random.seed(0)


class QBot:
    """Bot playing Frozen Lake with q-table (reinforcement learning)"""

    def __init__(self, num_games=400, name='Q bot', discount=0.8, lr=0.9):
        self.num_games = num_games
        self.name = name
        self.results = np.zeros(shape=num_games)
        self.games_played = 0
        self.discount = discount
        self.lr = lr
        self.q_table = None
        self.env = None
        self.seed = None

    def _play(self, seed=0):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        state = self.env.reset()

        game_over = False
        game_reward = 0

        while not game_over:
            noise = np.random.random(size=(1, self.env.action_space.n)) / (self.games_played + 1 ** 2)
            action = np.argmax(self.q_table[state] + noise)
            next_state, reward, game_over, _ = self.env.step(action)
            q_target = reward + self.discount * np.max(self.q_table[next_state])
            self.q_table[state, action] = (1 - self.lr) * self.q_table[state, action] + self.lr * q_target
            game_reward += reward
            state = next_state

        print(self.name, 'achieved a score of', game_reward, 'in game number', self.games_played + 1)
        self.results.put(self.games_played, game_reward)
        self.games_played += 1

    def simulate_bot(self):
        """Simulate the bot playing a couple of games"""

        self.env = gym.make(id='FrozenLake-v0')
        self.q_table = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))

        for idx in range(self.num_games):
            self._play(seed=idx)

    def display_results(self):
        """Average results over all the games played"""

        print(self.name, 'achieved an average score of',
              np.average(self.results), 'over', self.games_played, 'games')


def main():
    q_learning_agent = QBot()
    q_learning_agent.simulate_bot()


if __name__ == '__main__':
    main()
