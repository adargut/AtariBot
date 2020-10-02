import gym
import random
import numpy as np

random.seed(0)


class RandomBot:
    """Bot playing the Atari game at random.
    The idea is to compare the random bot to the trained bot for reference."""

    def __init__(self, num_games=10, name='Random Bot 2'):
        self.num_games = num_games
        self.name = name
        self.results = np.zeros(shape=num_games)
        self.games_played = 0
        self.env = None
        self.seed = None

    def _play(self, seed=0):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.reset()

        game_over = False
        game_reward = 0

        while not game_over:
            action = self.env.action_space.sample()
            _, reward, game_over, _ = self.env.step(action)
            game_reward += reward

        print(self.name, 'achieved a score of', game_reward, 'in game number', self.games_played + 1)
        self.results.put(self.games_played, game_reward)
        self.games_played += 1

    def simulate_bot(self):
        """Simulate the bot playing a couple of games"""
        self.env = gym.make(id='SpaceInvaders-v0')

        for idx in range(self.num_games):
            self._play(seed=idx)

    def display_results(self):
        """Average results over all the games played"""

        print(self.name, 'achieved an average score of',
              np.average(self.results), 'over', self.games_played, 'games')


def main():
    random_bot = RandomBot()
    random_bot.simulate_bot()
    random_bot.display_results()


if __name__ == '__main__':
    main()
