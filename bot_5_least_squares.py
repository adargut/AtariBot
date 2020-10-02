import gym
import random
import numpy as np
from typing import Tuple
from typing import Callable

random.seed(0)
np.random.seed(0)


class QBot:
    """Bot playing Frozen Lake using least squares method"""

    def __init__(self, num_games=5000, name='Q bot', discount=0.85, lr=0.9, report_interval=500, w_lr=0.5):
        self.num_games = num_games
        self.name = name
        self.results = np.zeros(shape=num_games)
        self.games_played = 0
        self.discount = discount
        self.lr = lr
        self.w_lr = w_lr
        self.report_interval = report_interval
        self.states = []
        self.labels = []
        self.W = None
        self.Q = None
        self.q_table = None
        self.env = None
        self.seed = None
        self.n_obs = None
        self.n_actions = None

    def makeQ(self, model: np.array) -> Callable[[np.array], np.array]:
        """Returns a lambda function which maps state to distribution over actions"""

        return lambda x: x.dot(model)

    def initialize(self, shape: Tuple):
        """Initialize a model from normal distribution"""

        W = np.random.normal(0.0, 0.1, shape)
        Q = self.makeQ(W)

        return W, Q

    def train(self, x: np.array, y: np.array, W: np.array):
        """Train the model using the solution to regression"""

        identity = np.eye(x.shape[1])
        new_W = np.linalg.inv(x.T.dot(x) + 10e-4 * identity).dot(x.T.dot(y))
        W = new_W * self.w_lr + W * (1-self.w_lr)
        Q = self.makeQ(W)

        return W, Q

    def one_hot(self, i, n):
        """Implement one hot encoding for easier handling of data"""

        return np.identity(n)[i]

    def play(self, seed=0):
        """Let the bot play Frozen Lake once, and update the Q table"""

        if len(self.states) >= 1000:
            self.states, self.labels = [], []

        self.env.seed(seed)
        self.env.action_space.seed(seed)
        state = self.one_hot(self.env.reset(), self.n_obs)

        game_over = False
        game_reward = 0

        while not game_over:
            self.states.append(state)
            noise = np.random.random(size=(1, self.n_actions)) / (self.games_played + 1)
            action = np.argmax(self.Q(state) + noise)
            next_state, reward, game_over, _ = self.env.step(action)
            oh_next_state = self.one_hot(next_state, self.n_obs)
            q_target = reward + self.discount * np.max(self.Q(oh_next_state))
            label = self.Q(state)
            label[action] = (1-self.lr) * label[action] + self.lr * q_target
            self.labels.append(label)
            game_reward += reward
            state = oh_next_state

            if self.games_played % 10 == 0:
                self.W, self.Q = self.train(np.array(self.states), np.array(self.labels), self.W)

        if (self.games_played + 1) % self.report_interval == 0:
            self.display_results()

        self.results.put(self.games_played, game_reward)
        self.games_played += 1

    def simulate_bot(self):
        """Simulate the bot playing a couple of games"""

        self.env = gym.make(id='FrozenLake-v0')
        self.n_obs = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.W, self.Q = self.initialize(shape=(self.n_obs, self.n_actions))
        # self.q_table = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))

        for idx in range(self.num_games):
            self.play(seed=idx)

    def display_results(self):
        """Average results over all the games played"""

        print(self.name, 'achieved an average score of',
              np.sum(self.results) / self.games_played, 'over', self.games_played + 1, 'games')


def main():
    q_learning_agent = QBot()
    q_learning_agent.simulate_bot()


if __name__ == '__main__':
    main()
