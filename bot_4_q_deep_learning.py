import gym
import random
import numpy as np
import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)


class QBot:
    """Bot uses Q-learning neural network to train"""

    def __init__(self, num_games=4000, name='Q bot', discount=0.99, lr=0.15, report_interval=500,
                 exploration_probability=lambda episode: 50. / (episode + 10)):
        self.num_games = num_games
        self.name = name
        self.results = np.zeros(shape=num_games)
        self.games_played = 0
        self.discount = discount
        self.lr = lr
        self.report_interval = report_interval
        self.q_table = None
        self.env = None
        self.seed = None
        self.session = None
        self.exploration_probability = exploration_probability
        self.n_obs, self.n_actions = self.env.observation_space.n, self.env.action_space.n

        # Neural network architecture

        self.obs_t_ph = None
        self.obs_tp1_ph = None
        self.act_ph = None
        self.rew_ph = None
        self.q_target_ph = None
        self.W = None
        self.q_current = None
        self.q_target = None

        self.q_target_max = None
        self.q_target_sa = None
        self.q_current_sa = None
        self.error = None
        self.pred_act_ph = None

        self.trainer = None
        self.update_model = None

    @staticmethod
    def _one_hot(i: int, n: int) -> np.array:
        return np.identity(n)[i].reshape((1, -1))

    def _play(self, seed=0):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        obs_t = self.env.reset()

        game_over = False
        game_reward = 0

        while not game_over:
            obs_t_oh = QBot._one_hot(obs_t, self.n_obs)
            action = self.session.run(self.pred_act_ph, feed_dict={self.obs_t_ph: obs_t_oh})[0]
            if np.random.rand(1) < self.exploration_probability(episode=self.games_played):
                action = self.env.action_space.sample()

            obs_tp1, reward, game_over, _ = self.env.step(action)

            # Train network

            obs_tp1_oh = QBot._one_hot(obs_tp1, self.n_obs)
            q_target_val = self.session.run(self.q_target, feed_dict={self.obs_tp1_ph: obs_tp1_oh})
            self.session.run(self.update_model, feed_dict={
                self.obs_t_ph: obs_t_oh,
                self.rew_ph: reward,
                self.q_target_ph: q_target_val,
                self.act_ph: action
            })
            game_reward += reward
            obs_t = obs_tp1

        if (self.games_played + 1) % self.report_interval == 0:
            self.display_results()

        self.results.put(self.games_played, game_reward)
        self.games_played += 1

    def _set_up_neural_network(self):
        self.obs_t_ph = tf.placeholder(shape=(1, self.n_obs), dtype=tf.float32)
        self.obs_tp1_ph = tf.placeholder(shape=(1, self.n_obs), dtype=tf.float32)
        self.act_ph = tf.placeholder(tf.int32, shape=())
        self.rew_ph = tf.placeholder(shape=(), dtype=tf.float32)
        self.q_target_ph = tf.placeholder(shape=(1, self.n_actions), dtype=tf.float32)
        self.W = tf.Variable(tf.random.uniform(shape=(self.n_obs, self.n_actions), maxval=0.01))
        self.q_current = tf.matmul(self.obs_t_ph, self.W)
        self.q_target = tf.matmul(self.obs_tp1_ph, self.W)

        self.q_target_max = tf.reduce_max(self.q_target_ph, axis=1)
        self.q_target_sa = self.rew_ph + self.discount * self.q_target_max
        self.q_current_sa = self.q_current[0, self.act_ph]
        self.error = tf.reduce_sum(tf.square(self.q_target_sa - self.q_current_sa))
        self.pred_act_ph = tf.argmax(self.q_current, axis=1)

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.update_model = self.trainer.minimize(loss=self.error)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def simulate_bot(self):
        """Simulate the bot playing a couple of games"""

        self.env = gym.make(id='FrozenLake-v0')
        self.q_table = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
        self.n_obs, self.n_actions = self.env.observation_space.n, self.env.action_space.n

        self._set_up_neural_network()

        for idx in range(self.num_games):
            self._play(seed=idx)

    def display_results(self):
        """Average results over all the games played"""

        print(self.name, 'achieved an average score of',
              np.sum(self.results) / self.games_played, 'over', self.games_played + 1, 'games')


def main():
    q_learning_agent = QBot()
    q_learning_agent.simulate_bot()


if __name__ == '__main__':
    main()
