import numpy as np


def compute_ranks(x):
    """
    code from open-ai
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    code from open-ai
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


class PEPG:
    '''
    Extension of PEPG with bells and whistles.
    '''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 sigma_alpha=0.20,  # learning rate for standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 learning_rate=0.01,  # learning rate for mean value
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.001,  # stop annealing learning rate
                 popsize=255,  # population size
                 rank_fitness=True,  # use rank rather than fitness numbers
                 mu=None):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.rank_fitness = rank_fitness
        self.popsize = popsize
        assert (self.popsize & 1), "Population size must be odd"
        self.batch_size = int((self.popsize - 1) / 2)
        self.batch_reward = np.zeros(self.batch_size * 2)
        if mu is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = mu
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])

        epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
        # trans to numpy array
        reward_table = np.array(reward_table_result)
        if self.rank_fitness:
            reward_table = compute_centered_ranks(reward_table)

        reward_offset = 1
        b = reward_table[0]  # baseline
        reward = reward_table[reward_offset:]
        idx = np.argsort(reward)[::-1]
        # print(idx[0])
        best_reward = reward[idx[0]]
        if best_reward > b:
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.best_mu = best_mu
        self.best_reward = best_reward

        stdev_reward = reward.std()
        epsilon = self.epsilon
        sigma = self.sigma
        S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
        reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
        rS = reward_avg - b
        delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

        # move mean to the average of the best idx means
        rM = (reward[:self.batch_size] - reward[self.batch_size:])
        change_mu = self.learning_rate * np.dot(rM, epsilon)
        self.mu += change_mu

        # adjust sigma according to the adaptive sigma calculation
        change_sigma = self.sigma_alpha * delta_sigma
        change_sigma = np.minimum(change_sigma, self.sigma)
        change_sigma = np.maximum(change_sigma, - 0.5 * self.sigma)
        self.sigma += change_sigma
        self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay
