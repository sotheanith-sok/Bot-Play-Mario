import numpy as np
from SumTree import SumTree


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, input_dimension, n_actions, discrete=False):
        self.discrete = discrete
        self.memory_size = max_size
        self.tree = SumTree(self.memory_size)
        self.n_actions = n_actions

        self.e = 0.01
        self.a = 0.6
        self.b = 0.4
        self.b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.0

        pass

    def store_trainsition(self, state, action, reward, new_state, done):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[action] = 1

        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, (state, one_hot_action, reward, new_state, done))

    def sample_buffer(self, batch_size):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = (
            np.empty((batch_size,), dtype=np.int32),
            np.empty((batch_size, 1), dtype=np.float32),
        )

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / batch_size  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.b = np.min(
            [1.0, self.b + self.b_increment_per_sampling]
        )  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity :]) / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.b)

        for i in range(batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = (
                np.power(batch_size * sampling_probabilities, -self.b) / max_weight
            )

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights


    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
