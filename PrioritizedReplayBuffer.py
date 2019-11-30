import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, input_dimension, n_actions, discrete=False):
        # Prevent overusage of memory
        self.memory_size = max_size
        self.memory_counter = 0

        # Check if input is continue or discrete values
        self.discrete = discrete
        self.states_memory = np.zeros(((self.memory_size,) + input_dimension),dtype=np.float32)
        self.new_states_memory = np.zeros(((self.memory_size,) + input_dimension), dtype = np.float32)
        dtype = np.int8 if self.discrete else np.float
        self.actions_memory = np.zeros(
            ((self.memory_size,) + (n_actions,)), dtype=dtype
        )
        self.rewards_memory = np.zeros(self.memory_size,dtype=np.float32)
        self.terminals_memory = np.zeros(self.memory_size, dtype=np.int8)

        self.priorities = np.zeros(self.memory_size, dtype= np.float32)

    def store_trainsition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.states_memory[index] = state
        self.actions_memory[index] = action
        if self.discrete:
            actions = np.zeros(np.shape(self.actions_memory)[1])
            actions[action] = 1.0
            self.actions_memory[index] = actions
        else:
            self.actions_memory[index] = action
        self.new_states_memory[index] = new_state
        self.terminals_memory[index] = 1 - int(done)

        if self.memory_counter == 0:
            self.priorities[index] = 1
        else:
            self.priorities[index] = np.max(self.priorities)

        self.memory_counter += 1

    def get_probabilities(self, priority_scale):
        scaled_priority = np.power(self.priorities, priority_scale)
        sample_probabilities = np.divide(scaled_priority, np.sum(scaled_priority))
        return sample_probabilities

    def get_importance(self, probabilities, probabilities_scale):
        importance = np.power(1 / self.memory_size * 1 / probabilities, probabilities_scale)
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample_buffer(self, batch_size, priority_scale=1.0, probabilities_scale=1.0):
        max_memory = min(self.memory_counter, self.memory_size)

        sample_probs = self.get_probabilities(priority_scale)

        indices = np.random.choice(max_memory, batch_size, p=sample_probs[0:max_memory])
        states = self.states_memory[indices]
        new_states = self.new_states_memory[indices]
        rewards = self.rewards_memory[indices]
        actions = self.actions_memory[indices]
        terminals = self.terminals_memory[indices]
        importances = self.get_importance(sample_probs[indices], probabilities_scale)
        return states, actions, rewards, new_states, terminals, importances, indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
