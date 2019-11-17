import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        # Prevent overusage of memory
        self.memory_size = max_size
        self.memory_counter = 0

        # Check if input is continue or discrete values
        self.discrete = discrete
        self.states_memory = np.zeros(((self.memory_size,) + input_shape))
        self.new_states_memory = np.zeros(((self.memory_size,) + input_shape))
        dtype = np.int8 if self.discrete else np.float
        self.actions_memory = np.zeros(((self.memory_size,) + (n_actions,)), dtype=dtype)
        self.rewards_memory = np.zeros(self.memory_size)
        self.terminals_memory = np.zeros(self.memory_size, dtype=np.float)

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

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.states_memory[batch]
        new_states = self.new_states_memory[batch]
        rewards = self.rewards_memory[batch]
        actions = self.actions_memory[batch]
        terminals = self.terminals_memory[batch]
        return states, actions, rewards, new_states, terminals
