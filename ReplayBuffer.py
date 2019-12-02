import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_dimension, n_actions, discrete=False):
        """Initialize replay buffer
        
        Arguments:
            max_size {int} -- [buffer capacity]
            input_dimension {numpy.array} -- [shape of observations]
            n_actions {int} -- [number of possible actions]
        
        Keyword Arguments:
            discrete {bool} -- [is action descrete or continue?] (default: {False})
        """
        # Memory counter uses to keep track of how many experience has been stored so far
        self.memory_counter = 0

        # Save buffer memory size
        self.memory_size = max_size
        
        # Initialize array for storing oberservations
        self.states_memory = np.zeros(((self.memory_size,) + input_dimension))
        self.new_states_memory = np.zeros(((self.memory_size,) + input_dimension))
       
        # Initialize array for storing actions, reward, and terminals
        self.discrete = discrete
        dtype = np.int8 if self.discrete else np.float
        self.actions_memory = np.zeros(
            ((self.memory_size,) + (n_actions,)), dtype=dtype
        )
        self.rewards_memory = np.zeros(self.memory_size)
        self.terminals_memory = np.zeros(self.memory_size, dtype=np.float)

    def store_trainsition(self, state, action, reward, new_state, done):
        """Store experience into memory
        
        Arguments:
            state {numpy.array} -- [current state]
            action {int or float} -- [action taken given current state]
            reward {float} -- [reward produces from given action]
            new_state {numpy.array} -- [new state]
            done {0 or 1} -- [is it the end of episode]
        """
        # Decide which slot to store experience
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
        self.memory_counter+=1

    def sample_buffer(self, batch_size):
        """Sample some experience from buffers
        
        Arguments:
            batch_size {int} -- [number of experiences to sample]
        """
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.states_memory[batch]
        new_states = self.new_states_memory[batch]
        rewards = self.rewards_memory[batch]
        actions = self.actions_memory[batch]
        terminals = self.terminals_memory[batch]
        return states, actions, rewards, new_states, terminals
