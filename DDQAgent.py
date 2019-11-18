from ReplayBuffer import ReplayBuffer
from models import build_model
from tensorflow.keras import models
import numpy as np

class DDQAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dimension, epsilon_dec=0.99, epsilon_min = 0.01, memory_size = 1000, filename= "myModel.h5",replace_target = 100):
        self.n_actions = n_actions
        self.actions_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.memory = ReplayBuffer(memory_size,input_dimension,n_actions,True)
        self.filename = filename
        self.replace_target = replace_target
        self.batch_size=batch_size
        self.q_evaluation = build_model(0.01,n_actions,input_dimension)
        self.q_target = build_model(0.01,n_actions,input_dimension)
    
    def remember(self, state, action, reward, new_state, done):
        action = self.game_input_to_integers(np.array([action]))[0]
        self.memory.store_trainsition(state,action,reward,new_state,done)
    
    def choose_action(self, state):
        rand = np.random.random_sample()
        state = state[np.newaxis,:]
        if rand<self.epsilon:
             action = np.random.choice(self.actions_space)
        else:
            actions= self.q_evaluation.predict(state)
            action = np.argmax(actions)
        return self.integers_to_game_input(np.array([action]))[0]
    
    def learn(self):
        if self.memory.memory_counter>self.batch_size:
            states, actions, rewards, new_states,dones =self.memory.sample_buffer(self.batch_size)
            actions_value  = np.array(self.actions_space,dtype=np.int16)
            action_indices = np.dot(actions,actions_value)

            q_next = self.q_target.predict(new_states)
            q_eval = self.q_evaluation.predict(new_states)

            q_pred= self.q_evaluation.predict(states)

            max_actions =np.argmax(q_eval, axis=1)

            q_target =q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index,action_indices]= rewards + self.gamma*q_next[batch_index,max_actions.astype(int)]*dones

            self.q_evaluation.fit(states, q_target, verbose =0)
            self.epsilon =self.epsilon*self.epsilon_dec if self.epsilon>self.epsilon_min else self.epsilon_min

            if(self.memory.memory_counter% self.replace_target==0):
                self.update_network_params()

    def update_network_params(self):
        self.q_target.set_weights(self.q_evaluation.get_weights())

    def save_model(self):
        self.q_evaluation.save(self.filename)
    def load_model(self):
        self.q_evaluation = models.load_model(self.filename)

        if self.epsilon<=self.epsilon_min:
            self.update_network_params()

    def game_input_to_integers(self,values):
        return values.dot(1 << np.arange(values.shape[-1]))

    def integers_to_game_input(self,values, bits_length=12):
        return (((values[:, None] & (1 << np.arange(bits_length)))) > 0).astype(int)