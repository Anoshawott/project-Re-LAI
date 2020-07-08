from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf

from player_interaction import PlayerAI

from collections import deque

import time
import numpy as np
import random

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MODEL_NAME = '256X2'

class AIEnv:
    RETURN_DATA = True
    ACTION_SPACE_SIZE = 1270
    OBSERVATION_SPACE_VALUES = (1270,711,3)
    # Reward and Penalty Values
    # also need a reward for mana and hp increases, but make this minimal compared to others 
    rewards = {'cs':1000, 'k':100000, 'd':100000, 'a':1000, 'hp':10, 'mana':10,
                'tur_outer':10, 'tur_inner':20, 'tur_inhib':30, 
                'inhib':30, 'tur_nex_1':40, 'tur_nex_2':40, 'nexus':40}
    cs_reward = 10

    def reset(self):
        self.play_ai = PlayerAI()

        self.episode_step = 0

        if self.RETURN_DATA:
            observation = self.play_ai.new_data()
        # else?
        return observation
    
    def step(self, action, last_obs=self.play_ai.new_data()):
        self.episode_step+=1

        self.play_ai.action(action)

        if self.RETURN_DATA:
            new_observation = self.play_ai.new_data()
        
        net_reward = 0
        # Reward and Penalty Conditions
        output_data_comp = new_observation['output_data'] & last_obs['output_data']

        if len(output_data_comp) != 4:
            for i in output_data_comp:
                del new_observation['output_data'][i]
                del last_obs['output_data'][i]
            for k in new_observation['output_data']:
                if k == 'cs' or k == 'level' or k == 'k' or k == 'hp' or k == 'mana':
                    try:
                        new = int(new_observation['output_data'][k])
                        old = int(last_obs['output_data'][k])    
                    except:
                        new = 0
                        old = 0   
                    delta = new - old
                    if delta < 0:
                        total_penalty = -self.rewards[k] * delta
                        net_reward += total_penalty
                    else:
                        total_reward = self.rewards[k] * delta
                        net_reward += total_reward
                else:
                    try:
                        new = int(new_observation['output_data'][k])    
                        old = int(last_obs['output_data'][k])
                    except:
                        new = 0
                        old = 0                        
                    delta = new - old
                    total_penalty = -self.rewards[k] * delta
                    net_reward += total_penalty
        
        tur_data_comp = new_observation['map_data']['tur_dist'] & last_obs['map_data']['tur_dist']
        if len(tur_data_comp) != 7:
            for i in tur_data_comp:
                del new_observation['map_data']['tur_dist'][i]
                del last_obs['map_data']['tur_dist'][i]
            for k in new_observation['map_data']['tur_dist']:
                try:
                    new = int(new_observation['map_data']['tur_dist'][k])   
                    old = int(last_obs['map_data']['tur_dist'][k]) 
                except:
                    new = 0
                    old = 0  
                delta = new - old
                if delta < 0:
                    total_reward = self.rewards[k] * -delta
                    net_reward += total_reward
        
        tur_hps = {'tur_outer':5000, 'tur_inner':3600, 'tur_inhib':3300, 
                    'inhib':4000, 'tur_nex_1':2700, 'tur_nex_2':2700, 'nexus':5500}
        tur_hp_data_comp = not new_observation['map_data']['tur_hp']
        if tur_hp_data_comp != True:
            for k in new_observation['map_data']['tur_dist']:
                try:
                    new = int(new_observation['map_data']['tur_dist'][k])
                    old = tur_hps[k]   
                except:
                    new = 0
                    old = 0  
                delta = new - old
                if delta < 0:
                    total_reward = self.rewards[k] * -delta
                    net_reward += total_reward
                if k == 'nexus' and new_observation['map_data']['tur_dist'][k] == 0:
                    done = True
                else:
                    done = False

        return new_observation, net_reward, done

env = AIEnv()

random.seed(1)
np.random.seed(1)
tf.set_random_seed


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):

        # main model --> gets trained every step
        self.model = self.create_model()

        # target model --> this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # handles batch samples so to attain stability in training; prevent overfitting
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')

        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model = Sequential()
        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1,*state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X=[]
        y=[]

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE,
                        verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1 
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0        
