from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf

from player_interaction import PlayerAI
from env_reset import EnvReset
from choice_create import ChoiceCreate

from collections import deque

from tqdm import tqdm

import time
import numpy as np
import random
import os
import subprocess
import pickle
import copy

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 8
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200

EPISODES = 20_000

# Consider changing epsilon??
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False

MODEL_NAME = '12X2'

class AIEnv:
    RETURN_DATA = True
    ACTION_SPACE_SIZE = 1271
    OBSERVATION_SPACE_VALUES = (711,1270,3)
    # Reward and Penalty Values
    # also need a reward for mana and hp increases, but make this minimal compared to others 
    rewards = {'cs':1000, 'k':100000, 'd':1000000, 'a':1000, 'hp':10, 'mana':10,
                'level':10000,'tur_outer':10, 'tur_inner':20, 'tur_inhib':30, 
                'inhib':30, 'tur_nex_1':40, 'tur_nex_2':40, 'nexus':40}

    def reset(self):
        # further this function so that the whole game resets with a new game each time an episode has past
        # use the following code to close windows from within this function: subprocess.call("taskkill /f /im notepad.exe", shell=True)
        # have timed intervals between each step in the reset process... --> just need to automate setting up a new game
        # need to also determine how to save each model between episodes...
        EnvReset().game_reset()

        ChoiceCreate().min_tur_save()
        ChoiceCreate().tur_status_save()

        self.play_ai = PlayerAI()

        self.episode_step = 0

        if self.RETURN_DATA:
            observation = self.play_ai.new_data()
        # else?
        return observation
    
    def step(self, action, last_obs=None): 
        self.episode_step+=1

        self.play_ai.action(action)
        
        done = False

        if self.RETURN_DATA:
            new_observation = self.play_ai.new_data()
        
        net_reward = 0
        # Reward and Penalty Conditions
        try:
            if int(new_observation['output_data']['hp']) == 0:
                done = True
        except:
            None
        

        tmp_new = copy.deepcopy(new_observation)
        tmp_old = copy.deepcopy(last_obs)

        # Have a list that appends turrets that have been destroyed to have them removed and then added to the distance left to turret
        # use min-max method to optimising distance reward reward since the ai rn is prioritising reaching the nexus before destroying
        # anything else first...
        output_data_comp = new_observation['output_data'].items() & last_obs['output_data'].items()

        if len(output_data_comp) != 4:
            for i in output_data_comp:
                del tmp_new['output_data'][i[0]]
                del tmp_old['output_data'][i[0]]
            for k in tmp_new['output_data']:
                if k == 'cs' or k == 'level' or k == 'k' or k == 'hp' or k == 'mana':
                    try:
                        new = int(new_observation['output_data'][k])
                        old = int(last_obs['output_data'][k])
                        delta = new - old
                        if delta < 0:
                            total_penalty = self.rewards[k] * delta
                            net_reward += total_penalty
                            print('1')
                        # elif delta > 0:
                        #     total_reward = self.rewards[k] * delta * 0.001
                        #     net_reward += total_reward
                        #     print(new)
                        #     print(old)
                        #     print('2')
                        if k == 'hp' and new == 0:
                            done = True    
                    except:
                        None
                    
                else:
                    try:
                        new = int(new_observation['output_data'][k])    
                        old = int(last_obs['output_data'][k])
                        delta = new - old
                        total_penalty = -self.rewards[k] * delta
                        net_reward += total_penalty
                        print('3')
                        if delta > 0:
                            done = True
                    except:
                        None                     


        tur_hps = {'tur_outer':5000, 'tur_inner':3600, 'tur_inhib':3300, 
                    'inhib':4000, 'tur_nex_1':2700, 'tur_nex_2':2700, 'nexus':5500}
        tur_hp_data_comp = not new_observation['map_data']['tur_hp']
        if tur_hp_data_comp != True:
            for k in new_observation['map_data']['tur_hp']:
                try:
                    new = int(new_observation['map_data']['tur_hp'][k])
                    old = tur_hps[k]
                    delta = new - old
                    if delta < 0:
                        total_reward = self.rewards[k] * -delta
                        net_reward += total_reward
                        print('4')
                    if new == 0:
                        tur_status = pickle.load(open('tur_status.pickle', 'rb'))
                        tur_status[k] = 0
                        try: 
                            tur_status[tur_status.index(k) + 1] = 1
                            pickle_out = open('tur_status.pickle','wb')
                            pickle.dump(tur_status, pickle_out)
                            pickle_out.close()
                        except: 
                            done = True     
                except:
                    None  
                
        # Reading min dictionary of values to read and update min distances
        tur_status = pickle.load(open('tur_status.pickle', 'rb'))
        # print(new_observation['output_data'], new_observation['map_data'])
        # print(last_obs['output_data'], new_observation['map_data'])
        # print('----------------------')
        # print(tmp_new['output_data'], tmp_new['map_data'])
        # print(tmp_old['output_data'], tmp_old['map_data'])
        for i in tur_status:
            if tur_status[i] == 0:
                del tmp_new['map_data']['tur_dist'][i]
                del tmp_old['map_data']['tur_dist'][i]

        tur_data_comp = tmp_new['map_data']['tur_dist'].items() & tmp_old['map_data']['tur_dist'].items()
        if len(tur_data_comp) != 7:
            for i in tur_data_comp:
                del tmp_new['map_data']['tur_dist'][i[0]]
                del tmp_old['map_data']['tur_dist'][i[0]]
            for k in tmp_new['map_data']['tur_dist']:
                try:
                    new = int(new_observation['map_data']['tur_dist'][k])   
                    old = int(last_obs['map_data']['tur_dist'][k]) 
                    min_tur = pickle.load(open('min_tur.pickle', 'rb'))
                    if new < min_tur[k]:    #new < min_tur[k]
                        print('yes/new:',new)
                        min_tur[k] = new
                        pickle_out = open('min_tur.pickle','wb')
                        pickle.dump(min_tur, pickle_out)
                        pickle_out.close()

                        delta = new - old
                        if delta < 0:
                            total_reward = self.rewards[k] * -delta * 10
                            net_reward += total_reward
                            print('5')
                    else:
                        if delta < 0:
                            total_reward = self.rewards[k] * -delta
                            net_reward += total_reward
                            print('5.1')
                        elif delta > 0:
                            total_penalty = self.rewards[k] * -delta
                            net_reward += total_penalty
                            print('6')
                    # if k == 'tur_outer' and new == 0:
                    #     done = True
                except:
                    None 

        return new_observation, net_reward, done

env = AIEnv()

ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('dql_models'):
    os.makedirs('dql_models')

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

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
    
    # def _write_logs(self, logs, index):
    #     for name, value in logs.items():
    #         if name in ['batch', 'size']:
    #             continue
    #         summary = tf.summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = value
    #         summary_value.tag = name
    #         self.writer.add_summary(summary, index)
    #     self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):

        # main model --> gets trained every step
        # self.model = self.create_model()

        # Loading from first run
        self.model = tf.keras.models.load_model('dql_models_per_5/12X2__ep___20.00_-200.00max_-4870.00avg_-9540.00min__1594721228.model')

        # target model --> this is what we .predict against every step
        # self.target_model = self.create_model() 
        self.target_model = tf.keras.models.load_model('dql_models_per_5/12X2__ep___20.00_-200.00max_-4870.00avg_-9540.00min__1594721228.model')
        self.target_model.set_weights(self.model.get_weights())

        # handles batch samples so to attain stability in training; prevent overfitting
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))

        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(12, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(12, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(8))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0]['img'] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3]['img'] for transition in minibatch])/255
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

            X.append(current_state['img'])
            y.append(current_qs)
        
        self.model.fit(np.divide(np.array(X), 255), np.array(y), batch_size=MINIBATCH_SIZE,
                        verbose=0, shuffle=False)
        # , callbacks=[self.tensorboard] if terminal_state else None

        if terminal_state:
            self.target_update_counter += 1 
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0        


agent = DQNAgent()

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state['img']))
        else:
            action = np.random.randint(0,env.ACTION_SPACE_SIZE)
        print(step)
        new_state, reward, done = env.step(action=action, last_obs=current_state)
        
        episode_reward += reward
        print('episode_reward:', episode_reward)
        print('-------------------')
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state.copy()
        step += 1
    
    # NEED TO ADD FUNCTION TO SAVE EVERY EPISODE!!! --> every 5 episodes?

    print('episode_reward:', episode_reward)
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'dql_best_avg/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    if episode%5 == 0:
        agent.model.save(f'dql_models_per_5/{MODEL_NAME}__ep_{episode+20:_>7.2f}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    print('avg_reward:', average_reward)
    print('min_reward:', min_reward)
    print('max_reward:', max_reward)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)