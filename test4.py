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

EPISODES = 100

def thing(last_obs=None, count=None): 
        rewards = {'cs':10000, 'k':300000, 'd':1000000, 'a':1000, 'hp':1, 'mana':1,
                'level':100000,'tur_outer':10, 'tur_inner':20, 'tur_inhib':30, 
                'inhib':30, 'tur_nex_1':40, 'tur_nex_2':40, 'nexus':40}
        print('huh')
        done = False

        play_ai = PlayerAI()

        new_observation = play_ai.new_data()
        
        net_reward = 0
        # Reward and Penalty Conditions
        try:
            if int(new_observation['output_data']['hp']) == 0:
                done = True
                new = int(new_observation['output_data']['d'])    
                old = int(last_obs['output_data']['d'])
                delta = new - old
                total_penalty = -rewards['d'] * delta
                net_reward += total_penalty
                print('DEAD!')
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
                if k == 'cs' or k == 'level' or k == 'k' or k == 'a' or k == 'hp' or k == 'mana':
                    if k == 'cs':
                        print(k)
                    try:
                        new = int(new_observation['output_data'][k])
                        old = int(last_obs['output_data'][k])
                        delta = new - old
                        if delta < 0:
                            total_penalty = rewards[k] * delta
                            net_reward += total_penalty
                            print('1')
                        elif delta > 0:
                            if k != 'cs':
                                total_reward = rewards[k] * delta * 0.001
                            else:
                                total_reward = rewards[k] * delta
                            net_reward += total_reward
                            print(new)
                            print(old)
                            print('2')  
                    except:
                        None
                    
                # else:
                #     try:
                #         new = int(new_observation['output_data'][k])    
                #         old = int(last_obs['output_data'][k])
                #         delta = new - old
                #         total_penalty = -rewards[k] * delta
                #         net_reward += total_penalty
                #         print('3')
                #         if delta > 0:
                #             done = True
                #     except:
                #         None                     
        

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
                        total_reward = rewards[k] * -delta
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

        if new_observation['map_data']['player_pos'] == [20,170]:
            count+=1
        else:
            count=0
        
        if count > 4:
            pos_penalty = -1
            net_reward += pos_penalty

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
                    if new < min_tur[k] and last_obs['map_data']['player_pos'] != [20,170]:    #new < min_tur[k]
                        print('yes/new:',new)
                        min_tur[k] = new
                        pickle_out = open('min_tur.pickle','wb')
                        pickle.dump(min_tur, pickle_out)
                        pickle_out.close()

                        delta = new - old
                        if delta < 0:
                            total_reward = rewards[k] * -delta * 2
                            net_reward += total_reward
                            print('5')
                    else:
                        delta = new - old
                        if delta < 0 and last_obs['map_data']['player_pos'] != [20,170]:
                            total_reward = rewards[k] * -delta
                            net_reward += total_reward
                            print('5.1')
                        elif delta > 0:
                            total_penalty = rewards[k] * -delta
                            net_reward += total_penalty
                            print('6')
                    # if k == 'tur_outer' and new == 0:
                    #     done = True
                except:
                    None 

        return new_observation, net_reward, done, count

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):

    play_ai = PlayerAI()
    episode_reward = 0
    step = 1
    current_state = play_ai.new_data()

    done = False
    
    num = 0
    while not done:
        print(step)
        new_state, reward, done, num = thing(last_obs=current_state, count=num)
        episode_reward += reward
        print('episode_reward:', episode_reward)
        print('-------------------')

        current_state = copy.deepcopy(new_state)
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

    if (episode+75)%5 == 0:
        agent.model.save(f'dql_models4/{MODEL_NAME}__ep_{episode+75:_>7.2f}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    print('avg_reward:', average_reward)
    print('min_reward:', min_reward)
    print('max_reward:', max_reward)

    # Decay epsilon 

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)