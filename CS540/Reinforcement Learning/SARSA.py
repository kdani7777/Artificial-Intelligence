from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
       
        #TODO perform SARSA learning
        obs = env.reset()

        curr_state, curr_action, curr_reward = None, None, None # our t+1
        prev_state, prev_action, prev_reward = None, None, None # our t

        done = False
        while done == False:

            curr_state = obs

            if random.uniform(0,1) < EPSILON:
                curr_action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                curr_action =  np.argmax(prediction)
            
            obs,reward,done,info = env.step(curr_action)
            episode_reward += reward
            curr_reward = reward

            # iterate through action space and find best possible action (the one with the best/max_reward)
            # max_reward = max([Q_table[(obs,i)] for i in range(env.action_space.n)])
            if prev_action == None:
                prev_state = curr_state
                prev_action = curr_action
                prev_reward = curr_reward
                continue 

            Q_table[(prev_state,prev_action)] += LEARNING_RATE *(prev_reward +
                (DISCOUNT_FACTOR*(Q_table[(curr_state,curr_action)])) - Q_table[(prev_state,prev_action)])
            
            if done:
                episode_reward_record.append(episode_reward)
                Q_table[(curr_state,curr_action)] += LEARNING_RATE*(curr_reward - Q_table[(curr_state,curr_action)])
            
            prev_state = curr_state
            prev_action = curr_action
            prev_reward = curr_reward

        EPSILON = EPSILON * EPSILON_DECAY

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



