# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 22:27:05 2020

@author: Cai
"""
from scdc.agents.base_agent import BaseAgent
from scdc.env.micro_env.mm_env import MMEnv
import time
import numpy as np
import argparse


class RandomAgents():
    
    def __init__(self, n_agents, env):
        self.n_agents = n_agents 
        self.env = env
    
    def step(self, obs, state):
        # obs. state should be returned by env
        # super(RandomAgents, self).step(obs)
        # time.sleep(0.5)

        actions = []
        for agent_id in range(self.n_agents):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)

        reward, terminated, _ = self.env.step(actions)
        
        return reward, terminated 
        

parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='half_6m_vs_full_4m', help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=8, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='7', help='The difficulty of built-in computer AI bot (default is "7").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=30, type=int, help='Number of episodes the game will run for.')

args = parser.parse_args()
        
if __name__ == "__main__":
    
    map_name = args.map_name
    step_mul = args.step_mul
    difficulty = args.difficulty
    reward_sparse = args.reward_sparse
    debug = args.debug 
    n_episodes = args.n_episodes

    env = MMEnv(map_name=map_name, step_mul=step_mul, difficulty=difficulty, reward_sparse=reward_sparse, debug=debug)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    
    ra = RandomAgents(n_agents, env)
    
    for e in range(n_episodes):
        env.reset()
        
        episode_reward = 0
        
        terminated = False
        
        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            
            reward, terminated  = ra.step(obs, state)
            
            episode_reward += reward 
            
        print("Total reward in episode {} = {}".format(e, episode_reward))
            
    env.close()
    
    
    