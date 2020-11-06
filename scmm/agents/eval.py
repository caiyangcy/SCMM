from scmm.agents.scripted.alternating_fire import AlternatingFire
from scmm.agents.scripted.hybrid_attack import HybridAttackHeal
from scmm.agents.scripted.focus_fire import FocusFire
from scmm.agents.scripted.kiting import Kiting
from scmm.agents.scripted.positioning import Positioning
from scmm.agents.scripted.wall_off import WallOff
from scmm.agents.scripted.dying_retreat import DyingRetreat
from scmm.agents.scripted.block_enemy import BlockEnemy

from scmm.env.micro_env.mm_env import MMEnv
from scmm.utils.game_show import game_show, game_show_adv

from scmm.env.micro_env.maps.mm_maps import get_scmm_map_registry
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys
import numpy as np
import random

'''
This file run all the agents on all the maps and collect the results(rewards)
'''
seed = 42
np.random.seed(seed)
random.seed(seed)

agents = [AlternatingFire, HybridAttackHeal, FocusFire, Kiting, Positioning, WallOff, DyingRetreat, BlockEnemy]

map_registry = get_scmm_map_registry()
map_names = map_registry.keys()

n_episodes = 10
map_names = ['3s_vs_3z', '3s_vs_4z', '3s_vs_5z']

all_rewards = np.zeros((len(map_names), n_episodes, ))
all_results = np.zeros((len(map_names), n_episodes, ))

for i, map_name in enumerate(map_names):

    env = MMEnv(map_name=map_name, step_mul=2, difficulty='A', reward_sparse=False, debug=False)
    env_info = env.get_env_info()
    
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    alpha = 0
    
    # agent = HybridAttackHeal(n_agents, alpha)
    agent = Kiting(n_agents, consuctive_attack_count=13)
    agent.fit(env)
    
    # all_rewards = np.zeros((n_episodes, ))
    # all_results = np.zeros((n_episodes, ))
    
    for e in range(n_episodes):
        agent.env.reset()
            
        episode_reward = 0
        terminated = False
        
        game_map = np.zeros((env.map_x, env.map_y))

        while not terminated:
            
            reward, terminated, info  = agent.step(0)
            episode_reward += reward 

        all_rewards[i, e] = reward
        all_results[i, e] = info['battle_won']
        
    env.close()    



fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
ax.vlines(x=np.arange(n_episodes)-0.25, ymin=0, ymax=all_rewards[0], color='#cc0000', alpha=0.7, linewidth=20, label='3s_vs_3z-Kiting')
ax.vlines(x=np.arange(n_episodes), ymin=0, ymax=all_rewards[1], color='#cc0066', alpha=0.7, linewidth=20, label='3s_vs_4z-Kiting')
ax.vlines(x=np.arange(n_episodes)+0.25, ymin=0, ymax=all_rewards[2], color='#cc6699', alpha=0.7, linewidth=20, label='3s_vs_5z-Kiting')

# Annotate Text
for i in range(all_rewards.shape[1]):
    ax.text(i-0.25, all_rewards[0][i]+0.01, round(all_rewards[0][i], 3), horizontalalignment='center', fontsize=12)
    ax.text(i, all_rewards[1][i]+0.01, round(all_rewards[1][i], 3), horizontalalignment='center', fontsize=12)
    ax.text(i+0.25, all_rewards[2][i]+0.01, round(all_rewards[2][i], 3), horizontalalignment='center', fontsize=12)



all_rewards = np.zeros((len(map_names), n_episodes, ))
all_results = np.zeros((len(map_names), n_episodes, ))

for i, map_name in enumerate(map_names):

    env = MMEnv(map_name=map_name, step_mul=2, difficulty='A', reward_sparse=False, debug=False)
    env_info = env.get_env_info()
    
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    alpha = 0
    
    agent = HybridAttackHeal(n_agents, alpha)
    agent.fit(env)
    
    for e in range(n_episodes):
        agent.env.reset()
            
        episode_reward = 0
        terminated = False
        
        game_map = np.zeros((env.map_x, env.map_y))

        while not terminated:
            
            reward, terminated, info  = agent.step(0)
            episode_reward += reward 

        all_rewards[i, e] = reward
        all_results[i, e] = info['battle_won']
        
    env.close()    


# fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
ax.vlines(x=np.arange(n_episodes)-0.25, ymin=0, ymax=all_rewards[0], color='#000099', alpha=0.7, linewidth=20, label='3s_vs_3z-AC')
ax.vlines(x=np.arange(n_episodes), ymin=0, ymax=all_rewards[1], color='#0000ff', alpha=0.7, linewidth=20, label='3s_vs_3z-AC')
ax.vlines(x=np.arange(n_episodes)+0.25, ymin=0, ymax=all_rewards[2], color='#3366ff', alpha=0.7, linewidth=20, label='3s_vs_3z-AC')

# Annotate Text
for i in range(all_rewards.shape[1]):
    ax.text(i-0.25, all_rewards[0][i]+0.01, round(all_rewards[0][i], 3), horizontalalignment='center', fontsize=12)
    ax.text(i, all_rewards[1][i]+0.01, round(all_rewards[1][i], 3), horizontalalignment='center', fontsize=12)
    ax.text(i+0.25, all_rewards[2][i]+0.01, round(all_rewards[2][i], 3), horizontalalignment='center', fontsize=12)


# Title, Label, Ticks and Ylim
ax.set_title('Kiting vs AC', fontdict={'size':22})
ax.set_ylabel('Final Rewards', fontsize=20)
ax.set_ylim(0, 2)

ax.set_xlabel('Episode', fontsize=20)
plt.xticks(np.arange(n_episodes), np.arange(n_episodes), horizontalalignment='right', fontsize=12)
plt.legend()
plt.show()
fig.savefig("ac_vs_kiting2.pdf")