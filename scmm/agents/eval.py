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
import matplotlib.pyplot as plt

import sys
import numpy as np
import random
from tqdm import tqdm

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
# map_names = ['3m', '3s_vs_4z_medium', '3s_vs_5z_medium']

# map_names = ['1m', '3m', '8m']

# map_names = ['bane_vs_bane', 'so_many_baneling', '2c_vs_64zg'] # Need to run on 1st and 3rd
# map_names = ['bane_vs_bane', '2c_vs_64zg'] # Need to run on 1st and 3rd
# map_names = ['bane_vs_bane']
# map_names = ['corridor']
# map_names = ['6s1s_vs_10r']
# map_names = ['2m_vs_1z', '2s_vs_1sc']

for i, map_name in tqdm(enumerate(map_names)):
    
    if i < 23:
        continue
    
    if map_name == '6m1r_vs_4g' or map_name == '12m2r_vs_7g':
        continue
    
    all_rewards = np.zeros((n_episodes, ))
    all_results = np.zeros((n_episodes, ))

    env = MMEnv(map_name=map_name, step_mul=2, difficulty='A', reward_sparse=False, debug=False)
    env_info = env.get_env_info()
    
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    alpha = 0
    
    # agent = HybridAttackHeal(n_agents, alpha)
    # agent = Kiting(n_agents, consuctive_attack_count=10)
    # agent = AlternatingFire(n_agents)
    # agent = FocusFire(n_agents)
    # agent = Positioning(n_agents)
    # agent = WallOff(n_agents)
    # agent = BlockEnemy(n_agents)
    
    
    # agent.fit(env)
    
    
    for e in range(n_episodes):
        # agent = FocusFire(n_agents, False)
        agent = DyingRetreat(n_agents)
        agent.fit(env)
        agent.env.reset()
            
        episode_reward = 0
        terminated = False
        
        game_map = np.zeros((env.map_x, env.map_y))

        while not terminated:
            
            reward, terminated, info  = agent.step(0)
            episode_reward += reward 

        all_rewards[e] = reward
        all_results[e] = info['battle_won']
        
    env.close()    

    all_results = all_results.astype(bool)
    fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)

    lines = ax.vlines(x=np.arange(n_episodes)[all_results], ymin=0, ymax=all_rewards[all_results], color='#00cc66', alpha=0.7, linewidth=20)
    lines = ax.vlines(x=np.arange(n_episodes)[~all_results], ymin=0, ymax=all_rewards[~all_results], color='firebrick', alpha=0.7, linewidth=20)

    # Annotate Text
    for i, y in enumerate(all_rewards):
        ax.text(i, y+0.01, round(y, 3), horizontalalignment='center')
    
    # Title, Label, Ticks and Ylim
    ax.set_title(f'Dying Retreat - {map_name}', fontdict={'size':22})
    ax.set(ylabel='Final Rewards', ylim=(0, 2))
    ax.set(xlabel='Episode')
    plt.xticks(np.arange(n_episodes), np.arange(n_episodes), horizontalalignment='right', fontsize=12)
    fig.savefig(f"plots/DR_{map_name}.pdf")
    plt.close(fig)
    