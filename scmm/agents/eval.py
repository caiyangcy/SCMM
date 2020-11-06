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
import matplotlib.pyplot as plt

import numpy as np
import random
from tqdm import tqdm
import os

'''
This file run the agents on some of maps and collect the results(rewards). 
This file is not formatted nicely and not included in pypi package.
Change the agent and its params along with the map name to run it.
Results are saved under plots folder
'''

seed = 42
np.random.seed(seed)
random.seed(seed)

agents = [AlternatingFire, HybridAttackHeal, FocusFire, Kiting, Positioning, WallOff, DyingRetreat, BlockEnemy]

map_registry = get_scmm_map_registry()
map_names = map_registry.keys()

n_episodes = 10

for i, map_name in tqdm(enumerate(map_names)):
    
    if map_name == '6m1r_vs_4g' or map_name == '12m2r_vs_7g':
        continue
    
    all_rewards = np.zeros((n_episodes, ))
    all_results = np.zeros((n_episodes, ))

    env = MMEnv(map_name=map_name, step_mul=2, difficulty='A', reward_sparse=False, debug=False)
    env_info = env.get_env_info()
    
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    alpha = 0
    
    for e in range(n_episodes):
        agent = HybridAttackHeal(n_agents, 0) # AC
        
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
    ax.set_title(f'Attack Closest - {map_name}', fontdict={'size':22})
    ax.set(ylabel='Final Rewards', ylim=(0, 2))#, fontdict={'size':20})
    ax.set(xlabel='Episode')#, fontdict={'size':20})
    plt.xticks(np.arange(n_episodes), np.arange(n_episodes), horizontalalignment='right')#, fontsize=15)
    
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    
    fig.savefig(f"plots/AC_{map_name}.pdf")
    plt.close(fig)
    