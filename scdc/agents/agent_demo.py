from scdc.agents.baseline_agents import *
from scdc.env.micro_env.mm_env import MMEnv
import time
import argparse


parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='bane_vs_bane', help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=2, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='A', help='The difficulty of built-in computer AI bot (default is "7").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=1, type=int, help='Number of episodes the game will run for.')
parser.add_argument('--agent', default="Positioning", type=str, help='Number of episodes the game will run for.')
parser.add_argument('--alpha', default=1, type=int, help='Parameter used for calculating score in HybridAttack.')

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
    
    if args.agent == 'HybridAttack':
        alpha = 0.5
        agent = globals()[args.agent](n_agents, env, alpha)
    else:
        agent = globals()[args.agent](n_agents, env)
        
    for e in range(n_episodes):
        agent.env.reset()
            
        episode_reward = 0
        
        terminated = False
        
        while not terminated:
            obs = agent.env.get_obs()
            state = agent.env.get_state()
            
            reward, terminated  = agent.step(obs, state)
            episode_reward += reward 
            
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
        # time.sleep(5)
        
    env.close()