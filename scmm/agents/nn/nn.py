from scmm.env.micro_env.mm_env import MMEnv
import numpy as np
import argparse
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt

class NN_Agent:
    def __init__(self, n_agents, n_epoches, evolving_low=0.99, evolving_high=1.01):
        self.n_agents = n_agents 
        self.map = env.map_name 
        self.params = torch.zeros(40, 1)
        self.net = [MLP(), MLP(), MLP()] # 3 nets in total
        self.n_epoches = n_epoches
        self.prev_move = self.prev_prev_move = None 
        self.first = True
        self.second = False
        self.evolving_low = evolving_low
        self.evolving_high = evolving_high
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, net_ind, plot_level):
        actions = []        

        for agent_id in range(self.n_agents):
            
            unit = self.env.get_unit_by_id(agent_id)
            self._fill_params(unit, self.prev_move, self.prev_prev_move)
            action = self._get_action(net_ind)
            self.prev_prev_move = self.prev_move
            self.prev_move = action
            if action == self.n_actions_no_attack:
                close_e_id = self.find_closest(unit)
                action = self.n_actions_no_attack+close_e_id
                
            actions.append(action)

        reward, terminated, info = self.env.step(actions)
        
        if self.first:
            self.first = False
            self.second = True
        elif self.second:
            self.second = False 
        
        
        if plot_level:
            return actions, reward, terminated, info
        
        return reward, terminated, info
    
    def find_closest(self, unit):
        unit_x, unit_y = unit.pos.x, unit.pos.y
        target_items = self.env.enemies.items()
        closest_id = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
                  
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                dist = self.env.distance(unit_x, unit_y, t_unit.pos.x, t_unit.pos.y)
                if dist < min_dist:
                    min_dist = dist 
                    closest_id = t_id
                        
        return closest_id
        
    def _fill_params(self, unit, prev_move=None, prev_prev_move=None):
        
        for i in range(1, 9):
            exec("enermy_avg_dist_"+str(i)+"=0")
            exec("ally_avg_dist_"+str(i)+"=0")
            exec("enermy_count_"+str(i)+"=0")
            exec("ally_count_"+str(i)+"=0")
        
        enermy_avg_dist_1 = enermy_avg_dist_2 = enermy_avg_dist_3 = enermy_avg_dist_4 = \
            enermy_avg_dist_5 = enermy_avg_dist_6 = enermy_avg_dist_7 = enermy_avg_dist_8 = 0     
            
        ally_avg_dist_1 = ally_avg_dist_2 = ally_avg_dist_3 = ally_avg_dist_4 = \
            ally_avg_dist_5 = ally_avg_dist_6 = ally_avg_dist_7 = ally_avg_dist_8 = 0    
            
        enermy_count_1 = enermy_count_2 = enermy_count_3 = enermy_count_4 = \
            enermy_count_5 = enermy_count_6 = enermy_count_7 = enermy_count_8 = 0    
            
        ally_count_1 = ally_count_2 = ally_count_3 = ally_count_4 = \
            ally_count_5 = ally_count_6 = ally_count_7 = ally_count_8 = 0    
            
        shoot_range = self.env.unit_shoot_range(unit)
        
        unit_x, unit_y = unit.pos.x, unit.pos.y 
        map_x, map_y = self.env.max_distance_x, self.env.max_distance_y
        self.params[32] = map_x-unit_x
        self.params[33] = unit_x 
        self.params[34] = map_y-unit_y 
        self.params[35] = unit_y 
        
        self.params[36] = self.env.unit_max_cooldown(unit)
        self.params[37] = self.env.unit_damage(unit)
        
        
        for agent_id in range(self.n_agents):
            ally = self.env.get_unit_by_id(agent_id)
            if ally.health > 0:
                a_pos_x, a_pos_y = ally.pos.x, ally.pos.y
                dist_to_ally = self.env.distance(unit_x, unit_y, a_pos_x, a_pos_y)
                
                if a_pos_x >= unit_x and a_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        ally_count_1 += 1
                        ally_avg_dist_1 += dist_to_ally
                    else:
                        ally_count_2 += 1
                        ally_avg_dist_2 += dist_to_ally
                elif a_pos_x < unit_x and a_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        ally_count_3 += 1
                        ally_avg_dist_3 += dist_to_ally
                    else:
                        ally_count_4 += 1
                        ally_avg_dist_4 += dist_to_ally
                elif a_pos_x >= unit_x and a_pos_y < unit_y:
                    if dist_to_ally <= shoot_range:
                        ally_count_7 += 1
                        ally_avg_dist_7 += dist_to_ally
                    else:
                        ally_count_8 += 1
                        ally_avg_dist_8 += dist_to_ally
                else:
                    if dist_to_ally <= shoot_range:
                        ally_count_6 += 1
                        ally_avg_dist_6 += dist_to_ally
                    else:
                        ally_count_5 += 1
                        ally_avg_dist_5 += dist_to_ally
        
        target_items = self.env.enemies.items()
        
        
        for e_id, e_unit in target_items:
            if e_unit.health > 0:
                e_pos_x, e_pos_y = e_unit.pos.x, e_unit.pos.y
                dist_to_ally = self.env.distance(unit_x, unit_y, e_pos_x, e_pos_y)
                
                if e_pos_x >= unit_x and e_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        enermy_count_1 += 1
                        enermy_avg_dist_1 += dist_to_ally
                    else:
                        enermy_count_2 += 1
                        enermy_avg_dist_2 += dist_to_ally
                elif e_pos_x < unit_x and e_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        enermy_count_3 += 1
                        enermy_avg_dist_3 += dist_to_ally
                    else:
                        enermy_count_4 += 1
                        enermy_avg_dist_4 += dist_to_ally
                elif e_pos_x >= unit_x and e_pos_y < unit_y:
                    if dist_to_ally <= shoot_range:
                        enermy_count_7 += 1
                        enermy_avg_dist_7 += dist_to_ally
                    else:
                        enermy_count_8 += 1
                        enermy_avg_dist_8 += dist_to_ally
                else:
                    if dist_to_ally <= shoot_range:
                        enermy_count_6 += 1
                        enermy_avg_dist_6 += dist_to_ally
                    else:
                        enermy_count_5 += 1
                        enermy_avg_dist_5 += dist_to_ally
            
        
        for i in range(8):
            self.params[i] = locals()["enermy_avg_dist_"+str(i+1)]/locals()["enermy_count_"+str(i+1)] if locals()["enermy_count_"+str(i+1)] > 0 else 0
        for i in range(8, 16):
            self.params[i] = locals()["ally_avg_dist_"+str(i-7)]/locals()["ally_count_"+str(i-7)] if locals()["ally_count_"+str(i-7)] > 0 else 0
        for i in range(16, 24):
            self.params[i] = locals()["enermy_count_"+str(i-15)]  
        for i in range(24, 32):
            self.params[i] = locals()["ally_count_"+str(i-23)]
        
        
    
    def _get_action(self, net_ind):
        out = self.net[net_ind](self.params)
        if out[-1] > 0.5:
            action = self.n_actions_no_attack
        else:
            out -= 0.5
            if out[0] >= out[1]: # move along x
                if out[0] > 0:
                    action = self.actions['East']
                else:
                    action = self.actions['West']
            else: # move along y
                if out[1] > 0:
                    action = self.actions['North']
                else:
                    action = self.actions['South']
        return action 
    
    
    def evolve(self):
        max_rewards, mean_rewards, min_rewards = [], [], []
        for epoch in range(self.n_epoches):
            rewards = []
            for net_ind in range(len(self.net)):
                
                terminated = False
                total_reward = 0
                self.env.reset()
                while not terminated:
                    reward, terminated, _ = self.step(net_ind, 0)
                    total_reward += reward 
                
                rewards.append(reward)
                 
            print('epoch: {} max reward: {} min reward: {} mean reward: {}'.format((epoch+1), np.max(rewards), \
                                                                                   np.min(rewards), np.mean(rewards)))
            max_rewards.append(np.max(rewards))
            mean_rewards.append(np.mean(rewards))
            min_rewards.append(np.min(rewards))
            self.topology_update(rewards)
            
        self.env.close()
        return max_rewards, mean_rewards, min_rewards
        
    def topology_update(self, rewards):
        r_argmax= np.argmax(rewards)
        best_net = self.net[r_argmax]
        
        for i in range(len(self.net)):
            if i == r_argmax:
                continue
            
            net_i = self.net[i]
            net_i.load_state_dict(best_net.state_dict())
            for layer in range(net_i.L):
                new_weight = net_i.FC_layers[layer].weight*np.random.uniform(self.evolving_low, self.evolving_high)
                net_i.FC_layers[layer].weight = nn.Parameter(new_weight, requires_grad=True)
                
            self.net[i] = net_i 
        
        ind = np.random.randint(len(self.net))
        while ind == r_argmax:
            ind = np.random.randint(len(self.net))
        self.net[ind] = MLP()
        
        
class MLP(nn.Module):
    '''
    A 4-layer NN with leaky relu for predicting actions
    '''
    def __init__(self, L=4, input_dim=40):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , 3 , bias=True ))
        
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x.T
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.leaky_relu(y)
        y = self.FC_layers[self.L](y)
        out = torch.sigmoid(y)
        return out.squeeze()
        
    
parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='2m_vs_1z', help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=2, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='A', help='The difficulty of built-in computer AI bot (default is "7").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes the game will run for.')
parser.add_argument('--agent', default="AlternatingFire", type=str, help='Number of episodes the game will run for.')
parser.add_argument('--opt', default=torch.optim.Adam, help='Optimizer class')

args = parser.parse_args()
        
if __name__ == "__main__":
    
    map_name = args.map_name
    step_mul = args.step_mul
    difficulty = args.difficulty
    reward_sparse = args.reward_sparse
    debug = args.debug 
    n_episodes = args.n_episodes
    lr = args.lr
    opt = args.opt

    env = MMEnv(map_name=map_name, step_mul=step_mul, difficulty=difficulty, reward_sparse=reward_sparse, debug=debug)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    nn_agent = NN_Agent(n_agents, n_episodes, opt)
    nn_agent.fit(env)
    max_rewards, mean_rewards, min_rewards = nn_agent.evolve()
    
    fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
    ax.plot(np.arange(len(max_rewards)), max_rewards, label='max reward')
    plt.plot(np.arange(len(min_rewards)), min_rewards, label='min reward')
    plt.plot(np.arange(len(mean_rewards)), mean_rewards, label='mean reward')
        
    ax.set_title(f'NN - {map_name}', fontdict={'size':22})
    ax.set(ylabel='Rewards', ylim=(0, 1.2))
    ax.set(xlabel='Episode')
    plt.legend()
    plt.show()
    fig.savefig(f"nn_{map_name}.png")
    plt.close(fig)