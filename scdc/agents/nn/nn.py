from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
# https://github.com/ddehueck/pytorch-neat
class NN_Agent:
    def __init__(self, n_agents, env, n_epoches, lr, opt):
        self.n_agents = n_agents 
        self.env = env
        self.map = env.map_name 
        self.params = torch.zeros(40, 1)
        self.net = [MLP(), MLP(), MLP()]
        self.n_epoches = n_epoches
        self.lr = lr
        self.opt = opt
        self.prev_move = self.prev_prev_move = None 
        self.first = True
        self.second = False
        
    def step(self, net_ind):
        actions = []        

        for agent_id in range(self.n_agents):
            
            unit = self.env.get_unit_by_id(agent_id)
            self._fill_params(unit, self.prev_move, self.prev_prev_move)
            action = self._get_action(net_ind)
            self.prev_prev_move = self.prev_move
            self.prev_move = action
            if action == 6:
                close_e_id = self.find_closest(unit)
                action = 6+close_e_id
                
            actions.append(action)

        reward, terminated, _ = self.env.step(actions)
        
        if self.first:
            self.first = False
            self.second = True
        elif self.second:
            self.second = False 
        
        return reward, terminated 
    
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
            action = 6
        else:
            out -= 0.5
            if out[0] >= out[1]: # move along x
                if out[0] > 0:
                    action = 4
                else:
                    action = 5
            else: # move along y
                if out[1] > 0:
                    action = 2
                else:
                    action = 3
        return action 
    
    
    def train(self):

        for epoch in range(self.n_epoches):
            rewards = []
            for net_ind in range(len(self.net)):
                
                terminated = False
                total_reward = 0
                self.env.reset()
                while not terminated:
                    reward, terminated = self.step(net_ind)
                    total_reward += reward 
                
                rewards.append(total_reward)
                 
            print('epoch: {} rewards: {}'.format((epoch+1), rewards))
            self.topology_update(rewards)
            
        self.env.close()
        
        
    def topology_update(self, rewards):
        r_argmax= np.argmax(rewards)
        best_net = self.net[r_argmax]
        
        lower = True
        
        for i in range(len(self.net)):
            if i == r_argmax:
                continue
            
            net_i = self.net[i]
            net_i.load_state_dict(best_net.state_dict())
            
            for layer in range(net_i.L):
                new_weight = net_i.FC_layers[layer].weight*0.99 if lower else net_i.FC_layers[layer].weight*1.01
                net_i.FC_layers[layer].weight = nn.Parameter(new_weight, requires_grad=True)
                
            self.net[i] = net_i 
            lower = False
        
class MLP(nn.Module):
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
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        out = torch.sigmoid(y)
        return out.squeeze()
        
    
parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='half_6m_vs_full_4m', help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=2, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='A', help='The difficulty of built-in computer AI bot (default is "7").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes the game will run for.')
parser.add_argument('--agent', default="AlternatingFire", type=str, help='Number of episodes the game will run for.')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--opt', default=torch.optim.Adam, help='Optimizer class')

# half_6m_vs_full_4m
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

    nn_agent = NN_Agent(n_agents, env, n_episodes, lr, opt)

    nn_agent.train()