from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class neat:
    def __init__(self, n_agents, env):
        self.n_agents = n_agents 
        self.env = env
        self.map = env.map_name 
        self.params = torch.zeros(40, 1)
        self.net = SCMLP()
        self.prev_move = self.prev_prev_move = None 
        self.first = True
        self.second = False
        
    def step(self):
        actions = []        

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            action = self._get_action(unit, self.prev_move, self.prev_prev_move)
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
            exec("enermy_avg_dist_"+str(i))
            exec("ally_avg_dist_"+str(i))
            exec("enermy_count_"+str(i))
            exec("ally_count_"+str(i))
            
        shoot_range = self.env.unit_shoot_range(unit)
        
        unit_x, unit_y = unit.pos.x, unit.pos.y 
        map_x, map_y = self.env.map_distance_x, self.env.map_distance_y
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
                
                if e_pos_x >= unit_x and e_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        ally_count_1 += 1
                        ally_avg_dist_1 += dist_to_ally
                    else:
                        ally_count_2 += 1
                        ally_avg_dist_2 += dist_to_ally
                elif e_pos_x < unit_x and e_pos_y >= unit_y:
                    if dist_to_ally <= shoot_range:
                        ally_count_3 += 1
                        ally_avg_dist_3 += dist_to_ally
                    else:
                        ally_count_4 += 1
                        ally_avg_dist_4 += dist_to_ally
                elif e_pos_x >= unit_x and e_pos_y < unit_y:
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
            self.params[i] = globals()["enermy_avg_dist_"+str(i+1)]/globals()["enermy_count_"+str(i+1)]  
        for i in range(8, 16):
            self.params[i] = globals()["ally_avg_dist_"+str(i-7)]/globals()["ally_count_"+str(i-7)]
        for i in range(16, 24):
            self.params[i] = globals()["enermy_count_"+str(i-15)]  
        for i in range(24, 32):
            self.params[i] = globals()["ally_count_"+str(i-23)]
        
        
    
    def _get_action(self):
        out = self.net(self.params)
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
    
    
    def _evolve(self):
        # evoling SCMLP using NEAT
        raise NotImplementedError
        
class SCMLP(nn.Module):
    def __init__(self, L=4, input_dim=40):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , 4 , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        
        return F.sigmoid(y)
        