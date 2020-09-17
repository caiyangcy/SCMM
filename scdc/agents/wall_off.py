from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class WallOff():
    '''
    hard-coded
    '''
    def __init__(self, n_agents, env):
        assert env.map_name == 'corridor', "WallOff trick only works for corridor maps."
        self.n_agents = n_agents 
        self.env = env
        self.destination_point = (9, 9) # this one is hard-coded 
        self.arrival = False 
        
    def step(self):
        actions = []
        if self.arrival:
            actions = self.find_close_k()
            reward, terminated, _ = self.env.step(actions)
            return reward, terminated      
            
        center_x, center_y = self.env.get_ally_center()
        if self.env.distance(center_x, center_y, self.destination_point[0], self.destination_point[1]) < 2:
            self.arrival = True
            actions = self.find_close_k()
        else:
            if center_x > center_y:
                actions = [5]*self.n_agents
            else:
                actions = [3]*self.n_agents
                
        
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated      
    
    
    def find_close_k(self):
        e_id_arr = []
        target_items = self.env.enemies.items()
        for agent_id in range(self.n_agents):
            
            unit = self.env.get_unit_by_id(agent_id)
            min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
            min_dist_id = -5
            
            for t_id, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_id = t_id 
            
            e_id_arr.append(min_dist_id+6)
        
        return e_id_arr