from scdc.agents.base_agent import BaseAgent
# import base_agent
from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class FocusFire():
    def __init__(self, n_agents, env):
        # Passive personality is never recommonded
        self.n_agents = n_agents 
        self.env = env
        
    def step(self, obs, state):                    
        all_closest_id = self.find_top_closest()
        actions = [6+all_closest_id]*self.n_agents
                    
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated 
    
    def find_top_closest(self):
        center_x, center_y = self.env.get_ally_center()
        
        target_items = self.env.enemies.items()
        all_closest_id = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            # print()
            # print(unit)
            # print()             
            for t_id, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    dist = self.env.distance(center_x, center_y, t_unit.pos.x, t_unit.pos.y)
                    if dist < min_dist:
                        min_dist = dist 
                        all_closest_id = t_id
                        
        return all_closest_id
    
    
class HybridAttack():
    def __init__(self, n_agents, env, alpha):
        self.n_agents = n_agents
        self.env = env
        self.alpha = alpha 
        
        
    def _calculate_score(self, alpha, health, distance):
        return alpha*health+(1-alpha)*distance 
    
    
    def step(self, obs, state):
        actions = []        
        target_items = self.env.enemies.items()

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            min_score_e_id = self.find_lowest_score(unit, target_items)
            action = 6+min_score_e_id
                
            actions.append(action)

        reward, terminated, _ = self.env.step(actions)
        
        return reward, terminated 
        
        
    def find_lowest_score(self, unit, target_items):
        min_score = None
        min_score_e_id = None
        for t_id, t_unit in target_items:
            if t_unit.health > 0:
                dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                score = self._calculate_score(self.alpha, t_unit.health, dist)
                if min_score is None or score < min_score :
                    min_score = score 
                    min_score_e_id = t_id
                    
        return min_score_e_id 
                
        
        
class AlternatingFire():
    # TODO: Implement alternating fire trick
    pass


class Kiting():
    # TODO: Implement kiting trick
    pass


class Positioning():
    # TODO: Implement positioning trick
    pass


class WallOff():
    # TODO: Implement wall off trick
    pass