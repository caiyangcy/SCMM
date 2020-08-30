from scdc.agents.base_agent import BaseAgent
# import base_agent
from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class FocusFire():
    def __init__(self, n_agents, env, personality='passive'):
        self.n_agents = n_agents 
        self.env = env
        assert personality == 'passive', "Only passive supported for now."
        self.personality = personality
        
    def step(self, obs, state):
        actions = []        

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            all_closest_id, avail_to_shoot = self.find_closest_to_center()
            closest_enermy = self.env.get_unit_by_id(all_closest_id)
            actions = []
            for agent_id in range(self.n_agents):
                if not avail_to_shoot[agent_id]: # This unit is too far from the closest one, need to move a distance
                    move_action = self.find_move_direction(unit, closest_enermy)
                    actions.append(move_action) if move_action else actions.append(1)
                else:
                    actions.append(6+all_closest_id)
        print(actions)
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated 
    
    def find_closest_to_all(self):
        target_items = self.env.enemies.items()
        all_closest_id = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        avail_to_shoot = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            shoot_range = self.env.unit_shoot_range(unit) #unit.shoot_range
            
            for t_id, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist < min_dist:
                        min_dist = dist 
                        all_closest_id = t_id
                        if dist <= shoot_range:
                            avail_to_shoot[agent_id] = 1
                        
            return all_closest_id, avail_to_shoot
    
    def find_closest_to_center(self):
        center_x, center_y = self.env.get_ally_center()
        
        target_items = self.env.enemies.items()
        all_closest_id = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        avail_to_shoot = [0]*self.n_agents
        
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            shoot_range = self.env.unit_shoot_range(unit) 
            
            for t_id, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    dist = self.env.distance(center_x, center_y, t_unit.pos.x, t_unit.pos.y)
                    if dist < min_dist:
                        min_dist = dist 
                        all_closest_id = t_id
                        if dist <= shoot_range:
                            avail_to_shoot[agent_id] = 1
                        
        return all_closest_id, avail_to_shoot
    
    def find_move_direction(self, unit, target):
        delta_x = target.pos.x - unit.pos.x
        delta_y = target.pos.y - unit.pos.y

        if abs(delta_x) > abs(delta_y): # east or west
            if delta_x > 0 and self.env.can_move(unit, Direction.EAST): # east
                return 4
            elif self.env.can_move(unit, Direction.WEST): # west
                return 5
        else: # north or south
            if delta_y > 0 and self.env.can_move(unit, Direction.NORTH): # north                        
                return 2
            elif self.env.can_move(unit, Direction.SOUTH): # south                        
                return 3
        return None
    
    
class AttackClosest(): # Always attcking
    def __init__(self, n_agents, env, personality='passive'):
        self.n_agents = n_agents 
        self.env = env
        assert personality == 'passive' or personality == 'aggressive', \
            "Unknown personality: it should be either passive or aggressive"
        self.personality = personality
        
    def step(self, obs, state):
        actions = []        
        target_items = self.env.enemies.items()

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            close_e_id, close_e_unit = self.find_closest_enermy(unit, target_items)
            
            if not close_e_id: # No closest enermy available Then move to the closest
                # print('close e id: ', close_e_id)
                # print('close e unit: ', close_e_unit)
                if self.personality != 'passive':
                    action = self.find_move_direction(unit, close_e_unit)
                else:
                    action = 1 # do nothing
            else:
                action = 6+close_e_id
                
            actions.append(action)

        reward, terminated, _ = self.env.step(actions)
        
        return reward, terminated 
        
    
    def find_closest_enermy(self, unit, target_items):
        shoot_range = self.env.unit_shoot_range(unit) #unit.shoot_range
        
        close_e_id = None 
        close_e_unit = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                if dist < min_dist:
                    min_dist = dist 
                    close_e_unit = t_unit 
                    if dist <= shoot_range:
                        close_e_id = t_id
                    
        return close_e_id, close_e_unit
    
    def find_move_direction(self, unit, target):
        delta_x = target.pos.x - unit.pos.x
        delta_y = target.pos.y - unit.pos.y

        if abs(delta_x) > abs(delta_y): # east or west
            if delta_x > 0 and self.env.can_move(unit, Direction.EAST): # east
                action = 4
            elif self.env.can_move(unit, Direction.WEST): # west
                action = 5
        else: # north or south
            if delta_y > 0 and self.env.can_move(unit, Direction.NORTH): # north                        
                action = 2
            elif self.env.can_move(unit, Direction.SOUTH): # south                        
                action = 3
        return action 





class AttackWeakest():
    def __init__(self, n_agents, env, personality='passive'):
        self.n_agents = n_agents 
        self.env = env
        assert personality == 'passive' or personality == 'aggressive', \
            "Unknown personality: it should be either passive or aggressive"
        self.personality = personality
        
    def step(self, obs, state):
        actions = []        
        target_items = self.env.enemies.items()

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            weakest_e_id, weakest_e_unit  = self.find_weakest_enermy(unit, target_items)
            
            if not weakest_e_id: # No closest enermy available
                # Then move to the closest
                if self.personality != 'passive':
                    action = self.find_move_direction(unit, weakest_e_unit)
                else:
                    action = 1 # do nothing
            else:
                action = 6+weakest_e_id
                
            actions.append(action)

        reward, terminated, _ = self.env.step(actions)
        
        return reward, terminated 
        
    
    def find_weakest_enermy(self, unit, target_items):
        shoot_range = self.env.unit_shoot_range(unit) #unit.shoot_range
        
        weakest_e_id = None
        weakest_e_unit = None
        min_hp = 99999

        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                if t_unit.health < min_hp:
                    weakest_e_unit = t_unit
                    min_hp = t_unit.health
                    
                    dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                
                    if dist <= shoot_range:
                        weakest_e_id = t_id
                    
        return weakest_e_id, weakest_e_unit

    
    def find_move_direction(self, unit, target):
        delta_x = target.pos.x - unit.pos.x
        delta_y = target.pos.y - unit.pos.y

        if abs(delta_x) > abs(delta_y): # east or west
            if delta_x > 0 and self.env.can_move(unit, Direction.EAST): # east
                action = 4
            elif self.env.can_move(unit, Direction.WEST): # west
                action = 5
        else: # north or south
            if delta_y > 0 and self.env.can_move(unit, Direction.NORTH): # north                        
                action = 2
            elif self.env.can_move(unit, Direction.SOUTH): # south                        
                action = 3
        return action 
    
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