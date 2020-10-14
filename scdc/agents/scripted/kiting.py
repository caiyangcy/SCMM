from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class Kiting():
    '''
    Kiting is only tested on 3s_vs_3z, 3s_vs_4z, 3s_vs_5z
    '''
    def __init__(self, n_agents, consuctive_attack_count=10):
        # assert env.map_name in {'3s_vs_3z', '3s_vs_4z', '3s_vs_5z', '3s_vs_3z_medium', '3s_vs_4z_medium', '3s_vs_5z_medium'}, \
        #     "Kiting trick only works for 3s_vs_3z, 3s_vs_4z, 3s_vs_5z maps."
        self.n_agents = n_agents 
        self.distination_point = 0
        self.move_direction = 2
        self.direction_map = {2:4, 3:5, 4:3, 5:2}
        self.consuctive_attack_count = consuctive_attack_count
        self.ready_for_attack = False
    
    def fit(self, env):
        self.env = env 
        self.n_actions_no_attack = self.env.n_actions_no_attack
        
    
    def step(self, adv_plot):
        closest_e_id, close_e_unit, closest_dist, move_direction = self.find_closest()
        actions = []
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            
            if self.ready_for_attack:
                actions.append(self.n_actions_no_attack+closest_e_id)
                self.consuctive_attack_count -= 1
                if self.consuctive_attack_count == 0:
                    self.ready_for_attack = False
                                
            elif self.env.unit_shoot_range(unit) >= closest_dist and self.env.unit_shoot_range(close_e_unit) < closest_dist:
            # if self.env.unit_shoot_range(unit) - self.env.unit_shoot_range(close_e_unit) > closest_dist:
                actions.append(self.n_actions_no_attack+closest_e_id)
                self.ready_for_attack = True
                self.consuctive_attack_count = 10
                
            elif self.env.unit_shoot_range(close_e_unit) >= closest_dist:
                actions.append(move_direction)
                
            else: # find enermy and make them give a chase
                # passive
                actions.append(1)
                
        
        reward, terminated, _ = self.env.step(actions)
        
        if adv_plot:
            return actions, reward, terminated 
        return reward, terminated 

    def find_closest(self):        
        target_items = self.env.enemies.items()
        all_closest_id, all_closest_unit = None, None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        move_direction = None
        
        for t_id, t_unit in target_items: # t_id starts from 0
            print('t_id: ', t_id)
            assert False
            if t_unit.health > 0:
                e_pos_x, e_pos_y = t_unit.pos.x, t_unit.pos.y
                
                for agent_id in range(self.n_agents):
                    unit = self.env.get_unit_by_id(agent_id)    
                    if unit.health > 0: 
                        dist = self.env.distance(unit.pos.x, unit.pos.y, e_pos_x, e_pos_y)
                
                        if dist < min_dist:
                            min_dist = dist 
                            all_closest_id = t_id
                            all_closest_unit = t_unit
        
                    
        e_pos_x, e_pos_y = self.env.get_enemy_center()      
        map_x, map_y = self.env.playable_x_max, self.env.playable_y_max
        # botton left is (0, 0)
        # N S E W: 2, 3, 4, 5
        
        min_x, min_y = self.env.playable_x_min, self.env.playable_y_min
        # distination_list = [(min_x, map_y), (map_x, map_y), (map_x, min_y),  (min_x, min_y)]
        
        move_direction = self.move_direction
        # distination_point = distination_list[self.distination_point]
        
        ally_pos = self.env.get_ally_positions()
        
        if self.distination_point == 0:
            dist = np.min(np.abs(ally_pos[:,1]-map_y))
            # dist = self.env.distance(0, center_y, 0, map_y)
        elif self.distination_point == 1:
            dist = np.min(np.abs(ally_pos[:,0]-map_x))
            # dist = self.env.distance(center_x, 0, map_x, 0)
        elif self.distination_point == 2:
            dist = np.min(np.abs(ally_pos[:,1]-min_y))
            # dist = self.env.distance(0, center_y, 0, offset_y)
        elif self.distination_point == 3:
            dist = np.min(np.abs(ally_pos[:,0]-min_x))
            # dist = self.env.distance(center_x, 0, offset_x, 0)
        
        # print('dist: ', min_dist)
        # print('distination_point: ', distination_point)
        # print('move_direction: ', move_direction)        
        
        if dist < 5: # check if close to destination
            self.distination_point = (self.distination_point+1)%4
            move_direction = self.direction_map[move_direction]
            self.move_direction = move_direction 
            
        return all_closest_id, all_closest_unit, min_dist, move_direction