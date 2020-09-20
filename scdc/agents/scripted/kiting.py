from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class Kiting():
    '''
    Kiting is only tested on 3s_vs_3z, 3s_vs_4z, 3s_vs_5z
    '''
    def __init__(self, n_agents, env, consuctive_attack_count=7):
        assert env.map_name in {'3s_vs_3z', '3s_vs_4z', '3s_vs_5z'}, "Kiting trick only works for 3s_vs_3z, 3s_vs_4z, 3s_vs_5z maps."
        self.n_agents = n_agents 
        self.env = env
        self.distination_point = 0
        self.move_direction = 2
        self.direction_map = {2:4, 3:5, 4:3, 5:2}
        self.consuctive_attack_count = consuctive_attack_count
        self.ready_for_attack = False
    
    def step(self):
        closest_e_id, close_e_unit, closest_dist, move_direction = self.find_closest()
        actions = []
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            
            if self.ready_for_attack:
                actions.append(6+closest_e_id)
                self.consuctive_attack_count -= 1
                if self.consuctive_attack_count == 0:
                    self.ready_for_attack = False

                
            elif self.env.unit_shoot_range(unit) >= closest_dist and self.env.unit_shoot_range(close_e_unit) < 1.2*closest_dist:
            # if self.env.unit_shoot_range(unit) - self.env.unit_shoot_range(close_e_unit) > closest_dist:
                actions.append(6+closest_e_id)
                self.ready_for_attack = True
                self.consuctive_attack_count = 7
                
            elif self.env.unit_shoot_range(close_e_unit) >= 1.2*closest_dist:
                actions.append(move_direction)
                
            else: # find enermy and make them give a chase
                # passive
                actions.append(1)
                
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated 

    def find_closest(self):
        center_x, center_y = self.env.get_ally_center()
        
        target_items = self.env.enemies.items()
        all_closest_id, all_closest_unit = None, None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        move_direction = None
        
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                
                e_pos_x, e_pos_y = t_unit.pos.x, t_unit.pos.y
                
                dist = self.env.distance(center_x, center_y, e_pos_x, e_pos_y)
                if dist < min_dist:
                    min_dist = dist 
                    all_closest_id = t_id
                    all_closest_unit = t_unit
        
                    
        e_pos_x, e_pos_y = self.env.get_enermy_center()      
        map_x, map_y = self.env.map_x, self.env.map_y
        # botton left is (0, 0)
        # N S E W: 2, 3, 4, 5
        
        # The first four are directly hard-coded
        discount_factor = 0.2
        
        offset_x, offset_y = map_x*discount_factor, map_y*discount_factor
        distination_list = [(offset_x, map_y-offset_y), (map_x-offset_x, map_y-offset_y), (map_x-offset_x,offset_y),  (offset_x, offset_y)]
        
        move_direction = self.move_direction
        distination_point = distination_list[self.distination_point]
        dist = self.env.distance(center_x, center_y, distination_point[0], distination_point[1])
        # print('dist: ', dist)
        if dist < 5: # check if close to destination
            self.distination_point = (self.distination_point+1)%4
            move_direction = self.direction_map[move_direction]
            self.move_direction = move_direction 
            
        return all_closest_id, all_closest_unit, min_dist, move_direction