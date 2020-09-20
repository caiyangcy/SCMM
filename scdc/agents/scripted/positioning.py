from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class Positioning():
    '''
    hard-coded
    '''
    def __init__(self, n_agents, env):
        assert env.map_name in {'bane_vs_bane', 'so_many_baneling', '2c_vs_64zg'}, "Not supported map"
        self.n_agents = n_agents 
        self.env = env
        self.map_name = env.map_name
        self.in_position = False
        
    def step(self):
        if self.map_name == 'bane_vs_bane':
            if self.in_position:
                actions = self.find_close_k()
            else:
                self.in_position, actions = self.partition()                
        elif self.map_name == 'so_many_baneling':
            spread_points = [(10, 18.5), (19.5, 12.5), (7.5, 15), (15, 8), (4, 12), (12, 4), (3, 3.5)]
            
            actions = []
            actions2 = []
            for i in range(self.n_agents):
                unit = self.env.get_unit_by_id(i)
                if unit.health > 0:
                    actions2.append([unit.pos.x, unit.pos.y])
                    if self.env.distance(unit.pos.x, unit.pos.y, spread_points[i][0], spread_points[i][1]) < 0.5:
                        close_e_dist, close_e_id = self.find_close(unit)
                        if close_e_dist <= self.env.unit_shoot_range(unit):
                            actions.append(close_e_id+6)
                        else:
                            actions.append(1)
                    else:
                        action = 1
                        if spread_points[i][0] > unit.pos.x or spread_points[i][1] > unit.pos.y:
                            # move top right
                            if spread_points[i][0]-unit.pos.x > spread_points[i][1]-unit.pos.y:
                                action = 4
                            else:
                                action = 2
                            
                        elif spread_points[i][0] < unit.pos.x or spread_points[i][1] < unit.pos.y:
                            if np.abs(spread_points[i][0]-unit.pos.x) > np.abs(spread_points[i][1]-unit.pos.y):
                                action = 5
                            else:
                                action = 3
                        actions.append(action)
                else:
                    actions.append(1)
        else:
            y_thres = 12.5
            actions = []
            e_center_x, e_center_y = self.env.get_enermy_center()
            
            e_higher_than_y = 0
            e_total = 0
            target_items = self.env.enemies.items()
            for _, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    e_total += 1
                    if t_unit.pos.y > y_thres:
                        e_higher_than_y += 1
            
            y1, y2 = self.env.get_unit_by_id(0).pos.y, self.env.get_unit_by_id(1).pos.y
            
            if e_higher_than_y <= e_total*0.7:
                if y1 >= 14 and y2 >= 14: # likely to be on the bridge
                    # find close and attack
                    for i in range(self.n_agents):
                        unit = self.env.get_unit_by_id(i)
                        close_e_dist, close_e_id = self.find_close(unit)
                        actions.append(close_e_id+6)
                elif e_higher_than_y >= e_total*0.4:
                    actions = [2, 2]
                else:
                    for i in range(self.n_agents):
                        unit = self.env.get_unit_by_id(i)
                        close_e_dist, close_e_id = self.find_close(unit)
                        actions.append(close_e_id+6)
                        
            elif e_higher_than_y > e_total*0.7:
                if y1 < 11 and y2 < 11: # likely to be on the plane
                    # find close and attack
                    for i in range(self.n_agents):
                        unit = self.env.get_unit_by_id(i)
                        close_e_dist, close_e_id = self.find_close(unit)
                        actions.append(close_e_id+6)
                else:
                    actions = [3, 3]
            
            print()
            
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated    
    
    def partition(self):
        if self.map_name == 'bane_vs_bane':
            # 5 partitions
            zerg_partition_counter = 1
            actions = []
            patition_done = True
            
            for agent_id in range(self.n_agents):
                unit = self.env.get_unit_by_id(agent_id)
                if unit.health > 0:
                    if unit.unit_type == self.env.baneling_id:
                        actions.append(1)
                    else:
                        # 20 zergs in total
                        pos = unit.pos 
                        if zerg_partition_counter <= 5: 
                            move_direction = self.get_move_direction(pos, (7, 11), zerg_partition_counter)
                        elif zerg_partition_counter <= 10: 
                            move_direction = self.get_move_direction(pos, (10, 8), zerg_partition_counter)
                        elif zerg_partition_counter <= 15: 
                            move_direction = self.get_move_direction(pos, (21, 11), zerg_partition_counter)
                        elif zerg_partition_counter <= 20: 
                            move_direction = self.get_move_direction(pos, (24, 8), zerg_partition_counter)
                            
                        zerg_partition_counter += 1
                        
                        actions.append(move_direction)
                        if actions[-1] != 1:
                            patition_done = False
                else:
                    actions.append(1)
                    
            return patition_done, actions
        
    
    def get_group_center(self, group):
        center_x, center_y = 0, 0
        for unit in group:
            center_x += unit.pos.x 
            center_y += unit.pos.y
        return (center_x/len(group), center_y/len(group))
        
        
    def get_move_direction(self, center, distination, zerg_partition_counter):
        center_x, center_y = center.x, center.y
        if self.env.distance(center_x, center_y, distination[0], distination[1]) < 3:
            return 1
        else:
            
            if zerg_partition_counter <= 5:
                return 5
            elif zerg_partition_counter <= 10:
                if np.abs(distination[0]-center_x) > np.abs(distination[1]-center_y):
                    action = 5
                else:
                    action = 3
                    
            elif zerg_partition_counter <= 15:
                return 4
                    
                    
            elif zerg_partition_counter <= 20:
                if np.abs(distination[0]-center_x) > np.abs(distination[1]-center_y):
                    action = 4
                else:
                    action = 3

        return action
    
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
    
    def find_close(self, unit):
        target_items = self.env.enemies.items()

        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        min_e_id = None
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                if dist < min_dist:
                    min_dist = dist
                    min_e_id = t_id
        return min_dist, min_e_id