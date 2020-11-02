import numpy as np
import math
from scmm.utils.game_utils import fine_closest_position

class Positioning():
    '''
    hard-coded positioning for 'bane_vs_bane', 'so_many_baneling', '2c_vs_64zg'
    '''
    def __init__(self, n_agents):
        self.n_agents = n_agents 
        self.in_position = False
        self.consecutive_waits = 5
        
    def fit(self, env):
        assert env.map_name in {'bane_vs_bane', 'so_many_baneling', '2c_vs_64zg'}, "Not supported map"
        self.env = env 
        self.map_name = env.map_name
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, plot_level):
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
                        close_e_id, _, close_e_dist = fine_closest_position(unit.pos, self.env.enemies)
                        if close_e_dist <= self.env.unit_shoot_range(unit):
                            actions.append(close_e_id+self.n_actions_no_attack)
                        else:
                            actions.append(self.actions['Stop'])
                    else:
                        action = self.actions['Stop']
                        if spread_points[i][0] > unit.pos.x or spread_points[i][1] > unit.pos.y:
                            # move top right
                            if spread_points[i][0]-unit.pos.x > spread_points[i][1]-unit.pos.y:
                                action = self.actions['East']
                            else:
                                action = self.actions['North']
                            
                        elif spread_points[i][0] < unit.pos.x or spread_points[i][1] < unit.pos.y:
                            if np.abs(spread_points[i][0]-unit.pos.x) > np.abs(spread_points[i][1]-unit.pos.y):
                                action = self.actions['West']
                            else:
                                action = self.actions['South']
                        actions.append(action)
                else:
                    actions.append(self.actions['No-Op'])
        else:
            y_thres = 12.5
            actions = []
            e_center_x, e_center_y = self.env.get_enemy_center()
            
            e_higher_than_y = 0
            e_total = 0
            target_items = self.env.enemies.items()
            for _, t_unit in target_items: # t_id starts from 0
                if t_unit.health > 0:
                    e_total += 1
                    if t_unit.pos.y > y_thres:
                        e_higher_than_y += 1
            
            y1, y2 = self.env.get_unit_by_id(0).pos.y, self.env.get_unit_by_id(1).pos.y

                        
            if e_higher_than_y < e_total*0.25: # majority on plane
                for a_id, y in enumerate([y1, y2]):
                    if y >= 13: 
                        unit = self.env.get_unit_by_id(a_id)
                        close_e_id, _, close_e_dist = fine_closest_position(unit.pos, self.env.enemies)
                        actions.append(close_e_id+self.n_actions_no_attack)
                    else:
                        actions.append(self.actions['North'])
            elif e_higher_than_y > e_total*0.75: # majority on high ground
                for a_id, y in enumerate([y1, y2]):
                    if y < 12: 
                        unit = self.env.get_unit_by_id(a_id)
                        close_e_id, _, close_e_dist = fine_closest_position(unit.pos, self.env.enemies)
                        actions.append(close_e_id+self.n_actions_no_attack)
                    else:
                        actions.append(self.actions['South'])
            else:
                for a_id in range(self.n_agents):
                    unit = self.env.get_unit_by_id(a_id)
                    close_e_id, _, close_e_dist = fine_closest_position(unit.pos, self.env.enemies)
                    actions.append(close_e_id+self.n_actions_no_attack)
           
        reward, terminated, info = self.env.step(actions)
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 
    
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
                        if self.env.distance(unit.pos.x, unit.pos.y, 16, 11.5) < 3:
                            actions.append(self.get_move_direction_new(unit.pos, (16, 11.5)))
                        else:
                            actions.append(1)
                    else:
                        # 20 zergs in total
                        pos = unit.pos 
                        if zerg_partition_counter <= 5: 
                            move_direction = self.get_move_direction(pos, (7, 11), zerg_partition_counter)
                        elif zerg_partition_counter <= 10: 
                            move_direction = self.get_move_direction(pos, (10, 8), zerg_partition_counter)
                        elif zerg_partition_counter <= 15: 
                            move_direction = self.get_move_direction(pos, (22, 10), zerg_partition_counter)
                        elif zerg_partition_counter <= 20: 
                            move_direction = self.get_move_direction(pos, (25, 8), zerg_partition_counter)
                            
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
            return self.actions['Stop']
        else:
            
            if zerg_partition_counter <= 5:
                return self.actions['West']
            elif zerg_partition_counter <= 10:
                if np.abs(distination[0]-center_x) > np.abs(distination[1]-center_y):
                    action = self.actions['West']
                else:
                    action = self.actions['South']
                    
            elif zerg_partition_counter <= 15:
                return self.actions['East']
                    
                    
            elif zerg_partition_counter <= 20:
                if np.abs(distination[0]-center_x) > np.abs(distination[1]-center_y):
                    action = self.actions['East']
                else:
                    action = self.actions['South']

        return action
    
    
    def get_move_direction_new(self, pos, distination):
        x, y = pos.x, pos.y 
        distination_x, distination_y = distination[0], distination[1]
        if np.abs(distination_x-x) > np.abs(distination_y-y):
            if distination_x > x:
                action = self.actions['East']
            else:
                action = self.actions['West']
        else:
            if distination_y > y:
                action = self.actions['North']
            else:
                action = self.actions['South']
        
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
            
            e_id_arr.append(min_dist_id+self.n_actions_no_attack)
        
        return e_id_arr