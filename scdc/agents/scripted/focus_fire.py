import numpy as np
import math

class FocusFire():
    def __init__(self, n_agents, no_over_kill=False):
        self.n_agents = n_agents 
        self.no_over_kill = no_over_kill
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        
    def step(self, plot_level):                    
       
        if self.no_over_kill:
            actions = self.find_focus_targets()
        else:
            all_closest_id = self.find_closest()
            actions = [self.n_actions_no_attack+all_closest_id]*self.n_agents
            
        reward, terminated, _ = self.env.step(actions)
        return reward, terminated 
        
    
    def find_closest(self):
        center_x, center_y = self.env.get_ally_center()
        target_items = self.env.enemies.items()
        all_closest_id = None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
                  
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                dist = self.env.distance(center_x, center_y, t_unit.pos.x, t_unit.pos.y)
                if dist < min_dist:
                    min_dist = dist 
                    all_closest_id = t_id
                        
        return all_closest_id
    
    def find_focus_targets(self):
        # Find top k closest targets and each ally unit doesn't do damage more than enough to kill them
        # Must use on homogeneous maps at the moment
        center_x, center_y = self.env.get_ally_center()
        
        target_items = self.env.enemies.items()
        
        # sort distances
        dist_arr = [] 
        hp_arr = []
        e_id_arr = []
        
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                dist = self.env.distance(center_x, center_y, t_unit.pos.x, t_unit.pos.y)
                dist_arr.append(dist)
                hp_arr.append(t_unit.health)        
                e_id_arr.append(t_id)
        
        dist_arr = np.array(dist_arr)
        hp_arr = np.array(hp_arr)
        e_id_arr = np.array(e_id_arr)
        
        ind = np.argsort(dist_arr)
        
        dist_arr = dist_arr[ind]
        hp_arr = hp_arr[ind]
        e_id_arr = e_id_arr[ind]
        
        attack_id = []
        enermy_counter, count = 0, 0
        single_unit_damage = self.env.unit_damage(self.env.get_unit_by_id(0))
        
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)    
            if unit.health > 0:
                count += 1
                attack_id.append(e_id_arr[enermy_counter]+6)
                
                if hp_arr[enermy_counter] <= single_unit_damage*count:
                    count = 0
                    
                    if enermy_counter+1 >= len(e_id_arr):
                        remaining = self.n_agents - agent_id
                        for _ in range(remaining):
                            attack_id.append(np.random.randint(0, len(e_id_arr))+6)
                            return attack_id 
                    else:
                        enermy_counter += 1
            else:
                attack_id.append(1)
        
        return attack_id 
    
    