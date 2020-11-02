import math

class WallOff():
    '''
    hard-coded Wall off for corridor map
    '''
    def __init__(self, n_agents):
        self.n_agents = n_agents 
        self.destination_point = (9.5, 9.5) # this one is hard-coded 
        self.arrival = False 
        
    def fit(self, env):
        assert env.map_name == 'corridor', "WallOff trick only works for corridor maps."
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, plot_level):
        actions = []
        if self.arrival:
            actions = self.find_close_k()
            reward, terminated, info = self.env.step(actions)
            if plot_level > 0:
                return actions, reward, terminated, info 
            return reward, terminated, info 
            
        center_x, center_y = self.env.get_ally_center()
        if self.env.distance(center_x, center_y, self.destination_point[0], self.destination_point[1]) < 2:
            self.arrival = True
            actions = self.find_close_k()
        else:
            if center_x > center_y:
                actions = [self.actions["West"]]*self.n_agents
            else:
                actions = [self.actions["South"]]*self.n_agents
                
        
        reward, terminated, info = self.env.step(actions)
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 
    
    
    def find_close_k(self):
        e_id_arr = []
        target_items = self.env.enemies.items()
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            if unit.health <= 0:
                e_id_arr.append(self.actions['No-Op'])
                continue
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