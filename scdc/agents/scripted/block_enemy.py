from scdc.utils.game_utils import fine_closest_position

class BlockEnemy():

    def __init__(self, n_agents):
        self.n_agents = n_agents 
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        # 224 comes from 7*32, 11s is roughly step_mul of 32 and 7 steps.
        # 11s is the duration for field 
        # Notice that this is just a rough estimation
        self.field_duration = 7*(32/self.env._step_mul)
        self.activated = False
        
    def step(self, plot_level):                    
       
        allies = self.env.agents.items()   
        enemies = self.env.enemies
        actions = []
        
        center_x, center_y  = self.env.get_ally_center()
        if self.activated:
            min_e_id = self.find_close_to_field_exit()
        else:
            min_e_id, _ = fine_closest_position((center_x, center_y), enemies)
                
        for a_id, a_unit in allies:
            if self.env.unit_to_name(a_unit) == 'sentry' and a_unit.energy >= 50: # 50 is the minimum energy level
                actions.append(6)
                self.activated = True
                    
            else:
                actions.append(self.n_actions_no_attack+min_e_id)
                
            if self.activated:
                self.field_duration -= 1
                if self.field_duration == 0:
                    self.activated = False
            
            
        reward, terminated, _ = self.env.step(actions)
        if plot_level > 0:
            return actions, reward, terminated 
        return reward, terminated 
       
    def find_close_to_field_exit(self):
        '''
        The allies can simply attack the closest one to the center.
        However, the closest one to the center is necessarily the one passing the corridor right now
        '''
        # Find lowest in y and leftmost in x
        enemies = self.env.enemies.items()
        close_e_id = None
        close_x = None
        for e_id, e_unit in enemies:
            if e_unit.health > 0:
                if close_x is None or e_unit.pos.x < close_x:
                    close_e_id = e_id
                    close_x = e_unit.pos.x 
        
        return close_e_id 