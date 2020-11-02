from scmm.utils.game_utils import fine_closest_position

class BlockEnemy():

    def __init__(self, n_agents):
        self.n_agents = n_agents 
        
    def fit(self, env):
        assert env.map_name in {'6s1s_vs_10r'}, "a\BlockEnemy only works for 6s1s_vs_10r"
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        # 224 comes from 7*32, 11s is roughly step_mul of 32 and 7 steps.
        # 11s is the duration for field 
        # Notice that this is just a rough estimation
        self.field_duration = 7*(32/self.env._step_mul)
        self.activated = False
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, plot_level):                    
       
        allies = self.env.agents.items()   
        enemies = self.env.enemies
        actions = []
        
        center_x, center_y  = self.env.get_ally_center()
        if self.activated:
            min_e_id = self.find_close_to_field_exit()
        else:
            min_e_id, _, _ = fine_closest_position((center_x, center_y), enemies)
                
        for a_id, a_unit in allies:
            if a_unit.health <= 0:
                actions.append(self.actions['No-Op'])
                continue
            
            if self.env.unit_to_name(a_unit) == 'sentry' and a_unit.energy >= 50: # 50 is the minimum energy level
                # If the Sentry has enough energy to release the force field
                # Then do it
                actions.append(self.actions['Force Field'])
                self.activated = True
                    
            else:
                actions.append(self.n_actions_no_attack+min_e_id)
                
            if self.activated:# Countdown of the activation period of force field
                self.field_duration -= 1
                if self.field_duration == 0:
                    self.activated = False
            
            
        reward, terminated, info = self.env.step(actions)
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 
       
    def find_close_to_field_exit(self):
        '''
        Find the enemy that is closest to the exit of force field.
        The allies can simply attack the closest one to the center.
        However, the closest one to the center is not necessarily the one passing the corridor right now
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