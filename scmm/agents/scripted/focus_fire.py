import numpy as np
from scmm.utils.game_utils import fine_closest_position, find_weakest_injured, fine_weakest_enemy

class FocusFire():
    '''
    Focus on attacking some enemies instead of spreading fire.
    No Overkill can be helpful in fast elimination.
    Notice that due to the delay of executing actions in games, 
    the actual overkill in this case does not work well
    '''
    def __init__(self, n_agents, no_over_kill=False):
        self.n_agents = n_agents 
        self.no_over_kill = no_over_kill
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, plot_level):                    
       
        if self.no_over_kill:
            actions = self.find_focus_targets()
        else:
            center_x, center_y = self.env.get_ally_center(True)
            # min_e_id, _, _ = fine_closest_position((center_x, center_y), self.env.enemies) # focus on attacking the closest one
            min_e_id, _ = fine_weakest_enemy(self.env.enemies) # Uncomment this line to attack on the weakest one
            actions = []

            for agent_id in range(self.n_agents):
                unit = self.env.get_unit_by_id(agent_id)    
                if unit.health <= 0:
                    actions.append(self.actions['No-Op'])
                    continue
                
                if self.env.unit_to_name(unit) == 'medivac':
                    min_a_id = find_weakest_injured(agent_id, self.env.agents)
                    if min_a_id is None:
                        actions.append(self.actions['Stop'])
                    else:
                        actions.append(self.n_actions_no_attack+min_a_id)
                        
                else:
                    actions.append(self.n_actions_no_attack+min_e_id)

        reward, terminated, info = self.env.step(actions)
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 
        
    
    def find_focus_targets(self):
        # Find top k closest targets and each ally unit doesn't do damage more than what is required to kill them
        # Must use on homogeneous maps at the moment
        center_x, center_y = self.env.get_ally_center()
        
        target_items = self.env.enemies.items()
        
        # sort distances
        dist_arr = [] 
        hp_arr = []
        e_id_arr = []
        
        for t_id, t_unit in target_items: # t_id starts from 0
            
            if t_unit.health > 0:
                e_id_arr.append(t_id)
                dist = self.env.distance(center_x, center_y, t_unit.pos.x, t_unit.pos.y)
                dist_arr.append(dist)
                hp_arr.append(t_unit.health)        
                
        
        dist_arr = np.array(dist_arr)
        hp_arr = np.array(hp_arr)
        e_id_arr = np.array(e_id_arr)
        
        ind = np.argsort(hp_arr)
        
        dist_arr = dist_arr[ind]
        hp_arr = hp_arr[ind]
        e_id_arr = e_id_arr[ind]
        
        actions = []
        enermy_counter, accum_damage = 0, 0
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)    
            if unit.health > 0:
                if self.env.unit_to_name(unit) == 'medivac':
                    min_a_id = find_weakest_injured(agent_id, self.env.agents)
                    if min_a_id is None:
                        actions.append(1)
                    else:
                        actions.append(self.n_actions_no_attack+min_a_id)
                    
                    continue
                
                accum_damage += self.env.unit_damage(unit)
                actions.append(e_id_arr[enermy_counter]+self.n_actions_no_attack)
                
                if hp_arr[enermy_counter] <= accum_damage:
                    accum_damage = 0
                    
                    if enermy_counter+1 >= len(e_id_arr):
                        remaining = self.n_agents - agent_id - 1
                        for _ in range(remaining):
                            actions.append(np.random.randint(0, len(e_id_arr))+self.n_actions_no_attack)
                            return actions 
                    else:
                        enermy_counter += 1
            else:
                actions.append(self.actions['No-Op'])
        
        return actions 