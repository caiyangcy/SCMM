from scmm.utils.game_utils import get_opposite_direction, fine_closest_position, find_weakest_injured
import numpy as np

class DyingRetreat():
    '''
    Units that have been marked as dying will retreat from the combat.
    '''
    def __init__(self, n_agents,  use_accum=True, dying_factor=1, consective_move=2):
        self.n_agents = n_agents 
        self.retreated = [0]*self.n_agents # has used up all retreat times or not
        self.use_accum = use_accum # Whether use accumulated damage from last step
        self.dying_factor = dying_factor # A factor that is used to check if a unit is dying
        
        self.max_move = consective_move # Max of retreat times. Units cannot keep retreating
        self.consective_move = [self.max_move]*self.n_agents
        
        if use_accum:
            # self.hp_thres will be ignores if use accumulation of damage
            self.accum_damage = np.zeros((self.n_agents, ))
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def step(self, plot_level):                    
       
        allies = self.env.agents.items()        
        actions = []
        
        for a_id, a_unit in allies:
            if a_unit.health > 0:
                
                if self.env.unit_to_name(a_unit) == 'medivac':
                    
                    min_a_id = find_weakest_injured(a_id, self.env.agents)
                    if min_a_id is None:
                        actions.append(self.actions['Stop'])
                    else:
                        actions.append(self.n_actions_no_attack+min_a_id)
                    continue 
                
                closest_id, closest_unit,_ = fine_closest_position(a_unit.pos, self.env.enemies)
                
                if closest_id is not None:
                    actions.append(self.n_actions_no_attack)
                    continue

                if 0 < self.consective_move[a_id] < self.max_move or (a_unit.health <= self.dying_factor*self.accum_damage[a_id] and not self.retreated[a_id]):
                    # If the unit is dying and it has remaining retreat steps, then it will retreat
                    opposite = get_opposite_direction(a_unit.pos, closest_unit.pos)
                    actions.append(opposite)
                    self.consective_move[a_id] -= 1
                    if self.consective_move[a_id] == 0:
                        self.retreated[a_id] = 1
                        
                else: # the ally can either attack or it has retreated and cannot move back again
                    actions.append(closest_id+self.n_actions_no_attack)
                    
            else:
                actions.append(self.actions["No-Op"])
        
       
        prev_hp = np.array([a_unit.health for a_id, a_unit in allies])
        
        reward, terminated, info = self.env.step(actions)
        
        if not terminated:
            new_hp = np.array([a_unit.health for a_id, a_unit in allies])
            self.accum_damage = prev_hp-new_hp
        
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 
    