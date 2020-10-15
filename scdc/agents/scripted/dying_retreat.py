from scdc.utils.game_utils import get_opposite_direction, fine_closest_position
import numpy as np

class DyingRetreat():
    def __init__(self, n_agents, hp_thres=0.2, use_accum=True, dying_factor=1, consective_move=2):
        self.n_agents = n_agents 
        self.retreated = [0]*self.n_agents
        self.hp_thres = hp_thres
        self.use_accum = use_accum
        self.dying_factor = dying_factor
        
        self.max_move = consective_move
        self.consective_move = [self.max_move]*self.n_agents
        
        if use_accum:
            # self.hp_thres will be ignores if use accumulation of damage
            self.accum_damage = np.zeros((self.n_agents, ))
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        
    def step(self, plot_level):                    
       
        allies = self.env.agents.items()        
        enemies = self.env.enemies
        actions = []
        
        for a_id, a_unit in allies:
            if a_unit.health > 0:
                closest_id, closest_unit = fine_closest_position(a_unit.pos, enemies)
                # print('a_id: {} accum damage: {} remaining HP: {}'. format(a_id, self.accum_damage[a_id], a_unit.health))

                if 0 < self.consective_move[a_id] < self.max_move or (a_unit.health <= self.dying_factor*self.accum_damage[a_id] and not self.retreated[a_id]):
                    opposite = get_opposite_direction(a_unit.pos, closest_unit.pos)
                    actions.append(opposite)
                    self.consective_move[a_id] -= 1
                    if self.consective_move[a_id] == 0:
                        self.retreated[a_id] = 1
                        
                else: # the ally can either attack or it has retreated and cannot move back again
                    actions.append(closest_id+6)
                    
            else:
                actions.append(1)
        
       
        prev_hp = np.array([a_unit.health for a_id, a_unit in allies])
        reward, terminated, _ = self.env.step(actions)
        
        if not terminated:
            new_hp = np.array([a_unit.health for a_id, a_unit in allies])
            self.accum_damage = prev_hp-new_hp
        
        if plot_level > 0:
            return actions, reward, terminated 
        return reward, terminated 
    