from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class InformationTable():
    '''
    Adv. verson of HybridAttack
    '''
    def __init__(self, n_agents, env, w1, w2, c, delta, r, e, potential_field_mearure='health'):
        self.n_agents = n_agents
        self.env = env
        self.w1 = w1 # constants related to unit health
        self.w2 = w2 # constants related to unit health
        self.delta = delta # influence decay factor
        self.r = r # maximum disntace that influence will be calculated on
        self.c = c # coefficient used for calculating field
        self.e = e # expoenet used for calculating field
        self.potential_field_mearure = potential_field_mearure
        
    def _calculate_score(self, alpha, health, distance):
        return alpha*health+(1-alpha)*distance 
    
    def _init_table(self):
        rows, cols = self.env.n_agents, self.env.n_enemies
        self.info_table = np.zeros((rows, cols))
        self.potential_field_table = np.zeros((rows, cols))
        
    def _update_table(self):
        target_items = self.env.enemies.items()
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            if unit.health < 0:
                continue
            for t_id, t_unit in target_items:
                if t_unit.health > 0: 
                    dist_e_unit = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist_e_unit > self.r:
                        self.info_table[agent_id, t_id] = 0
                    else:
                        unit_hp_fraction = unit.health/unit.max_health
                        e_hp_fraction = t_unit.health/t_unit.max_health
                        
                        influence = self.w1*e_hp_fraction+self.w2 
                        dist_decay = self.delta**self.dist_e_unit # this may be too aggressive
                        influence *= dist_decay 
                        
                        if self.potential_field_mearure == 'health':
                            base = t_unit.health 
                        elif self.potential_field_mearure == 'distance':
                            base = dist_e_unit
                        elif self.potential_field_mearure == 'cooling_down':
                            base = t_unit.cool_down
                        self.potential_field_table[agent_id, t_id] = self.c*base**self.e 
                        
                        self.info_table[agent_id, t_id] = influence

    def step(self):
        actions = []        

        for agent_id in range(self.n_agents):
            
            unit = self.env.get_unit_by_id(agent_id)
            if unit.health > 0:
                a_table = self.info_table[agent_id]
                max_inf = np.max(a_table)
                e_id = np.argmax(a_table)
                # check if there are same values
                if (a_table == max_inf).sum() > 1:
                    ind = np.arange(self.env.n_enemies)[a_table == max_inf]
                    pf_table = self.potential_field_table[agent_id][ind]
                    max_pf = np.max(pf_table)
                    for i in range(self.env.n_enemies):
                        if pf_table[i] == max_pf:
                            e_id = i
                            break 
                                                
                actions.append(e_id+6)
            else:
                actions.append(1)

        reward, terminated, _ = self.env.step(actions)
        
        return reward, terminated 
        
        