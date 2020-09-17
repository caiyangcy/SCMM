from scdc.env.micro_env.mm_env import MMEnv, Direction
import time
import numpy as np
import argparse
import math

class AlternatingFire():
    def __init__(self, n_agents, env):
        assert env.map_name in {'2m_vs_1z', '2s_vs_1sc'}, "Alternating Fire trick only supports 2m_vs_1z and 2s_vs_1sc map"
        self.n_agents = n_agents 
        self.env = env
        self.attacking_agent = np.random.randint(0, 2)
        self.attack_count = 2
        self.init = True
        self.map = env.map_name 
        
    def step(self):
        if self.map == '2m_vs_1z':
            a1, a2 = self.env.get_unit_by_id(0), self.env.get_unit_by_id(1)

            _, e_unit = list(self.env.enemies.items())[0]                
            e_to_a1 =  self.env.distance(a1.pos.x, a1.pos.y, e_unit.pos.x, e_unit.pos.y)
            e_to_a2 =  self.env.distance(a2.pos.x, a2.pos.y, e_unit.pos.x, e_unit.pos.y)
            
            a_shoot_range = self.env.unit_shoot_range(a1)
            min_dist = 1
            
            if e_to_a1 > a_shoot_range and e_to_a2 > a_shoot_range:        
                actions = [1, 1]
            else:
                # The further one should fire
                if e_to_a1 > min_dist and e_to_a2 > min_dist and self.init:
                    self.init = False
                    if e_to_a1 <= e_to_a2:
                        # a1 closer and it fires
                        actions = [6, 1]
                    else:
                        actions = [1, 6]
                else:
                    if e_to_a1 <= min_dist:
                        actions = [1, 6]
                    elif e_to_a2 <= 1:
                        actions = [6, 1]
                    else:
                        if e_to_a1 >= e_to_a2:
                            # close one fire to attract
                            actions = [6, 1] #[1, 6]
                        else:
                            actions = [1, 6] #[6, 1]
            reward, terminated, _ = self.env.step(actions)
            return reward, terminated 
        else:
            a1, a2 = self.env.get_unit_by_id(0), self.env.get_unit_by_id(1)

            _, e_unit = list(self.env.enemies.items())[0]                
            e_to_a1 =  self.env.distance(a1.pos.x, a1.pos.y, e_unit.pos.x, e_unit.pos.y)
            e_to_a2 =  self.env.distance(a2.pos.x, a2.pos.y, e_unit.pos.x, e_unit.pos.y)
            
            actions = []
            if a1.shield < 30:
                if e_to_a1 < 9:
                    actions.append(3)
                else:
                    actions.append(1)
            else:
                actions.append(6)
                
            if a2.shield < 30:
                if e_to_a2 < 9:
                    actions.append(3)
                else:
                    actions.append(1)
            else:
                actions.append(6)

            reward, terminated, _ = self.env.step(actions)
            return reward, terminated 