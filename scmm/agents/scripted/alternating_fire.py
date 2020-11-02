import numpy as np

class AlternatingFire():
    '''
    Attacking enemy in an alternative manner
    '''
    def __init__(self, n_agents):
        self.n_agents = n_agents 
        self.attacking_agent = np.random.randint(0, 2)
        self.attack_count = 2
        self.init = True
        
    def fit(self, env):
        assert env.map_name in {'2m_vs_1z', '2s_vs_1sc'}, "Alternating Fire trick only supports 2m_vs_1z and 2s_vs_1sc map"
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.map = env.map_name 
        self.actions = self.env.get_non_attack_action_set()
        self.shield_thres = 30
    
    def step(self, plot_level=0):
        if self.map == '2m_vs_1z': # There is different strategy for different map
            '''
            The idea when Marines shoot at Zealot will be:
            1. One Marine starts attacking Zealot and the other remains still
            2. If Zealot gets close to the Marine which is currently attacking, the attacking Marine stops shooting and the other one starts shooting
            '''
        
            a1, a2 = self.env.get_unit_by_id(0), self.env.get_unit_by_id(1)

            _, e_unit = list(self.env.enemies.items())[0]                
            e_to_a1 =  self.env.distance(a1.pos.x, a1.pos.y, e_unit.pos.x, e_unit.pos.y)
            e_to_a2 =  self.env.distance(a2.pos.x, a2.pos.y, e_unit.pos.x, e_unit.pos.y)
            
            a_shoot_range = self.env.unit_shoot_range(a1)
            min_dist = 1
            
            if e_to_a1 > a_shoot_range and e_to_a2 > a_shoot_range:        
                actions = [self.actions["Stop"], self.actions["Stop"]]
            else:
                # The further one should fire
                if e_to_a1 > min_dist and e_to_a2 > min_dist and self.init:
                    self.init = False
                    if e_to_a1 <= e_to_a2:
                        # a1 closer and it fires
                        actions = [self.n_actions_no_attack, self.actions["Stop"]]
                    else:
                        actions = [self.actions["Stop"], self.n_actions_no_attack]
                else:
                    if e_to_a1 <= min_dist:
                        actions = [self.actions["Stop"], self.n_actions_no_attack]
                    elif e_to_a2 <= 1:
                        actions = [self.n_actions_no_attack, self.actions["Stop"]]
                    else:
                        if e_to_a1 >= e_to_a2:
                            # close one fire to attract
                            actions = [self.n_actions_no_attack, self.actions["Stop"]] 
                        else:
                            actions = [self.actions["Stop"], self.n_actions_no_attack] 
                            
    
        else:
            '''
            Stalkers here move close to Spine Crawler to attack. 
            Given Stalkers have shields so they will keep attacking until the shield is going to run out
            Then Stalkers retreat and recover the shiled
            Repeat the above
            '''
            e_unit = self.env.get_enermy_by_id(0)

            actions = []
            
            for a_id in range(self.n_agents):
                unit = self.env.get_unit_by_id(a_id)
                dist = self.env.distance(unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y)
                
                if unit.shield < self.shield_thres:
                    if dist < 9:
                        actions.append(self.actions["South"])
                    else:
                        actions.append(self.actions["Stop"])
                else:
                    actions.append(self.n_actions_no_attack)


        reward, terminated, info = self.env.step(actions)
        
        if plot_level > 0:
            return actions, reward, terminated, info 
        return reward, terminated, info 