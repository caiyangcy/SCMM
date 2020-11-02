from scmm.utils.game_utils import find_weakest_injured

class HybridAttackHeal():
    '''
    Attack or Heal Enemy based on a score function
    '''
    def __init__(self, n_agents, alpha):
        self.n_agents = n_agents
        self.alpha = alpha 
        
    def fit(self, env):
        self.env = env 
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def _calculate_score(self, alpha, health, distance):
        return alpha*health+(1-alpha)*distance 
    
    
    def step(self, adv_plot=False):
        actions = []        
        target_items = self.env.enemies.items()
        
        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            action = 0
            if unit.health > 0:
                if self.env.unit_to_name(unit) == 'medivac':
                    
                    min_a_id = find_weakest_injured(agent_id, self.env.agents)
                    if min_a_id is None:
                        action = self.actions['Stop']
                    else:
                        action = self.n_actions_no_attack+min_a_id
                
                else:
                    min_score_e_id = self.find_lowest_score(unit, target_items)
                    if min_score_e_id is None:
                        action = self.actions['Stop']
                    else:
                        action = self.n_actions_no_attack+min_score_e_id
                    
            actions.append(action)
            
        reward, terminated, info = self.env.step(actions)
        
        if adv_plot:
            return actions, reward, terminated, info 
        
        return reward, terminated, info
        
        
    def find_lowest_score(self, unit, target_items):
        '''
        Find the unit with lowest score.
        Score = alpha*health+(1-alpha)*distance 
        '''
        min_score = None
        min_score_e_id = None
        for t_id, t_unit in target_items:
            if t_unit.health > 0:
                dist = self.env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                score = self._calculate_score(self.alpha, t_unit.health, dist)
                if min_score is None or score < min_score :
                    min_score = score 
                    min_score_e_id = t_id
                    
        return min_score_e_id 
    
    