from scmm.env.micro_env.mm_env import MMEnv
from scmm.utils.game_utils import fine_closest_position
import numpy as np
import argparse


class PotentialField:
    def __init__(self, ally_attract_thres, enemy_repulsive_thres, enemy_attract_thres, force_params):
        self.ally_attract_thres = ally_attract_thres
        assert enemy_repulsive_thres <= enemy_attract_thres, "Repulsive threshold should be less than attractive one."
        self.enemy_attract_thres = enemy_attract_thres
        self.enemy_repulsive_thres = enemy_repulsive_thres
        self.env = env
        # first three control enemy repulsive
        # fourth controls enemy attractive
        # last three controls ally attracitve
        self.force_params = force_params 
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        self.actions = self.env.get_non_attack_action_set()
        
    def _calc_direction(self, unit_id):
        '''
        Approximate moving directions based on force, which are all linear formulas here.
        This function is only used when there is no enemy within shooting range of the unit
        '''
        force = self._calc_force(unit_id)
        force_x, force_y = force[0], force[1]
        
        direction = None
        # N S E W: 2, 3, 4, 5
        if np.abs(force_x) > np.abs(force_y):
            if force_x > 0:
                # move east
                direction = self.actions['East']
            else:
                # move west
                direction = self.actions['West']
        else:
            if force_y > 0:
                # move north
                direction = self.actions['North']
            else:
                # move south
                direction = self.actions['South']
                
        return direction
    
    def _calc_force(self, curr_a_id):
        '''
        Given that the ally units are always together,
        I didn't use ally attractive force
        '''
        allies = self.env.agents.items()
        enemies = self.env.enemies.items()
        force = np.zeros((2, )) 
        curr_unit = self.env.get_unit_by_id(curr_a_id)
        
        ally_center_x, ally_center_y = 0, 0
        for a_id, a_unit in allies:
            if a_id == curr_a_id:
                continue
            
            if a_unit.health > 0:
                ally_center_x += a_unit.pos.x
                ally_center_y += a_unit.pos.y 
                
        ally_alive = self.env.count_alive_units()-1
        ally_center_x /= ally_alive
        ally_center_y /= ally_alive
        dit_to_ally_center = self.env.distance(curr_unit.pos.x, curr_unit.pos.y, ally_center_x, ally_center_y)
        
        if dit_to_ally_center <= self.ally_attract_thres:
            ally_attractive = self.force_params[3]/curr_unit.health + self.force_params[4]*ally_alive + self.force_params[5]*dit_to_ally_center
            direction = np.abs(np.array([ally_center_x-curr_unit.pos.x, ally_center_y-curr_unit.pos.y]))
            normalised_direction = direction/np.linalg.norm(direction)
            ally_attractive = ally_attractive*normalised_direction
        else:
            ally_attractive = 0    
        
        
        enemy_attractive = 0
        enemy_repulsive = 0
        
        for e_id, e_unit in enemies:
            if e_unit.health > 0:
                dist = self.env.distance(curr_unit.pos.x, curr_unit.pos.y, e_unit.pos.x, e_unit.pos.y)
                if dist > self.enemy_attract_thres:
                    direction = np.abs(np.array([e_unit.pos.x-curr_unit.pos.x, e_unit.pos.x-curr_unit.pos.y]))
                    normalised_direction = direction/np.linalg.norm(direction)
                    enemy_attractive += normalised_direction*self.force_params[-1]
                    
                    
                if dist < self.enemy_repulsive_thres:
                    enemy_repulsive_i = self.force_params[0]/curr_unit.health + self.force_params[1]*self.env.unit_damage(e_unit)
                    direction = np.abs(np.array([e_unit.pos.x-curr_unit.pos.x, e_unit.pos.x-curr_unit.pos.y]))
                    normalised_direction = direction/np.linalg.norm(direction)
                    enemy_repulsive += -normalised_direction*enemy_repulsive_i
        
        force += ally_attractive+enemy_attractive+enemy_repulsive
        return force
    
    def step(self, adv_plot=False):
        actions = []        

        for agent_id in range(self.env.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            action = 1
            if unit.health > 0:
                min_dist_id, _, _ = fine_closest_position(unit.pos, self.env.enemies)
                if min_dist_id is not None:
                    action = self.n_actions_no_attack+min_dist_id
                else:
                    action = self._calc_direction(agent_id)
            actions.append(action)
            
        reward, terminated, info = self.env.step(actions)
        
        if adv_plot:
            return actions, reward, terminated, info
        
        return reward, terminated, info
        
    
    
parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='3m', help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=2, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='7', help='The difficulty of built-in computer AI bot (default is "7").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes the game will run for.')
parser.add_argument('--alpha', default=0.5, type=int, help='Parameter used for calculating score in HybridAttack.')
parser.add_argument('--adv_plot', default=False, help='Whether using advanced plot or not')

args = parser.parse_args()
    

if __name__ == "__main__":
    
    map_name = args.map_name
    step_mul = args.step_mul
    difficulty = args.difficulty
    reward_sparse = args.reward_sparse
    debug = args.debug 
    n_episodes = args.n_episodes
    adv_plot = args.adv_plot

    env = MMEnv(map_name=map_name, step_mul=step_mul, difficulty=difficulty, reward_sparse=reward_sparse, debug=debug)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    # This is just an example of hyperparameters
    # Due to the difficulty of finding the best hyperparameters
    # This algorithm is not tested
    ally_attract_thres = 0.5
    enemy_repulsive_thres = 2
    enemy_attract_thres = 6
    force_params = [1, 3, 2, 3, 1, 2, 1]
    
    
    agent = PotentialField(ally_attract_thres, enemy_repulsive_thres, enemy_attract_thres, force_params)
    agent.fit(env)
    
    for e in range(n_episodes):
        agent.env.reset()
            
        episode_reward = 0
        terminated = False
        
        while not terminated:
            
            reward, terminated, info  = agent.step(adv_plot)
            episode_reward += reward 
                                        
        print("Total reward in episode {} = {}".format(e, reward))
        
    env.close()