from scmm.env.micro_env.mm_env import MMEnv
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt 

class GA_agent:
    def __init__(self, n_agents, chromosome):
        self.n_agents = n_agents
        # move dist-first attack hp-first attack
        self.chromosome = chromosome
        
    def fit(self, env):
        self.env = env
        self.n_actions_no_attack = self.env.n_actions_no_attack
        
    def step(self, plot_level=0):
        actions = []        

        for agent_id in range(self.n_agents):
            unit = self.env.get_unit_by_id(agent_id)
            if unit.health > 0:
                action = self._action_decision(unit)
                actions.append(action)
            else:
                actions.append(1)
        
        reward, terminated, info = self.env.step(actions)
            
        if plot_level > 0:
            return actions, reward, terminated, info
        return reward, terminated, info 
    
    def _calc_actual(self, ally):
        # v1: range of unit - range of closest enemy
        # v2: sum of enermies within unit range
        # v3: weakest enemy HP divied by unit damage
        # v4: the number of enemy - the number of ally
        ally_shoot_range = self.env.unit_shoot_range(ally)
        target_items = self.env.enemies.items()
        
        v2, v3 = 0, 0
        min_e_id, min_e_unit = None, None
        min_dist = math.hypot(self.env.max_distance_x, self.env.max_distance_y)
        
        min_hp_e_id, min_hp_e_unit = None, None
        min_hp = None
        
        
        for e_id, e_unit in target_items:
            if e_unit.health > 0:
                
                dist = self.env.distance(ally.pos.x, ally.pos.y, e_unit.pos.x, e_unit.pos.y)
                if dist <= ally_shoot_range:
                    v2 += 1
                if dist < min_dist:
                    min_dist = dist
                    min_e_id = e_id
                    min_e_unit = e_unit 
                    
                if min_hp is None or min_hp < e_unit.health:
                    min_hp_e_id = e_id
                    min_hp_e_unit = e_unit 
                    min_hp = e_unit.health
                
        v1 = ally_shoot_range-self.env.unit_shoot_range(min_e_unit)
        v3 = min_hp/self.env.unit_damage(ally)
        v4 = self.env.count_alive_units('ally')-self.env.count_alive_units('enemy')
        
        return [v1, v2, v3, v4], [min_e_id, min_hp_e_id], [min_e_unit, min_hp_e_unit]
    
    def _action_decision(self, unit):
        '''
        select actions based on threshold and priority
        '''
        move_value = np.sum(self.chromosome[:4])
        dist_first_attack = np.sum(self.chromosome[4:8])
        hp_first_attack = np.sum(self.chromosome[8:12])
        
        v, ids, units = self._calc_actual(unit)
        v = sum(v)
        decision = np.zeros(3)
        action = None
        if v<move_value:
            decision[0] = 1
        if v>dist_first_attack:
            decision[1] = 1
        if v<hp_first_attack:
            decision[2] = 1
        if np.any(decision):
            if (decision==1).sum() > 1:
                priority = self.chromosome[-3]
                if priority == 1 or priority == 2: # move closest or weakest depending on the priority
                    e_unit = units[int(priority)-1]
                    if np.abs(e_unit.pos.x-unit.pos.x) > np.abs(e_unit.pos.y-unit.pos.y):
                        action = 5-(e_unit.pos.x>unit.pos.x)
                    else:
                        action = 3-(e_unit.pos.y>unit.pos.y)
                elif priority == 3: # attack closest
                    action = self.n_actions_no_attack+ids[0]
                else: # attack weakest
                    action = self.n_actions_no_attack+ids[1]
            else:
                if decision[0]: # move closest
                    e_unit = units[0]
                    if np.abs(e_unit.pos.x-unit.pos.x) > np.abs(e_unit.pos.y-unit.pos.y):
                        action = 5-(e_unit.pos.x>unit.pos.x)
                    else:
                        action = 3-(e_unit.pos.y>unit.pos.y)
                elif decision[1]:
                    action = 6+ids[0]
                else:
                    action = 6+ids[1]
        else:
            action = 1
            
        return action
            
class GA:
    def __init__(self, env, n_iterations, population_size, n_agents, mutation_rate=0.01, crossover_rate=1.0):
        self.n_iter = n_iterations
        self.pop_size = population_size
        self.pop = np.zeros((population_size, 20))
        self.mutation_rate = mutation_rate 
        self.corss_rate = crossover_rate
        self.env = env
        self.map = self.env.map_name
        self.n_agents = n_agents
        self.init_DNA()
        
    def run(self):
        chromo_agent = GA_agent(self.n_agents, None)
        chromo_agent.fit(self.env)
        
        max_reward_change, min_reward_change, mean_reward_change = [], [], []
        for i_iter in range(self.n_iter):
            pop_fitness = np.empty((self.pop_size, ))
            
            for p in range(self.pop_size):
                chromo_agent.chromosome = self.pop[p]
                # run chromo agent
                fitness = 0
                terminated = False
                
                chromo_agent.env.reset()
                
                while not terminated:
        
                    reward, terminated, _  = chromo_agent.step()
                    fitness = reward 
                    
                pop_fitness[p] = fitness
            
            max_reward_change.append(np.max(pop_fitness))
            min_reward_change.append(np.min(pop_fitness))
            mean_reward_change.append(np.mean(pop_fitness))
            print('iter {} max fitness: {} min fitness: {} mean fitness: {}'.format(i_iter+1, np.max(pop_fitness), \
                                                                                    np.min(pop_fitness), np.mean(pop_fitness)))
                
            # crossover
            # select better population as parent 1
            pop_max = self.pop[np.argmax(pop_fitness)]
            pop = self.select(self.pop, pop_fitness)
            # make another copy as parent 2
            pop_copy = pop.copy()
            
            for parent in pop:
                # produce a child by crossover operation
                child = self.crossover(parent, pop_copy)
                # mutate child
                child =self.mutate(child)
                # replace parent with its child
                parent[:] = child  
            
            self.pop = pop 
            self.pop[np.argmax(pop_fitness)] = pop_max
                        
        chromo_agent.env.close()
        
        return max_reward_change, min_reward_change, mean_reward_change
        
    def init_DNA(self):
        
        self.pop[:,[0,4,8,12]] = np.random.uniform(low=-10.25, high=11.75, size=(self.pop_size, 4))
        self.pop[:,[1,5,9,13]] = np.random.randint(low=-0, high=6, size=(self.pop_size, 4))
        self.pop[:,[2,6,10,14]] = np.random.uniform(low=0, high=5, size=(self.pop_size, 4))
        self.pop[:,[3,7,11,15]] = np.random.randint(low=-1, high=21, size=(self.pop_size, 4))
        
        # v1: range of unit - range of closest enemy
        # v2: sum of enermies within unit range
        # v3: weakest enemy HP divied by unit damage
        # v4: the number of enemy - the number of ally
        for p in range(self.pop_size):
            # v1: [-10.25, 10.75] --> [-10, 10]
            # v2: [0, 5]
            # v3: [0, 1]
            # v4: [-1, 20]
            if self.map in {'corridor', '27m_vs_30m', '25m', 'so_many_baneling', 'bane_vs_bane', '2c_vs_64zg'}:
                pass

            self.pop[p, -4:] = np.random.permutation(4)
    
    def crossover(self, parent, pop):
        if np.random.rand() < self.corss_rate:
            # randomly select another individual from population
            i = np.random.randint(0, self.pop_size, size=1)    
            # choose crossover points(bits)
            cross_points = np.random.randint(0, 2, size=20).astype(np.bool)
            # produce one child
            parent[cross_points] = pop[i, cross_points]  
            
        return parent
    
    def mutate(self, child):      
        for point in [0,4,8,12]:
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.uniform(low=-10.25, high=11.75)
                
        for point in [1,5,9,13]:
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(low=0, high=6)
        
        for point in [2,6,10,14]:
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.uniform(low=0, high=1)
        
        for point in [3,7,11,15]:
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(low=-1, high=21)
        
        return child

    
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]


parser = argparse.ArgumentParser(description='Run an agent with actions randomly sampled.')
parser.add_argument('--map_name', default='25m', type=str, help='The name of the map. The full list can be found by running bin/map_list.')
parser.add_argument('--step_mul', default=2, type=int, help='How many game steps per agent step (default is 8). None indicates to use the default map step_mul..')
parser.add_argument('--difficulty', default='A', help='The difficulty of built-in computer AI bot (default is "A").')
parser.add_argument('--reward_sparse', default=False, help='Receive 1/-1 reward for winning/loosing an episode (default is False). The rest of reward parameters are ignored if True.')
parser.add_argument('--debug', default=True, help='Log messages about observations, state, actions and rewards for debugging purposes (default is False).')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes the game will run for.')
parser.add_argument('--population_size', default=40, type=int, help='Population size')
args = parser.parse_args()
        
if __name__ == "__main__":
    
    map_name = args.map_name
    step_mul = args.step_mul
    difficulty = args.difficulty
    reward_sparse = args.reward_sparse
    debug = args.debug 
    n_episodes = args.n_episodes
    population_size = args.population_size

    env = MMEnv(map_name=map_name, step_mul=step_mul, difficulty=difficulty, reward_sparse=reward_sparse, debug=debug)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    

    ga = GA(env, n_episodes, population_size, n_agents, mutation_rate=0.01, crossover_rate=1.0)

    max_reward_change, min_reward_change, mean_reward_change = ga.run()
    
    fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
    ax.plot(np.arange(len(max_reward_change)), max_reward_change, label='max fitness')
    plt.plot(np.arange(len(max_reward_change)), min_reward_change, label='min fitness')
    plt.plot(np.arange(len(max_reward_change)), mean_reward_change, label='mean fitness')
    
    ax.set_title(f'GA - {map_name}', fontdict={'size':22})
    
    ax.set_ylabel('Rewards', fontsize=20)
    ax.set_ylim(0, 1.5)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.set_xlabel('Episode', fontsize=20)
    plt.xticks(np.arange(n_episodes), np.arange(n_episodes), horizontalalignment='right', fontsize=15)
    
    plt.legend()
    plt.show()
    fig.savefig(f"plots/ga_{map_name}.png")
    plt.close(fig)