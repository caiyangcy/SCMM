# Config
import torch 
from nn import SCMLP
from scdc.env.micro_env.mm_env import MMEnv

class Config:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 40
    NUM_OUTPUTS = 3
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 15.0 

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Allow episode lengths of > than 200
    def __init__(self, env):
        self.env = env 
    # env = MMEnv(map_name=map_name, step_mul=step_mul, difficulty=difficulty, reward_sparse=reward_sparse, debug=debug)

    def fitness_fn(self, genome):
        # OpenAI Gym
        done = False
        self.env.reset()

        fitness = 0
        phenotype = SCMLP(genome, self)

        while not done:
            # TODO: Modify get_obs()
            input = self.env.get_obs().to(self.DEVICE)

            pred = round(float(phenotype(input)))
            reward, terminated = self.env.step(pred)

            fitness += reward
            
        self.env.close()

        return fitness