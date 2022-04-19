# General
from itertools import product
import numpy as np
import random

# Display Stuff
from IPython.display import clear_output
from IPython import display
from ipywidgets import interact
import matplotlib.pyplot as plt

# Color Definitions
tree_colors = np.array(
    [[104/256, 230/256, 158/256],
    [114/256, 134/256, 38/256],
    [70/256, 126/256, 4/256],
    [42/256, 70/256, 8/256],
    [142/256, 170/256, 72/256]])

ground_colors = np.array(
    [[65 / 256, 105 / 256, 225 / 256],
    [51 / 256, 230 / 256, 255 / 256],
    [238 / 256, 214 / 256, 175 / 256],
    [160 / 256, 82 / 256, 45 / 256],
    [139 / 256, 69 / 256, 19 / 256],
    [139 / 256, 137 / 256, 137 / 256],
    [255 / 256, 250 / 256, 250 / 256]])

ideal_colors = np.array(
    [[65 / 256, 105 / 256, 225 / 256],
    [104/256, 230/256, 158/256],
    [114/256, 134/256, 38/256],
    [70/256, 126/256, 4/256],
    [42/256, 70/256, 8/256],
    [142/256, 170/256, 72/256],
    [255 / 256, 250 / 256, 250 / 256]])

# Miscellaneous Definitions
example_fitness = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ground_elevations = np.array([0.01, 0.075, 0.15, 0.2, 0.3, 0.4, 1.0])
h_o = (0, 0.5, 0.05)
h = (0,0.5, 0.05)
r = (0.0, 1)


class AbstractTree:
    def __init__(self, dna=None):
        self.traits_dim = 5
            
        if dna is None:
            self.initialize_random_dna(self.traits_dim)
        else:
            self.dna = dna
        
    def random_dna(self):
        raise "Not implemented"
    
    def generate_offspring(self, mutation_var):
        raise "Not implemented"
        
    def get_color(self):
        color = np.matmul(self.dna, tree_colors)
        return color
    
    def prefered_location(self):
        pref_loc = np.argmax(self.dna)
        return pref_loc
    
    def calc_fitness(self, env_type):
        if env_type == -1:
            fitness = self.dna[0]
        elif env_type == 5:
            fitness = 0
        else:
            fitness = self.dna[env_type]
        return fitness


class AbstractSquare:
    def __init__(self,
                 height,
                 soil_color,
                 mutation_var,
                 traits_dim,
                 comp_constant,
                 life_prob,
                 max_seeds):

        self.traits_dim = traits_dim
        self.soil_color = soil_color
        self.env_type = find_bucket(height) - 1
        self.mutation_var = mutation_var
        self.max_seeds = max_seeds
        self.life_prob = life_prob
        self.comp_constant = comp_constant
        self.tree = None
        self.seeds = []
        self.future_seeds = []
        
    def sample_boltzmann_distribution(self, fitness, return_dist=False):
        raise "Not implemented"
        
    def simulate_life_creation(self):
        raise "Not implemented"
        
    def generate_offspring(self):
        raise "Not implemented"
       
    def age_tree(self):
        raise "Not implemented"

    def contains_tree(self):
        return self.tree is not None
    
    def terminate_tree(self):
        self.tree = None

    def is_ocean(self):
        return self.env_type == -1

    def is_habitable(self):
        return self.env_type not in [-1, 5]

    def env_step(self):
        self.simulate_life_creation()

        if self.is_habitable():
            self.age_tree()
            self.fitness_competition()
        else:
            self.age_seeds()
        self.seeds, self.future_seeds = self.future_seeds, []

    def get_seeds(self):
        if self.is_ocean():
            seeds, self.seeds = self.seeds, []
        else:
            seeds = self.generate_offspring()
        return seeds

    def age_seeds(self):
        """
        Because seeds degrade over time, we need to simulate
        the aging of seeds, and remove those that die
        """
        seeds, self.seeds = self.seeds, []
        for s in seeds:
            survival_prob = s.calc_fitness(self.env_type)
            if np.random.uniform() < survival_prob:
                self.seeds.append(s)
        
    def fitness_competition(self):
        """
        When multiple seeds land on the same square, they will need to compete
        for the chance to grow. To simulate this, sample the winner from the bolzmann distribution.
        """
        competitors = self.seeds
        if self.contains_tree():
            competitors.append(self.tree)

        if len(competitors) == 0:
            return
        
        fitness = np.array([competitors[i].calc_fitness(self.env_type)
             for i in range(len(competitors))])
        
        weighted_fitness = self.comp_constant * fitness
        weighted_fitness = weighted_fitness - np.max(weighted_fitness)
        fitness = weighted_fitness / self.comp_constant
   
        winner_index = self.sample_boltzmann_distribution(fitness)
        
        self.tree = competitors[winner_index]
    
    def plant_seed(self, seed):
        self.future_seeds.append(seed)

    def convergence(self):
        """
        If there is no tree, return 0
        Otherwise, return whether it has converged to its enviorment
        
        A tree has converged if its highest trait corresponds to its locations enviorment type
        ie. max(self.tree) == self.env_type
        """
        if self.contains_tree():
            return self.tree.prefered_location() == self.env_type
        return 0

    def fitness(self):
        """
        If there is no tree, return 0
        Otherwise, return the fitness of the tree
        """
        if self.contains_tree():
            return self.tree.calc_fitness(self.env_type)
        return 0

    def get_square_color(self):
        """
        Return the color of the square
        If there is no tree present, return self.soil_color
        Otherwise, calculate the color by multiplying the trait
        values by the corresponding colors, which is defined as tree_colors above.
        
        ie. color = self.tree[0]*tree_colors[0] + ... + self.tree[4]*tree_colors[4]
        """
        if self.contains_tree():
            color = self.tree.get_color()
        else:
            color = self.soil_color
        
        return color
    

class AbstractWorld:
    def __init__(self,
                 generate_world,
                 Square,
                 dim=75,
                 mutation_var=0.01,
                 wind_var=2,
                 waves_var=5,
                 life_prob=0.0001,
                 comp_constant=100,
                 max_seeds=10
                 ):
        self.Square = Square
        self.dim = dim
        self.life_prob = life_prob
        self.comp_constant = comp_constant
        self.traits_dim = 5
        self.mutation_var = mutation_var
        self.wind_var = wind_var
        self.waves_var = waves_var
        self.max_seeds = max_seeds
        self.trait_grid = generate_world(dim)
        self.color_grid = get_uninhabited_world(self.trait_grid)
        self.ideal_grid = get_ideal_world(self.trait_grid)
        self.initial_world = self.color_grid.copy()
        self.world = self.initialize_world()
        self.fitness_progress = np.zeros((1000,))
        self.spread_progress = np.zeros((1000,))
        self.convergence_progress = np.zeros((1000,))
        self.timestep = 0
        plt.ion()
        
    def simulate_movement(self, x, y, var):
        raise "Not Implemented"
    
    def spread_seeds(self, i, j):
        raise "Not Implemented"

    def initialize_world(self):
        grid_world = []
        for n in range(self.dim):
            slab = []
            for m in range(self.dim):
                curr_loc = self.Square(
                    self.trait_grid[n][m],
                    self.color_grid[n][m],
                    self.mutation_var,
                    self.traits_dim,
                    self.comp_constant,
                    self.life_prob,
                    self.max_seeds)
                slab.append(curr_loc)
            grid_world.append(slab)
        return grid_world

    def step_square(self, x, y, stats):
        square = self.world[x][y]
        square.env_step()
        self.spread_seeds(x, y)
        self.color_grid[x][y] = square.get_square_color()

        stats["total_trees"] += square.contains_tree()
        stats["total_habitable"] += square.is_habitable()
        stats["total_fitness"] += square.fitness()
        stats["total_convergence"] += square.convergence()

    def update_statistics(self, stats):
        self.fitness_progress[self.timestep] = stats["total_fitness"] / max(stats["total_trees"], 1)
        self.spread_progress[self.timestep] = stats["total_trees"] / max(stats["total_habitable"], 1)
        self.convergence_progress[self.timestep] = stats["total_convergence"] / max(stats["total_trees"], 1)

    def env_step(self):
        self.display_visuals()
        stats = {"total_trees": 0,
                 "total_habitable": 0,
                 "total_fitness": 0,
                 "total_convergence": 0}

        for m in range(self.dim):
            for n in range(self.dim):
                self.step_square(m, n, stats)

        self.update_statistics(stats)
        self.timestep += 1

    def display_visuals(self):
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        clear_output(wait=True)
        ax[0].imshow(self.initial_world)
        ax[0].set_title('Initial World')
        ax[1].imshow(self.color_grid)
        ax[1].set_title('Current World')
        ax[2].imshow(self.ideal_grid)
        ax[2].set_title('Ideal World')
        ax[3].set_ylim((0, 1))
        ax[3].set_title('Progress Statistics')
        x_axis = np.arange(0, self.timestep)
        ax[3].plot(x_axis, self.fitness_progress[:self.timestep], 'r')
        ax[3].plot(x_axis, self.spread_progress[:self.timestep], 'g')
        ax[3].plot(x_axis, self.convergence_progress[:self.timestep], 'b')
        ax[3].legend(["Fitness", "Coverage", "Convergence"], loc="upper left", fontsize="small")
        plt.show()
        plt.pause(0.1)


# Miscellaneous Functions
def find_bucket(elevation, hb=np.array([0.01, 0.075, 0.15, 0.2, 0.3, 0.4, 1.0])):
    index = 0
    while elevation > hb[index]:
        index += 1
    return index

def get_uninhabited_world(grid):
    colors = np.zeros(grid.shape + (3,))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            colors[i][j] = ground_colors[find_bucket(grid[i][j])]
    return colors

def get_ideal_world(grid):
    colors = np.zeros(grid.shape + (3,))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            colors[i][j] = ideal_colors[find_bucket(grid[i][j])]
    return colors

def visualize_dna(tree):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].bar(np.arange(5), tree.dna, color=tree_colors)
    ax[0].set_title("DNA Visualization", fontsize=16, fontweight='bold')
    ax[0].set_xticks([r for r in range(5)], ["Shallows", "Beach", "Dirt", "Inland", "Mountains"])
    ax[0].set_xlabel('Environment Type', fontweight='bold')
    ax[0].set_ylabel('Survival Probability')
    ax[0].set_ylim((0, 1))
    color = tree.get_color()
    rectangle = plt.Rectangle((0,0), 1, 1, fc=color, ec=color)
    ax[1].add_patch(rectangle)
    ax[1].set_title('Resultant Tree Color')
    ax[1].tick_params(axis='both', which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
    plt.show()

def visualize_competition(dist, ent):
    plt.figure()
    plt.bar(np.arange(dist.shape[0]), dist)
    plt.xticks(np.arange(dist.shape[0]), example_fitness)
    
    ent = np.round(ent, 2)
    plt.title("Competition Visualization (Entropy={0})".format(ent), fontsize=16,
              fontweight='bold')
    plt.xlabel('Tree Fitness', fontweight='bold')
    plt.ylabel('Victory Probability')
    plt.show()

def create_secret_square(var, comp_constant, Square):
    return Square(0.2, (0,0,0), var, 5, comp_constant, 1, 10)

def psuedo_env_step(square, Tree):
    tree = square.tree
    square.env_step()
    for c in square.generate_offspring():
        square.plant_seed(c)
    if not square.contains_tree():
        square.tree = Tree()
    visualize_dna(square.tree)
    
def visualize_height_values(heights, grid):
    colors = np.zeros(grid.shape + (3,))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            colors[i][j] = ground_colors[find_bucket(grid[i][j], hb=heights)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(colors)
    ax[0].set_title('Colorized World')
    ax[1].imshow(grid)
    ax[1].set_title('Perlin Noise')
    
def entropy(dist):
    return - np.log2(dist).T @ dist
