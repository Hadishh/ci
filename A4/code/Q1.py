#Q1_graded
import random
INFINITY = 10000000
class Chromosome:
  def __init__(self, graph, genomes=None):
    self.genome_count = len(graph)
    self.graph = graph
    if(genomes is not None):
      self.genomes = genomes
      return
    genomes = []
    for i in range(self.genome_count):
      rand_int = random.randint(0, self.genome_count - 1)
      while (rand_int in genomes):
        rand_int = random.randint(0, self.genome_count - 1)
      genomes.append(rand_int)
    self.genomes = genomes
  def fitness_value(self):
    cost = 0
    before = 0
    for i in range(1, len(self.genomes)):
      if(self.graph[before][self.genomes[i]] == -1):
        return INFINITY
      cost += self.graph[before][self.genomes[i]]
      before = self.genomes[i]
    if(self.graph[before][self.genomes[0]] == -1):
        return INFINITY
    cost += self.graph[before][self.genomes[0]]
    return cost

#Q1_graded
class Population:
  def __init__(self, graph, population_size):
    #initialize population
    self.chromosomes = [Chromosome(graph) for _ in range(population_size)]
    self.min_observed_fitness = min([c.fitness_value() for c in self.chromosomes])
    self.population_size = population_size
    self.graph = graph
  def sort_by_fitness(self):
    self.chromosomes = sorted(self.chromosomes, key=lambda c: c.fitness_value())
  
  def __crossover(self, c1, c2, cross_over_point=3):
    new_genome = c1.genomes[:cross_over_point]
    for gen in c2.genomes:
      if (gen not in new_genome):
        new_genome.append(gen)
    new_chromosome = Chromosome(graph, genomes=new_genome)
    return new_chromosome
  
  def __mutate(self, c):
    new_genome = c.genomes.copy()
    i = random.randint(0, c.genome_count - 1)
    j = random.randint(0, c.genome_count - 1)
    while i == j:
      j = random.randint(0, c.genome_count - 1)
    temp = new_genome[i]
    new_genome[i] = new_genome[j]
    new_genome[j] = temp
    return Chromosome(graph, new_genome)

  def do_crossover(self,k=50):
    good_chromosomes = self.chromosomes[:k]
    for i in range(len(good_chromosomes)):
      for j in range(i + 1, len(good_chromosomes)):
        new_chromosome = self.__crossover(good_chromosomes[i], good_chromosomes[j])
        self.chromosomes.append(new_chromosome)
  
  def do_mutation(self, k=50):
    good_chromosomes = self.chromosomes[:k]
    for c in good_chromosomes:
      new_chromosome = self.__mutate(c)
      self.chromosomes.append(new_chromosome)
  def __compute_gamma(self, c):
    f =  c.fitness_value()
    self.min_observed_fitness = min(self.min_observed_fitness, f)
    return 1 / (1 + f - self.min_observed_fitness)
  def do_selection(self):
    distribution = [self.__compute_gamma(c) for c in self.chromosomes]
    sum_ = sum(distribution)
    distribution = [d / sum_ for d in distribution]
    new_population = random.choices(self.chromosomes, distribution, k=self.population_size)
    self.chromosomes = new_population
    self.min_observed_fitness = min([c.fitness_value() for c in self.chromosomes])

#Q1_graded
graph = [
         [-1, 12, 10, -1, -1, -1, 12],
         [12, -1, 8, 12, -1, -1, -1],
         [10, 8, -1, 11, 3, -1, 9],
         [-1, 12, 11, -1, 11, 10, -1],
         [-1, -1, 3, 11, -1, 6, 7],
         [-1, -1, -1, 10, 6, -1, 9],
         [12, -1, 9, -1, 7, 9, -1]
]
n_nodes = 7
population = Population(graph, 1000)
iterations = 100
mutation_prob = 0.001
for iter in range(iterations):
  population.sort_by_fitness()
  population.do_crossover()
  r = random.random()
  if r < mutation_prob:
    population.do_mutation()
  population.do_selection()
population.sort_by_fitness()
print(population.chromosomes[0].fitness_value())

