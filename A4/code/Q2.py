#Q2_graded
class Chromosome:
  def __init__(self,genome=None):
    self.genome_count = 1
    self.multipliers = [-13257.2, 15501.2, -7227.94, 1680.1, -194.7, 9]
    if(genome is not None):
      self.genome = genome
      return
    self.genome = random.randint(-500, 500)
  def fitness_value(self):
    value = 0
    for i in range(len(self.multipliers)):
      value += self.multipliers[i] * (self.genome ** i)
    return value**2

#Q2_graded
class Population:
  def __init__(self, population_size):
    #initialize population
    self.chromosomes = [Chromosome() for _ in range(population_size)]
    self.min_observed_fitness = min([c.fitness_value() for c in self.chromosomes])
    self.population_size = population_size
  def sort_by_fitness(self):
    self.chromosomes = sorted(self.chromosomes, key=lambda c: c.fitness_value())
  
  def __crossover(self, c1, c2):
    new_genome = (c1.genome + c2.genome) / 2
    new_chromosome = Chromosome(genome=new_genome)
    return new_chromosome
  
  def __mutate(self, c):
    new_genome = c.genome
    r = random.random() * 2 - 1
    new_genome += r
    return Chromosome(new_genome)

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

n_nodes = 7
population = Population(1000)
iterations = 1000
mutation_prob = 0.05
for iter in range(iterations):
  population.sort_by_fitness()
  population.do_crossover()
  r = random.random()
  if r < mutation_prob:
    population.do_mutation()
  population.do_selection()
population.sort_by_fitness()
print(population.chromosomes[0].fitness_value())
print(population.chromosomes[0].genome)

