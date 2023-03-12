##Importing Packages
import numpy as np

##Defining the Parameter
cities = 10
population_size = 50
total_generations = 50
elite_number = 2
total_cultures = 5
total_immigrants = 2
mutate_prob = 0.02
crossover_prob = 0.7

##Creating the distance matrix
def generate_distance_matrix(min_distance,max_distance):
    np.random.seed(42)
    distances = np.random.randint(min_distance,max_distance,size=(cities,cities))
    np.fill_diagonal(distances,0)
    return distances

distances = generate_distance_matrix(1, 101)
#print(distances)

##Creating a fitness function which will take the order in which the cities are visited
def fitness(individual):
    fitness_value = -np.sum([distances[individual[i],individual[(i+1)%cities]] for i in range(cities)])
    return fitness_value

##Creating a selection function using the roulette wheel selection
def roulette_wheel_selection(population,fitness_value):
    fitness_sum = np.sum(fitness_value)
    fitness_prob = fitness_value / fitness_sum
    cum_prob = np.cumsum(fitness_prob)
    selected_index = []
    for i in range(len(population)):
        val = np.random.rand()
        for j in range(len(cum_prob)):
            if val <= cum_prob[j]:
                selected_index.append(j)
                break
    return [population[i] for i in selected_index]

##Creating the single point cross-over function
def crossover(parent1, parent2):
    child = [-1] * cities
    crossover_point = np.random.randint(1, cities)
    child[:crossover_point] = parent1[:crossover_point]
    for i in range(crossover_point, cities):
        if parent2[i] not in child:
            child[i] = parent2[i]
    for i in range(crossover_point):
        if parent2[i] not in child:
            for j in range(cities):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child


##Creating the mutation function
def mutate(individual):
    if np.random.rand() < mutate_prob:
        i = np.random.randint(cities)
        j = np.random.randint(cities)
        individual[i],individual[j] = individual[j],individual[i]
    return individual

##Initalizing the population for each culture
populations = []
for i in range(total_cultures):
    population = [np.random.permutation(cities) for _ in range(population_size)]
    populations.append(population)
   
##Starting the Evolution process
for generation in range(total_generations):
    elites = []
    for population in populations:
        fitness_value = [fitness(individual) for individual in population]
        elites_index = np.argsort(fitness_value)[-elite_number:]
        elites.append([population[i] for i in elites_index])
    #combine the elites into a single population
    elite_population = [individual for elite in elites for individual in elite]
    #Generate the immigrants
    immigrants = [np.random.permutation(cities) for _ in range(total_immigrants)]
    #Combine the population and immigrants
    populations = [elite_population] + populations[:-1] + [immigrants]
   
   
    ##Perform Re-Combination
    new_populations = []
    for population in populations:
        new_population = []
        while(len(new_population) < population_size):
            fitness_value = [fitness(individual) for individual in population]
            sele_indi = roulette_wheel_selection(population, fitness_value)
            individual1 = sele_indi[0]
            individual2 = sele_indi[1]
            if np.random.rand() < crossover_prob:
                child = crossover(individual1, individual2)
            else:
                child = individual1
            new_population.append(child)
        new_populations.append(new_population)
   
    populations = new_populations
   
    ##Performs Mutation
    for population in populations:
        for individual in population:
            if np.random.rand() < mutate_prob:
                individual = mutate(individual)
           
    ##Display the best fitness in each culture
for i,population in enumerate(populations):
    fitness_values = [fitness(individual) for individual in population]
    print(f"Culture {i}: Best Fitness = {np.max(fitness_values)}")
       

all_solutions = [individual for population in populations for individual in population]
best_solution = max(all_solutions, key=fitness)
print("Best solution found:", best_solution)
print("Fitness of the best solution:", fitness(best_solution))
