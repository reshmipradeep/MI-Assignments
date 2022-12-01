import numpy
import pygad
import pygad.nn
import pygad.gann

global GANN_instance, inputs, outputs
def fitnessFunction(soln, soln_idx):
    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[soln_idx], data_inputs = inputs)
    correct_pred = numpy.where(predictions == outputs)[0].size
    fitness = (correct_pred/outputs.size)*100
    return fitness

def callbackGeneration(ga_instance):
    popln_matrices = pygad.gann.population_as_matrices(population_networks = GANN_instance.population_networks, population_vectors = ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights = popln_matrices)
    print("Generation: {generation}".format(generation = ga_instance.generations_completed))
    print("Accuracy: {fitness}".format(fitness = ga_instance.best_solution()[1]))

inputs = numpy.array([[1, 0],[0, 0],[0, 1],[1, 1]])
outputs = numpy.array([1, 0, 1, 1])

GANN_instance = pygad.gann.GANN(num_solutions=10, num_neurons_input = 2, num_neurons_hidden_layers = [2], num_neurons_output = 2, hidden_activations = ["relu"], output_activation = "softmax")
popln_vectors = pygad.gann.population_as_vectors(population_networks = GANN_instance.population_networks)

ga_instance = pygad.GA(num_generations = 50, num_parents_mating = 3,initial_population = popln_vectors.copy(), fitness_func = fitnessFunction, mutation_percent_genes = 10, callback_generation=callbackGeneration)
ga_instance.run()

solution, fitness, idx = ga_instance.best_solution()
print("Solution: ",solution)
print("Fitness:",fitness)
print("Idx:",idx)