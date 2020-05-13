#########################################################################
# Defines population and individual methods
#########################################################################
from structure import Module, Blueprint, StructureBuilder
from controller_net import BlueprintController, ModuleController

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Individual:
    """
    An individual topology/network

    -An individual possesses a blueprint, a model (which is an implementation of the blueprint), a score and a name
    """
    def __init__(self, blueprint, model=None, name=None, score = 0):
        super().__init__()
        self.blueprint = blueprint
        self.model = model
        self.name = name
        self.score = score
    
    def update_model(self):
        print("Not Implemented")
    
    def create_model(self):
        print("Not Implemented")

    def fit(self, trainloader, num_epochs, save = False, verbose = True):
        """
        Trains the model on the given dataset
        """
        net = self.model
        net.to(device)

        # define the loss
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # loop over the training set
        for epoch in range(num_epochs):  
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % 2000 == 1999 and verbose == True):    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        if (save):
            #save model
            PATH = './cifar_net.pth'
            torch.save(net.state_dict(), PATH)
    
    def score(self, dataloader):
        """
        Computes performance of the model on the given dataset
        """
        net = self.model
        net.to(device)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = ( correct / total)
        print('Accuracy of the network on the validation set: ', accuracy)
        self.score = accuracy
        return accuracy
        

class Ecosystem:
    """
    Represents the populations of topologies, modules and individuals
    """
    def __init__(self, datasets=None, individuals=[], blueprints=[], modules=[], hyperparameters=[], input_shape=None, population_size=1, compiler=None):
        self.datasets = datasets
        self.individuals = individuals
        self.blueprints = blueprints
        self.modules = modules
        self.hyperparameters = hyperparameters
        self.module_species = None
        self.blueprint_species = None
        self.input_shape = input_shape
        self.population_size = population_size
        self.compiler = compiler
    
    def create_module_population(self, global_configs, possible_components, possible_complementary_components):
        """
        Creates a population of modules to be used in blueprint populations.
        Can be evolved over generations.
        """

        new_modules = []

        for n in range(self.population_size):
            mark = self.historical_marker.mark_module()
            new_module = StructureBuilder().random_module(global_configs, 
                                                        possible_components, 
                                                        possible_complementary_components,
                                                        name=mark)
            new_module.mark = mark
            new_modules.append(new_module)
        
        self.modules = new_modules

    def create_blueprint_population(self, size, global_configs, possible_components, possible_complementary_components,
                                    input_configs, possible_inputs, possible_complementary_inputs,
                                    output_configs, possible_outputs, possible_complementary_outputs):
        """
        Creates a population of blueprints to be used in individual populations.
        Can be evolved over generations.
        """

        new_blueprints = []
        # Blueprints must always be created with an appropriate input layer
        self.blueprints = new_blueprints
    
    def create_individual_population(self, size=1, compiler=None):
        """
        Creates a population of individuals to be compared.
        Can be evolved over generations.
        """
        new_individuals = []
        self.individuals = new_individuals

class GeneticAlgorithm:
    """
    Main class which runs the overall algorithm to generate and train networks
    """
    def __init__(self, trainloader, validationloader, input_shape, num_epochs = 10, pop_size = 5, num_generations = 20, initial_depth = 4):
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.num_generations= num_generations
    
    def fit_genetics(self, num_epochs = 10, pop_size = 5, num_generations = 20):
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.num_generations= num_generations

        # Initialization
        best = None

        ecosystem = Ecosystem(population_size = self.pop_size)

        ecosystem.create_module_population
        ecosystem.create_blueprint_population
        ecosystem.create_individual_population

        blueprint_controller = BlueprintController()
        module_controller = ModuleController()

        # Actual genetic algorithm loop
        for generation in range(num_generations):
            # Train and compute fitnesses
            for individual in ecosystem.individuals:
                individual.fit(self.trainloader)
                individual.score(self.validationloader)
            
            # Crossover and mutate new population
            if generation != num_generations:
                module_controller.mutate_modules(ecosystem.modules)
                blueprint_controller.mutate_blueprints(ecosystem.individuals)

            
            # Update controllers
        
        # Return best performing network
        return best

        """
        logging.info(f"Iterating over {generations} generations")
        iteration = None
        iterations = []
        best_scores = []
        csv_history = open(f"{basepath}iterations.csv", "w", newline="")
        csv_history.write("indiv,blueprint,scores,features,species,generation\n")
        csv_history.close()

        for generation in range(generations):
            logging.info(f" -- Iterating generation {generation} -- ")
            print(f" -- Iterating generation {generation} -- ")
            logging.log(21, f"Currently {len(self.modules)} modules, {len(self.blueprints)} blueprints, latest iteration: {iteration}")
            logging.log(21, f"Current modules: {[item.mark for item in self.modules]}")
            logging.log(21, f"Current blueprints: {[item.mark for item in self.blueprints]}")

            # Create representatives of blueprints
            self.create_individual_population(self.population_size, compiler=self.compiler)
            logging.log(21, f"Created individuals for blueprints: {[(item.name, item.blueprint.mark) for item in self.individuals]}")

            # Iterate fitness and record the iteration results
            iteration = self.iterate_fitness(training_epochs, validation_split, current_generation=generation)
            with open(f"{basepath}iterations.csv", "a", newline="") as csv_history:
                csv_history_writer = csv.writer(csv_history)
                csv_history_writer.writerows(iteration)
            iterations.append(iteration)
            logging.log(21, f"This iteration: {iteration}")

            # Update weighted scores of species
            self.update_shared_fitness()
            self.reset_usage()

            # Save best model
            # [name, blueprint_mark, score[test_loss, test_val], history]
            best_fitting = max(iteration, key=lambda x: (x[2][1], -x[2][0]))
            best_scores.append([f"generation {generation}", best_fitting])
            logging.log(21, f"Best model chosen: {best_fitting}")
            
            try:
                self.return_individual(best_fitting[0]).model.save(f"{basepath}/models/best_generation_{generation}.h5")
            except:
                logging.error(f"Model from generation {generation} could not be saved.")

            # Summarize execution
            self.summary(generation)

            # Crossover, mutate and update species.
            if generation != generations:
                self.crossover_modules(crossover_rate)
                self.mutate_modules(mutation_rate, elitism_rate, possible_components, possible_complementary_components)
                self.update_module_species()

                self.crossover_blueprints(crossover_rate)
                self.mutate_blueprints(mutation_rate, elitism_rate, possible_components, possible_complementary_components)
                self.update_blueprint_species()

        return best_scores
        """