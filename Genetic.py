#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'davinellulinvega'
import random
import multiprocessing
from deap import base, creator, tools, algorithms
import gym
import numpy as np
from PrimEmoArch.Configurations import genetic as gen_conf
from PrimEmoArch.Network import Network
from PrimEmoArch.Layer import LayerType
from PrimEmoArch.Connection import ConType
from PrimEmoArch.Group import ConPattern
from Main import Agent, play
import settings

# Declare the Fitness and Individual classes before anything else so that other functions and classes can use them later
creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def mut_gauss(individual, mu, sigma, indpb):
    """
    Cycle through the rows of an individual and randomly apply gaussian mutation to each element of a row.
    :param individual: A list of lists representing a weight matrix.
    :param mu: A float representing the mean of the gaussian distribution.
    :param sigma: A float representing the standard deviation of the gaussian distribution.
    :param indpb: A float indicating the independent probability of a given element to mutate.
    :return: A tuple containing the mutated individual.
    """

    # Cycle through the individual's rows and mutate each element if necessary
    for idx, row in enumerate(individual):
        individual[idx], = tools.mutGaussian(row, mu=mu, sigma=sigma, indpb=indpb)

    # Return the mutated individual
    return individual,


def crossover_two_points(ind1, ind2):
    """
    Perform a two points crossover over the rows of the given individuals.
    :param ind1: A list of lists representing the synaptic strength of a connection.
    :param ind2: A list of lists representing the synaptic strength of a connection.
    :return: A tuple, containing the crossed over individuals.
    """

    # Cycle through the rows of both ind1 and ind2
    for idx, (p1, p2) in enumerate(zip(ind1, ind2)):
        if len(p1) < 2 or len(p2) < 2:  # Those will only learn through mutation (e.g.: mc)
            continue
        # Mate
        ind1[idx], ind2[idx] = tools.cxTwoPoint(p1, p2)

    # Return the children
    return ind1, ind2


def crossover_uniform(ind1, ind2):
    """
    Perform a uniform crossover over the rows of the given individuals.
    See: http://www.tomaszgwiazda.com/uniformX.htm, for more details on the algorithm behind the uniform crossover.
    :param ind1: A list of lists representing the synaptic strength of a connection.
    :param ind2: A list of lists representing the synaptic strength of a connection.
    :return: A tuple, containing the crossed over individuals.
    """

    # Cycle through the rows of both ind1 and ind2
    for idx, (p1, p2) in enumerate(zip(ind1, ind2)):
        # Mate
        # The last parameter indicates the independent probability for each element to be swapped
        ind1[idx], ind2[idx] = tools.cxUniform(p1, p2, 0.3)

    # Return the children
    return ind1, ind2


def crossover_blend(ind1, ind2):
    """
    Perform a uniform crossover over the rows of the given individuals.
    See: http://www.tomaszgwiazda.com/blendX.htm, for more details on the algorithm behind the blend crossover.
    :param ind1: A list of lists representing the synaptic strength of a connection.
    :param ind2: A list of lists representing the synaptic strength of a connection.
    :return: A tuple, containing the crossed over individuals.
    """

    # Cycle through the rows of both ind1 and ind2
    for idx, (p1, p2) in enumerate(zip(ind1, ind2)):
        # Mate
        # The last parameter indicates the extent of the interval in which the new values can be drawn for each
        # attribute on both side of the parents’ attributes
        ind1[idx], ind2[idx] = tools.cxUniform(p1, p2, 0.1)

    # Return the children
    return ind1, ind2


class Genetic:
    """
    A simple cooperative co-evolving genetic algorithm to learn the pole balancing task using the PrimEmoArch project.
    """

    def __init__(self):
        """
        Declare and initialize the required attributes for the co-evolving genetic algorithm.
        """

        # Initialize the object's attributes
        self._species = []
        self._representatives = []
        self._environment = gym.make("CartPole-v0")

        # Initialize DEAP's attributes
        self._toolbox = base.Toolbox()
        self._toolbox.register("mate", crossover_uniform)
        self._toolbox.register("mutate", mut_gauss, mu=0.5, sigma=0.25, indpb=0.01)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("get_best", tools.selBest, k=1)
        self._pool = multiprocessing.Pool()
        self._toolbox.register("map", self._pool.map)

    def main(self):
        """
        The core method of the Genetic class, where species evolve to adapt to the task the GeneticAgent have to
        perform.
        :return: Nothing.
        """

        # Initialize all the species
        self._init_species()

        # Initialize the list of representatives
        self._representatives = [random.choice(specie) for specie in self._species]

        # Initialize a list of the next representatives
        next_repr = [None] * len(self._species)

        # Initialize the generation meter
        gen = 0

        # And here we ... go. Loop over the genetic algorithm until the last generation has been reached
        while gen < settings.MAX_GENERATION:
            # Message to the user
            print("Evaluating generation number {}.".format(gen))

            # Loop over the species
            for sp_idx, specie in enumerate(self._species):
                # Crossover and mutation
                # The last two parameters correspond to crossover and mutation probabilities
                specie = algorithms.varAnd(specie, self._toolbox, 0.6, 0.1)

                # Get the representatives for the other species
                repr_left, repr_right = self._representatives[0:sp_idx], self._representatives[sp_idx+1:]

                # Evaluate each individual
                for individual in specie:
                    # Concatenate the list of representatives with the current individual
                    reprs = repr_left + [individual] + repr_right
                    # Get the GeneticAgent corresponding to these representatives
                    agent = self._new_agent(reprs)
                    # Evaluate the agent on the given task
                    individual.fitness.values = play(self._environment, agent, render=True)

                # Select the individuals that will form the next generation
                self._species[sp_idx] = self._toolbox.map(self._toolbox.clone, self._toolbox.select(specie, len(specie)))
                # Select the individual that will be this specie's representative from now on
                next_repr[sp_idx] = self._toolbox.get_best(specie)[0]

            # Increase the generation meter
            gen += 1

            # Store the next representatives into the list of representatives
            self._representatives = next_repr

            # Reinitialize the next representatives
            next_repr = [None] * len(self._species)

        # End message for the user
        print("Evaluated {} generations.")

    def shutdown(self):
        """
        Do some clean up before gracefully shutting down.
        :return: Nothing.
        """

        # Terminate the multiprocessing pool
        self._pool.terminate()

        # Close the environment
        self._environment.close()

    def _init_species(self):
        """
        Fill in the _species attribute with random initial value, following the architecture's configuration.
        :return: Nothing.
        """

        # Get the architecture from the genetic configuration
        arch = gen_conf.ARCH

        # For each layer defined in the architecture
        for lay_name, post_conf in arch.items():
            # Input layers have no inputs, therefore no connections
            if post_conf.get('type') == LayerType.INPUT:
                continue

            # Each connection defines a new specie
            for pre_name, con_conf in post_conf.get('inputs').items():
                # Modulation and bias connections have no weight, so just skip them
                if con_conf.get('type') in [ConType.MODULATION_P, ConType.MODULATION_N, ConType.REC_MOD_P,
                                            ConType.REC_MOD_N, ConType.BIAS_P, ConType.BIAS_N, ConType.REC_BIAS_N,
                                            ConType.REC_BIAS_P]:
                    continue

                # Get the configuration corresponding to the pre-synaptic layer
                pre_conf = arch.get(pre_name)

                # Extract some recurrent configurations
                pre_type = pre_conf.get('type')
                post_size = post_conf.get('size')

                if post_conf.get('type') == LayerType.GROUP:
                    if pre_type == LayerType.GROUP:  # And now patterns are a thing
                        # Get the layer sizes of the pre group
                        pre_layer_sizes = pre_conf.get('layer_sizes')
                        if not isinstance(pre_layer_sizes, list):
                            pre_layer_sizes = [pre_layer_sizes] * pre_conf.get('size')

                        # Get the layer sizes of the post group
                        post_layer_sizes = post_conf.get('layer_sizes')
                        if not isinstance(post_layer_sizes, list):
                            post_layer_sizes = [post_layer_sizes] * post_size

                        # Add new species depending on the connection pattern
                        if con_conf.get('pattern') == ConPattern.ONE_TO_ONE:
                            # Add a new specie for each layer in both pre and post group
                            for in_size, out_size in zip(pre_layer_sizes, post_layer_sizes):
                                self._new_specie(in_size, out_size)
                        else:  # DENSE
                            for out_size in post_layer_sizes:
                                for in_size in pre_layer_sizes:
                                    self._new_specie(in_size, out_size)

                    else:  # LayerType.HIDDEN and LayerType.INPUT, no connection pattern taken into account
                        # Get the size of each layer in the post group
                        layer_sizes = post_conf.get('layer_sizes')
                        if not isinstance(layer_sizes, list):
                            layer_sizes = [layer_sizes] * post_size

                        # Add a new specie for each layer in the post group
                        pre_size = pre_conf.get('size')
                        for layer_size in layer_sizes:
                            self._new_specie(pre_size, layer_size)

                else:  # LayerType.HIDDEN and LayerType.INPUT
                    if pre_type == LayerType.GROUP:
                        layer_sizes = pre_conf.get('layer_sizes')
                        if not isinstance(layer_sizes, list):
                            layer_sizes = [layer_sizes] * pre_conf.get('size')
                        for layer_size in layer_sizes:
                            self._new_specie(layer_size, post_size)

                    else:  # LayerType.HIDDEN and LayerType.INPUT
                        self._new_specie(pre_conf.get('size'), post_size)

    def _new_specie(self, pre_size, post_size):
        """
        Append a new specie, filled with individuals of the given shape, to the list of all species.
        :param pre_size: An integer representing the size of the pre-synaptic layer.
        :param post_size: An integer representing the size of the post-synaptic layer.
        :return: Nothing.
        """

        # Append an new list of individuals
        # Each individual being a weight matrix of shape (pre_size, post_size)
        self._species.append([
            creator.Individual(
                [
                    [random.gauss(0.5, 0.5) for _ in range(pre_size)]
                    for _ in range(post_size)
                ]
            )
            for _ in range(settings.POPULATION_SIZE)
        ])

    def _new_agent(self, individuals):
        """
        Configure a new GeneticAgent whose performance will reflect the fitness of the given individuals.
        :param individuals: A list of representatives (i.e.: weight matrices) for each species.
        :return: An instance of the GeneticAgent class.
        """

        # Get the architecture's configuration from the specific module
        arch = gen_conf.ARCH

        # For each connection between layers specify the initial value of the weight matrix
        # The weights are extracted from the list of individuals
        for post_name, post_conf in arch.items():
            # Extract the layer's type
            post_type = post_conf.get('type')

            # If the Layer is an InputLayer, just skip it
            if post_type == LayerType.INPUT:
                continue

            # For each connection
            for pre_name, con_conf in post_conf.get('inputs').items():
                # If the connection is of type bias or modulation, just skip it, since they have no weight
                if con_conf.get('type') in [ConType.MODULATION_N, ConType.MODULATION_P, ConType.REC_MOD_N,
                                            ConType.REC_MOD_P, ConType.BIAS_N, ConType.BIAS_P, ConType.REC_BIAS_N,
                                            ConType.REC_BIAS_P]:
                    continue

                # Extract the configuration of the pre-synaptic layer
                pre_conf = arch.get(pre_name)
                pre_type = pre_conf.get('type')
                pre_size = pre_conf.get('size')

                # Assign the weights depending on the pre- and post-layers types and on the connection pattern
                if post_type == LayerType.GROUP:
                    post_size = post_conf.get('size')
                    if pre_type == LayerType.GROUP:
                        # Check the connection pattern
                        if con_conf.get('pattern') == ConPattern.ONE_TO_ONE:
                            # Extract the list of weights
                            # In this instance post_size == pre_size
                            weights = [[weight] for weight in individuals[0:post_size]]
                            # Remove the values from the original individuals
                            del individuals[0:post_size]
                        else:  # ConPattern.DENSE
                            # Extract the list of weights
                            weights = [[weight for weight in individuals[i * pre_size:(i+1) * pre_size]]
                                       for i in range(post_size)]
                            # Remove the weight from the original individuals
                            del individuals[0:pre_size * post_size]
                    else:  # LayerType.HIDDEN and LayerType.INPUT
                        # Extract the list of weights
                        weights = [individuals[0:post_size]]
                        # Remove the weights from the original list
                        del individuals[0:post_size]
                else:  # LayerType.HIDDEN and LayerType.INPUT
                    if pre_type == LayerType.GROUP:
                        # Get a list of individuals matching the connections between the pre-group's layers and the post
                        # layer
                        weights = individuals[0:pre_size]
                        # Remove the extracted individuals from the original list
                        del individuals[0:pre_size]
                    else:
                        # Get a single individual representing the connection's weight
                        # See HiddenLayer.add_input() for more details on why weight is put inside a list
                        weights = [individuals.pop(0)]

                # Assign the weight to the connection's configuration
                arch[post_name]['inputs'][pre_name]['weight'] = weights

        # Return a new GeneticAgent initialized with the given architecture
        return GeneticAgent(arch, self._environment.action_space.n, self._environment.observation_space.high[0],
                            self._environment.observation_space.high[2])


class GeneticAgent(Agent):
    """
    Define a simple agent used by the Genetic class to evaluate the performance of a given individual.
    """

    def __init__(self, arch_conf, out_size, max_pos, max_angle):
        """
        Declare and initialize the required attributes.
        :param arch_conf: A dictionary containing the configuration to build the brain.
        :param out_size: An integer representing the size of the output vector.
        :param max_pos: A float representing the maximum position, the cart can occupy on the horizontal axis.
        :param max_angle: A float representing the maximum angle in radians, the pole can lean towards the right or
        left, before it is considered as fallen.
        """

        # Initialize the agent's brain
        self._brain = Network(lesion=False)
        self._brain.build(arch_conf)

        # Initialize all the other required attributes
        self._out_size = out_size
        # Divide by two since it was multiplied by the same amount in the source code, for no apparent reasons
        self._max_cart_pos = max_pos / 2
        self._max_pole_angle = max_angle / 2

    def next_action(self, observation, reward):
        """
        The main function of this class, learn from your past mistakes and choose the next action to perform.
        :param observation: A list of values representing: the cart's position and velocity, as well as the pole's
        angular position and velocity.
        :param reward: An integer representing the reward obtained for performing the task.
        :return: Int. An integer representing the next action to perform. 0: left, 1: right.
        """

        # Normalize the sensory inputs and activate the brain
        curr_ext_ins = np.array([[observation[1]], [observation[3]]], dtype=np.float64)
        curr_int_ins = np.array([[observation[0] / self._max_cart_pos], [observation[2] / self._max_pole_angle]],
                                dtype=np.float64)
        self._brain.activate({'external': curr_ext_ins, 'internal': curr_int_ins, 'reward': [[reward / 10.]]}, "mc")

        # Choose the bes action to perform next
        action = np.argmax(self._brain.layers['mc'].output)

        # Return the chosen action
        return action

    def replay(self):
        pass


if __name__ == "__main__":
    # Instantiate a genetic supervisor
    supervisor = Genetic()

    try:
        # Launch the main method, that goes through the evolution of the different species
        supervisor.main()
    except KeyboardInterrupt:
        print("Exiting the program on user's request ...")
    finally:
        # Gracefully shut the supervisor down
        supervisor.shutdown()

