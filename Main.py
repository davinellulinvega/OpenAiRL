#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'davinellulinvega'
import importlib
import argparse
import random as rnd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import gym
import settings

# TODO: Add cProfile and pstats calls to profile the run time of each controller and optimize the code before running it on AWS.
# See: https://docs.python.org/3.6/library/profile.html, for more details on how to do that.


class Agent:
    """
    Defines an abstract agent, which should be overridden to implement a specific agent.
    """

    def next_action(self, observation, reward):
        """
        This is where the core of any agent lives, since the method takes a list of observations and a reward for the
        previously performed action and expect the agent to choose what to do next.
        :param observation: A list of sensory data, describing the current state of the environment.
        :param reward: A float indicating how good or bad the previous action was.
        :return: An integer, indicating the next action to perform.
        """

        raise NotImplementedError("Class {} does not implement the next_action() "
                                  "method.".format(self.__class__.__name__))

    def replay(self):
        """
        Randomly relive a batch of memories gathered during training, so as not to forget what was learned a long time
        ago.
        :return: Nothing.
        """

        raise NotImplementedError("Class {} does not implement the next_action() "
                                  "method.".format(self.__class__.__name__))

class Qlearning(Agent):
    """
    Implement a simple Q-learning algorithm to solve the Cart pole classical control problem.
    """

    def __init__(self, in_size, out_size):
        """
        Build the neural network responsible for learning the control task.
        :param in_size: The size of the input space.
        :param out_size: The size of the action space.
        """

        # Initialize the environment
        self._old_obs = None
        self._old_action = 0
        self._out_size = out_size

        # Initialize the neural network
        self._network = Sequential()
        self._network.add(Dense(30, kernel_initializer="lecun_uniform", input_dim=in_size))
        self._network.add(LeakyReLU())
        self._network.add(Dense(20, kernel_initializer="lecun_uniform"))
        self._network.add(LeakyReLU())
        self._network.add(Dense(10, kernel_initializer="lecun_uniform"))
        self._network.add(LeakyReLU())
        self._network.add(Dense(5, kernel_initializer="lecun_uniform"))
        self._network.add(LeakyReLU())
        self._network.add(Dense(out_size, kernel_initializer="lecun_uniform", activation="linear"))

        optimizer = Adam(lr=0.001)
        self._network.compile(loss="mse", optimizer=optimizer)

        # Initialize the elements required for the Q-learning
        self._replay_stack = []

    def next_action(self, observation, reward):
        """
        Given the current state of the environment, choose the an action to perform.
        :param observation: An array containing the state of the environment.
        :param reward: A float representing the positive/negative feedback from the environment, corresponding to the
        previously performed action.
        :return: Int. An integer representing the next action to perform. 0: left, 1: right.
        """

        # Get the current observation
        curr_obs = np.reshape(observation, (1, 4))

        if self._old_obs is not None:
            # Fill in the replay stack
            self._replay_stack.append((self._old_obs, self._old_action, reward, curr_obs))
            if len(self._replay_stack) > settings.REPLAY_STACK_SIZE:
                self._replay_stack.pop(0)

            # Get the current action values
            curr_act_vals = self._network.predict(curr_obs, batch_size=1)

            # Compute the target
            target_value = reward + settings.LEARNING_RATE * np.max(curr_act_vals)
            target = self._network.predict(self._old_obs, batch_size=1)
            target[0, self._old_action] = target_value

            # Have the network learn
            self._network.fit(self._old_obs, target, batch_size=1, epochs=1, verbose=0)

        # Choose the next action to perform following an e-greedy policy
        if rnd.uniform(0, 1) <= settings.EPSILON:
            self._old_action = rnd.randint(0, self._out_size - 1)
        else:
            # Get the current action values
            curr_act_vals = self._network.predict(curr_obs, batch_size=1)

            self._old_action = np.argmax(curr_act_vals)

        # Save the environment's state
        self._old_obs = curr_obs

        # Return the next action to perform
        return self._old_action

    def replay(self):
        """
        Using a random batch of states gathered so far, relearn the associations.
        :return: Nothing.
        """

        # Get a random batch of inputs
        # Use of the '//' operator is required since rnd.sample needs an int as the population size
        batch = rnd.sample(self._replay_stack, min(settings.REPLAY_STACK_SIZE // 4, len(self._replay_stack)))

        # Initialize the training sets
        x_train = []
        y_train = []

        # For each memory in the batch
        for old_obs, old_act, reward, curr_obs in batch:
            # Compute the target
            old_act_vals = self._network.predict(old_obs, batch_size=1)
            curr_act_vals = self._network.predict(curr_obs, batch_size=1)
            max_q = np.max(curr_act_vals)

            target_value = reward + settings.LEARNING_RATE * max_q
            old_act_vals[0][old_act] = target_value

            # Add the input and ouptput to the training sets
            x_train.append(np.reshape(old_obs, (4,)))
            y_train.append(np.reshape(old_act_vals, (2,)))

        # Train all over again
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self._network.fit(x_train, y_train, batch_size=len(batch), epochs=200, verbose=0)


class AClearning(Agent):
    """
    Implement a simple Actor-Critic algorithm to solve the Cart Pole classical control problem.
    """

    def __init__(self, in_size, out_size):
        """
        Build the neural network responsible for learning and initialize all variables required to tackle this task.
        :param obs: An array containing the original state of the environment.
        :param in_size: An integer representing the size of the input space.
        :param out_size: An integer representingt the size of the output space.
        """

        # Initialize the environment
        self._old_obs = None
        self._old_action = 0
        self._out_size = out_size

        # Initialize both neural networks
        self._actor = Sequential()
        self._actor.add(Dense(30, kernel_initializer="lecun_uniform", input_dim=in_size))
        self._actor.add(LeakyReLU())
        self._actor.add(Dense(20, kernel_initializer="lecun_uniform"))
        self._actor.add(LeakyReLU())
        self._actor.add(Dense(10, kernel_initializer="lecun_uniform"))
        self._actor.add(LeakyReLU())
        self._actor.add(Dense(5, kernel_initializer="lecun_uniform"))
        self._actor.add(LeakyReLU())
        self._actor.add(Dense(out_size, kernel_initializer="lecun_uniform", activation="linear"))

        optimizer = Adam(lr=0.001)
        self._actor.compile(loss="mse", optimizer=optimizer)

        self._critic = Sequential()
        self._critic.add(Dense(30, kernel_initializer="lecun_uniform", input_dim=in_size))
        self._critic.add(LeakyReLU())
        self._critic.add(Dense(20, kernel_initializer="lecun_uniform"))
        self._critic.add(LeakyReLU())
        self._critic.add(Dense(10, kernel_initializer="lecun_uniform"))
        self._critic.add(LeakyReLU())
        self._critic.add(Dense(5, kernel_initializer="lecun_uniform"))
        self._critic.add(LeakyReLU())
        self._critic.add(Dense(1, kernel_initializer="lecun_uniform", activation="linear"))

        optimizer = Adam(lr=0.001)
        self._critic.compile(loss="mse", optimizer=optimizer)

        # Initialize the elements required for the policy and memory
        self._replay_stack = []

    def next_action(self, obs, reward):
        """
        Given the current state of the environment, choose the next action to perform. Also learn from the given reward.
        :param obs: An array containing the current state of the environment.
        :param reward: An integer representing the value of taking the previous action in the previous state.
        :return: Int. An integer representing the action to perform next.
        """

        # Reshape the current observations
        curr_obs = np.reshape(obs, (1, 4))

        if self._old_obs is not None:
            # Fill in the replay stack
            self._replay_stack.append((self._old_obs, self._old_action, reward, curr_obs))
            if len(self._replay_stack) > settings.REPLAY_STACK_SIZE:
                self._replay_stack.pop(0)

            # Get the current Q-values
            curr_state_val = self._critic.predict(curr_obs, batch_size=1, verbose=0)

            # Compute the target value of the critic
            old_state_val = self._critic.predict(self._old_obs, batch_size=1, verbose=0)
            critic_target = reward + settings.LEARNING_RATE * curr_state_val

            # Compute the actor target
            error = reward + settings.LEARNING_RATE * curr_state_val[0, 0] - old_state_val[0, 0]
            actor_target = self._actor.predict(self._old_obs, batch_size=1, verbose=0)
            actor_target[0, self._old_action] = error

            # Have both networks learn
            self._critic.fit(self._old_obs, critic_target, batch_size=1, epochs=1, verbose=0)
            self._actor.fit(self._old_obs, actor_target, batch_size=1, epochs=1, verbose=0)

        # Store the old state
        self._old_obs = curr_obs

        # Choose what action to perform next using a greedy policy
        if rnd.uniform(0, 1) < settings.EPSILON:
            self._old_action = rnd.randint(0, self._out_size - 1)
        else:
            curr_act_vals = self._actor.predict(curr_obs, batch_size=1, verbose=0)
            self._old_action = np.argmax(curr_act_vals)

        # Return the next action
        return self._old_action

    def replay(self):
        """
        Relearn past associations from memory.
        :return: Nothing.
        """

        # Randomly select a batch to learn from
        # Use of the '//' operator is required since rnd.sample needs an int as the population size
        batch = rnd.sample(self._replay_stack, min(settings.REPLAY_STACK_SIZE // 4, len(self._replay_stack)))

        # Initialize the training sets
        x_train = []
        y_train_crit = []
        y_train_act = []

        # For each memory
        for old_state, old_action, reward, curr_state in batch:
            # Compute the values for both current and old state and actions
            old_state_val = self._critic.predict(old_state, batch_size=1, verbose=0)
            curr_state_val = self._critic.predict(curr_state, batch_size=1, verbose=0)
            old_acts_vals = self._actor.predict(old_state, batch_size=1, verbose=0)

            # Compute the actor's and critic's target
            critic_target = reward + settings.LEARNING_RATE * curr_state_val

            error = reward + settings.LEARNING_RATE * curr_state_val[0, 0] - old_state_val[0, 0]
            actor_target = old_acts_vals
            actor_target[0, old_action] = error

            # Store the input and training outputs into their respective sets
            x_train.append(np.reshape(old_state, (4, )))
            y_train_crit.append(np.reshape(critic_target, (1, )))
            y_train_act.append(np.reshape(actor_target, (2, )))

        # Train the model on the batch
        x_train = np.array(x_train)
        y_train_act = np.array(y_train_act)
        y_train_crit = np.array(y_train_crit)
        self._critic.fit(x_train, y_train_crit, batch_size=len(batch), epochs=200, verbose=0)
        self._actor.fit(x_train, y_train_act, batch_size=len(batch), epochs=200, verbose=0)


def play(env, agent, render=False):
    """
    Have the agent given in parameter, accomplish a task forever. Or at least until the user sends a SIGSTOP signal.
    :param env: An OpenAi gym environment defining a task the agent will have to perform.
    :param agent: A learning artificial agent which can perform tasks.
    :param render: A boolean indicating whether to render the task while the agent is performing it or not.
    :return: A tuple representing the agent's performance on the task.
    """

    # Initialize some variables
    done = False
    t = 0
    obs = env.reset()
    reward = 0

    # Perform the task until completion or failure
    while not done:
        # Get the next action given the current environment's state
        next_action = agent.next_action(obs, reward)

        # Perform the next action
        obs, reward, done, info = env.step(next_action)

        # Render the environment
        if render:
            env.render()

        # Check if the end has been reached
        if done and t < 195:
            reward = -5

        # Increase the performance meter
        t += 1

    # Call the next_action method a last time so that the agent can learn from its mistake
    agent.next_action(obs, reward)

    # Return the measure of the agent's performance
    return t,


if __name__ == "__main__":
    # Declare and parse command line arguments
    parser = argparse.ArgumentParser()
    excl_grp = parser.add_mutually_exclusive_group()
    excl_grp.add_argument("-a", help="Use the Actor-Critic learning algorithm.", dest="ac", action="store_true")
    excl_grp.add_argument("-q", help="Use the Q-learning algorithm.", dest="q", action="store_true")
    excl_grp.add_argument("-p", help="Use the PrimEmo architecture.", dest="p", action="store_true")
    parser.add_argument("-r", "--render", help="If specified, the task will be rendered while the agent is learning.",
                        dest="render", action="store_true")
    parser.add_argument("-m", "--module", help="The name (relative to the PrimEmoArch.Configurations sub-module)  of "
                                               "the module containing the configuration to be loaded for the PrimEmo "
                                               "architecture..", dest="mod_name")
    args = parser.parse_args()

    # Initialize the environment
    env = gym.make("CartPole-v0")

    if args.q:
        # Build a Q-learning agent
        agent = Qlearning(env.observation_space.shape[0], env.action_space.n)
    elif args.ac:
        # Build an Actor-Critic agent
        agent = AClearning(env.observation_space.shape[0], env.action_space.n)
    elif args.p:
        # The sizes of the internal and external input layers have to be set in the settings module
        agent = PrimEmo(env.action_space.n, env.observation_space.high[0], env.observation_space.high[2], args.mod_name)
    else:
        print("You must choose at least one learning algorithm to apply to this task.")
        exit(-1)

    try:
        while True:
            # Let the agent perform its task
            t_steps,  = play(env, agent, args.render)
            # Some nice message for the user
            print("Task performed in {} time steps.".format(t_steps))

            # Let the agent learn from memory
            agent.replay()
            # Reset the environment
            env.reset()
    except KeyboardInterrupt:
        print("Exiting on user's request ...")
    finally:
        # Clear Keras' session
        K.clear_session()
        # Close the environment
        env.close()
