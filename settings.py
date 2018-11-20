#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables related to the different controller algorithms
REPLAY_STACK_SIZE = 1000
EPSILON = 0.1  # Should be decaying?
LEARNING_RATE = 0.9

POPULATION_SIZE = 20
MAX_GENERATION = 100
