#########################################################################
# Defines graph and component structures
#########################################################################
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

class ModuleComposition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    CONV = "conv2d"
    DENSE = "dense"
    OUTPUT = "output"
    COMPILER = "compiler"

class ComponentParameters(Enum):
    CONV2D = (nn.Conv2d, {"filters": ([8, 16], 'int'), "kernel": ([3, 5, 7], 'list'), "stride": ([1, 2, 3], 'list')})
    MAXPOOLING2D = (nn.MaxPool2d, {"kernel":[3, 5, 7]})
    FLATTEN = []
    DENSE = (nn.Linear, {"units":128, "activation":"relu"})   # Dense size should be variable


class Component:
    """
    Component corresponds to a single layer/structure, components make up modules
    """
    def __init__(self, representation, component_type=None, keras_component=None, complementary_component=None, keras_complementary_component=None):
        self.representation = representation
        self.keras_component = keras_component
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component
        self.component_type = component_type

class Module:
    """
    Module corresponds to a combination of layers, modules are used to make up networks in blueprints
    """
    def __init__(self, components:dict, layer_type:ModuleComposition=ModuleComposition.INPUT, mark=None, component_graph=None, parents=None):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.mark = mark
        self.weighted_scores = [99,0]
        self.score_log = []
        self.species = None
        self.parents = parents
        self.use_count = 0

class Blueprint:
    """
    Blueprints contain information on the topology of a netowrk, linking modules
    """
    def __init__(self, modules):
        self.modules = modules
        self.input_shape = input_shape
        self.module_graph = module_graph
        self.mark = mark
        self.weighted_scores = [99,0]
        self.score_log = []
        self.species = None

class HistoricalMarker:
    """
    Historical markers are used in codeepneat to cross models and to split blueprints into species
    """
    
    def __init__(self):
        self.module_counter = 0
        self.blueprint_counter = 0
        self.individual_counter = 0
    
    def mark_module(self):
        self.module_counter += 1
        return self.module_counter

    def mark_blueprint(self):
        self.blueprint_counter += 1
        return self.blueprint_counter
    
    def mark_individual(self):
        self.individual_counter += 1
        return self.individual_counter