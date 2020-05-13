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
    DENSE = (nn.Linear, {"units":128, "activation":"relu"})   # TODO: Dense size should be variable


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

class StructureBuilder():

    def random_component(self, possible_components, possible_complementary_components = None):
        component_type = random.choice(list(possible_components))
        component_def, possible_parameters = possible_components[component_type]

        parameter_def = {}
        for parameter_name in possible_parameters:
            parameter_def[parameter_name] = self.random_parameter_def(possible_parameters, parameter_name)

        # TODO: what is a complementary component
        """
        if possible_complementary_components != None:
            compl_component_def, possible_compl_parameters = possible_complementary_components[random.choice(list(possible_complementary_components))]

            compl_parameter_def = {}
            for parameter_name in possible_compl_parameters:
                compl_parameter_def[parameter_name] = self.random_parameter_def(possible_compl_parameters, parameter_name)
            complementary_component = [compl_component_def, compl_parameter_def]
            keras_complementary_component = compl_component_def(**compl_parameter_def)
        else:
            complementary_component = None
            keras_complementary_component = None
        """

        complementary_component = None

        new_component = Component(representation=[component_def, parameter_def],
                                    keras_component=None,#component_def(**parameter_def),
                                    complementary_component=complementary_component,
                                    keras_complementary_component=None,#keras_complementary_component,
                                    component_type=component_type)
        return new_component
    
    def random_module(self, global_configs, possible_components, possible_complementary_components, name=0, layer_type=ModuleComposition.INTERMED):

        node_range = self.random_parameter_def(global_configs, "component_range")

        graph = self.random_graph(node_range=node_range,
                                            node_content_generator=self.random_component,
                                            args = {"possible_components": possible_components,
                                                    "possible_complementary_components": possible_complementary_components})

        self.count+=1
        new_module = Module(None, layer_type=layer_type, component_graph=graph)

        return new_module
    
    def random_graph(self, node_range, node_content_generator, args=None):

        new_graph = nx.DiGraph()

        for node in range(node_range):
            node_def = node_content_generator(**args)
            new_graph.add_node(node, node_def=node_def)

            if node == 0:
                pass
            elif node > 0 and (node < node_range-1 or node_range <= 2):
                precedent = random.randint(0, node-1)
                new_graph.add_edge(precedent, node)
            elif node == node_range-1:
                leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                root_node = min([node for node in new_graph.nodes() if new_graph.in_degree(node) == 0])
                leaf_nodes.remove(node)

                while (len(leaf_nodes) > 0):
                    if len(leaf_nodes) <= 2:
                        leaf_node = random.choice(leaf_nodes)
                        new_graph.add_edge(leaf_node, node)
                    else:
                        leaf_nodes.append(root_node)
                        random_node1 = random.choice(leaf_nodes)
                        simple_paths = [node for path in nx.all_simple_paths(new_graph, root_node, random_node1) for node in path]
                        leaf_nodes.remove(random_node1)
                        random_node2 = random.choice(leaf_nodes)
                        if (new_graph.in_degree(random_node2) >= 1 and random_node2 not in simple_paths and random_node2 != root_node):
                            new_graph.add_edge(random_node1, random_node2)
                    leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                    leaf_nodes.remove(node)

        return new_graph
    
    def random_blueprint(self, global_configs, possible_components, possible_complementary_components,
                        input_configs, possible_inputs, possible_complementary_inputs,
                        output_configs, possible_outputs, possible_complementary_outputs,
                        input_shape, node_content_generator=None, args={}, name=0):

        node_range = self.random_parameter_def(global_configs, "module_range")

        if (node_content_generator == None):
            node_content_generator = self.random_module
            args = {"global_configs": global_configs,
                    "possible_components": possible_components,
                    "possible_complementary_components": possible_complementary_components}

        input_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": input_configs,
                                                    "possible_components": possible_inputs,
                                                    "possible_complementary_components": None,
                                                    "layer_type": ModuleComposition.INPUT})
        #self.save_graph_plot(f"blueprint_{name}_input_module.png", input_node)

        intermed_graph = self.random_graph(node_range=node_range,
                                            node_content_generator=node_content_generator,
                                            args = args)
        #self.save_graph_plot(f"blueprint_{name}_intermed_module.png", intermed_graph)

        output_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": output_configs,
                                                    "possible_components": possible_outputs,
                                                    "possible_complementary_components": possible_complementary_outputs,
                                                    "layer_type": ModuleComposition.OUTPUT})
        #self.save_graph_plot(f"blueprint_{name}_output_module.png", output_node)

        graph = nx.union(input_node, intermed_graph, rename=("input-", "intermed-"))
        graph = nx.union(graph, output_node, rename=(None, "output-"))
        graph.add_edge("input-0", "intermed-0")
        graph.add_edge(f"intermed-{max(intermed_graph.nodes())}", "output-0")
        self.save_graph_plot(f"blueprint_{name}_module_level_graph.png", graph)

        new_blueprint = Blueprint(None, input_shape, module_graph=graph)

        return new_blueprint
