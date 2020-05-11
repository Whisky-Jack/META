#########################################################################
# Defines graph and component structures
#########################################################################
import networkx as nx

# Component corresponds to a single layer/structure, components make up modules
class Component:
    def __init__(self):
        super().__init__()

# Module corresponds to a combination of layers, modules are used to make up networks in blueprints
class Module:
    def __init__(self):
        super().__init__()

# Blueprints contain information on the topology of a netowrk, linking modules
class Blueprint:
    def __init__(self):
        super().__init__()

