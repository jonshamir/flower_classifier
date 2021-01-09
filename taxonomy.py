import json
import numpy as np
from anytree import Node, RenderTree

# taxonomy tree
RANKS = ['kingdom', 'clade', 'order', 'family', 'genus', 'species']
ALLOWED_CLADES = ['monocots', 'eudicots', 'nymphaeales', 'magnoliids']

rank_nodes = {rank: [] for rank in RANKS}
rank_names = {rank: [] for rank in RANKS}

# root of the tree
plantae = Node(0)
rank_nodes['kingdom'].append(plantae)
rank_names['kingdom'].append('plantae')



def get_node(rank, rank_name, parent):
    node_list = rank_nodes[rank]
    name_list = rank_names[rank]

    try:
        rank_id = name_list.index(rank_name)
    except ValueError:
        rank_id = len(name_list)
        name_list.append(rank_name)
        node_list.append(Node(rank_id, parent=parent))

    return node_list[rank_id]


# Construct taxonomic tree data structure
with open('data/flower_data.json') as json_file:
    flower_data = json.load(json_file)
    for flower in flower_data:
        # print('\n' + flower['name'])
        for rank_entry in flower['taxonomy']:
            rank = list(rank_entry.keys())[0]
            rank_name = rank_entry[rank]
            if rank == 'clade' and rank_name in ALLOWED_CLADES:
                curr_node = get_node(rank, rank_name, plantae)
            elif rank in ['order', 'family', 'genus']:
                curr_node = get_node(rank, rank_name, curr_node)

        get_node('species', flower['name'], curr_node)


# Print tree
for pre, fill, node in RenderTree(plantae):
    node_rank = RANKS[node.depth]
    node_name = rank_names[node_rank][node.name]
    print("%s%s" % (pre, node_name))

# Rank statistics
for rank_name, node_list in rank_nodes.items():
    print(rank_name + ": " + str(len(node_list)))


node = next((node for node in species_nodes if node.name == flower_id), None)

species_nodes = rank_nodes['species']


species_supclasses = []
for species_id in range(len(species_nodes)):
    species_node = next((node for node in species_nodes if node.name == species_id), None)
    order_id = species_node.ancestors[RANKS.index('order')].name
    family_id = species_node.ancestors[RANKS.index('family')].name
    species_supclasses.append((order_id, family_id))


 rank_names['order'][species_supclasses[12][0]]
 rank_names['family'][species_supclasses[12][1]]



species_supclasses = np.array(species_supclasses)

# list of classes [c1,c2,c3,...] -> lists of supclasses
# species_subclasses[[12,12,1]]
