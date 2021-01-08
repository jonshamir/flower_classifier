import json
from anytree import Node, RenderTree

# taxonomy tree
plantae = Node('plantae')

rank_names = ['kingdom', 'clade', 'order', 'family', 'genus']
rank_nodes = {
    'kingdom': [plantae],
    'clade': [],
    'order': [],
    'family': [],
    'genus': [],
    'species': []
}

clade_names = ['monocots', 'eudicots', 'nymphaeales', 'magnoliids']

def get_node(rank_name, node_name, parent):
    node_list = rank_nodes[rank_name]
    node = next((node for node in node_list if node.name == node_name), None)
    if node == None:
        node = Node(node_name, parent=parent)
        node_list.append(node)
    return node


# Construct taxonomic tree data structure
with open('flower_data.json') as json_file:
    flower_data = json.load(json_file)
    for flower in flower_data:
        # print('\n' + flower['name'])
        for rank in flower['taxonomy']:
            rank_name = list(rank.keys())[0]
            rank_value = rank[rank_name]
            if rank_name == 'clade' and rank_value in clade_names:
                curr_node = get_node(rank_name, rank_value, plantae)
            elif rank_name in ['order', 'family', 'genus']:
                curr_node = get_node(rank_name, rank_value, curr_node)

        get_node('species', flower['name'], curr_node)


# Print tree
for pre, fill, node in RenderTree(plantae): print("%s%s" % (pre, node.name))

for rank_name, node_list in rank_nodes.items():
    print(rank_name + ": " + str(len(node_list)))
