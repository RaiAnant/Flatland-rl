
""" ###################### EDGE ####################"""

class g_edge:
    def __init__(self, node1, node2):
        self.start = node1
        self.end = node2
        self.cost_triples = []

class g_vertex:
    def __init__(self, node):
        self.id = node
        self.edges = []
        self.transitions = []


class Global_Graph:
    def __init__(self):
        self.vert_dict = {}
        self.edge_dict = []
        self.num_vertices = 0
        self.num_edges = 0

    def add_vertex(self, node):
        if node not in self.vert_dict.keys():
            print("adding vertex ", self.num_vertices, node)
            self.num_vertices = self.num_vertices + 1
            new_vertex = g_vertex(node)
            self.vert_dict[node] = new_vertex
            return new_vertex
        return self.vert_dict[node]

    def add_edge(self, frm, to, traj):
        # an edge can be added as follows
        #   both the vertices must exist
        print("adding edge between ", self.num_edges, frm, to)

        self.edge_dict.append([frm, to, traj, []])
        self.num_edges += 1




if __name__ == "__main__":
    # create a graphof 4 nodes
    #
    # if a node is added - only node list is updated
    # call graph insert method
    # if an edge is added - possibly two nodes will be added
    g = Global_Graph()
    g.add_vertex('a')
    g.add_edge('a','b')
    g.add_edge('a','c')
    g.add_edge('b','c')
    g.add_edge('b','d')
    g.add_edge('c','d')

    source_vert = g.vert_dict['a']
    for edge in g.vert_dict['a'].edges:
        if edge.end == 'c':
            edge.cost_triples.append([1,2,3])
            print("found")
    #edge_temp = g.edge_dict['ab']
    #edge_temp.cost_triples.append([1,2,3])
    #g.add_vertex('b')
    #g.add_vertex('c')
    #g.add_vertex('d')
    #g.add_vertex('e')
    print("done")