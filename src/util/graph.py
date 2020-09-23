
""" ###################### EDGE ####################"""

class edge:
    def __init__(self, node1, node2):
        self.start = node1
        self.end = node2
        self.cost_triples = []

class vertex:
    def __init__(self, node):
        self.id = node
        self.edges = []


class Graph:
    def __init__(self):
        self.vert_dict = {}
        #self.edge_dict = {}
        self.num_vertices = 0

    def add_vertex(self, node):
        if node not in self.vert_dict.keys():
            self.num_vertices = self.num_vertices + 1
            new_vertex = vertex(node)
            self.vert_dict[node] = new_vertex
            return new_vertex
        return self.vert_dict[node]

    def add_edge(self, frm, to):
        frmob = self.add_vertex(frm)
        toob = self.add_vertex(to)
        new_edge = edge(frm, to)
        #self.edge_dict[frm+to] = new_edge

        frmob.edges.append(new_edge)
        toob.edges.append(new_edge)



if __name__ == "__main__":
    # create a graphof 4 nodes
    #
    # if a node is added - only node list is updated
    # call graph insert method
    # if an edge is added - possibly two nodes will be added
    g = Graph()
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



""" ###################### VERTEX ####################"""
"""
class Vertex:
    def __init__(self, node, mat):
        self.id = node
        self.mcv = mat
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        # add a multi-dim value here that represent the number of agents on that edge
        self.adjacent[neighbor] = weight

    def del_neighbor(self, neighbor):
        if neighbor in self.adjacent.keys():
            del self.adjacent[neighbor]


    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_mcv(self):
        return self.mcv

    def set_mcv(self, tempMCV):
        self.mcv = tempMCV

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]
"""


""" ###################### GRAPH ####################"""

"""
class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node, mat):
        if node not in self.vert_dict.keys():
            self.num_vertices = self.num_vertices + 1
            new_vertex = Vertex(node, mat)
            self.vert_dict[node] = new_vertex
            return new_vertex
        return None


    def del_vertex(self, node):
        self.num_vertices = self.num_vertices - 1
        del self.vert_dict[node]


    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def get_vertex_mcv(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n].get_mcv()
        else:
            return None

    def set_vertex_mcv(self, n, tempMCV):
        self.vert_dict[n].set_mcv(tempMCV)

    def add_edge(self, frm, frm_mat, to, to_mat, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm, frm_mat)
        if to not in self.vert_dict:
            self.add_vertex(to, to_mat)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def del_edge(self, frm, to):

        self.vert_dict[frm].del_neighbor(self.vert_dict[to])
        self.vert_dict[to].del_neighbor(self.vert_dict[frm])

    def get_vertices(self):
        return self.vert_dict.keys()

    def is_neighbour(self, frm, to):
        vertex = self.vert_dict[to]
        for w in vertex.get_connections():
            if frm == w.get_id():
                return True
        return False

"""
