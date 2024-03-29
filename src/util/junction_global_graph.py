
""" ###################### EDGE ####################"""
import numpy as np
from collections import defaultdict


class vertex:
    def __init__(self, type, node, id):
        """

        :param node:
        """
        self.id = id
        self.Type = type
        self.Cells = node

        self.Trains = []
        self.TrainsDir = []  # 0 = A->B, 1 = B->A
        self.Links = []
        self.TrainsTraversal = defaultdict(list)

        self.is_signal_on = False
        self.is_starting_edge = False

        self.is_safe = True
        self.occupancy = 0
        self.extended_capacity = len(node)
        self.capacity = len(node)

    def __str__(self):
        """

        :return:
        """
        return 'Type : ' + str(self.Type) \
                + '; Trains: ' + str(self.Trains) \
                + '; Safety Status: ' + str(self.is_safe)

    def other_end(self, first):
        return self.Cells[0] if self.Cells[-1] == first else self.Cells[-1]

    def setCosts(self):
        """

        :return:
        """

        #self.setCollision()

        #self.CostCollisionLockTotal = 0
        #self.CostTransitionTimeTotal = 0
        #self.CostDeadLockTotal = 0
        #self.CostTotal = 0
        #self.CostPerTrain = []
        #self.DeadlockCostPerTrain = []

        #if self.signal_time > 1:
        #    self.signal_time -= 1
        #elif self.signal_time == 1:
        #    self.signal_time -= 1
        #    self.signal_deadlocks = []

        #start_times = [item[0] for item in self.TrainsTime]
        agent_dirs = np.unique(self.TrainsDir)
        if self.is_starting_edge:
            self.is_safe = True
        else:
            self.is_safe = True if len(agent_dirs) <= 1 else False

        #self.is_safe = True if len(np.unique(self.TrainsDir)) <= 1 else False


    def setExtendedCapacity(self):
        """

        :return:
        """

        if self.is_safe:
            pending_to_explore = []
            explored = []
            explored.append(self.id)
            for vertex in self.Links:
                if vertex[1].is_safe:
                    pending_to_explore.append(vertex[1])

            capacity = 0
            while len(pending_to_explore):
                vertex = pending_to_explore.pop()
                explored.append(vertex.id)
                capacity += len(vertex.Cells)
                for next_vertex in vertex.Links:
                    if next_vertex[1].is_safe and next_vertex[1].id not in explored:
                        pending_to_explore.append(next_vertex[1])
            self.extended_capacity = capacity

        if self.extended_capacity > 2*self.capacity:
            self.extended_capacity = 2*self.capacity


class Global_Graph:
    def __init__(self):
        """
        """
        self.vertices = {}
        self.num_vertices = 0
        #self.Deadlocks = []
        #self.LastUpdated = 0

        #self.CostTotalEnv = 0

    def __str__(self):
        """

        :return:
        """
        return 'Cost: ' + str(self.CostTotalEnv) + ' Deadlocks: ' + str(self.Deadlocks)


    def setCosts(self):
        """

        :return:
        """
        #cost = 0
        for vertex in self.vertices:
            if len(self.vertices[vertex].Trains):
                self.vertices[vertex].setCosts()
            #cost += self.vertices[vertex].CostTotal

        #for vertex in self.vertices:
        #    if len(self.vertices[vertex].Trains):
        #        self.vertices[vertex].setExtendedCapacity()

        #self.CostTotalEnv = cost


    def add_edge_vertex(self, type, cells):
        """

        :param node:
        :return:
        """
        if str(cells[0])[1:-1]+","+str(cells[-1])[1:-1] not in self.vertices\
                and str(cells[-1])[1:-1]+","+str(cells[0])[1:-1] not in self.vertices:

            new_edge_vertex = vertex(type, cells, str(cells[0])[1:-1]+","+str(cells[-1])[1:-1])
            self.vertices[str(cells[0])[1:-1]+","+str(cells[-1])[1:-1]] = new_edge_vertex
            self.num_vertices += 1

            return new_edge_vertex

        elif str(cells[0])[1:-1]+","+str(cells[-1])[1:-1] in self.vertices:

            return self.vertices[str(cells[0])[1:-1]+","+str(cells[-1])[1:-1]]

        elif str(cells[-1])[1:-1]+","+str(cells[0])[1:-1] in self.vertices:

            return self.vertices[str(cells[-1])[1:-1]+","+str(cells[0])[1:-1]]


    def add_signal_vertex(self, type, node):
        """

        :param node:
        :return:
        """
        if str(node)[1:-1] not in self.vertices:
            new_vertex = vertex(type, [node], str(node)[1:-1])
            self.vertices[str(node)[1:-1]] = new_vertex
            self.num_vertices = self.num_vertices + 1
            return new_vertex
        return self.vertices[str(node)[1:-1]]


if __name__ == "__main__":
    # create a graph of 4 nodes
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
            #print("found")
    #edge_temp = g.edge_dict['ab']
    #edge_temp.cost_triples.append([1,2,3])
    #g.add_vertex('b')
    #g.add_vertex('c')
    #g.add_vertex('d')
    #g.add_vertex('e')
    #print("done")