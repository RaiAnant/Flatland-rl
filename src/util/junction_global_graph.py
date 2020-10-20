
""" ###################### EDGE ####################"""
import numpy as np



class vertex:
    def __init__(self, type, node):
        """

        :param node:
        """
        self.Type = type
        self.Cells = node

        self.Trains = []
        self.TrainsTime = []
        self.TrainsDir = []  # 0 = A->B, 1 = B->A
        self.Links = []

        self.CollisionLockMatrix = []  # train 0 with train 1 and train 1 with train 0

        self.CostPerTrain = []

        self.CostTotal = 0


    def __str__(self):
        """

        :return:
        """
        return 'Type : ' + str(self.Type) + '; Cost: ' + str(self.CostTotal) + '; Trains: ' + str(self.Trains)


    def setCosts(self):
        """

        :return:
        """

        self.CostCollisionLockTotal = 0
        self.CostTransitionTimeTotal = 0
        self.CostTotal = 0
        self.CostPerTrain = []

        self.CollisionLockMatrix = np.zeros((len(self.Trains),len(self.Trains)),dtype=np.uint8)

        if self.Type == "junction1":
            for agent_pos_id, agent_id in enumerate(self.Trains):

                # find direction of first agent
                agent_dir = self.TrainsDir[agent_pos_id]

                # find the agents which are not in teh same direction (can be many if junction)
                opposing_agents = [num for num, item in enumerate(self.TrainsDir) if item != agent_dir]

                for opp_agent in opposing_agents:
                    if self.TrainsTime[agent_pos_id][0] == self.TrainsTime[opp_agent][0]:
                        self.CollisionLockMatrix[agent_pos_id][opp_agent] = 1

            for i, item in enumerate(self.TrainsTime):

                if item[0] != 0:
                    self.CostPerTrain.append(
                        np.count_nonzero(self.CollisionLockMatrix[i]) * 10000 + abs(item[1] +1 - item[0]))
                    self.CostCollisionLockTotal += np.count_nonzero(self.CollisionLockMatrix[i]) * 5000

                else:
                    self.CostPerTrain.append(abs(item[1] + 1 - item[0]))

                self.CostTransitionTimeTotal += abs(item[1] +1 - item[0])

            self.CostTotal = self.CostCollisionLockTotal + self.CostTransitionTimeTotal

        elif self.Type == "edge" or self.Type == "junction":
            for agent_pos_id, agent_id in enumerate(self.Trains):

                # find direction of first agent
                agent_dir = self.TrainsDir[agent_pos_id]

                # find the agents which are not in teh same direction (can be many if junction)
                opposing_agents = [num for num, item in enumerate(self.TrainsDir) if item != agent_dir]


                if len(opposing_agents):
                    for opp_agent in opposing_agents:

                        bitmap = np.zeros((np.max(self.TrainsTime) + 1))
                        bitmap[self.TrainsTime[opp_agent][0]:self.TrainsTime[opp_agent][1]+1] = 1

                        if np.any(bitmap[self.TrainsTime[agent_pos_id][0]:self.TrainsTime[agent_pos_id][1]+1] > 0):
                            self.CollisionLockMatrix[agent_pos_id][opp_agent] = 1


            for i, item in enumerate(self.TrainsTime):

                if item[0] != 0:
                    self.CostPerTrain.append(
                        np.count_nonzero(self.CollisionLockMatrix[i]) * 10000 + abs(item[1] - item[0]))
                    self.CostCollisionLockTotal += np.count_nonzero(self.CollisionLockMatrix[i]) * 5000

                else:
                    self.CostPerTrain.append(abs(item[1] - item[0]))

                self.CostTransitionTimeTotal += abs(item[1] - item[0])

            self.CostTotal = self.CostCollisionLockTotal + self.CostTransitionTimeTotal
        else:
            print("What kind of node is this ?")


class Global_Graph:
    def __init__(self):
        """
        """
        self.vertices = {}
        self.num_vertices = 0

        self.CostTotalEnv = 0

    def setCosts(self):
        """

        :return:
        """
        for vertex in self.vertices:
            self.vertices[vertex].setCosts()

        cost = 0
        for vertex in self.vertices:
            cost += self.vertices[vertex].CostTotal

        self.CostTotalEnv = cost


    def add_edge_vertex(self, type, cells):
        """

        :param node:
        :return:
        """
        if str(cells[0])[1:-1]+","+str(cells[-1])[1:-1] not in self.vertices\
                and str(cells[-1])[1:-1]+","+str(cells[0])[1:-1] not in self.vertices:

            new_edge_vertex = vertex(type, cells)
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
            new_vertex = vertex(type, [node])
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