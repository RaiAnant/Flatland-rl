
""" ###################### EDGE ####################"""
import numpy as np

class g_edge:
    def __init__(self, node1, node2, node1_id, node2_id, traj):
        """

        :param node1:
        :param node2:
        :param node1_id:
        :param node2_id:
        :param traj:
        """
        self.A = node1
        self.B = node2
        self.A_node = node1_id
        self.B_node = node2_id
        self.Cells = traj
        self.Trains = []
        self.TrainsDir = []  # 0 = A->B, 1 = B->A
        self.TrainsTime = []
        self.CollisionLockMatrix = []  # train 0 with train 1 and train 1 with train 0
        self.CostCollisionLockTotal = 0
        self.CostPerTrain = []
        self.CostTransitionTimeTotal = 0
        self.CostTotal = 0
        self.Delay = []

    def __str__(self):
        """

        :return:
        """
        return ' Cost: ' + str(self.CostTotal) + ' Trains: ' + str(self.Trains)# + ' Cells: ' + str(self.Cells)

    def setCosts(self):
        """

        :return:
        """

        self.CollisionLockMatrix = []  # train 0 with train 1 and train 1 with train 0
        self.CostCollisionLockTotal = 0
        self.CostTransitionTimeTotal = 0
        self.CostTotal = 0

        self.CollisionLockMatrix = np.zeros((len(self.Trains),len(self.Trains)),dtype=np.uint8)
        for t_num, t_id in enumerate(self.Trains):
            for c_t_num, c_t_id in enumerate(self.Trains):
                # if the train is not compared with itself and
                # they have opposing direction
                if t_id != c_t_id and self.TrainsDir[t_num] != self.TrainsDir[c_t_num]:
                    # check the amount of time overlap

                    # find the max time for first Train
                    if self.TrainsTime[t_num][1] > self.TrainsTime[c_t_num][1]:
                        tmp = np.zeros((self.TrainsTime[t_num][1]+1))
                    else:
                        tmp = np.zeros((self.TrainsTime[c_t_num][1] + 1))

                    for i in range(self.TrainsTime[t_num][0], self.TrainsTime[t_num][1]+1):
                        tmp[i] += 1

                    for i in range(self.TrainsTime[c_t_num][0], self.TrainsTime[c_t_num][1]+1):
                        tmp[i] += 1

                    if np.max(tmp) > 1 and self.TrainsTime[t_num][0] != 0:
                        self.CollisionLockMatrix[t_num][c_t_num] = 1


                    # find the max time for second Train
        # surely trains are on the same section
        # check if they are in opposite direction
        # if yes check if they have overlap of time

        for i, item in enumerate(self.TrainsTime):

            if item[0] != 0:
                self.CostPerTrain.append(np.count_nonzero(self.CollisionLockMatrix[i])*10000 + abs(item[1] - item[0]))
                self.CostCollisionLockTotal += np.count_nonzero(self.CollisionLockMatrix[i]) * 5000

            else:
                self.CostPerTrain.append(np.count_nonzero(self.CollisionLockMatrix[i])*10000 + abs(item[1] - item[0]))
                self.CostCollisionLockTotal += np.count_nonzero(self.CollisionLockMatrix[i]) * 5000

            self.CostTransitionTimeTotal += abs(item[1] - item[0])

        self.CostTotal = self.CostCollisionLockTotal + self.CostTransitionTimeTotal



class g_vertex:
    def __init__(self, node):
        """

        :param node:
        """
        self.id = node
        self.edges = []
        self.transitions = []

    def __str__(self):
        """

        :return:
        """
        return ' ID: ' + str(self.id)


class Global_Graph:
    def __init__(self):
        """
        """
        self.vert_dict = {}
        self.edge_ids = []

        self.num_vertices = 0
        self.num_edges = 0

        self.CostTotalEnv = 0

    def setCosts(self):
        """

        :return:
        """
        for edge in self.edge_ids:
            edge.setCosts()

        cost = 0
        for edge in self.edge_ids:
            cost += edge.CostTotal

        self.CostTotalEnv = cost


    def add_vertex(self, node):
        """

        :param node:
        :return:
        """
        if node not in self.vert_dict.keys():
            self.num_vertices = self.num_vertices + 1
            new_vertex = g_vertex(node)
            self.vert_dict[node] = new_vertex
            return new_vertex
        return self.vert_dict[node]



    def add_edge(self, frm, to, frm_id, to_id, traj):
        """

        :param frm:
        :param to:
        :param frm_id:
        :param to_id:
        :param traj:
        :return:
        """
        # an edge can be added as follows
        #   both the vertices must exist

        found = False
        for item in self.edge_ids:
            if (item.A == frm and item.B == to and item.Cells == traj) \
                    or (item.A == to and item.B == frm and item.Cells == traj[::-1]):
                found = True

        if not found:
            temp = g_edge(frm, to, frm_id, to_id, traj)
            self.edge_ids.append(temp)
            return temp

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