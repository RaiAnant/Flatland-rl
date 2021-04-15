"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).
+ `reset()` is called after each environment reset (i.e. at the beginning of a new episode), to allow for pre-computing relevant data.
+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.
"""

import collections
from typing import Optional, List, Dict, Tuple
import queue
import numpy as np
from collections import defaultdict
from itertools import groupby
import math
import copy

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail, position_to_coordinate
from flatland.utils.ordered_set import OrderedSet


from src.priority import assign_priority
from src.util.graph import Graph
from src.util.junction_global_graph import Global_Graph

class GraphObsForRailEnv(ObservationBuilder):
    """
    Build graph observations.
    """

    def __init__(self, predictor, bfs_depth):
        super(GraphObsForRailEnv, self).__init__()
        self.predictor = predictor
        self.bfs_depth = bfs_depth
        self.max_prediction_depth = 0
        self.prediction_dict = {}  # Dict handle : list of tuples representing prediction steps
        self.predicted_pos = {}  # Dict ts : int_pos_list
        self.predicted_pos_list = {} # Dict handle : int_pos_list
        self.predicted_pos_coord = {}  # Dict ts : coord_pos_list
        self.predicted_dir = {}  # Dict ts : dir (float)
        self.num_active_agents = 0
        self.cells_sequence = None
        self.forks_coords = None
        self.base_graph = None
        self.observations = None
        self.ts = -1
        self.agent_position_data = defaultdict(list)
        self.agent_initial_positions = defaultdict(list)
        self.agent_init_edge_list = []
        self.cur_pos_list = []
        self.signal_time = defaultdict()
        self.signal_deadlocks = defaultdict(list)


    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            # Use set_env available in PredictionBuilder (parent class)
            self.predictor.set_env(self.env)


    def reset(self):
        """
        Inherited method used for pre computations.
        :param:
        :return:
        """
        self.forks_coords = self._find_forks()
        self.ts = -1
        self.cur_pos_list = []
        self.agent_position_data = defaultdict(list)
        self.agent_initial_positions = defaultdict(list)
        self.base_graph = None
        self.observations = None


    def get_prev_vert(self, cur_vertex, cur_pos_on_traj, a):
        # go back in trajectory until the beginning
        # check if a previous vertex exists

        if len(cur_vertex.Cells) == 1:

            prev_pos_on_traj = cur_pos_on_traj - 1
            agent_prev_position = self.cells_sequence[a.handle][prev_pos_on_traj]
            prev_vertex = [vert[1] for vert in cur_vertex.Links
                           if (vert[1].Cells[0][0] == agent_prev_position[0]
                               and vert[1].Cells[0][1] == agent_prev_position[1])
                           or (vert[1].Cells[-1][0] == agent_prev_position[0]
                               and vert[1].Cells[-1][1] == agent_prev_position[1])]
            if len(prev_vertex):
                prev_vertex = prev_vertex[0]
            else:
                prev_vertex = None

        else:

            found = False
            while cur_pos_on_traj > 0:

                cur_pos_on_traj -= 1

                prev_item = self.cells_sequence[a.handle][cur_pos_on_traj]

                if prev_item not in cur_vertex.Cells:
                    found = True
                    break

            if found:
                agent_prev_position = self.cells_sequence[a.handle][cur_pos_on_traj]

                prev_vertex = [vert[1] for vert in cur_vertex.Links
                           if (vert[1].Cells[0][0] == agent_prev_position[0]
                               and vert[1].Cells[0][1] == agent_prev_position[1])
                           or (vert[1].Cells[-1][0] == agent_prev_position[0]
                               and vert[1].Cells[-1][1] == agent_prev_position[1])]

                if len(prev_vertex):
                    prev_vertex = prev_vertex[0]
                else:
                    prev_vertex = None
            else:
                prev_vertex = None


        return prev_vertex



    def set_agents_state(self):
        """
        :param observations:
        :return:
        """
        # find current positions of the agents
        # find next positions of teh agent
        # if the next positions are signals then update the graph
        # update if an agent want to enter a signal section

        if self.ts == 0:
            self.cur_pos_list = []
            for a in self.env.agents:
                self.cur_pos_list.append([self.cells_sequence[a.handle][0], self.cells_sequence[a.handle][1]])

        for a in self.env.agents:
            cur_pos = self.cur_pos_list[a.handle][0]
            next_pos = self.cur_pos_list[a.handle][1]

            if a.status == RailAgentStatus.DONE_REMOVED:
                self.cur_pos_list[a.handle] = [tuple((0,0)), tuple((0,0)), False, [], True, False, None]
                continue


            if cur_pos[0] == 0 and cur_pos[1] == 0:

                self.cur_pos_list[a.handle] = [cur_pos, next_pos, False, [], True, True, None]

            elif next_pos[0] == 0 and next_pos[1] == 0:

                self.cur_pos_list[a.handle] = [cur_pos, next_pos, False, [], True, True, None]

            else:

                cur_pos_on_traj_list = [num for num, cell in enumerate(self.cells_sequence[a.handle])
                                        if cell[0] == cur_pos[0] and cell[1] == cur_pos[1]]

                for cur_pos_on_traj_cand in cur_pos_on_traj_list:
                    if self.cells_sequence[a.handle][cur_pos_on_traj_cand + 1] == self.cur_pos_list[a.handle][1]:
                        cur_pos_on_traj = cur_pos_on_traj_cand
                        break

                if a.position == None or a.position == cur_pos:
                    cur_pos = self.cells_sequence[a.handle][cur_pos_on_traj]
                    next_pos = self.cells_sequence[a.handle][cur_pos_on_traj + 1]
                else:
                    cur_pos = self.cells_sequence[a.handle][cur_pos_on_traj + 1]
                    next_pos = self.cells_sequence[a.handle][cur_pos_on_traj + 2]
                    cur_pos_on_traj = cur_pos_on_traj +1

                for signals in self.observations.vertices:

                    if cur_pos in self.observations.vertices[signals].Cells:
                        cur_vertex = self.observations.vertices[signals]
                        if cur_vertex != None:
                            cur_vertex.occupancy += 1
                        else:
                            print("Here")

                    if next_pos in self.observations.vertices[signals].Cells:
                        next_vertex = self.observations.vertices[signals]

                if cur_pos_on_traj == 0:
                    prev_vertex = None
                else:
                    prev_vertex = self.get_prev_vert(cur_vertex, cur_pos_on_traj, a)


                if next_vertex == None:
                    self.cur_pos_list[a.handle] = [cur_pos, next_pos, False, [], True, True, cur_vertex]
                    continue

                if cur_vertex.id == next_vertex.id:

                    vert_list = []
                    decision = [cur_pos, next_pos, False, vert_list, True, cur_vertex.is_safe, cur_vertex]

                    if next_vertex.TrainsTraversal[a.handle][0][1] != None:

                        while True:

                            next_vertex = [vert[1] for vert in cur_vertex.TrainsTraversal[a.handle]
                                           if vert[0] == prev_vertex][0]
                            if next_vertex == None:
                                break
                            elif next_vertex.is_safe:

                                vert_list.append(next_vertex)
                                break
                            else:
                                vert_list.append(next_vertex)

                            prev_vertex = cur_vertex
                            cur_vertex = next_vertex

                    else:
                        vert_list.append(next_vertex)

                    decision[3] = vert_list
                    self.cur_pos_list[a.handle] = decision

                else:

                    vert_list = []
                    decision = []
                    if cur_vertex.is_safe and next_vertex.is_safe:
                        decision = [cur_pos, next_pos, False, vert_list, True, cur_vertex.is_safe, cur_vertex]
                    elif not cur_vertex.is_safe and not next_vertex.is_safe:
                        decision = [cur_pos, next_pos, False, vert_list, True, cur_vertex.is_safe, cur_vertex]
                    elif not cur_vertex.is_safe and next_vertex.is_safe:
                        decision = [cur_pos, next_pos, False, vert_list, True, cur_vertex.is_safe, cur_vertex]
                    elif cur_vertex.is_safe and not next_vertex.is_safe:
                        decision = [cur_pos, next_pos, True, vert_list, False, cur_vertex.is_safe, cur_vertex]

                    while True:

                        if next_vertex == None:
                            break
                        elif next_vertex.is_safe:
                            vert_list.append(next_vertex)
                            break
                        else:
                            vert_list.append(next_vertex)
                            try:
                                next_vertex = [vert[1] for vert in cur_vertex.TrainsTraversal[a.handle]
                                           if vert[0] == prev_vertex][0]
                            except:
                                raise Exception("Here")

                        prev_vertex = cur_vertex
                        cur_vertex = next_vertex

                    decision[3] = vert_list
                    self.cur_pos_list[a.handle] = decision





    def _find_forks(self):
        """
        A fork (in the map) is either a switch or a diamond crossing.
        :return:
        """

        forks = set() # Set of nodes as tuples/coordinates
        # Identify cells hat are nodes (have switches)
        for i in range(self.env.height):
            for j in range(self.env.width):

                is_switch = False
                is_crossing = False

                # Check if diamond crossing
                transitions_bit = bin(self.env.rail.get_full_transitions(i, j))
                if int(transitions_bit, 2) == int('1000010000100001', 2):
                    is_crossing = True

                else:
                    # Check if switch
                    for direction in (0, 1, 2, 3):  # 0:N, 1:E, 2:S, 3:W
                        possible_transitions = self.env.rail.get_transitions(i, j, direction)
                        num_transitions = np.count_nonzero(possible_transitions)
                        if num_transitions > 1:
                            is_switch = True

                if is_switch or is_crossing:
                    forks.add((i, j))

        return forks


    def get_many(self, handles) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """
        return self.get()


    def get(self) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """
        self.ts += 1

        if self.base_graph is None:
            self.build_global_graph()

        self.prediction_dict = self.predictor.get()
        if self.cells_sequence is None:
            local = self.predictor.compute_cells_sequence(self.prediction_dict)
            self.cells_sequence = local
            self.max_prediction_depth = self.predictor.max_depth

        if self.ts > 0:
            for a in self.env.agents:
                current_position = a.position if a.position is not None \
                                 else a.initial_position if a.status is not RailAgentStatus.DONE_REMOVED \
                                 else tuple((0,0))
                self.agent_position_data[a.handle].append(current_position)

        #observations = copy.deepcopy(self.base_graph)
        #if self.ts == 0:
        self.populate_graph()
        self.observations.setCosts()
        self.set_agents_state()

        return self.observations




    def populate_graph(self):
        """
        Inherited method used for pre computations.
        :return:
        """
        self.observations = copy.deepcopy(self.base_graph)
        agent_initial_positions = defaultdict(list)
        for a in self.env.agents:
            new_edge = [self.observations.vertices[vertex] for vertex in self.observations.vertices \
                                                if a.initial_position in self.observations.vertices[vertex].Cells][0]
            new_edge.is_starting_edge = True
            agent_initial_positions[a.handle] = [a.initial_position, new_edge]


        for a in range(self.env.number_of_agents):

            # this tells up how far we are in the trajectory
            # how many time steps we spent on each cell
            # it considers every train to be on the initial position until they move
            # weather READY_TO_MOVE or ACTIVE

            # Build one vector of time spent on already travelled trajectory
            # and the planned one

            # initial state
            agent_current_vertex = agent_initial_positions[a][1]
            agent_prev_vertex = None
            agent_trajectory = self.cells_sequence[a]


            if len(agent_current_vertex.Cells) > 1:
                index = [num for num,cell in enumerate(agent_current_vertex.Cells)
                         if cell[0] == agent_trajectory[0][0] and cell[1] == agent_trajectory[0][1]]
                if not len(index):
                    raise Exception("agent start position not found while populating the graph")

                trajectory_to_begin = agent_current_vertex.Cells[0:index[0]+1][::-1]
                trajectory_to_end = agent_current_vertex.Cells[index[0]:]

                if trajectory_to_end == agent_trajectory[:len(trajectory_to_end)]:

                    res = [[num, vert[1]] for num,vert in enumerate(agent_current_vertex.Links)
                                         if vert[0][0] == trajectory_to_end[-1][0]
                                         and vert[0][1] == trajectory_to_end[-1][1]][0]
                    if not len(res):
                        raise Exception("vertex not found in links")

                    agent_dir, agent_next_vertex = res[0], res[1]
                    agent_pos_on_traj = len(trajectory_to_end)

                if trajectory_to_begin == agent_trajectory[:len(trajectory_to_begin)]:
                    res = [[num, vert[1]] for num,vert in enumerate(agent_current_vertex.Links)
                                         if vert[0][0] == trajectory_to_begin[-1][0]
                                         and vert[0][1] == trajectory_to_begin[-1][1]][0]
                    if not len(res):
                        raise Exception("vertex not found in links")

                    agent_dir, agent_next_vertex = res[0], res[1]
                    agent_pos_on_traj = len(trajectory_to_begin)

            else:
                agent_next_position = agent_trajectory[1]

                res = [[num, vert[1]] for num, vert in enumerate(agent_current_vertex.Links)
                                     if (vert[1].Cells[0][0] == agent_next_position[0]
                                         and vert[1].Cells[0][1] == agent_next_position[1])
                                     or (vert[1].Cells[-1][0] == agent_next_position[0]
                                         and vert[1].Cells[-1][1] == agent_next_position[1])][0]
                agent_dir, agent_next_vertex = res[0], res[1]
                agent_pos_on_traj = 1


            agent_current_vertex.Trains.append(a)
            agent_current_vertex.TrainsTraversal[a] = [[None, agent_next_vertex]]
            agent_current_vertex.TrainsDir.append(agent_dir)

            agent_prev_vertex = agent_current_vertex
            agent_current_vertex = agent_next_vertex

            # start with the beginning
            # find next exit on trajectory
            while(True):

                # check what sort of vertex the agent is at right now.
                if agent_current_vertex.Type == "junction" \
                        or (agent_current_vertex.Type == "edge" and \
                        len(agent_current_vertex.Cells) == 1):

                    if agent_pos_on_traj+1 >= len(agent_trajectory)-1:
                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsTraversal[a] = [[agent_prev_vertex, None]]

                        agent_dir = [num for num,item in enumerate(agent_current_vertex.Links)
                                     if item[1] != agent_prev_vertex][0]
                        agent_current_vertex.TrainsDir.append(agent_dir)
                        break

                    agent_next_pos_on_traj = agent_pos_on_traj+1
                    agent_next_position = agent_trajectory[agent_next_pos_on_traj]

                    agent_next_vertex = [[num,vert[1]] for num,vert in enumerate(agent_current_vertex.Links)
                                         if (vert[1].Cells[0][0] == agent_next_position[0]
                                         and vert[1].Cells[0][1] == agent_next_position[1])
                                         or (vert[1].Cells[-1][0] == agent_next_position[0]
                                         and vert[1].Cells[-1][1] == agent_next_position[1])]

                    agent_next_vertex = agent_next_vertex[0]

                    agent_current_vertex.Trains.append(a)

                    if a in agent_current_vertex.TrainsTraversal.keys():
                        temp = agent_current_vertex.TrainsTraversal[a]
                        temp.append([agent_prev_vertex, agent_next_vertex[1]])
                        agent_current_vertex.TrainsTraversal[a] = temp
                    else:
                        agent_current_vertex.TrainsTraversal[a] = [[agent_prev_vertex, agent_next_vertex[1]]]

                    agent_current_vertex.TrainsDir.append(agent_next_vertex[0])
                    agent_next_vertex = agent_next_vertex[1]

                elif agent_current_vertex.Type == "edge":

                    if agent_pos_on_traj+len(agent_current_vertex.Cells) >= len(agent_trajectory)-1:
                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsTraversal[a] = [[agent_prev_vertex, None]]

                        agent_dir = [num for num,item in enumerate(agent_current_vertex.Links)
                                     if item[1] != agent_prev_vertex][0]
                        agent_current_vertex.TrainsDir.append(agent_dir)
                        break

                    agent_next_pos_on_traj = agent_pos_on_traj+len(agent_current_vertex.Cells)
                    agent_next_position = agent_trajectory[agent_next_pos_on_traj]

                    agent_next_vertex = [[num,vert[1]] for num,vert in enumerate(agent_current_vertex.Links)
                                         if (vert[1].Cells[0][0] == agent_next_position[0]
                                         and vert[1].Cells[0][1] == agent_next_position[1])
                                         or (vert[1].Cells[-1][0] == agent_next_position[0]
                                         and vert[1].Cells[-1][1] == agent_next_position[1])]

                    agent_next_vertex = agent_next_vertex[0]

                    agent_current_vertex.Trains.append(a)

                    if a in agent_current_vertex.TrainsTraversal.keys():
                        temp = agent_current_vertex.TrainsTraversal[a]
                        temp.append([agent_prev_vertex, agent_next_vertex[1]])
                        agent_current_vertex.TrainsTraversal[a] = temp
                    else:
                        agent_current_vertex.TrainsTraversal[a] = [[agent_prev_vertex, agent_next_vertex[1]]]

                    agent_current_vertex.TrainsDir.append(agent_next_vertex[0])
                    agent_next_vertex = agent_next_vertex[1]

                agent_prev_vertex = agent_current_vertex
                agent_current_vertex = agent_next_vertex
                agent_pos_on_traj = agent_next_pos_on_traj


    # For Global Graph
    def build_global_graph(self):
        """
        :return:
        """

        self.base_graph = Global_Graph()

        for junctions in self.forks_coords:
            vertex = self.base_graph.add_signal_vertex("junction", junctions)
            #vertex.DeadLockMatrix = np.zeros((self.env.number_of_agents, self.env.number_of_agents), dtype=np.uint8)

            path_list = self._step(junctions)
            # there is following data in the node list
            #   1) the immediate neighbours of the fork
            #       extract possible transitions
            #   2) path to the next neighbour
            #       add it as the edge node
            #   3) the last cell is the next fork
            #       it will be added later

            for path in path_list:

                if len(path[2]) > 2:
                    edge_vertex_cells = path[2][1:-1]
                    edge_vertex = self.base_graph.add_edge_vertex("edge", edge_vertex_cells)
                    #edge_vertex.DeadLockMatrix = np.zeros((self.env.number_of_agents, self.env.number_of_agents),
                    #                                 dtype=np.uint8)

                    vertex.Links.append([junctions, edge_vertex])
                    edge_vertex.Links.append([edge_vertex_cells[0], vertex])

                else:
                    if str(path[0])[1:-1] in self.base_graph.vertices:

                        vertex.Links.append([junctions, self.base_graph.vertices[str(path[0])[1:-1]]])
                        self.base_graph.vertices[str(path[0])[1:-1]].Links.append([path[0],vertex])


    def _step_extend(self, current, direction):
        """
        :return:
        """

        traj = []

        position = get_new_position(current, direction)
        if not (position[0] > self.env.height-1 or position[1] > self.env.width-1 or
                position[0] < 0 or position[1] < 0):
            if self.env.rail.grid[position[0]][position[1]] != 0:
                traj.append(position)
        else:
            return None, None

        while True:

            full_cell_transition = self.env.rail.get_full_transitions(*position)
            cell_transitions_bitmap = bin(full_cell_transition)
            total_transitions = cell_transitions_bitmap.count("1")

            if total_transitions == 1:
                print("Dead End")
                return [position, total_transitions], traj
            elif total_transitions == 2:
                cell_transitions_bitmap = np.asarray([int(item) for item in cell_transitions_bitmap[2:]])
                pos = np.where(cell_transitions_bitmap == 1)
                pos = [item + (16 - len(cell_transitions_bitmap)) for item in pos[0]]

                if pos[0] in [0, 5, 10, 15] and pos[1] in [0, 5, 10, 15]:  # simple straight
                    position = get_new_position(position, direction)
                    if not (position[0] > self.env.height-1 or position[1] > self.env.width-1 or
                            position[0] < 0 or position[1] < 0):
                        if self.env.rail.grid[position[0]][position[1]] != 0:
                            traj.append(position)

                else:
                    if pos[0] in [1, 14] or pos[1] in [1, 14]:  # simple right
                        if direction == 0:
                            direction = (direction + 1) % 4
                        elif direction == 3:
                            direction = (direction - 1) % 4
                    elif pos[0] in [3, 6] or pos[1] in [3, 6]:  # simple right
                        if direction == 0:
                            direction = (direction - 1) % 4
                        elif direction == 1:
                            direction = (direction + 1) % 4
                    elif pos[0] in [11, 4] or pos[1] in [11, 4]:  # simple right
                        if direction == 2:
                            direction = (direction + 1) % 4
                        elif direction == 1:
                            direction = (direction - 1) % 4
                    elif pos[0] in [12, 9] or pos[1] in [12, 9]:  # simple right
                        if direction == 3:
                            direction = (direction + 1) % 4
                        elif direction == 2:
                            direction = (direction - 1) % 4
                    else:
                        print("Failed")

                    temp, temp1 = self._step_extend(position, direction)

                    return temp, traj+temp1
            else:
                return [position, total_transitions], traj


    def _step(self, current):
        """
        :return:
        """

        init_position = current
        node_list = []

        for direction in [0,1,2,3]:

            traj = []
            traj.append(init_position)

            position = get_new_position(init_position, direction)
            if not (position[0] > self.env.height-1 or position[1] > self.env.width-1 or
                    position[0] < 0 or position[1] < 0):
                if self.env.rail.grid[position[0]][position[1]] != 0:
                    traj.append(position)
            else:
                continue

            while True:

                # if number of total transition is 1
                # return as dead end
                # if number of total transition is > 2
                # return as node
                # else transition to new

                full_cell_transition = self.env.rail.get_full_transitions(*position)
                cell_transitions_bitmap = bin(full_cell_transition)
                total_transitions = cell_transitions_bitmap.count("1")

                if total_transitions == 1:
                    print("Dead End")
                    node_list.append([position, total_transitions, traj])
                    break
                elif total_transitions == 2:
                    # special case
                    # if straight : means dead end
                    # if simple left : explore recursively by changing direction to left
                    # if simple right : explore recursively by changing direction to right

                    cell_transitions_bitmap = np.asarray([int(item) for item in cell_transitions_bitmap[2:]])
                    pos = np.where(cell_transitions_bitmap == 1)
                    pos = [item + (16 - len(cell_transitions_bitmap)) for item in pos[0]]

                    if pos[0] in [0, 5, 10, 15] and pos[1] in [0, 5, 10, 15]:  # simple straight
                        position = get_new_position(position, direction)
                        if not (position[0] > self.env.height-1 or position[1] > self.env.width-1 or
                                position[0] < 0 or position[1] < 0):
                            if self.env.rail.grid[position[0]][position[1]] != 0:
                                traj.append(position)

                    else:
                        if pos[0] in [1,14] or pos[1] in [1, 14]:  # simple right
                            if direction == 0:
                                direction = (direction + 1) % 4
                            elif direction == 3:
                                direction = (direction - 1) % 4
                        elif pos[0] in [3, 6] or pos[1] in [3, 6]:  # simple right
                            if direction == 0:
                                direction = (direction - 1) % 4
                            elif direction == 1:
                                direction = (direction + 1) % 4
                        elif pos[0] in [11, 4] or pos[1] in [11, 4]:  # simple right
                            if direction == 2:
                                direction = (direction + 1) % 4
                            elif direction == 1:
                                direction = (direction - 1) % 4
                        elif pos[0] in [12, 9] or pos[1] in [12, 9]:  # simple right
                            if direction == 3:
                                direction = (direction + 1) % 4
                            elif direction == 2:
                                direction = (direction - 1) % 4
                        else:
                            print("Failed")

                        temp, temp1 = self._step_extend(position, direction)

                        if not (temp == None and temp1 == None):
                            traj = traj + temp1
                            node_list.append([temp[0], temp[1], traj])

                        break
                else:
                    node_list.append([position, total_transitions, traj])
                    break

        return node_list

