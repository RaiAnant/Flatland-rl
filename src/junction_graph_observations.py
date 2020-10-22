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


    def is_update_required(self,observations):
        """

        :param observations:
        :return:
        """
        # find current positions of the agents
        # find next positions of teh agent
        # if the next positions are signals then update the graph
        # update if an agent want to enter a signal section

        self.cur_pos_list = []

        Status = False
        for a in self.env.agents:
            localStatus = False
            cur_pos = a.position if a.position is not None else a.initial_position \
                           if a.status != RailAgentStatus.DONE_REMOVED \
                           else a.target
            try:
                cur_pos_on_traj = [num for num,cell in enumerate(self.cells_sequence[a.handle])
                               if np.all(cell == cur_pos)][0]
            except:
                print(cur_pos, self.cells_sequence[a.handle])
            next_pos_on_traj = cur_pos_on_traj + 1
            next_pos = self.cells_sequence[a.handle][next_pos_on_traj]

            for signals in observations.vertices:
                if next_pos == observations.vertices[signals].Cells[0] \
                        and observations.vertices[signals].Type == "junction":
                    Status = True
                    localStatus = True

            self.cur_pos_list.append([cur_pos, next_pos, localStatus])



        if self.ts == 0:
            return True

        return Status


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
        local = self.predictor.compute_cells_sequence(self.prediction_dict)
        if self.cells_sequence is None:
            self.cells_sequence = local
            self.max_prediction_depth = self.predictor.max_depth

        if self.ts > 0:
            for a in self.env.agents:
                current_position = a.position if a.position is not None \
                                 else a.initial_position if a.status is not RailAgentStatus.DONE_REMOVED \
                                 else tuple((0,0))
                self.agent_position_data[a.handle].append(current_position)

        """
        if self.ts == 0:
            self.observations = copy.deepcopy(self.base_graph)
            observations = self.observations
            self.observations = self.populate_graph(observations)
            self.observations.setCosts()


        observations = self.observations
        status = self.is_update_required(observations)

        if status:
            self.observations = self.update_for_delay(observations)
            self.observations.setCosts()
        """

        observations = copy.deepcopy(self.base_graph)
        status = self.is_update_required(observations)

        if status:
            self.observations = self.populate_graph(observations)
            self.observations.setCosts()


        return self.observations


    def populate_graph(self, observations):
        """
        Inherited method used for pre computations.

        :return:
        """

        agent_initial_positions = defaultdict(list)
        for a in self.env.agents:
            new_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                                                if a.initial_position in observations.vertices[vertex].Cells][0]
            agent_initial_positions[a.handle] = [a.initial_position, new_edge]


        for a in range(self.env.number_of_agents):

            # this tells up how far we are in the trajectory
            # how many time steps we spent on each cell
            # it considers every train to be on the initial position until they move
            # weather READY_TO_MOVE or ACTIVE

            # Build one vector of time spent on already travelled trajectory
            # and the planned one
            agent_time_stepwise = [int(1/self.env.agents[a].speed_data['speed'])]*len(self.cells_sequence[a])
            time_data = [len(list(i_list)) for i, i_list in groupby(self.agent_position_data[a])]
            agent_time_stepwise = time_data + agent_time_stepwise[len(time_data):]

            # initial state
            start_timestamp = 0
            agent_current_vertex = agent_initial_positions[a][1]
            agent_prev_vertex = None
            agent_trajectory = self.cells_sequence[a]
            agent_pos_on_traj = 0
            end_timestamp = 0

            # start with the beginning
            # find next exit on trajectory
            while(True):

                # check what sort of vertex the agent is at right now.
                if agent_current_vertex.Type == "junction":

                    agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                    agent_next_pos_on_traj = agent_pos_on_traj+1

                    agent_next_vertex, agent_next_dir = \
                                [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                        if agent_next_position in item[1].Cells][0]


                    agent_current_vertex.Trains.append(a)
                    agent_current_vertex.TrainsDir.append(agent_next_dir)

                    end_timestamp = start_timestamp + \
                                    np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                    agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                    #end_timestamp += 1

                    #if agent_prev_vertex != None:
                    #    if agent_next_vertex.Type != "junction":
                    #        end_timestamp += 1

                elif agent_current_vertex.Type == "edge" and \
                        len(agent_current_vertex.Cells) == 1:

                    agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                    agent_next_pos_on_traj = agent_pos_on_traj+1

                    if agent_current_vertex.Cells[0] != agent_trajectory[-2]:
                        agent_next_vertex, agent_next_dir = \
                                    [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                            if agent_next_position in item[1].Cells][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)
                        #end_timestamp = start_timestamp + \
                        #                np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, start_timestamp])
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                        # try both the ends of the cells sequence
                    else:
                        break

                    #if agent_prev_vertex != None:
                    #    if agent_next_vertex.Type != "junction":
                    #        end_timestamp += 1


                elif agent_current_vertex.Type == "edge":

                    if agent_current_vertex.Cells[0] in agent_trajectory[agent_pos_on_traj+1:]:

                        agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                             if item[0] == agent_current_vertex.Cells[0]][0]

                        agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj =  agent_pos_on_traj + \
                                                  [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                    if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        #end_timestamp = start_timestamp + np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                    elif agent_current_vertex.Cells[-1] in agent_trajectory[agent_pos_on_traj+1:]:

                        agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                             if item[0] == agent_current_vertex.Cells[-1]][0]

                        agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj =  agent_pos_on_traj + \
                                                  [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                    if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        #end_timestamp = start_timestamp + np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])
                        #end_timestamp = start_timestamp
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                    else:
                        break

                start_timestamp = end_timestamp
                agent_prev_vertex = agent_current_vertex
                agent_current_vertex = agent_next_vertex
                agent_pos_on_traj = agent_next_pos_on_traj

        return observations





    def populate_graph(self, observations):
        """
        Inherited method used for pre computations.

        :return:
        """

        agent_initial_positions = defaultdict(list)
        for a in self.env.agents:
            new_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                                                if a.initial_position in observations.vertices[vertex].Cells][0]
            agent_initial_positions[a.handle] = [a.initial_position, new_edge]


        for a in range(self.env.number_of_agents):

            # this tells up how far we are in the trajectory
            # how many time steps we spent on each cell
            # it considers every train to be on the initial position until they move
            # weather READY_TO_MOVE or ACTIVE

            # Build one vector of time spent on already travelled trajectory
            # and the planned one
            agent_time_stepwise = [int(1/self.env.agents[a].speed_data['speed'])]*len(self.cells_sequence[a])
            time_data = [len(list(i_list)) for i, i_list in groupby(self.agent_position_data[a])]
            agent_time_stepwise = time_data + agent_time_stepwise[len(time_data):]

            # initial state
            start_timestamp = 0
            agent_current_vertex = agent_initial_positions[a][1]
            agent_prev_vertex = None
            agent_trajectory = self.cells_sequence[a]
            agent_pos_on_traj = 0
            end_timestamp = 0

            # start with the beginning
            # find next exit on trajectory
            while(True):

                # check what sort of vertex the agent is at right now.
                if agent_current_vertex.Type == "junction":

                    agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                    agent_next_pos_on_traj = agent_pos_on_traj+1

                    agent_next_vertex, agent_next_dir = \
                                [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                        if agent_next_position in item[1].Cells][0]


                    agent_current_vertex.Trains.append(a)
                    agent_current_vertex.TrainsDir.append(agent_next_dir)

                    end_timestamp = start_timestamp + \
                                    np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                    agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                    #end_timestamp += 1

                    #if agent_prev_vertex != None:
                    #    if agent_next_vertex.Type != "junction":
                    #        end_timestamp += 1

                elif agent_current_vertex.Type == "edge" and \
                        len(agent_current_vertex.Cells) == 1:

                    agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                    agent_next_pos_on_traj = agent_pos_on_traj+1

                    if agent_current_vertex.Cells[0] != agent_trajectory[-2]:
                        agent_next_vertex, agent_next_dir = \
                                    [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                            if agent_next_position in item[1].Cells][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)
                        #end_timestamp = start_timestamp + \
                        #                np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, start_timestamp])
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                        # try both the ends of the cells sequence
                    else:
                        break

                    #if agent_prev_vertex != None:
                    #    if agent_next_vertex.Type != "junction":
                    #        end_timestamp += 1


                elif agent_current_vertex.Type == "edge":

                    if agent_current_vertex.Cells[0] in agent_trajectory[agent_pos_on_traj+1:]:

                        agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                             if item[0] == agent_current_vertex.Cells[0]][0]

                        agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj =  agent_pos_on_traj + \
                                                  [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                    if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        #end_timestamp = start_timestamp + np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                    elif agent_current_vertex.Cells[-1] in agent_trajectory[agent_pos_on_traj+1:]:

                        agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                             if item[0] == agent_current_vertex.Cells[-1]][0]

                        agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj =  agent_pos_on_traj + \
                                                  [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                    if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        #end_timestamp = start_timestamp + np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        #agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])
                        #end_timestamp = start_timestamp
                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                        #end_timestamp += 1

                    else:
                        break

                start_timestamp = end_timestamp
                agent_prev_vertex = agent_current_vertex
                agent_current_vertex = agent_next_vertex
                agent_pos_on_traj = agent_next_pos_on_traj

        return observations


    # For Global Graph
    def build_global_graph(self):
        """

        :return:
        """

        self.base_graph = Global_Graph()

        for junctions in self.forks_coords:
            vertex = self.base_graph.add_signal_vertex("junction", junctions)

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
        traj.append(position)

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
            traj.append(position)

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
                    #position = get_new_position(position, direction)

                    cell_transitions_bitmap = np.asarray([int(item) for item in cell_transitions_bitmap[2:]])
                    pos = np.where(cell_transitions_bitmap == 1)
                    pos = [item + (16 - len(cell_transitions_bitmap)) for item in pos[0]]

                    if pos[0] in [0, 5, 10, 15] and pos[1] in [0, 5, 10, 15]:  # simple straight
                        position = get_new_position(position, direction)
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
                        traj = traj + temp1
                        node_list.append([temp[0], temp[1], traj])
                        break
                else:
                    node_list.append([position, total_transitions, traj])
                    break

        return node_list



    def update_for_delay_edge_only(self, observations, a, vert_type):
        """
        Inherited method used for pre computations.

        :return:
        """
        agent_trajectory = self.cells_sequence[a]

        agent = self.env.agents[a]
        is_last_edge = False

        current_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                            if agent.initial_position in observations.vertices[vertex].Cells][0]

        # collect all junctions until next edge
        #next_edge = None
        next_vertex = current_edge

        target_pos = agent_trajectory[-2]

        while not is_last_edge:


            #if target_pos in current_edge.Cells:
            #    is_last_edge = True
            #    continue

            junc_list = []

            while not is_last_edge:

                agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]

                junc_list.append([next_vertex.Type, next_vertex.TrainsTime[agent_index]])

                next_vertex = next_vertex.Links[next_vertex.TrainsDir[agent_index]][1]

                if next_vertex.Type != "junction":
                    try:
                        agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                    except:
                        is_last_edge = True
                        continue

                    junc_list.append([next_vertex.Type, next_vertex.TrainsTime[agent_index]])

                    #next_edge = next_vertex
                    break

            if is_last_edge:
                continue

            #print(junc_list)
            #print("----------")
            #
            next_list = []

            if vert_type == "junction":
                current_time_val = junc_list[::-1][1][1][1]

                next_list.append([[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]], junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:

                    next_list.append([[current_time_val-1,current_time_val], item[0]])
                    current_time_val -= 1

                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])
            elif vert_type == "edge":
                current_time_val = junc_list[::-1][0][1][0]

                next_list.append([[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]], junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:

                    next_list.append([[current_time_val-1,current_time_val], item[0]])
                    current_time_val -= 1

                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])

            #print(next_list)
            next_list = next_list[::-1]
            #print(next_list)

            #print("Here")

            next_vertex = current_edge

            filler_index = 0

            while True:

                agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]

                next_vertex.TrainsTime[agent_index] = next_list[filler_index][0]
                filler_index += 1

                next_vertex = next_vertex.Links[next_vertex.TrainsDir[agent_index]][1]

                if next_vertex.Type != "junction":
                    agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                    next_vertex.TrainsTime[agent_index] = next_list[filler_index][0]
                    current_edge = next_vertex
                    break

        return observations




    def update_for_delay(self, observations, a, vert_type):
        """
        Inherited method used for pre computations.

        :return:
        """
        agent_trajectory = self.cells_sequence[a]

        agent = self.env.agents[a]
        is_last_edge = False

        current_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                            if agent.initial_position in observations.vertices[vertex].Cells][0]

        # collect all junctions until next edge
        #next_edge = None
        next_vertex = current_edge

        target_pos = agent_trajectory[-2]

        initial_lapse = 0

        while not is_last_edge:


            #if target_pos in current_edge.Cells:
            #    is_last_edge = True
            #    continue

            junc_list = []

            while not is_last_edge:

                agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]

                junc_list.append([next_vertex.Type, next_vertex.TrainsTime[agent_index]])

                next_vertex = next_vertex.Links[next_vertex.TrainsDir[agent_index]][1]

                if next_vertex.Type != "junction":
                    try:
                        agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                    except:
                        is_last_edge = True
                        continue

                    junc_list.append([next_vertex.Type, next_vertex.TrainsTime[agent_index]])

                    #next_edge = next_vertex
                    break

            if is_last_edge:
                continue

            #print(junc_list)
            #print("----------")
            #
            next_list = []

            if initial_lapse != 0:
                pass
                #    print("here")
            elif vert_type == "junction":
                current_time_val = junc_list[::-1][1][1][1]

                #if current_time_val > current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]:
                #    print("here")
                next_list.append([[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]], junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:

                    next_list.append([[current_time_val-1,current_time_val], item[0]])
                    current_time_val -= 1

                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])

                if junc_list[::-1][-1][1][0] > current_time_val:
                    pass
                    #print("Here")
            elif vert_type == "edge":
                current_time_val = junc_list[::-1][0][1][0]

                #if current_time_val > current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]:
                #    print("here")

                next_list.append([[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]], junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:

                    next_list.append([[current_time_val-1,current_time_val], item[0]])
                    current_time_val -= 1



                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])

                #if junc_list[::-1][-1][1][0] > current_time_val:
                #    print("here")

            #print(next_list)

            next_list = next_list[::-1]


            #next_list_temp = [item[0] for item in next_list]

            if initial_lapse == 0:
                junc_list_temp = [item[1][1] - item[1][0] for item in junc_list]
                next_list_temp = [item[0][1] - item[0][0] for item in next_list]
                initial_lapse = abs(((np.sum(junc_list_temp)) - ((np.sum(next_list_temp)))))
            #if initial_lapse > 0:
            #    print("here")
            #print(initial_lapse)

            #next_list_temp = np.asarray(next_list_temp)
            #next_list_temp = next_list_temp + initial_lapse

            #print(vert_type)
            #print(junc_list)
            #print(next_list)

            #print(next_list)

            #print("Here")

            next_vertex = current_edge

            filler_index = 0

            while True:
                agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]

                if len(next_list):

                    next_vertex.TrainsTime[agent_index][0] = next_list[filler_index][0][0]
                    next_vertex.TrainsTime[agent_index][1] = next_list[filler_index][0][1]

                    filler_index += 1

                    #if next_vertex.Type != "junction":
                    #if len(next_list):
                    #    agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                    #    #next_vertex.TrainsTime[agent_index] = next_list[filler_index][0]
                    #    next_vertex.TrainsTime[agent_index][0] = next_list[filler_index][0][0]
                    #    next_vertex.TrainsTime[agent_index][1] = next_list[filler_index][0][1]

                else:

                    if filler_index > 0:
                        agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                        # next_vertex.TrainsTime[agent_index] = next_list[filler_index][0]
                        next_vertex.TrainsTime[agent_index][0] += initial_lapse
                        next_vertex.TrainsTime[agent_index][1] += initial_lapse
                        #print(next_vertex.TrainsTime[agent_index])


                    filler_index += 1


                if len(junc_list) == filler_index:
                    current_edge = next_vertex
                    break
                else:
                    next_vertex = next_vertex.Links[next_vertex.TrainsDir[agent_index]][1]

        return observations






    def update_for_delay_1(self, observations, a):
        """
        Inherited method used for pre computations.

        :return:
        """

        agent_initial_positions = defaultdict(list)
        agent = self.env.agents[a]
        #for a in self.env.agents:
        new_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                                            if agent.initial_position in observations.vertices[vertex].Cells][0]
        agent_initial_positions[agent.handle] = [agent.initial_position, new_edge]


        #for a in range(self.env.number_of_agents):

        # this tells up how far we are in the trajectory
        # how many time steps we spent on each cell
        # it considers every train to be on the initial position until they move
        # weather READY_TO_MOVE or ACTIVE

        # Build one vector of time spent on already travelled trajectory
        # and the planned one
        agent_time_stepwise = [int(1/self.env.agents[a].speed_data['speed'])]*len(self.cells_sequence[a])
        time_data = [len(list(i_list)) for i, i_list in groupby(self.agent_position_data[a])]
        agent_time_stepwise = time_data + agent_time_stepwise[len(time_data):]

        # initial state
        start_timestamp = 0
        agent_first_edge = agent_initial_positions[a][1]
        agent_prev_vertex = None
        agent_trajectory = self.cells_sequence[a]
        agent_pos_on_traj = 0
        end_timestamp = 0




        # start with the beginning
        # find next exit on trajectory
        while(True):

            agent_next_vertex = None

            # move all the delay to previous edge : No matter what

            # check what sort of vertex the agent is at right now.
            if agent_current_vertex.Type == "junction":

                agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                agent_next_pos_on_traj = agent_pos_on_traj+1

                agent_next_vertex, agent_next_dir = \
                            [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                    if agent_next_position in item[1].Cells][0]

            elif agent_current_vertex.Type == "edge" and \
                    len(agent_current_vertex.Cells) == 1:

                agent_next_position = agent_trajectory[agent_pos_on_traj+1]
                agent_next_pos_on_traj = agent_pos_on_traj+1

                if agent_current_vertex.Cells[0] != agent_trajectory[-2]:
                    agent_next_vertex, agent_next_dir = \
                                [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                        if agent_next_position in item[1].Cells][0]

                else:
                    break

            elif agent_current_vertex.Type == "edge":

                if agent_current_vertex.Cells[0] in agent_trajectory[agent_pos_on_traj+1:]:

                    agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                         if item[0] == agent_current_vertex.Cells[0]][0]

                    agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                    agent_next_pos_on_traj =  agent_pos_on_traj + \
                                              [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]


                elif agent_current_vertex.Cells[-1] in agent_trajectory[agent_pos_on_traj+1:]:

                    agent_next_vertex, agent_next_dir = [[item[1],num] for num, item in enumerate(agent_current_vertex.Links)
                                         if item[0] == agent_current_vertex.Cells[-1]][0]

                    agent_next_position = [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                    agent_next_pos_on_traj =  agent_pos_on_traj + \
                                              [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                if cell[0] == agent_next_position[0] and cell[1] == agent_next_position[1]][0]

                else:
                    break


            # found next vertex
            if agent_next_vertex != None:
                if a in agent_next_vertex.Trains:

                    pos_on_first_edge = [num for num,item in enumerate(agent_current_vertex.Trains) if item == a][0]
                    pos_on_second_edge = [num for num,item in enumerate(agent_next_vertex.Trains) if item == a][0]

                    first_time = agent_current_vertex.TrainsTime[pos_on_first_edge]
                    second_time = agent_next_vertex.TrainsTime[pos_on_second_edge]

                    if second_time[0] > first_time[1]:
                        # a delay is probably added in the next node
                        # propagate it back by
                        first_time[1] = second_time[0]

                    elif second_time[0] < first_time[1]:
                        shift = first_time[1] - second_time[0]
                        second_time[0] += shift
                        second_time[1] += shift
                    else:
                        print("here")

                else:
                    break
            else:
                break




            start_timestamp = end_timestamp
            agent_prev_vertex = agent_current_vertex
            agent_current_vertex = agent_next_vertex
            agent_pos_on_traj = agent_next_pos_on_traj

        return observations

