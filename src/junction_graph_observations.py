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
        self.predicted_pos_list = {}  # Dict handle : int_pos_list
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

    def is_update_required(self, observations):
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
                cur_pos_on_traj = [num for num, cell in enumerate(self.cells_sequence[a.handle])
                                   if np.all(cell == cur_pos)][0]
            except:
                print(a.handle, cur_pos, self.cells_sequence[a.handle])
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

        forks = set()  # Set of nodes as tuples/coordinates
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
                    else tuple((0, 0))
                self.agent_position_data[a.handle].append(current_position)

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
            agent_time_stepwise = [int(1 / self.env.agents[a].speed_data['speed'])] * len(self.cells_sequence[a])
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
            while (True):

                # check what sort of vertex the agent is at right now.
                if agent_current_vertex.Type == "junction" \
                        or (agent_current_vertex.Type == "edge" and \
                            len(agent_current_vertex.Cells) == 1):

                    agent_next_position = agent_trajectory[agent_pos_on_traj + 1]
                    agent_next_pos_on_traj = agent_pos_on_traj + 1

                    if (agent_current_vertex.Type == "edge" and agent_current_vertex.Cells[0] != agent_trajectory[-2]) \
                            or agent_current_vertex.Type == "junction":
                        try:
                            agent_next_vertex, agent_next_dir = \
                                [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                                 if agent_next_position in item[1].Cells][0]
                        except:
                            print("debug")

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                    else:
                        break


                elif agent_current_vertex.Type == "edge":
                    #TODO CHECK 21 WITH GAURAV
                    if agent_current_vertex.Cells[0] in agent_trajectory[agent_pos_on_traj + 1:agent_pos_on_traj + 21]:

                        agent_next_vertex, agent_next_dir = \
                        [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                         if item[0] == agent_current_vertex.Cells[0]][0]

                        agent_next_position = \
                        [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj = agent_pos_on_traj + \
                                                 [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                                  if
                                                  cell[0] == agent_next_position[0] and cell[1] == agent_next_position[
                                                      1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                    elif agent_current_vertex.Cells[-1] in agent_trajectory[agent_pos_on_traj + 1:]:

                        agent_next_vertex, agent_next_dir = \
                        [[item[1], num] for num, item in enumerate(agent_current_vertex.Links)
                         if item[0] == agent_current_vertex.Cells[-1]][0]

                        agent_next_position = \
                        [item[0] for item in agent_next_vertex.Links if item[1] == agent_current_vertex][0]

                        agent_next_pos_on_traj = agent_pos_on_traj + \
                                                 [num for num, cell in enumerate(agent_trajectory[agent_pos_on_traj:]) \
                                                  if
                                                  cell[0] == agent_next_position[0] and cell[1] == agent_next_position[
                                                      1]][0]

                        agent_current_vertex.Trains.append(a)
                        agent_current_vertex.TrainsDir.append(agent_next_dir)

                        end_timestamp = start_timestamp + \
                                        np.sum(agent_time_stepwise[agent_pos_on_traj:agent_next_pos_on_traj])
                        agent_current_vertex.TrainsTime.append([start_timestamp, end_timestamp])

                    elif agent_trajectory[-2] in agent_current_vertex.Cells:
                        # print("Trajectory End")
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
            vertex.DeadLockMatrix = np.zeros((self.env.number_of_agents, self.env.number_of_agents), dtype=np.uint8)

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
                    edge_vertex.DeadLockMatrix = np.zeros((self.env.number_of_agents, self.env.number_of_agents),
                                                          dtype=np.uint8)

                    vertex.Links.append([junctions, edge_vertex])
                    edge_vertex.Links.append([edge_vertex_cells[0], vertex])

                else:
                    if str(path[0])[1:-1] in self.base_graph.vertices:
                        vertex.Links.append([junctions, self.base_graph.vertices[str(path[0])[1:-1]]])
                        self.base_graph.vertices[str(path[0])[1:-1]].Links.append([path[0], vertex])

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

                    return temp, traj + temp1
            else:
                return [position, total_transitions], traj

    def _step(self, current):
        """

        :return:
        """

        init_position = current
        node_list = []

        for direction in [0, 1, 2, 3]:

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
                    # position = get_new_position(position, direction)

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
                        traj = traj + temp1
                        node_list.append([temp[0], temp[1], traj])
                        break
                else:
                    node_list.append([position, total_transitions, traj])
                    break

        return node_list

    def setDeadLocks(self, obs):

        # find the agent's remaining trajectory

        # find agents on shared paths
        # some might have exclusive paths as well, who knows !!
        agent_pos = []
        agent_traj = []
        for a in self.env.agents:
            # agent_pos.append(a.position if a.position is not None
            #                 else a.initial_position if a.status is not RailAgentStatus.DONE_REMOVED
            #                 else a.target)
            agent_pos_on_traj = [num for num, item in enumerate(self.cells_sequence[a.handle])
                                 if item[0] == self.cur_pos_list[a.handle][0][0]
                                 and item[1] == self.cur_pos_list[a.handle][0][1]][0]
            agent_traj.append(self.cells_sequence[a.handle][agent_pos_on_traj:])

        dead_lock_matrix = np.zeros((self.env.number_of_agents, self.env.number_of_agents), dtype=np.uint8)

        for agent_id, trajectory in enumerate(agent_traj):
            for other_agent_id, other_trajectory in enumerate(agent_traj):
                if agent_id != other_agent_id:

                    if not self.cur_pos_list[agent_id][2] and len(trajectory) > 3:
                        is_next_in_queue = [num for num, item in enumerate(self.cur_pos_list)
                                            if item[0][0] == trajectory[1][0]
                                            and item[0][1] == trajectory[1][1]]

                        if len(is_next_in_queue):
                            new_dead_locks = []
                            for deadlock in obs.Deadlocks:
                                if deadlock[0] == is_next_in_queue[0]:
                                    if agent_traj[deadlock[0]][1][0] == trajectory[2][0] \
                                            and agent_traj[deadlock[0]][1][1] == trajectory[2][1] \
                                            and agent_traj[deadlock[0]][2][0] == trajectory[3][0] \
                                            and agent_traj[deadlock[0]][2][1] == trajectory[3][1]:

                                        vertex_end_1 = obs.vertices[deadlock[2]].Cells[0]
                                        vertex_end_2 = obs.vertices[deadlock[2]].Cells[-1]

                                        end_pos = [num for num, item in enumerate(trajectory)
                                                   if (item[0] == vertex_end_1[0]
                                                       and item[1] == vertex_end_1[1])
                                                   or (item[0] == vertex_end_2[0]
                                                       and item[1] == vertex_end_2[1])]

                                        if len(end_pos):
                                            new_dead_locks.append([agent_id, deadlock[1], deadlock[2]])

                            for deadlock in new_dead_locks:
                                if deadlock not in obs.Deadlocks:
                                    obs.Deadlocks.append(deadlock)

                    # while True:
                    #
                    #     if len(trajectory) > 3:
                    #         is_next_in_queue = [num for num,item in enumerate(self.cur_pos_list)
                    #                                if item[0][0] == trajectory[1][0]
                    #                                and item[0][1] == trajectory[1][1]]
                    #
                    #         break_inner = False
                    #
                    #         if len(is_next_in_queue):
                    #             for deadlock in obs.Deadlocks:
                    #                 if deadlock[0] == is_next_in_queue[0]:
                    #                     if agent_traj[deadlock[0]][1][0] == trajectory[2][0] \
                    #                             and agent_traj[deadlock[0]][1][1] == trajectory[2][1] \
                    #                             and agent_traj[deadlock[0]][2][0] == trajectory[3][0] \
                    #                             and agent_traj[deadlock[0]][2][1] == trajectory[3][1]:
                    #                         trajectory = trajectory[1:]
                    #                         break_inner = True
                    #                         break
                    #
                    #     else:
                    #         break
                    #
                    #     if not break_inner:
                    #         break
                    #

                    # if len(is_next_in_queue) \
                    #        and self.env.agents[agent_id].position == self.env.agents[agent_id].old_position\
                    #        and self.env.agents[agent_id].position != self.env.agents[agent_id].initial_position\
                    #        and self.env.agents[agent_id].position is not None:
                    #    trajectory = trajectory[1:]

                    else:

                        pos_first_in_second = [num for num, item in enumerate(trajectory)
                                               if item[0] == other_trajectory[0][0]
                                               and item[1] == other_trajectory[0][1]]

                        if len(pos_first_in_second):
                            first_agent_traj_from_overlap = trajectory[1:pos_first_in_second[0] + 1][::-1]

                            if len(first_agent_traj_from_overlap) > 1:
                                incre = 0

                                while True:

                                    # other agent ends before reaching current position of agent
                                    # it would not have to cross teh agent
                                    if incre == len(other_trajectory):
                                        break
                                    elif incre == len(first_agent_traj_from_overlap):

                                        dead_lock_matrix[agent_id][other_agent_id] = 1
                                        first_agent_traj_from_overlap = first_agent_traj_from_overlap[::-1]

                                        pos_on_conflict_section = 0

                                        outer = True
                                        while outer:

                                            for vertex in obs.vertices:

                                                is_pos_on_vertex = [num for num, item in
                                                                    enumerate(obs.vertices[vertex].Cells)
                                                                    if first_agent_traj_from_overlap[
                                                                        pos_on_conflict_section][0]
                                                                    == item[0] and
                                                                    first_agent_traj_from_overlap[
                                                                        pos_on_conflict_section][1]
                                                                    == item[1]]

                                                if len(is_pos_on_vertex):
                                                    if obs.vertices[vertex].Type == "edge":
                                                        if obs.vertices[vertex].update_ts < self.ts:
                                                            obs.vertices[vertex].DeadLockMatrix = np.zeros(
                                                                (self.env.number_of_agents, self.env.number_of_agents),
                                                                dtype=np.uint8)
                                                            obs.vertices[vertex].update_ts = self.ts

                                                        obs.vertices[vertex].DeadLockMatrix[agent_id][
                                                            other_agent_id] = 1

                                                        exit_point = obs.vertices[vertex].other_end(
                                                            first_agent_traj_from_overlap[pos_on_conflict_section])

                                                        exit_position_on_conflict_section = [num for num, item in
                                                                                             enumerate(
                                                                                                 first_agent_traj_from_overlap)
                                                                                             if item[0] == exit_point[0]
                                                                                             and item[1] == exit_point[
                                                                                                 1]]

                                                        if len(exit_position_on_conflict_section):

                                                            if ([agent_id, other_agent_id, vertex]
                                                                    not in obs.Deadlocks):
                                                                obs.Deadlocks.append([agent_id, other_agent_id, vertex])

                                                            if len(first_agent_traj_from_overlap) > \
                                                                    exit_position_on_conflict_section[0] + 1:
                                                                pos_on_conflict_section = \
                                                                exit_position_on_conflict_section[0] + 1
                                                                break
                                                            else:
                                                                outer = False
                                                                break
                                                        else:
                                                            # print("possibly wrong calculation in deadlock")

                                                            outer = False
                                                            break
                                                    else:

                                                        if obs.vertices[vertex].update_ts < self.ts:
                                                            obs.vertices[vertex].DeadLockMatrix = np.zeros(
                                                                (self.env.number_of_agents, self.env.number_of_agents),
                                                                dtype=np.uint8)
                                                            obs.vertices[vertex].update_ts = self.ts

                                                        obs.vertices[vertex].DeadLockMatrix[agent_id][
                                                            other_agent_id] = 1

                                                        if len(first_agent_traj_from_overlap) > \
                                                                pos_on_conflict_section + 1:
                                                            pos_on_conflict_section += 1
                                                            break
                                                        else:
                                                            outer = False
                                                            break
                                        break
                                    elif not (first_agent_traj_from_overlap[incre][0] == other_trajectory[incre][0] \
                                              and first_agent_traj_from_overlap[incre][1] == other_trajectory[incre][
                                                  1]):
                                        # this is the point of seperation
                                        # might not be safe.

                                        next_exit_point = other_trajectory[incre]

                                        last_is_occupied = [num for num, item in enumerate(self.cur_pos_list)
                                                            if item[0][0] == next_exit_point[0]
                                                            and item[0][1] == next_exit_point[1]]

                                        if len(last_is_occupied):
                                            dead_lock_matrix[agent_id][agent_id] = 1
                                        break

                                    incre += 1

    def update_for_delay(self, observations, a, vert_type):
        """
        Inherited method used for pre computations.

        :return:
        """

        agent = self.env.agents[a]
        is_last_edge = False

        current_edge = [observations.vertices[vertex] for vertex in observations.vertices \
                        if agent.initial_position in observations.vertices[vertex].Cells][0]

        # collect all junctions until next edge
        next_vertex = current_edge

        initial_lapse = 0

        while not is_last_edge:

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
                    break

            if is_last_edge:
                continue

            next_list = []

            if initial_lapse != 0:
                pass

            elif vert_type == "junction":
                current_time_val = junc_list[::-1][1][1][1]

                next_list.append(
                    [[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]],
                     junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:
                    next_list.append([[current_time_val - 1, current_time_val], item[0]])
                    current_time_val -= 1

                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])

                if junc_list[::-1][-1][1][0] > current_time_val:
                    pass
            elif vert_type == "edge":
                current_time_val = junc_list[::-1][0][1][0]

                next_list.append(
                    [[current_time_val, current_time_val + junc_list[::-1][0][1][1] - junc_list[::-1][0][1][0]],
                     junc_list[::-1][0][0]])

                for item in junc_list[::-1][1:-1]:
                    next_list.append([[current_time_val - 1, current_time_val], item[0]])
                    current_time_val -= 1

                next_list.append([[junc_list[::-1][-1][1][0], current_time_val], junc_list[::-1][-1][0]])

            next_list = next_list[::-1]

            if initial_lapse == 0:
                junc_list_temp = [item[1][1] - item[1][0] for item in junc_list]
                next_list_temp = [item[0][1] - item[0][0] for item in next_list]
                initial_lapse = abs(((np.sum(junc_list_temp)) - ((np.sum(next_list_temp)))))

            next_vertex = current_edge

            filler_index = 0

            while True:
                agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]

                if len(next_list):

                    next_vertex.TrainsTime[agent_index][0] = next_list[filler_index][0][0]
                    next_vertex.TrainsTime[agent_index][1] = next_list[filler_index][0][1]

                elif filler_index > 0:
                    agent_index = [num for num, item in enumerate(next_vertex.Trains) if item == a][0]
                    next_vertex.TrainsTime[agent_index][0] += initial_lapse
                    next_vertex.TrainsTime[agent_index][1] += initial_lapse

                filler_index += 1

                if len(junc_list) == filler_index:
                    current_edge = next_vertex
                    break
                else:
                    next_vertex = next_vertex.Links[next_vertex.TrainsDir[agent_index]][1]

        return observations
