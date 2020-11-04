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

from itertools import groupby

from src.priority import assign_priority
from src.util.graph import Graph
from src.util.global_graph import Global_Graph


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
        self.time_at_cell = None
        self.forks_coords = None
        self.base_graph = None
        self.cur_graph = None
        self.ts = -1
        self.agent_prev = defaultdict(list)
        self.agent_position_data = defaultdict(list)
        self.cur_pos_edge = defaultdict()

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            # Use set_env available in PredictionBuilder (parent class)
            self.predictor.set_env(self.env)

    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.

        :param direction:
        :return:
        """
        return int((direction + 2) % 4)

    def reset(self):
        """
        Inherited method used for pre computations.

        :param:
        :return:
        """
        self.forks_coords = self._find_forks()
        self.ts = -1
        self.agent_prev = defaultdict(list)
        self.cells_sequence = None
        self.time_at_cell = None
        self.base_graph = None
        self.cur_graph = None
        self.agent_position_data = defaultdict(list)
        #self.base_graph_edge = defaultdict()

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

    def get(self, handles) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """
        return self._get()

    def get_many(self, handles) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """
        return self._get()

    def _get(self):
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """
        self.ts += 1

        self.prepare_global_var()

        observations = copy.deepcopy(self.base_graph)
        observations = self.populate_graph(observations)

        return observations






    def prepare_global_var(self) -> {}:
        """

        :return:
        """

        if self.base_graph is None:
            self.build_global_graph()

        self.prediction_dict = self.predictor.get()
        local = self.predictor.compute_cells_sequence(self.prediction_dict)
        if self.cells_sequence is None:
            self.cells_sequence = local

            self.cells_sequence = local

            self.time_at_cell = defaultdict(list)
            self.malfunction_time = defaultdict(list)
            self.introduced_delay_time = defaultdict(list)

            for a in self.cells_sequence:
                step_size = 1 / self.env.agents[a].speed_data['speed']
                self.time_at_cell[a] = [step_size]*len(self.cells_sequence[a])
                self.malfunction_time[a] = [0]*len(self.cells_sequence[a])
                self.introduced_delay_time[a] = [0]*len(self.cells_sequence[a])


            # check if multiple trains start at the same spot. introduce intial delay
            start_pos = []
            for a in self.env.agents:
                self.introduced_delay_time[a.handle][0] += start_pos.count(a.initial_position)
                start_pos.append(a.initial_position)


            self.max_prediction_depth = self.predictor.max_depth

        for a in self.env.agents:
            current_position = a.position if a.position is not None \
                             else a.initial_position if a.status is not RailAgentStatus.DONE_REMOVED \
                             else tuple((0,0))
            self.agent_position_data[a.handle].append(current_position)

        if len(self.agent_position_data[0]) > 1:
            for a in self.env.agents:
                data = [len(list(i_list))-1 for i, i_list in groupby(self.agent_position_data[a.handle])]
                self.malfunction_time[a.handle][0:len(data)] = data
                self.introduced_delay_time[a.handle] = [0]*(len(data)-1)+self.introduced_delay_time[a.handle][len(data)-1:]


    def populate_graph(self, observations):
        """
        Inherited method used for pre computations.

        :return:
        """
        """
        for every agent
           get its planned trajectory
               find which edge is represented by the trajectory
               Also if a half edge if represented then consider it till the end of that edge
               (right before the end of section)
           now enter for the edge
               [train_ID, direction [start_cell, end_cell], time stamp [start, end]]
                   (EWNS might not work hence direction from source to dest)
         Return this as observation
        """

        # find current positions of agents
        cur_pos = []
        for a in self.env.agents:
            cur_pos.append(a.initial_position)

        # find the edges where the starting points are
        cur_pos_edge = defaultdict()
        for edge in observations.edge_ids:
            for agent_id, item in enumerate(cur_pos):
                if item in edge.Cells:
                    cur_pos_edge[agent_id] = [item, edge]

        # now start to fillup the cell sequence
        for a in self.env.agents:

            #print(" *********************** Agent ", a.handle, "\n")

            start_timestamp = 0
            agent_pos_on_traj = 0

            agent_trajectory = self.cells_sequence[a.handle]
            #agent_current_pos = cur_pos_edge[a.handle][0]
            agent_current_pos = agent_trajectory[agent_pos_on_traj]
            agent_edge = cur_pos_edge[a.handle][1]

            trajectory_ended = False
            while True:

                #print(agent_pos_on_traj)
                #print("\n", agent_current_pos, "\n", agent_edge.Cells, "\n", agent_trajectory)

                # find agent position in the trajectory
                try:
                    agent_pos_on_edge = [num for num, item in enumerate(agent_edge.Cells)
                                     if item[0] == agent_current_pos[0] and item[1] == agent_current_pos[1]][0]
                except:
                    print(agent_current_pos, agent_edge.Cells)

                # find agent direction
                if 0 < agent_pos_on_edge < len(agent_edge.Cells) - 1:
                    if agent_trajectory[1] == agent_edge.Cells[agent_pos_on_edge - 1]:
                        agent_dir_on_edge = 1
                    elif agent_trajectory[1] == agent_edge.Cells[agent_pos_on_edge + 1]:
                        agent_dir_on_edge = 0
                elif 0 == agent_pos_on_edge:
                    agent_dir_on_edge = 0
                elif agent_pos_on_edge == len(agent_edge.Cells) - 1:
                    agent_dir_on_edge = 1

                # find end timestamp
                # either the end of the edge will be in the trajectory
                if agent_dir_on_edge == 0:
                    next_node = agent_edge.B_node
                    if agent_edge.B in agent_trajectory:
                        cell_difference = [num for num, item in enumerate(agent_trajectory[agent_pos_on_traj:])
                                     if item[0] == agent_edge.B[0] and item[1] == agent_edge.B[1]][0]+1
                    elif agent_trajectory[-2] in agent_edge.Cells:
                        cell_difference = [num for num, item in enumerate(agent_edge.Cells)
                                     if item[0] == agent_trajectory[-2][0] and item[1] == agent_trajectory[-2][1]][0]+1
                        trajectory_ended = True
                    else:
                        print("Bug : because the trajectory and cell ends do not match while populating graph \n",
                              agent_pos_on_edge, "\n", agent_edge.B,"\n", agent_trajectory[-2],"\n", agent_edge.Cells,
                              "\n",agent_trajectory,"\n")

                # or the end of the trajectory has to be on the edge
                elif agent_dir_on_edge == 1:
                    next_node = agent_edge.A_node
                    if agent_edge.A in agent_trajectory:
                        cell_difference = [num for num, item in enumerate(agent_trajectory[agent_pos_on_traj: ])
                                     if item[0] == agent_edge.A[0] and item[1] == agent_edge.A[1]][0]+1
                    elif agent_trajectory[-2] in agent_edge.Cells:
                        cell_difference = [num for num, item in enumerate(agent_edge.Cells[::-1])
                                     if item[0] == agent_trajectory[-2][0] and item[1] == agent_trajectory[-2][1]][0]+1
                        trajectory_ended = True

                    else:
                        print("Bug : because the trajectory and cell ends do not match while populating graph \n",
                              agent_pos_on_edge, "\n", agent_edge.A,"\n", agent_trajectory[-2],"\n", agent_edge.Cells,
                              "\n",agent_trajectory,"\n")

                # update edge
                agent_edge.Trains.append(a.handle)
                agent_edge.TrainsDir.append(agent_dir_on_edge)

                try:
                    steps = np.sum(self.time_at_cell[a.handle][int(agent_pos_on_traj) \
                                                        :int(agent_pos_on_traj) + cell_difference]) \
                            + np.sum(self.malfunction_time[a.handle][int(agent_pos_on_traj) \
                                                              :int(agent_pos_on_traj) + cell_difference]) \
                            + np.sum(self.introduced_delay_time[a.handle][int(agent_pos_on_traj) \
                                                                   :int(agent_pos_on_traj) + cell_difference])
                except:
                    print(agent_pos_on_traj, cell_difference, self.time_at_cell[a.handle][int(agent_pos_on_traj) \
                                                        :int(agent_pos_on_traj) + cell_difference],\
                          self.malfunction_time[a.handle][int(agent_pos_on_traj) \
                                                      :int(agent_pos_on_traj) + cell_difference],\
                          self.introduced_delay_time[a.handle][int(agent_pos_on_traj) \
                                                      :int(agent_pos_on_traj) + cell_difference])

                end_timestamp = int(start_timestamp + steps * 1 / self.env.agents[a.handle].speed_data['speed'])

                agent_edge.TrainsTime.append([start_timestamp, end_timestamp])

                agent_pos_on_traj += cell_difference-1

                # now update agent_edge
                # agent_current_pos
                agent_current_pos = agent_trajectory[agent_pos_on_traj]

                start_timestamp = end_timestamp
                # or end of trajectory should be within the edge

                if not trajectory_ended:
                    for next_edge in next_node.edges:
                        if next_edge is not None:
                            if agent_trajectory[agent_pos_on_traj+1] in next_edge.Cells:
                                agent_edge = next_edge
                else:
                    break

        observations.setCosts()
        observations = copy.deepcopy(observations)

        #sorted_list = sorted(observations.edge_ids, key=lambda x: x.CostTotal, reverse=True)
        #optimization_candidate = [edge for edge in sorted_list if edge.CostTotal > 10000]


        return observations











    # For Global Graph
    def build_global_graph(self):
        """

        :return:
        """

        self.base_graph = Global_Graph()

        # Step 1 : Collect all the vertices
        #
        # Find the first agent's position
        #   check how many options are there in this cell
        #       (CAUTION: Check if stations are similar to forks or to dead ends
        #                   for now we consider them to be forks
        #                   if dead-end post process to merge them)
        #       If 1 : Its a dead end
        #               Add as a node
        #       If 2 : Normal way
        #               proceed to next cell and Continue
        #       If >2 : forks
        #               Add as a node
        #               Add all the possible next positions with respective directions
        # Step 2 : Connect edges and make transition table

        #print(self.forks_coords)
        if not len(self.env.agents):
            return "No active Agents"
        init_pos = next(iter(self.forks_coords))
        pending_to_explore = [init_pos]
        added_vertex = []

        full_cell_transition = self.env.rail.get_full_transitions(*init_pos)
        cell_transitions_bitmap = bin(full_cell_transition)
        total_transitions = cell_transitions_bitmap.count("1")

        # add the starting node if the agent starts from a fork which is highly improbable

        if total_transitions > 2:
            self.base_graph.add_vertex(tuple((init_pos[0], init_pos[1])))

            added_vertex.append(init_pos)

        while len(pending_to_explore):
            current = pending_to_explore.pop()
            next_pos = self._step(current)


            for item in next_pos:
                if item[0] not in added_vertex:
                    if item[1] == 1:
                        self.base_graph.add_vertex(tuple((item[0][0], item[0][1])))
                        added_vertex.append(item[0])

                    elif item[1] > 2:
                        self.base_graph.add_vertex(tuple((item[0][0], item[0][1])))
                        added_vertex.append(item[0])
                        pending_to_explore.append(item[0])

                if item[1] == 1 or item[1] > 2:
                    source_node = self.base_graph.vert_dict[tuple((current[0], current[1]))]
                    dest_node = self.base_graph.vert_dict[tuple((item[0][0], item[0][1]))]

                    added_edge = self.base_graph.add_edge(tuple((current[0], current[1])),
                                                          tuple((item[0][0], item[0][1])), source_node, dest_node, item[2])

                    source_node.edges.append(added_edge)
                    dest_node.edges.append(added_edge)

        return "built base graph"

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

        """
        for direction in [0,1,2,3]:
            position = get_new_position(init_position, direction)
            next_actions = get_valid_move_actions_(direction, position, self.env.distance_map.rail)
            for next_action in next_actions:
                next_action_distance = self.env.distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                print(next_action_distance)
        """

        #all_actions = []
        for direction in [0,1,2,3]:

            traj = []
            traj.append(init_position)

            position = get_new_position(init_position, direction)

            # check if this new position transition is possible

            next_actions = get_valid_move_actions_(direction, position, self.env.distance_map.rail)
            #all_actions.append(next_actions)

            if len(next_actions) > 0:
                traj.append(position)

                while True:

                    # if number of total transition is 1
                    # return as dead end
                    # if number of total transition is > 2
                    # return as node
                    # else transition to new

                    full_cell_transition = self.env.rail.get_full_transitions(*position)
                    cell_transition = self.env.rail.get_transitions
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

                            next_actions = get_valid_move_actions_(direction, position, self.env.distance_map.rail)
                            # all_actions.append(next_actions)

                            if len(next_actions) > 0:
                                traj.append(position)
                            #traj.append(position)
                            else:
                                break

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

    def is_new_route(self, node, cell_passed):
        if (len(node.path) != 0 and node.path[0] != cell_passed) or (
                len(node.path) == 0 and cell_passed != node.children[0].node_id):
            return True
        return False

    def get_alternate_paths(self, agent_id, node, collision_node, idx):
        if node.is_blocked:
            return None
        is_new_route = self.is_new_route(node, self.cells_sequence[agent_id][idx + 1])
        if node.node_id == collision_node and is_new_route:  # if at node that connects to collsion edge but the path doesnt lead to the collsion edge, add it as a possible route
            return [node]

        elif node.node_id == collision_node:  # if the node is the one with collision, block it from further exploration
            node.is_blocked = True  # TODO : CHANGE BLOCKED IN TIME
            return None

        if idx != 0 and is_new_route:  # if the current route is not explored in cell sequence, add it as a possible route
            return [node]

        if len(node.path) != 0 and \
                (node.node_id[0] - collision_node[0]) * (node.path[-1][0] - collision_node[0]) <= 0 and \
                (node.node_id[1] - collision_node[1]) * (node.path[-1][1] - collision_node[
            1]) <= 0:  # check if the collision cell is present in the current tree node, and if it is, return None
            node.is_blocked = True
            return None

        list = []
        for child in node.children:  # explore the childrnen of the given node
            if idx == 0:
                val = self.get_alternate_paths(agent_id, child, collision_node, idx + len(node.path)+1)
            else:
                val = self.get_alternate_paths(agent_id, child, collision_node, idx + len(node.path) + 1)

            if val is not None:
                if len(val) == 1:
                    list = val + list
                else:
                    list = list + val

        return list

    def sort_key(self, x):
        min_val = 100000
        if len(x.TrainsTime) == 0 or x.CostTotal < 100:
            return 100000
        for time in x.TrainsTime:
            if min_val > min(time[0], time[1]):
                min_val = min(time[0], time[1])

        # val /= len(x.TrainsTime)

        return min_val

    def rerouting(self, observations, agent_trees):
        """

        :return:
        """

        ob_list = []
        time_update_list = []
        ob_list.append(observations)
        # comp_observations = copy.deepcopy(observations)
        check_again = True
        check_again_counter = 0

        route_list = []

        while check_again and check_again_counter < 20:
            # check if the cost is within limits
            check_again = False
            check_again_counter += 1
            observations.edge_ids.sort(key=self.sort_key)  # sort the edges on the basis of occourence of collision
            agent_id = None
            for edge in observations.edge_ids:
                if edge.CostTotal > 1000:
                    check_again = True
                    # find the train to be stopped
                    id = np.argmax(edge.CostPerTrain)

                    t_id = edge.Trains[id]  # train id to re-route

                    agent_tree = agent_trees[t_id]

                    agent_id = t_id
                    # found the edge and the train for collision
                    collision_entry_point = edge.Cells[0] if edge.TrainsDir[id] == 0 else edge.Cells[-1]
                    prev_exit_point = [
                        num if item[0] == collision_entry_point[0] and item[1] == collision_entry_point[1]
                        else 0
                        for num, item in enumerate(self.cells_sequence[t_id])]

                    # pos =
                    # find the entering cell and one before that
                    # find the edge which contains this cell
                    # manipulate this to stop the train on this edge before exit

                    # time = edge.TrainsTime[id][0]

                    paths = self.get_alternate_paths(t_id, agent_tree.root, collision_entry_point,
                                                     0)  # get all the nodes in the tree with alternate path
                    selected_path = paths[-1]  # pick the one at the max depth
                    new_cell_seq = []
                    idx = 0

                    current_cell = self.cells_sequence[t_id][idx]

                    while selected_path.node_id != current_cell:  # copy the cell_seq as it is until we reach the point of re-route
                        new_cell_seq.append(current_cell)
                        idx += 1
                        current_cell = self.cells_sequence[t_id][idx]

                    new_cell_seq.append(selected_path.node_id)  # add the new path cells
                    new_cell_seq += selected_path.path
                    temp = selected_path

                    while len(temp.children) != 0 and len(new_cell_seq) < self.max_prediction_depth:
                        temp = temp.children[0]
                        new_cell_seq.append(temp.node_id)
                        new_cell_seq += temp.path

                    self.time_at_cell[t_id] = [1 / self.env.agents[t_id].speed_data['speed']] * len(
                        new_cell_seq)  # update the cell seq and time at cells values
                    self.cells_sequence[t_id] = new_cell_seq
                    self.cells_sequence[t_id].append(agent_tree.target)
                    self.cells_sequence[t_id].append((0, 0))
                    break

            # make a copy of base_graph for reusability
            observations = copy.deepcopy(self.base_graph)
            observations = self.populate_graph(observations)  # create new observation from the changes

            route_list.append((agent_id, self.cells_sequence[agent_id]))
            local_copy = copy.deepcopy(self.time_at_cell)

        return check_again_counter < 100, route_list
