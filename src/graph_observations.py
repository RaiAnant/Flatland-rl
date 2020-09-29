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
        self.forks_coords = None
        self.base_graph = None

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            # Use set_env available in PredictionBuilder (parent class)
            self.predictor.set_env(self.env)

    def reset(self):
        """
        Inherited method used for pre computations.
        :return:
        """

        self.forks_coords = self._find_forks()

    # ########################################################################################
    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.
        :param direction:
        :return:
        """
        return int((direction + 2) % 4)

    def _find_forks(self):
        """
        A fork (in the map) is either a switch or a diamond crossing.
        :return:
        """
        print("_find_forks()")

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
    # ########################################################################################
    # #################################### FPR GLOBAL GRAPH ##################################

    def build_global_graph(self):
        print("build_global_graph()")

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
            self.base_graph.add_vertex(str(init_pos[0])+","+str(init_pos[1]))
            added_vertex.append(init_pos)

        while len(pending_to_explore):
            current = pending_to_explore.pop()
            next_pos = self._step(current)

            for item in next_pos:
                if item[0] not in added_vertex:
                    if item[1] == 1:
                        self.base_graph.add_vertex(str(item[0][0])+","+str(item[0][1]))
                        added_vertex.append(item[0])

                    elif item[1] > 2:
                        self.base_graph.add_vertex(str(item[0][0])+","+str(item[0][1]))
                        added_vertex.append(item[0])
                        pending_to_explore.append(item[0])

                if item[1] == 1:
                    self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])

                elif item[1] > 2:
                    self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])

        return "built base graph"

    def _step_extend(self, current, direction):

        print("_step_extend()")

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
        print("_step()")

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
                        #traj.append(*temp1)

                        break
                else:
                    #print(" node found")
                    node_list.append([position, total_transitions, traj])
                    break

        return node_list

    # ########################################################################################

    def prepare_global_var(self) -> {}:

        if self.base_graph == None:
            print(self.build_global_graph())

        self.num_active_agents = 0
        for a in self.env.agents:
            if a.status == RailAgentStatus.ACTIVE:
                self.num_active_agents += 1
        self.prediction_dict = self.predictor.get()
        # Useful to check if occupancy is correctly computed
        self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
        self.max_prediction_depth = self.predictor.max_depth
        """
        if self.prediction_dict:
            self.max_prediction_depth = self.predictor.max_depth
            for t in range(self.max_prediction_depth):
                pos_list = []
                dir_list = []
                for a in range(self.env.number_of_agents):
                    if self.prediction_dict[a] is None:
                        continue
                    pos_list.append(self.prediction_dict[a][t][1:3])
                    dir_list.append(self.prediction_dict[a][t][3])
                self.predicted_pos_coord.update({t: pos_list})
                self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                #self.predicted_dir.update({t: dir_list})

            #for a in range(len(self.env.agents)):
            #    pos_list = []
            #    for ts in range(self.max_prediction_depth):
            #        pos_list.append(self.predicted_pos[ts][a])  # Use int positions
            #    #self.predicted_pos_list.update({a: pos_list})
        """

    def populate_graph(self, observations):
        # ************************************

        # for every agent
        #   get its planned trajectory
        #       find which edge is represented by the trajectory
        #       Also if a half edge if represented then consider it till the end of that edge
        #       (right before the end of section)
        #   now enter for the edge
        #       [train_ID, direction [start_cell, end_cell], time stamp [start, end]]
        #           (EWNS might not work hence direction from source to dest)
        # Return this as observation
        import itertools
        for a in range(self.env.number_of_agents):

            # manipulate trajectory for the agent to be modified
            traj = copy.deepcopy(self.cells_sequence[a][:-2])
            traj.insert(0,self.env.agents[a].position if self.env.agents[a].position is not None else self.env.agents[a].initial_position )
            #traj = [tuple((5,9)), tuple((5,8)), tuple((5,8)), tuple((5,8))]


            start_timestamp = 0
            while len(traj) > 1:
                #print(traj)
                start_timestamp_copy = copy.deepcopy(start_timestamp)
                for edge in observations.edge_ids:
                    #print(edge.Cells)
                    if traj[0] in edge.Cells and traj[1] in edge.Cells:
                        edge_pos = [i for i, tupl in enumerate(edge.Cells) if tupl[0] == traj[0][0] and tupl[1] == traj[0][1]][0]

                        # step_dir is the value is either 1 or -1 being the step in either direction of the edge
                        if 0 < edge_pos < len(edge.Cells) - 1:
                            if traj[1] == edge.Cells[edge_pos - 1]:
                                step_dir = -1
                            elif traj[1] == edge.Cells[edge_pos + 1]:
                                step_dir = 1
                        elif 0 == edge_pos:
                            step_dir = 1
                        elif edge_pos == len(edge.Cells) - 1:
                            step_dir = -1

                        edge_pos_end = 0
                        edge_cells = edge.Cells[edge_pos:len(edge.Cells)] if step_dir == 1 else edge.Cells[0:edge_pos+1][::-1]

                        traj_pos_end = 0
                        touched = False
                        while True:

                            # If predicted trajectory is remaining
                            #   check if next pos in the predicted trajectory is equal to the current cell of edge
                            #
                            #   if yes
                            #       increment the traj_pos_end
                            #   else
                            #       if edge has cells remaining
                            #           if next pos in the predicted trajectory is equal to the next cell of the edge
                            #               increment traj_pos_end
                            #               step edge_pos_end
                            #           else
                            #               break

                            if len(traj) > traj_pos_end and len(edge_cells) > edge_pos_end:
                                touched = True

                                # if the current positions are equal then proceed in trajectory
                                if traj[traj_pos_end] == edge_cells[edge_pos_end]:
                                    traj_pos_end += 1
                                elif edge_pos_end < len(edge_cells)-1:
                                    if traj[traj_pos_end] == edge_cells[edge_pos_end+1]:
                                        edge_pos_end += 1
                                    else:
                                        break
                                else:
                                    break
                            else:
                                break

                        print("here")
                        # so a section is found that has part of desired trajectory
                        # we have the agent ID
                        # we have the direction,
                        # we can find the relevant order for start and end for direction
                        # we can also find the number of time steps (number of cells * 1/speed)
                        if touched:
                            traj_pos_end -= 1
                            end_timestamp = start_timestamp + int((edge_pos_end) * 1 / self.env.agents[a].speed_data['speed'])

                            if start_timestamp != end_timestamp:
                                edge.Trains.append(a)
                                edge.TrainsTime.append(sorted([start_timestamp, end_timestamp]))
                                edge.TrainsDir.append(0 if step_dir == 1 else 1)
                                start_timestamp = end_timestamp
                        else:
                            break

                traj = traj[traj_pos_end:]

        # Now Build the the collision lock matrix
        observations.setCosts()

        return observations

    def get_many(self, handles) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """

        self.prepare_global_var()
        # make a copy of base_graph for reusability
        # ************************************
        # copy construction for the object
        observations = copy.deepcopy(self.base_graph)

        observations = self.populate_graph(observations)

        return observations

    # TODO Optimize considering that I don't need obs for those agents who don't have to pick actions
    def get(self) -> {}:
        """
        Returns obs for one agent, obs are a single array of concatenated values representing:
        - occupancy of next prediction_depth cells,
        - agent priority/speed,
        - number of malfunctioning agents (encountered),
        - number of agents that are ready to depart (encountered).
        :param handle:
        :return:
        """

        self.prepare_global_var()
        # make a copy of base_graph for reusability
        # ************************************
        # copy construction for the object
        observations = copy.deepcopy(self.base_graph)

        return self.populate_graph(observations)


    # ########################################################################################

    def optimize(self, observations):

        #comp_observations = copy.deepcopy(observations)
        check_again = True
        check_again_counter = 0

        while check_again and check_again_counter < 10:
            # check if the cost is within limits
            check_again = False
            check_again_counter += 1
            for edge in observations.edge_ids:
                if edge.CostTotal > 100:
                    check_again = True
                    # trim the cell sequence here
                    # find the train to be stopped
                    id = np.argmax(edge.CostPerTrain)
                    t_id = edge.Trains[id]
                    time = edge.TrainsTime[id][0]

                    self.cells_sequence[t_id][time-1:] = [self.cells_sequence[t_id][time]] * (len(self.cells_sequence[t_id])-time)
                    #self.cells_sequence[t_id] = self.cells_sequence[t_id][:time+1]


            # make a copy of base_graph for reusability
            # ************************************
            # copy construction for the object
            observations = copy.deepcopy(self.base_graph)
            observations = self.populate_graph(observations)

            print("here")

            # set check again to False
            # else: optimize

        return observations