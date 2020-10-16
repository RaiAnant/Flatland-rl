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
        self.time_at_cell = None
        self.forks_coords = None
        self.base_graph = None
        self.cur_graph = None
        self.ts = -1
        self.agent_prev = defaultdict(list)

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
        self.agent_prev = defaultdict(list)
        self.cells_sequence = None
        self.time_at_cell = None
        self.base_graph = None
        self.cur_graph = None


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

    def align_lapse(self):
        # if the graph says that the agent should be here

        # if first step; populate prev agent position
        # else check and edit delays


        # however a malfunction might happen at any other point.
        # Hence align based on the following thought
        #
        # where is teh agent
        # where is it in the trajectory
        # where should it be as per the graph
        # print both if there is a difference
        if self.ts > 0:
            for a in self.env.agents:
                if a.status == RailAgentStatus.ACTIVE or a.status == RailAgentStatus.READY_TO_DEPART:

                    cur_pos = a.position if a.position is not None else a.initial_position
                    pos_on_traj = [num for num, item in enumerate(self.cells_sequence[a.handle])
                                   if item[0] == cur_pos[0] and item[1] == cur_pos[1]][0]
                    acc_delay = np.sum(self.time_at_cell[a.handle][:pos_on_traj+1]) + \
                            np.sum(self.introduced_delay_time[a.handle][:pos_on_traj+1])+ \
                            np.sum(self.malfunction_time[a.handle][:pos_on_traj+1])

                    # if the total time taken to reach here is more than planned as per the graph
                    #        then something went wrong: its going slow and the graph should (add)update with
                    #        a malfunction delay to fill the gap of time is to be added
                    #        the graph will update without stopping the train
                    # for that we will add the lapse time to current cell and stop it or
                    # Align it
                    if acc_delay < self.ts+1:
                        #self.malfunction_time[a.handle][:pos_on_traj] = [0]*pos_on_traj
                        #self.introduced_delay_time[a.handle][:pos_on_traj] = [0]*(pos_on_traj

                        self.malfunction_time[a.handle][pos_on_traj] = acc_delay - np.sum(self.time_at_cell[a.handle][:pos_on_traj])


                    # if the total time taken to reach here is less than planned as per the graph
                    #        and the current cell is not where the train is spending the remaining time
                    #        then something went wrong: its going fast and should stop now
                    # for that we will add the lapse time to current cell and stop it or
                    # Align it
                    if acc_delay > self.ts+1  \
                            and self.malfunction_time[a.handle][pos_on_traj] < acc_delay - self.ts:
                        #self.malfunction_time[a.handle][:pos_on_traj+1] = [0]*pos_on_traj
                        #self.introduced_delay_time[a.handle][:pos_on_traj+1] = [0]*pos_on_traj

                        self.malfunction_time[a.handle][pos_on_traj] = acc_delay - np.sum(self.time_at_cell[a.handle][:pos_on_traj+1])


    def prepare_global_var(self) -> {}:
        """

        :return:
        """

        """
        if self.ts == 0:
            for a in self.env.agents:
                self.agent_prev[a.handle] = a.initial_position
        else:
            for a in self.env.agents:
                if a.position is None or a.position is a.initial_position:
                    self.malfunction_time[a.handle][0] += 1
                elif self.agent_prev[a.handle] == a.position:
                    # find the position to introduce delay due to malfunction
                    step = [i for i, cell in enumerate(self.cells_sequence[a.handle]) if np.all(cell == a.old_position)][0]
                    self.malfunction_time[a.handle][step] += 1
                self.agent_prev[a.handle] = a.position
        """

        if self.base_graph is None:
            self.build_global_graph()

        self.prediction_dict = self.predictor.get()
        local = self.predictor.compute_cells_sequence(self.prediction_dict)
        if self.cells_sequence is None:
            self.cells_sequence = local

            #self.prediction_dict = self.predictor.get()
            #self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
            self.cells_sequence = local

            self.time_at_cell = defaultdict(list)
            self.malfunction_time = defaultdict(list)
            self.introduced_delay_time = defaultdict(list)

            for a in self.cells_sequence:
                step_size = 1 / self.env.agents[a].speed_data['speed']
                self.time_at_cell[a] = [step_size]*len(self.cells_sequence[a])
                self.malfunction_time[a] = [0]*len(self.cells_sequence[a])
                self.introduced_delay_time[a] = [0]*len(self.cells_sequence[a])

            self.max_prediction_depth = self.predictor.max_depth


        # check for malfunction in the beginning;
        # Since this is not intoduced delay but happens due to many trains starting at the same point,
        # it has to be noted in some form
        for a in self.env.agents:
            if a.status == RailAgentStatus.READY_TO_DEPART and self.ts >= 0:
                self.malfunction_time[a.handle][0] +=1


        #self.align_lapse()


    def populate_graph(self, observations):
        """
        Inherited method used for pre computations.

        :return:
        """

        #self.align_lapse()

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

            while True:

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
                        cell_difference = [num for num, item in enumerate(agent_trajectory)
                                     if item[0] == agent_edge.B[0] and item[1] == agent_edge.B[1]][0]-agent_pos_on_traj
                    elif agent_trajectory[-2] in agent_edge.Cells:
                        cell_difference = [num for num, item in enumerate(agent_edge.Cells)
                                     if item[0] == agent_trajectory[-2][0] and item[1] == agent_trajectory[-2][1]][0]
                    else:
                        print("Bug : because the trajectory and cell ends do not match while populating graph \n",
                              agent_pos_on_edge, "\n", agent_edge.B,"\n", agent_trajectory[-2],"\n", agent_edge.Cells,
                              "\n",agent_trajectory,"\n")

                        for edge in observations.edge_ids:
                            print(edge.Cells)

                        return observations
                        exit(0)
                # or the end of the trajectory has to be on the edge
                elif agent_dir_on_edge == 1:
                    next_node = agent_edge.A_node
                    if agent_edge.A in agent_trajectory:
                        cell_difference = [num for num, item in enumerate(agent_trajectory)
                                     if item[0] == agent_edge.A[0] and item[1] == agent_edge.A[1]][0]-agent_pos_on_traj
                    elif agent_trajectory[-2] in agent_edge.Cells:
                        cell_difference = [num for num, item in enumerate(agent_edge.Cells[::-1])
                                     if item[0] == agent_trajectory[-2][0] and item[1] == agent_trajectory[-2][1]][0]
                    else:
                        for edge in observations.edge_ids:
                            print(edge.Cells)

                        return observations
                        exit(0)

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

                agent_pos_on_traj += cell_difference
                agent_edge = None
                if agent_pos_on_traj != len(agent_trajectory):
                    for next_edge in next_node.edges:
                        if next_edge is not None:
                            if agent_trajectory[agent_pos_on_traj+1] in next_edge.Cells:
                                agent_edge = next_edge
                # now update agent_edge
                # agent_current_pos
                agent_current_pos = agent_trajectory[agent_pos_on_traj]

                start_timestamp = end_timestamp
                # or end of trajectory should be within the edge

                if agent_edge is None:
                    break

        observations.setCosts()
        observations = copy.deepcopy(observations)

        return observations

        """
        print()
        for a in range(self.env.number_of_agents):

            # manipulate trajectory for the agent to be modified
            traj = copy.deepcopy(self.cells_sequence[a])

            # timestamp of execution environment
            # set at the time when agent enters the edge
            start_timestamp = 0
            #
            traj_pos_end_counter = 0

            while len(traj) > 1:
                for edge in observations.edge_ids:
                    if traj[0] in edge.Cells \
                            and traj[1] in edge.Cells:

                        agent_pos_on_edge = [i for i, tupl in enumerate(edge.Cells)
                                    if tupl[0] == traj[0][0] and tupl[1] == traj[0][1]][0]

                        # step_dir is the value is either 1 or -1 being the step in either direction of the edge
                        if 0 < agent_pos_on_edge < len(edge.Cells) - 1:
                            if traj[1] == edge.Cells[agent_pos_on_edge - 1]:
                                step_dir = -1
                            elif traj[1] == edge.Cells[agent_pos_on_edge + 1]:
                                step_dir = 1
                        elif 0 == agent_pos_on_edge:
                            step_dir = 1
                        elif agent_pos_on_edge == len(edge.Cells) - 1:
                            step_dir = -1

                        edge_cells = edge.Cells[agent_pos_on_edge:len(edge.Cells)] if step_dir == 1 else edge.Cells[0:agent_pos_on_edge+1][::-1]

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

                            if len(traj) > traj_pos_end and len(edge_cells) > traj_pos_end:
                                touched = True

                                # if the current positions are equal then proceed in trajectory
                                if traj[traj_pos_end] == edge_cells[traj_pos_end]:
                                    traj_pos_end += 1
                                else:
                                    break
                            else:
                                break

                        # so a section is found that has part of desired trajectory
                        # we have the agent ID
                        # we have the direction,
                        # we can find the relevant order for start and end for direction
                        # we can also find the number of time steps (number of cells * 1/speed)
                        if touched:
                            traj_pos_end -= 1
                            steps = np.sum(self.time_at_cell[a][int(traj_pos_end_counter) \
                                                :int(traj_pos_end_counter)+traj_pos_end]) \
                                        + np.sum(self.malfunction_time[a][int(traj_pos_end_counter) \
                                                :int(traj_pos_end_counter)+traj_pos_end]) \
                                        + np.sum(self.introduced_delay_time[a][int(traj_pos_end_counter) \
                                                :int(traj_pos_end_counter)+traj_pos_end])

                            end_timestamp = int(start_timestamp + steps * 1 / self.env.agents[a].speed_data['speed'])

                            if start_timestamp != end_timestamp:
                                edge.Trains.append(a)
                                edge.TrainsTime.append(sorted([start_timestamp, end_timestamp]))
                                edge.TrainsDir.append(0 if step_dir == 1 else 1)
                                start_timestamp = end_timestamp

                        traj_pos_end_counter += traj_pos_end
                        break

                traj = traj[traj_pos_end:]

        # Now Build the the collision lock matrix
        observations.setCosts()
        self.cur_graph = copy.deepcopy(observations)
        """

    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.

        :param direction:
        :return:
        """
        return int((direction + 2) % 4)

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

            if current == tuple((14,6)) or current == tuple((22,6)):
                print("Here")

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

    def get_action_dict(self, observations):
        """
        Takes an agent handle and returns next action for that agent following shortest path:
        - if agent status == READY_TO_DEPART => agent moves forward;
        - if agent status == ACTIVE => pick action using shortest_path.py() fun available in prediction utils;
        - if agent status == DONE => agent does nothing.
        :param ts: timestep at which teh action is to be looked up from graph for all agents
        :return:
        """
        actions = defaultdict()
        cur_pos = defaultdict()

        for a in self.env.agents:
            if a.status == RailAgentStatus.ACTIVE or a.status == RailAgentStatus.READY_TO_DEPART:
                cur_pos[a.handle] = (a.position if a.position is not None else a.initial_position)

        next_pos = defaultdict()
        for a in cur_pos:
            agent_pos_on_traj = [num for num, item in enumerate(self.cells_sequence[a])
                                 if item[0] == cur_pos[a][0] and item[1] == cur_pos[a][1]][0]

            if agent_pos_on_traj < len(self.cells_sequence[a]):
                next_pos[a] = [cur_pos[a], self.cells_sequence[a][agent_pos_on_traj+1]]

        candidate_edges = [edge for edge in observations.edge_ids if len(edge.Trains) > 0]

        for edge in candidate_edges:
            for a in next_pos:
                if a in edge.Trains:
                    if next_pos[a][0] in edge.Cells and next_pos[a][1] in edge.Cells:

                        cur_position = next_pos[a][0]
                        next_position = next_pos[a][1]

                        id = [num for num, item in enumerate(edge.Trains) if item == a][0]

                        # now either the train is leaving
                        if (next_pos[a][1] == edge.Cells[0] or next_pos[a][1] == edge.Cells[-1]) and self.ts <= edge.TrainsTime[id][1]+1:
                            actions[a] = 4
                        # or it is entering or normally travelling
                        else:

                            cur_direction = self.env.agents[a].direction
                            if cur_position[0] == next_position[0]:
                                if cur_position[1] > next_position[1]:
                                    next_direction = 3
                                elif cur_position[1] < next_position[1]:
                                    next_direction = 1
                                else:
                                    next_direction = cur_direction
                            elif cur_position[1] == next_position[1]:
                                if cur_position[0] > next_position[0]:
                                    next_direction = 0
                                elif cur_position[0] < next_position[0]:
                                    next_direction = 2
                                else:
                                    next_direction = cur_direction

                            if (cur_direction + 1) % 4 == next_direction:
                                actions[a] = 3
                            elif (cur_direction - 1) % 4 == next_direction:
                                actions[a] = 1
                            elif next_direction == cur_direction:
                                actions[a] = 2
                            else:
                                print("Bug")

        print(actions)
        return actions

    def find_non_link(self, observations, collision_entry_point, agent_id):
        """
        find timestamp of last section where train can be stopped without blocking multiple sections

        :param collision_entry_point:
        :param agent_id:
        :return: ts
        """
        agent_pos = [num for num, item in enumerate(self.cells_sequence[agent_id])
                     if item[0] == collision_entry_point[0] and item[1] == collision_entry_point[1]][0]

        while True:
            if agent_pos-1 >= 1:
                prev = self.cells_sequence[agent_id][agent_pos-1]
                prev_prev = self.cells_sequence[agent_id][agent_pos-2]

            elif agent_pos-1 >= 0:
                prev = self.cells_sequence[agent_id][agent_pos-1]
                prev_prev = prev

            for edge in observations.edge_ids:
                if prev in edge.Cells and prev_prev in edge.Cells and len(edge.Cells) > 2:
                    return agent_pos-1
                elif prev in edge.Cells and prev_prev in edge.Cells and len(edge.Cells) == 2:
                    agent_pos -= 2
                    break

        return ""



    def optimize(self, observations):
        """

        :return: List of observations containing resulting graph after each optimization step
        """

        check_again = True
        check_again_counter = 0

        while check_again and check_again_counter < 50:
            # check if the cost is within limits
            check_again = False
            check_again_counter += 1

            sorted_list = sorted(observations.edge_ids, key=lambda x: x.CostTotal, reverse=True)
            optimization_candidate = [edge for edge in sorted_list if edge.CostTotal > 10000]

            for edge in optimization_candidate:

                edge_copy = copy.deepcopy(edge.CostPerTrain)
                trains_count = len(edge.CostPerTrain)
                opt_try = 0

                while True:

                    check_again = True

                    # find the train to be stopped
                    id = np.argmax(edge_copy)
                    agent_id = edge.Trains[id]

                    # found the edge and the train for collision
                    collision_entry_point = edge.Cells[0] if edge.TrainsDir[id] == 0 else edge.Cells[-1]

                    # find a cell in the trajectory which is not a two point link to clear the region.
                    cell_seq_safe = self.find_non_link(observations, collision_entry_point, agent_id)

                    # but also note the imeediate one
                    cell_seq_current = [num for num, item in enumerate(self.cells_sequence[agent_id])
                                    if item[0] == collision_entry_point[0] and item[1] == collision_entry_point[1]][0]-1

                    agent_cur_pos = self.env.agents[agent_id].position if self.env.agents[agent_id].position is not None \
                                    else self.env.agents[agent_id].initial_position if self.env.agents[agent_id].status \
                                    is not RailAgentStatus.DONE_REMOVED else self.env.agents[agent_id].target

                    agent_cell_seq_current = [num for num, item in enumerate(self.cells_sequence[agent_id]) \
                                                                             if item[0] == agent_cur_pos[0] and \
                                                                             item[1] == agent_cur_pos[1]][0]

                    if agent_cell_seq_current <= cell_seq_current:

                        # stop all the trains in this group
                        time = edge.TrainsTime[id][0]
                        dir_opp = 1 if edge.TrainsDir[id] == 0 else 0
                        end_time = [item for num, item in enumerate(edge.TrainsTime) if edge.TrainsDir[num] == dir_opp ]

                        bitmap = np.zeros((len(self.env.agents), self.max_prediction_depth),dtype=np.uint8)

                        for num, item in enumerate(edge.TrainsTime):
                            if num != id:
                                try:
                                    bitmap[num][item[0]:item[1]+1] = np.ones((item[1]-item[0]+1))#[1]*(item[1]-item[0]+1)
                                except:
                                    print(item, item[1]-item[0]+1)

                        occupancy = np.sum(bitmap,axis=0)

                        for number in range(edge.TrainsTime[id][0], self.max_prediction_depth \
                                                 - edge.TrainsTime[id][1]-edge.TrainsTime[id][0]+1):
                            if np.all(occupancy[number:number+edge.TrainsTime[id][1]-edge.TrainsTime[id][0]+1] == 0):
                                self.introduced_delay_time[agent_id][cell_seq_current] = number - edge.TrainsTime[id][0] +1
                                break

                        break

                    elif opt_try < trains_count:
                        edge_copy[id] = 100
                    else:
                        #print("No agent on this edge can be stopped safely")
                        break

                    opt_try += 1


            # make a copy of base_graph for reusability
            observations = copy.deepcopy(self.base_graph)
            observations = self.populate_graph(observations)

        return observations