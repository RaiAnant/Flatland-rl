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

    Node = collections.namedtuple('Node',
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target')  # Whether agent's target is in this cell


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

    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles:
        :return:
        """

        if self.base_graph == None:
            print(self.build_global_graph())

        #temp = self._bfs_graph()
        #temp_1 = self._bfs_graph_1()
        self.num_active_agents = 0
        for a in self.env.agents:
            if a.status == RailAgentStatus.ACTIVE:
                self.num_active_agents += 1
        self.prediction_dict = self.predictor.get()
        # Useful to check if occupancy is correctly computed
        self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)

        if self.prediction_dict:
            self.max_prediction_depth = self.predictor.max_depth
            for t in range(self.max_prediction_depth):
                pos_list = []
                dir_list = []
                for a in handles:
                    if self.prediction_dict[a] is None:
                        continue
                    pos_list.append(self.prediction_dict[a][t][1:3])
                    dir_list.append(self.prediction_dict[a][t][3])
                self.predicted_pos_coord.update({t: pos_list})
                self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                self.predicted_dir.update({t: dir_list})

            for a in range(len(self.env.agents)):
                pos_list = []
                for ts in range(self.max_prediction_depth):
                    pos_list.append(self.predicted_pos[ts][a])  # Use int positions
                self.predicted_pos_list.update({a: pos_list})
        """
        observations = {}
        for a in handles:
            # TODO: Instead of below existing get()
            # make another function like first few lines of get()
            #               (get() checks for conflict in position,
            #               then by the virtue of direction of agents
            #               it finds out if there is a conflict in time)
            #
            # Hence, to find the time overlap of any other agent with the current agent under consideration
            #       Get     The cell ID's where conflict happens
            #               The ID's of the conflicting agent
            #               and it's Direction
            #

            #conflicting_agents, overlapping_paths_cellID = self.get_1(a)

            # make a graph with vertices and edges
            #
            # vertex is an object with
            #           cell coordinates str('x,y') as ID
            #           A list of edges connected to it
            #
            # edge is an object with
            #           start as the source vertex
            #           end as target vertex in a direct graph
            #           cost_triple (structure is list of triples) [conflicting_agent_id, direction, timestamp of conflict]
            #              (The above triple will be added based on the fact that when the map is traversed
            #               to generate the egocentric view for an agent, the cell coordinates are available.
            #               If this coordinate is in the conflict coordinate then the corresponding edge will have
            #               the value of triple added with relevant information)
            #
            # the object graph maintains a list of vertices in the graph
            #           given any vertex as a source vertex in this graph
            #           one can find the edges connected to it
            #           given an edge one can find the cost triple
            #
            # Hence, return this graph as the observation
            observations[a] = self.get(a)
        """

        #cell_seq = []
        #for a in range(self.env.number_of_agents):
        #    cell_seq.append(self.cells_sequence[a])

        # make a copy of base_graph for reusability
        # ************************************
        # copy construction for the object
        observations = copy.deepcopy(self.base_graph)
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
            traj = copy.deepcopy(self.cells_sequence[a][:-2])


            start_timestamp = 0
            while len(traj) > 1:
                #print(traj)
                for edge in observations.edge_ids:
                    #print(edge.Cells)
                    if traj[0] in edge.Cells and traj[1] in edge.Cells:
                        #pos = np.where(traj[0] == edge[2])
                        #pos = list(itertools.takewhile(lambda x: x[0] == traj[0][0] and x[1] == traj[0][1], edge[2]))
                        pos = [i for i, tupl in enumerate(edge.Cells) if tupl[0] == traj[0][0] and tupl[1] == traj[0][1]][0]

                        # found but now decide if the edge coordinates match on left or right
                        if 0 < pos < len(edge.Cells)-1:
                            if traj[1] == edge.Cells[pos-1]:
                                end_pos = 0
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[-1]
                            elif traj[1] == edge.Cells[pos+1]:
                                end_pos = len(edge.Cells)-1
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[0]
                        elif 0 == pos:
                            if traj[1] == edge.Cells[pos+1]:
                                end_pos = len(edge.Cells)-1
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[0]
                            else:
                                end_pos = 0
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[-1]
                        elif pos == len(edge.Cells)-1:
                            if traj[1] == edge.Cells[pos-1]:
                                end_pos = 0
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[-1]
                            else:
                                end_pos = len(edge.Cells)-1
                                end_id = edge.Cells[end_pos]
                                start_id = edge.Cells[0]

                        # so a section is found that has part of desired trajectory
                        # we have the agent ID
                        #print(a)

                        # we have the direction,
                        #print(start_id, end_id)
                        # we can find the relevant order for start and end for direction
                        # we can also find the number of time steps (number of cells * 1/speed)

                        end_timestamp = int(abs(pos - end_pos) * 1/self.env.agents[a].speed_data['speed'])
                        #print(start_timestamp, end_timestamp)

                        traj = traj[abs(pos - end_pos):]


                        edge.Trains.append(a)
                        edge.TrainsTime.append(sorted([start_timestamp,end_timestamp]))
                        if (str(start_id[0])+","+str(start_id[1]) == edge.A or str(end_id[0])+","+str(end_id[1]) == edge.B):
                            edge.TrainsDir.append(0)
                        elif (str(start_id[0])+","+str(start_id[1]) == edge.B or str(end_id[0])+","+str(end_id[1]) == edge.A):
                            edge.TrainsDir.append(1)

                        #edge[3].append([a,[start_id, end_id],[start_timestamp,end_timestamp]])
                        #edge.Triples.append([a,[start_id, end_id],[start_timestamp,end_timestamp]])
                        #print("Here \n \n")

                        start_timestamp = end_timestamp
                        #print("cost", edge.CostCollisionLockTotal)
                        break

        # Now Build the the collision lock matrix
        for edge in observations.edge_ids:
            edge.setCosts()

        cost = 0
        for edge in observations.edge_ids:
            cost += edge.CostTotal


        print("Total cost of the Graph", cost)

        return observations


    # TODO Optimize considering that I don't need obs for those agents who don't have to pick actions
    def get(self, handle: int = 0) -> {}:
        """
        Returns obs for one agent, obs are a single array of concatenated values representing:
        - occupancy of next prediction_depth cells,
        - agent priority/speed,
        - number of malfunctioning agents (encountered),
        - number of agents that are ready to depart (encountered).
        :param handle:
        :return:
        """

        agents = self.env.agents
        agent = agents[handle]

        # Occupancy
        occupancy, conflicting_agents = self._fill_occupancy(handle)
        # Augment occupancy with another one-hot encoded layer: 1 if this cell is overlapping and the conflict span was already entered by some other agent
        second_layer = np.zeros(self.max_prediction_depth, dtype=int) # Same size as occupancy
        for ca in conflicting_agents:
            if ca != handle:
                # Find ts when conflict occurred
                ts = [x for x, y in enumerate(self.cells_sequence[handle]) if y[0] == agents[ca].position[0] and y[1] == agents[ca].position[1]] # Find index/ts for conflict
                # Set to 1 conflict span which was already entered by some agent - fill left side and right side of ts
                if len(ts) > 0:
                    i = ts[0] # Since the previous returns a list of ts
                    while 0 <= i < self.max_prediction_depth:
                        second_layer[i] = 1 if occupancy[i] > 0 else 0
                        i -= 1
                    i = ts[0]
                    while i < self.max_prediction_depth:
                        second_layer[i] = 1 if occupancy[i] > 0 else 0
                        i += 1

        occupancy = np.append(occupancy, second_layer)

        #print('Agent {}'.format(handle))
        #print('Occupancy, first layer: {}'.format(occupancy))
        #print('Occupancy, second layer: {}'.format(second_layer))

        # Bifurcation points, one-hot encoded layer of predicted cells where 1 means that this cell is a fork
        # (globally - considering cell transitions not depending on agent orientation)
        forks = np.zeros(self.max_prediction_depth, dtype=int)
        # Target
        target = np.zeros(self.max_prediction_depth, dtype=int)
        for index in range(self.max_prediction_depth):
            # Fill as 1 if transitions represent a fork cell
            cell = self.cells_sequence[handle][index]
            if cell in self.forks_coords:
                forks[index] = 1
            if cell == agent.target:
                target[index] = 1

        # print('Forks: {}'.format(forks))
        # print('Target: {}'.format(target))

        #  Speed/priority
        is_conflict = True if len(conflicting_agents) > 0 else False
        priority = assign_priority(self.env, agent, is_conflict)
        max_prio_encountered = 0
        if is_conflict:
            conflicting_agents_priorities = [assign_priority(self.env, agents[ca], True) for ca in conflicting_agents]
            max_prio_encountered = np.min(conflicting_agents_priorities)  # Max prio is the one with lowest value

        #print('Priority: {}'.format(priority))
        #print('Max priority encountered: {}'.format(max_prio_encountered))

        # Malfunctioning obs
        # Counting number of agents that are currently malfunctioning (globally) - experimental
        n_agents_malfunctioning = 0  # in TreeObs they store the length of the longest malfunction encountered
        for a in agents:
            if a.malfunction_data['malfunction'] != 0:
                n_agents_malfunctioning += 1  # Considering ALL agents

        #print('Num malfunctioning agents (globally): {}'.format(n_agents_malfunctioning))

        # Agents status (agents ready to depart) - it tells the agent how many will appear - encountered? or globally?
        n_agents_ready_to_depart = 0
        for a in agents:
            if a.status in [RailAgentStatus.READY_TO_DEPART]:
                n_agents_ready_to_depart += 1  # Considering ALL agents

        #print('Num agents ready to depart (globally): {}'.format(n_agents_ready_to_depart))
        # shape (prediction_depth * 4 + 4, )
        agent_obs = np.append(occupancy, forks)
        agent_obs = np.append(agent_obs, target)
        agent_obs = np.append(agent_obs, (priority, max_prio_encountered, n_agents_malfunctioning, n_agents_ready_to_depart))

        # With this obs the agent actually decides only if it has to move or stop



        # TODO: By Gaurav
        # After getting this observation for any number of steps in future for all the agents
        #
        # make an egocentric graph per agent
        # using above conflict info :
        #           mark on this egocentric graph over which edge is the conflict
        #           with which agent is the conflict
        #           at what time step is the conflict
        # return observation and graph both


        res = self._bfs_graph(handle)

        return agent_obs


    # TODO Stop when shortest_path() says that rail is disrupted
    def _get_shortest_path_action(self, handle):
        """
        Takes an agent handle and returns next action for that agent following shortest path:
        - if agent status == READY_TO_DEPART => agent moves forward;
        - if agent status == ACTIVE => pick action using shortest_path.py() fun available in prediction utils;
        - if agent status == DONE => agent does nothing.
        :param handle:
        :return:
        """

        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:

            if self.num_active_agents < 10:  # TODO
                # This could be reasonable since agents never start on switches - I guess
                action = RailEnvActions.MOVE_FORWARD
            else:
                action = RailEnvActions.DO_NOTHING


        elif agent.status == RailAgentStatus.ACTIVE:
            # This can return None when rails are disconnected or there was an error in the DistanceMap
            shortest_paths = self.predictor.get_shortest_paths()

            if shortest_paths[handle] is None:  # Railway disrupted
                action = RailEnvActions.STOP_MOVING
            else:
                step = shortest_paths[handle][0]
                next_action_element = step[2][0]  # Get next_action_element

                # Just to use the correct form/name
                if next_action_element == 1:
                    action = RailEnvActions.MOVE_LEFT
                elif next_action_element == 2:
                    action = RailEnvActions.MOVE_FORWARD
                elif next_action_element == 3:
                    action = RailEnvActions.MOVE_RIGHT

        else:  # If status == DONE or DONE_REMOVED
            action = RailEnvActions.DO_NOTHING

        return action


    def choose_railenv_action(self, handle, network_action):
        """
        Choose action to perform from RailEnvActions, namely follow shortest path or stop if DQN network said so.

        :param handle:
        :param network_action:
        :return:
        """

        if network_action == 1:
            return RailEnvActions.STOP_MOVING
        else:
            return self._get_shortest_path_action(handle)


    def _possible_conflict(self, handle, ts):
        """
        Function that given agent (as handle) and time step, returns a counter that represents the sum of possible conflicts with
        other agents at that time step.
        Possible conflict is computed considering time step (current, pre and stop), direction, and possibility to enter that cell
        in opposite direction (w.r.t. to current agent).
        Precondition: 0 <= ts <= self.max_prediction_depth - 1.
        Exclude READY_TO_DEPART agents from this count, namely, check conflicts only with agents that are already active.

        :param handle: agent id
        :param ts: time step
        :return occupancy_counter, conflicting_agents
        """
        occupancy_counter = 0
        cell_pos = self.predicted_pos_coord[ts][handle]
        int_pos = self.predicted_pos[ts][handle]
        pre_ts = max(0, ts - 1)
        post_ts = min(self.max_prediction_depth - 1, ts + 1)
        int_direction = int(self.predicted_dir[ts][handle])
        cell_transitions = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int_direction)
        conflicting_agents_ts = set()

        # Careful, int_pos, predicted_pos are not (y, x) but are given as int
        if int_pos in np.delete(self.predicted_pos[ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[pre_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[pre_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[pre_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[pre_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[post_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[post_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[post_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[post_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        return occupancy_counter, conflicting_agents_ts


    def _fill_occupancy(self, handle):
        """
        Returns encoding of agent occupancy as an array where each element is
        0: no other agent in this cell at this ts (free cell)
        >= 1: counter (probably) other agents here at the same ts, so conflict, e.g. if 1 => one possible conflict, 2 => 2 possible conflicts, etc.
        :param handle: agent id
        :return: occupancy, conflicting_agents
        """
        occupancy = np.zeros(self.max_prediction_depth, dtype=int)
        conflicting_agents = set()
        overlapping_paths = self._compute_overlapping_paths(handle) # Overlap in position but not in time
        # cells_sequence = self.cells_sequence[handle]
        # span_cells = []

        for ts in range(self.max_prediction_depth):
            if self.env.agents[handle].status in [RailAgentStatus.READY_TO_DEPART, RailAgentStatus.ACTIVE]:
                occupancy[ts], conflicting_agents_ts = self._possible_conflict(handle, ts)
                conflicting_agents.update(conflicting_agents_ts)

        # If a conflict is predicted, then it makes sense to populate occupancy with overlapping paths
        # But only with THAT agent
        # Because I could have overlapping paths but without conflict (TODO improve)
        if len(conflicting_agents) != 0: # If there was conflict
            for ca in conflicting_agents:
                for ts in range(self.max_prediction_depth):
                    occupancy[ts] = overlapping_paths[ca, ts] if occupancy[ts] == 0 else 1

        # Occupancy is 0 for agents that are done - they don't perform actions anymore

        # TODO: By Gaurav
        # also add id of the conflicting agent, its direction and timestamp of conflict

        return occupancy, conflicting_agents


    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.
        :param direction:
        :return:
        """
        return int((direction + 2) % 4)

    # More than overlapping paths, this function computes cells in common in the predictions
    def _compute_overlapping_paths(self, handle):
        """
        Function that checks overlapping paths, where paths take into account shortest path prediction, so time/speed,
        but not the fact that the agent is moving or not.
        :param handle: agent id
        :return: overlapping_paths is a np.array that computes path overlapping for pairs of agents, where 1 means overlapping.
        Each layer represents overlapping with one particular agent.
        """
        overlapping_paths = np.zeros((self.env.get_num_agents(), self.max_prediction_depth), dtype=int)
        cells_sequence = self.predicted_pos_list[handle]
        for a in range(len(self.env.agents)):
            if a != handle:
                i = 0
                other_agent_cells_sequence = self.predicted_pos_list[a]
                for pos in cells_sequence:
                    if pos in other_agent_cells_sequence:
                        overlapping_paths[a, i] = 1
                    i += 1
        return overlapping_paths


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


    def _is_following(self, handle1, handle2):
        """
        Checks whether one agent is (probably) following the other one.
        :param handle1:
        :param handle2:
        :return:
        """
        agent1 = self.env.agents[handle1]
        agent2 = self.env.agents[handle2]
        virtual_position1 = agent1.initial_position if agent1.status == RailAgentStatus.READY_TO_DEPART else agent1.position
        virtual_position2 = agent2.initial_position if agent2.status == RailAgentStatus.READY_TO_DEPART else agent2.position

        if agent1.initial_position == agent2.initial_position \
                and agent1.initial_direction == agent2.initial_direction \
                and agent1.target == agent2.target \
                and (abs(virtual_position1[0] - virtual_position2[0]) <= 2 or abs(virtual_position1[1] - virtual_position2[1]) <= 2):
                return True
        else:
            return False


    def _bfs_graph(self, handle: int = 0) -> {}:
        """
        Build a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
        (that are tuples that represent the cell position, eg (11, 23))
        :param handle: agent id
        :return:
        """

        obs_graph = defaultdict(list)  # Dict node (as pos) : adjacent nodes
        visited_nodes = set()  # set
        bfs_queue = []
        done = False  # True if agent has reached its target

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
            done = True
        else:
            return None

        agent_current_direction = agent.direction

        # Push root node into the queue
        root_node_obs = GraphObsForRailEnv.Node(cell_position=agent_virtual_position,
                                                agent_direction=agent_current_direction,
                                                is_target=done)
        bfs_queue.append(root_node_obs)

        # Perform BFS of depth = bfs_depth
        for i in range(1, self.bfs_depth + 1):
            # Temporary queue to store nodes that must be appended at the next pass
            tmp_queue = []
            while not len(bfs_queue) == 0:
                current_node = bfs_queue.pop(0)
                agent_position = current_node[0]

                # Init node in the obs_graph (if first time)
                if not agent_position in obs_graph.keys():
                    obs_graph[agent_position] = []

                agent_current_direction = current_node[1]
                # Get cell transitions given agent direction
                possible_transitions = self.env.rail.get_transitions(*agent_position, agent_current_direction)

                orientation = agent_current_direction
                possible_branch_directions = []
                # Build list of possible branching directions from cell
                for j, branch_direction in enumerate([(orientation + j) % 4 for j in range(-1, 3)]):
                    if possible_transitions[branch_direction]:
                        possible_branch_directions.append(branch_direction)
                for branch_direction in possible_branch_directions:
                    # Gets adjacent cell and start exploring from that for possible fork points
                    neighbour_cell = get_new_position(agent_position, branch_direction)
                    adj_node = self._explore_path(handle, neighbour_cell, branch_direction)
                    if not (*adj_node[0], adj_node[1]) in visited_nodes:
                        # For now I'm using as key the agent_position tuple
                        obs_graph[agent_position].append(adj_node)
                        visited_nodes.add((*adj_node[0], adj_node[1]))
                        tmp_queue.append(adj_node)
            # Add all the nodes of the next level to the BFS queue
            for el in tmp_queue:
                bfs_queue.append(el)

        # After the last pass add adj nodes to the obs graph wih empty lists
        for el in bfs_queue:
            if not el[0] in obs_graph.keys():
                obs_graph[el[0]] = []
                # visited_nodes.add((*el[0], el[1]))
        # For obs rendering
        # self.env.dev_obs_dict[handle] = [(node[0], node[1]) for node in visited_nodes]

        # Build graph with graph-tool library for visualization
        # g = build_graph(obs_graph, handle)

        return obs_graph


    def _bfs_graph_1(self, handle: int = 0) -> {}:
        """
        Build a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
        (that are tuples that represent the cell position, eg (11, 23))
        :param handle: agent id
        :return:
        """


        obs_graph = Graph()
        #obs_graph = defaultdict(list)  # Dict node (as pos) : adjacent nodes
        visited_nodes = set()  # set
        bfs_queue = []
        done = False  # True if agent has reached its target

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
            done = True
        else:
            return None

        agent_current_direction = agent.direction

        # after getting agent Virtual position
        next_node = self._explore_path(handle,agent_virtual_position,agent_current_direction)
        # Find the nearest node
        # start from there



        # Push root node into the queue
        root_node_obs = GraphObsForRailEnv.Node(cell_position=agent_virtual_position,
                                                agent_direction=agent_current_direction,
                                                is_target=done)
        bfs_queue.append(root_node_obs)

        # Perform BFS of depth = bfs_depth
        for i in range(1, self.bfs_depth + 1):
            # Temporary queue to store nodes that must be appended at the next pass
            tmp_queue = []
            while not len(bfs_queue) == 0:
                current_node = bfs_queue.pop(0)
                agent_position = current_node[0]

                # Init node in the obs_graph (if first time)
                if not agent_position in obs_graph.vert_dict.keys():
                    #obs_graph[agent_position] = []
                    obs_graph.add_vertex(agent_position)

                agent_current_direction = current_node[1]
                # Get cell transitions given agent direction
                possible_transitions = self.env.rail.get_transitions(*agent_position, agent_current_direction)

                orientation = agent_current_direction
                possible_branch_directions = []
                # Build list of possible branching directions from cell
                for j, branch_direction in enumerate([(orientation + j) % 4 for j in range(-1, 3)]):
                    if possible_transitions[branch_direction]:
                        possible_branch_directions.append(branch_direction)
                for branch_direction in possible_branch_directions:
                    # Gets adjacent cell and start exploring from that for possible fork points
                    neighbour_cell = get_new_position(agent_position, branch_direction)
                    adj_node = self._explore_path(handle, neighbour_cell, branch_direction)
                    if not (*adj_node[0], adj_node[1]) in visited_nodes:
                        # For now I'm using as key the agent_position tuple
                        #obs_graph[agent_position].append(adj_node)
                        obs_graph.add_edge(agent_position, adj_node.cell_position)
                        visited_nodes.add((*adj_node[0], adj_node[1]))
                        tmp_queue.append(adj_node)
            # Add all the nodes of the next level to the BFS queue
            for el in tmp_queue:
                bfs_queue.append(el)

        # After the last pass add adj nodes to the obs graph wih empty lists
        for el in bfs_queue:
            if not el[0] in obs_graph.keys():
                obs_graph[el[0]] = []
                # visited_nodes.add((*el[0], el[1]))
        # For obs rendering
        # self.env.dev_obs_dict[handle] = [(node[0], node[1]) for node in visited_nodes]

        # Build graph with graph-tool library for visualization
        # g = build_graph(obs_graph, handle)

        return obs_graph


    def _explore_path(self, handle, position, direction):
        """
        Given agent handle, current position, and direction, explore that path until a new branching point is found.
        :param handle: agent id
        :param position: agent position as cell
        :param direction: agent direction
        :return: a tuple Node with its features
        """

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        # 4 different cases to have a branching point:
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        last_is_target = False  # target was reached
        agent = self.env.agents[handle]
        visited = OrderedSet()

        while True:

            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            cell_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = np.count_nonzero(cell_transitions)
            cell_transitions_bitmap = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = cell_transitions_bitmap.count("1")

            if num_transitions == 1:
                # Check if dead-end (1111111111111111), or if we can go forward along direction
                if total_transitions == 1:
                    last_is_dead_end = True
                    break

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)

            elif num_transitions > 1:
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break
        # Out of while loop - a branching point was found

        # TODO Here to save more features in a node
        node = GraphObsForRailEnv.Node(cell_position=position,
                                       agent_direction=direction,
                                       is_target=last_is_target)

        return node


    # #################################### FPR GLOBAL GRAPH ##################################

    def _step_extend(self, current, direction):

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

    def build_global_graph(self):

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
                        #print("adding vertex ", str(item[0][0])+","+str(item[0][1]))
                        self.base_graph.add_vertex(str(item[0][0])+","+str(item[0][1]))
                        #print("adding edge between ", str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]))
                        #self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])
                        added_vertex.append(item[0])

                    elif item[1] > 2:
                        #print("adding vertex ", str(item[0][0])+","+str(item[0][1]))
                        self.base_graph.add_vertex(str(item[0][0])+","+str(item[0][1]))
                        #print("adding edge between ", str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]))
                        #self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])
                        added_vertex.append(item[0])
                        pending_to_explore.append(item[0])

                if item[1] == 1:
                    #print("adding edge between ", str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]))
                    self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])

                elif item[1] > 2:
                    #print("adding edge between ", str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]))
                    self.base_graph.add_edge(str(current[0])+","+str(current[1]), str(item[0][0])+","+str(item[0][1]), item[2])


        return "built base graph"

    # ########################################################################################
    # ####################################    NEW  CODE    ###################################

    def get_1(self, handle: int = 0) -> {}:
        print("Inside get_1()")

        conflicting_agents, overlapping_paths_cellID = self._fill_occupancy_1(handle)
        # the overlapping_paths_cellID is the data that means,
        #       if there is a non zero value in the 2d array,
        # it means that the current agent will conflict with the considered agent
        #       at the given position
        #       on the reletive Timestamp represented by index

        return conflicting_agents, overlapping_paths_cellID

    def _fill_occupancy_1(self, handle):
        """
        Returns encoding of agent occupancy as an array where each element is
        0: no other agent in this cell at this ts (free cell)
        >= 1: counter (probably) other agents here at the same ts, so conflict, e.g. if 1 => one possible conflict, 2 => 2 possible conflicts, etc.
        :param handle: agent id
        :return: occupancy, conflicting_agents
        """
        conflicting_agents = set()
        contributing_handles = []
        overlapping_paths, overlapping_paths_cellID, overall_paths_cellID \
            = self._compute_overlapping_paths_1(handle) # Overlap in position but not in time
        # cells_sequence = self.cells_sequence[handle]
        # span_cells = []

        for ts in range(self.max_prediction_depth):
            if self.env.agents[handle].status in [RailAgentStatus.READY_TO_DEPART, RailAgentStatus.ACTIVE]:
                _, conflicting_agents_ts = self._possible_conflict_1(handle, ts)
                contributing_handles.append(conflicting_agents_ts)
                conflicting_agents.update(conflicting_agents_ts)

        # If a conflict is predicted, then it makes sense to populate occupancy with overlapping paths
        # But only with THAT agent
        # Because I could have overlapping paths but without conflict (TODO improve)
        if len(conflicting_agents) != 0: # If there was conflict
            for number, item in enumerate(contributing_handles):
                if len(item) == 0:
                    overlapping_paths_cellID[:,number] = 0
                else:
                    for inner_item in range(overlapping_paths_cellID.shape[0]):
                        if inner_item not in item:
                            overlapping_paths_cellID[inner_item, number] = 0
        else:
            overlapping_paths_cellID = np.zeros(overlapping_paths_cellID.shape, dtype=np.uint8)

        # Occupancy is 0 for agents that are done - they don't perform actions anymore

        # TODO: By Gaurav
        # modify the overlapping_paths_cellID to make sure
        # only the agents in the conflicting_agents should have non zero values.
        # that too for the section that was marked in occupancy

        return conflicting_agents, overlapping_paths_cellID

    def _possible_conflict_1(self, handle, ts):
        """
        Function that given agent (as handle) and time step, returns a counter that represents the sum of possible conflicts with
        other agents at that time step.
        Possible conflict is computed considering time step (current, pre and stop), direction, and possibility to enter that cell
        in opposite direction (w.r.t. to current agent).
        Precondition: 0 <= ts <= self.max_prediction_depth - 1.
        Exclude READY_TO_DEPART agents from this count, namely, check conflicts only with agents that are already active.

        :param handle: agent id
        :param ts: time step
        :return occupancy_counter, conflicting_agents
        """
        occupancy_counter = 0
        cell_pos = self.predicted_pos_coord[ts][handle]
        int_pos = self.predicted_pos[ts][handle]
        pre_ts = max(0, ts - 1)
        post_ts = min(self.max_prediction_depth - 1, ts + 1)
        int_direction = int(self.predicted_dir[ts][handle])
        cell_transitions = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int_direction)
        conflicting_agents_ts = set()

        # Careful, int_pos, predicted_pos are not (y, x) but are given as int
        if int_pos in np.delete(self.predicted_pos[ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[pre_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[pre_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[pre_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[pre_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[post_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[post_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[post_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[post_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        return occupancy_counter, conflicting_agents_ts

    # More than overlapping paths, this function computes cells in common in the predictions
    def _compute_overlapping_paths_1(self, handle):
        """
        Function that checks overlapping paths, where paths take into account shortest path prediction, so time/speed,
        but not the fact that the agent is moving or not.
        :param handle: agent id
        :return: overlapping_paths is a np.array that computes path overlapping for pairs of agents, where 1 means overlapping.
        Each layer represents overlapping with one particular agent.
        """
        overlapping_paths = np.zeros((self.env.get_num_agents(), self.max_prediction_depth), dtype=int)
        overlapping_paths_cellID = np.zeros((self.env.get_num_agents(), self.max_prediction_depth), dtype=int)
        overall_paths_cellID = np.zeros((self.env.get_num_agents(), self.max_prediction_depth), dtype=int)

        cells_sequence = self.predicted_pos_list[handle]
        overall_paths_cellID[handle] = cells_sequence

        for a in range(len(self.env.agents)):
            if a != handle:
                i = 0
                other_agent_cells_sequence = self.predicted_pos_list[a]
                overall_paths_cellID[a] = other_agent_cells_sequence
                for pos in cells_sequence:
                    if pos in other_agent_cells_sequence:
                        overlapping_paths[a, i] = 1
                        overlapping_paths_cellID[a, i] = pos
                    i += 1
        return overlapping_paths, overlapping_paths_cellID, overall_paths_cellID

    # ########################################################################################

