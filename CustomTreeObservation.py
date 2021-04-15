from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
import numpy as np
import collections
from typing import Optional, List, Dict, Tuple
from flatland.utils.ordered_set import OrderedSet

Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                      'dist_other_target_encountered '
                                      'dist_other_agent_encountered '
                                      'dist_potential_conflict '
                                      'dist_unusable_switch '
                                      'dist_to_next_branch '
                                      'dist_min_to_target '
                                      'num_agents_same_direction '
                                      'num_agents_opposite_direction '
                                      'num_agents_malfunctioning '
                                      'speed_min_fractional '
                                      'num_agents_ready_to_depart '
                                      'childs '
                                      'start '
                                      'end '
                                      'cells '
                                      'dir '
                                      'safe_zones '
                                      'tl_pos '
                                      'other_target_location '
                                      'other_agent_location '
                                      'location_of_potential_conflict '
                                      'location_of_agents_in_same_direction '
                                      'location_of_agents_in_opposite_direction '
                                      'location_of_agents_malfunctioning '
                                      'AB_future')


class collections_alt(Node):

    def __repr__(self):
        return 'Node' + str(self.tl_pos if self.tl_pos is not None else '--')

    def __str__(self):
        return 'Node' + str(self.tl_pos if self.tl_pos is not None else '--')


def find_new_direction(direction, action):  # returns new direction after taking action
    new_direction = direction
    if action == 1:
        new_direction = direction - 1


    elif action == 3:
        new_direction = direction + 1

    new_direction %= 4

    return new_direction


class CustomTreeObservation(TreeObsForRailEnv):
    safe_map = {}
    location_and_dir_map = {}
    branches = set()

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__(max_depth, predictor)
        self.obs_dict = {}
        self.time_steps = -1

    def add_to_safe_map(self, pos, dir, handle):
        if pos == (9, 15):
            print('here')
        if pos not in self.safe_map:
            self.safe_map[pos] = []
        if (pos, dir) not in self.location_and_dir_map:
            self.location_and_dir_map[(pos, dir)] = []
        if dir not in self.safe_map[pos]:
            self.safe_map[pos].append(dir)
        if handle not in self.location_and_dir_map[(pos, dir)]:
            self.location_and_dir_map[(pos, dir)].append(handle)

    def find_safe_edges(self, env):
        for idx, agent in enumerate(env.agents):

            current_pos = agent.position
            current_dir = agent.direction
            if current_pos is None:
                current_pos = agent.initial_position

            self.add_to_safe_map(current_pos, current_dir, agent)
            while not current_pos == agent.target:

                possible_transitions = env.rail.get_transitions(*current_pos, current_dir)
                min_distances = []
                min_distance = None

                for direction in [(current_dir + i) % 4 for i in range(-1, 2)]:

                    if possible_transitions[direction]:
                        new_position = get_new_position(current_pos, direction)
                        min_distances.append(env.distance_map.get()[idx, new_position[0], new_position[1], direction])
                        if min_distance is None or min_distance > min_distances[-1]:
                            min_distance = min_distances[-1]
                            next_pos = new_position
                    else:
                        min_distances.append(np.inf)

                transition_bit = bin(self.env.rail.get_full_transitions(*current_pos))
                total_transitions = transition_bit.count("1")
                if total_transitions == 4:
                    self.branches.add(current_pos)
                # if len([x for x in min_distances if x != np.inf]) > 1:
                #     self.branches.add(current_pos)
                # else:
                #     print("")
                current_dir = find_new_direction(current_dir, np.argmin(min_distances) + 1)
                current_pos = next_pos
                self.add_to_safe_map(current_pos, current_dir, agent)

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get(self, handle):
        return super().get(handle)

    def get(self, handle: int = 0) -> collections_alt:

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        root_node_observation = collections_alt(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                                other_target_location=None,
                                                dist_other_agent_encountered=0, other_agent_location=None,
                                                dist_potential_conflict=0, location_of_potential_conflict=None,
                                                dist_unusable_switch=0, dist_to_next_branch=0,
                                                dist_min_to_target=distance_map[
                                                    (handle, *agent_virtual_position,
                                                     agent.direction)],
                                                num_agents_same_direction=0, location_of_agents_in_same_direction=[],
                                                num_agents_opposite_direction=0,
                                                location_of_agents_in_opposite_direction=[],
                                                num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                                location_of_agents_malfunctioning=None,
                                                speed_min_fractional=agent.speed_data['speed'],
                                                num_agents_ready_to_depart=0,
                                                childs={}, start=None, end=None, cells=[], dir=None, safe_zones=None,
                                                tl_pos=None, AB_future=None)

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, agent_virtual_position, branch_direction, orientation, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, collections_alt]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)
        self.time_steps+=1
        for h in observations.keys():
            if (observations[h] is not None and min(observations[h].childs['F'][4],
                                                    observations[h].childs['F'][5]) == 1) and \
                    (h not in self.obs_dict or observations[h].childs['F'].tl_pos != self.obs_dict[h][-1][1][h].childs[
                        'F'].tl_pos):
                if h not in self.obs_dict:
                    self.obs_dict[h] = []
                self.obs_dict[h].append((self.time_steps, observations.copy()))

        return observations

    def _explore_branch(self, handle, position, tl_pos, direction, tl_dir, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """
        original_pos = position
        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0
        safe_zones = []
        cells = []
        encountered_agent_loc = None
        other_target_location = None
        other_agent_location = None
        location_of_potential_conflict = None
        location_of_agents_in_same_direction = []

        if tl_pos in self.location_has_agent and self.location_has_agent_direction[tl_pos] == tl_dir:
            location_of_agents_in_same_direction.append(tl_pos)

        location_of_agents_in_opposite_direction = []
        if tl_pos in self.location_has_agent and self.location_has_agent_direction[tl_pos] == (tl_dir+2)%4:
            location_of_agents_in_opposite_direction.append(tl_pos)

        location_of_agents_malfunctioning = None
        AB_future = []
        # dirs = []

        while exploring:
            # ############################# ############################# Modify here to compute any useful data
            # required to build the end node's features. This code is called for each cell visited between the
            # previous branching node and the next switch / target / dead-end.('Node', 'dist_own_target_encountered '
            # 'dist_other_target_encountered ' 'other_target_location ' 'dist_other_agent_encountered '
            # 'other_agent_location' 'dist_potential_conflict ' 'location_of_potential_conflict '
            # 'dist_unusable_switch ' 'dist_to_next_branch ' 'dist_min_to_target ' 'num_agents_same_direction '
            # 'location_of_agents_in_same_direction ' 'num_agents_opposite_direction '
            # 'location_of_agents_in_opposite_direction ' 'num_agents_malfunctioning '
            # 'location_of_agents_malfunctioning ' 'speed_min_fractional ' 'num_agents_ready_to_depart ' 'childs '
            # 'start ' 'end ' 'cells ' 'dir ' 'safe_zones ' 'tl_pos')
            cells.append(position)
            # if direction not in dirs:
            #     dirs.append(direction)
            #     if len(dirs) >1:
            #         print('')
            if position in self.location_has_agent:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist
                    encountered_agent_loc = position

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]
                    location_of_agents_malfunctioning = position

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1
                    location_of_agents_in_same_direction.append(position)

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    # Attention this counts to many agents as a few might be going off on a switch.
                    other_agent_opposite_direction += self.location_has_agent[position]
                    location_of_agents_in_opposite_direction.append(position)

                # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                    and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                    and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                    self.predicted_dir[post_step][ca])] == 1 \
                                    and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                location_of_potential_conflict = position

            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist
                    other_target_location = position

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    if position not in self.safe_map or len(self.safe_map[position]) == 1:
                        safe_zones.append(position)
                    if tl_pos == (9, 14):
                        print('', end='')
                    if (position, direction) in self.location_and_dir_map and (
                            len(self.location_and_dir_map[(position, direction)]) > 1 or handle not in
                            self.location_and_dir_map[(position, direction)]):
                        AB_future.append(position)
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)

                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        dir = 1 if original_pos[1] < position[1] or (
                original_pos[1] == position[1] and original_pos[0] < position[0]) else 0
        node = collections_alt(dist_own_target_encountered=own_target_encountered,
                               dist_other_target_encountered=other_target_encountered,
                               dist_other_agent_encountered=other_agent_encountered,
                               dist_potential_conflict=potential_conflict,
                               dist_unusable_switch=unusable_switch,
                               dist_to_next_branch=dist_to_next_branch,
                               dist_min_to_target=dist_min_to_target,
                               num_agents_same_direction=other_agent_same_direction,
                               num_agents_opposite_direction=other_agent_opposite_direction,
                               num_agents_malfunctioning=malfunctioning_agent,
                               speed_min_fractional=min_fractional_speed,
                               num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                               other_target_location=other_target_location,
                               other_agent_location=other_agent_location,
                               location_of_potential_conflict=location_of_potential_conflict,
                               location_of_agents_in_same_direction=location_of_agents_in_same_direction,
                               location_of_agents_in_opposite_direction=location_of_agents_in_opposite_direction,
                               location_of_agents_malfunctioning=location_of_agents_malfunctioning,
                               childs={},
                               start=original_pos,
                               end=position,
                               cells=cells,
                               dir=dir,
                               safe_zones=safe_zones,
                               tl_pos=tl_pos,
                               AB_future=AB_future)

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          position,
                                                                          (branch_direction + 2) % 4,
                                                                          direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          position,
                                                                          branch_direction,
                                                                          direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited
