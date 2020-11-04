import copy
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict
from itertools import groupby



def optimize_old_graph(observation_builder, observations):
    """

    :return: List of observations containing resulting graph after each optimization step
    """

    check_again = True
    check_again_counter = 0

    while check_again and check_again_counter < 100:
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

                #collision_entry_point =

                # find a cell in the trajectory which is not a two point link to clear the region.
                cell_seq_safe = find_non_link(observation_builder, observations, collision_entry_point, agent_id)

                # but also note the imeediate one
                cell_seq_current = [num for num, item in
                                    enumerate(observation_builder.cells_sequence[agent_id])
                                    if item[0] == collision_entry_point[0]
                                    and item[1] == collision_entry_point[1]][0]

                agent_cur_pos = observation_builder.env.agents[agent_id].position \
                    if observation_builder.env.agents[agent_id].position is not None \
                    else observation_builder.env.agents[agent_id].initial_position \
                    if observation_builder.env.agents[agent_id].status \
                    is not RailAgentStatus.DONE_REMOVED else \
                    observation_builder.env.agents[agent_id].target

                agent_cell_seq_current = [num for num, item in
                                          enumerate(observation_builder.cells_sequence[agent_id]) \
                                          if item[0] == agent_cur_pos[0] and \
                                          item[1] == agent_cur_pos[1]][0]

                if agent_cell_seq_current <= cell_seq_current:

                    # stop all the trains in this group
                    dir_opp = 1 if edge.TrainsDir[id] == 0 else 0

                    bitmap = np.zeros((len(observation_builder.env.agents),
                                       observation_builder.max_prediction_depth * 4),
                                      dtype=np.uint8)

                    for num, item in enumerate(edge.TrainsTime):
                        if num != id \
                                and edge.TrainsDir[id] != edge.TrainsDir[num]\
                                and observation_builder.env.agents[edge.Trains[num]].status is not \
                            RailAgentStatus.DONE_REMOVED:
                            try:
                                bitmap[num][item[0]:item[1] + 1] = np.ones(
                                    (item[1] - item[0] + 1))  # [1]*(item[1]-item[0]+1)
                            except:
                                print(item, item[1] - item[0] + 1)

                    occupancy = np.sum(bitmap, axis=0)

                    # we do not want a delay for the final edge if the number of matching cells in the trajectory is not just the final one
                    for number in range(edge.TrainsTime[id][0],
                                        observation_builder.max_prediction_depth - (edge.TrainsTime[id][1] - edge.TrainsTime[id][0]) + 1):
                        if np.all(occupancy[number:number + (edge.TrainsTime[id][1] - edge.TrainsTime[id][0]) + 1] == 0):
                            observation_builder.introduced_delay_time[agent_id][cell_seq_current-1] += number - edge.TrainsTime[id][0]
                            observation_builder.introduced_delay_time[agent_id][cell_seq_current:] = [0]*\
                                                                                (len(observation_builder.introduced_delay_time[agent_id])- \
                                                                                 cell_seq_current )
                            break
                    break

                elif opt_try < trains_count:
                    edge_copy[id] = 100
                else:
                    break

                opt_try += 1

        # make a copy of base_graph for reusability
        observations = copy.deepcopy(observation_builder.base_graph)
        observations = observation_builder.populate_graph(observations)

    return observations



def optimize(observation_builder, obs, node_type="edge"):
    """

    :return: List of observations containing resulting graph after each optimization step
    """

    #conflict_status =
    #observation_builder.setDeadLocks(obs)
    #obs.setCosts()

    #conflict_status = obs.Deadlocks
    #for

    # deadlocks cannot be optimized and hence should be populated straightforward
    # but the cost cannot change for the deadlocks even if they are resolved in the process

    max_CostTotalEnv = obs.CostTotalEnv
    check_again_counter = 0

    while True:
        # check if the cost is within limits

        observations = copy.deepcopy(obs)

        check_again_counter += 1

        sorted_list = sorted(observations.vertices, key=lambda x: observations.vertices[x].CostTotal, reverse=True)
        optimization_candidate = [vertex for vertex in sorted_list
                                  if observations.vertices[vertex].CostTotal > 10000
                                  and observations.vertices[vertex].Type == node_type]

        if not len(optimization_candidate):
            break

        for vert in optimization_candidate:

            observations = copy.deepcopy(obs)
            #trains = observations.vertices[vert].Trains
            inner_observation_structure = []

            # candidate_trains = []
            # candidate_trains_1 = [[id, agent_id] for id, agent_id in enumerate(observations.vertices[vert].Trains)
            #                     if observations.vertices[vert].DeadlockCostPerTrain[id] > 0 ]
            # candidate_trains_2 = [[id, agent_id] for id, agent_id in enumerate(observations.vertices[vert].Trains)
            #                     if observations.vertices[vert].CostPerTrain[id] > 10000]
            # if len(candidate_trains_1) and len(candidate_trains_2):
            #     candidate_trains = candidate_trains_1 + candidate_trains_2
            # elif len(candidate_trains_1):
            #     candidate_trains = candidate_trains_1
            # elif len(candidate_trains_2):
            #     candidate_trains = candidate_trains_2

            candidate_trains = [[id, agent_id]
                                for id, agent_id in enumerate(observations.vertices[vert].Trains)
                                if observations.vertices[vert].CostPerTrain[id] > 10000]

            #max_cost = np.max(observations.vertices[vert].CostPerTrain)

            for candidate in candidate_trains:

                id = candidate[0]
                agent_id = candidate[1]

                #if observations.vertices[vert].CostPerTrain[id] != max_cost:
                #    continue

                observations = copy.deepcopy(obs)
                vertex = observations.vertices[vert]

                if vertex.CostPerTrain[id] < 10000:
                    continue

                collision_entry_point = vertex.other_end(vertex.Links[vertex.TrainsDir[id]][0])

                # but also note the immediate one
                cell_seq_current = [num for num, item in
                                    enumerate(observation_builder.cells_sequence[agent_id])
                                    if item[0] == collision_entry_point[0]
                                    and item[1] == collision_entry_point[1]][0]

                agent_cur_pos = observation_builder.env.agents[agent_id].position \
                    if observation_builder.env.agents[agent_id].position is not None \
                    else observation_builder.env.agents[agent_id].initial_position \
                    if observation_builder.env.agents[agent_id].status \
                    is not RailAgentStatus.DONE_REMOVED else \
                    observation_builder.env.agents[agent_id].target

                agent_cell_seq_current = [num for num, item in
                                          enumerate(observation_builder.cells_sequence[agent_id]) \
                                          if item[0] == agent_cur_pos[0] and \
                                          item[1] == agent_cur_pos[1]][0]

                if agent_cell_seq_current <= cell_seq_current:

                    bitmap = np.zeros((len(observation_builder.env.agents),
                                       observation_builder.max_prediction_depth * 4),
                                      dtype=np.uint8)

                    for num, item in enumerate(vertex.TrainsTime):
                        if num != id \
                                and vertex.TrainsDir[id] != vertex.TrainsDir[num]\
                                and observation_builder.env.agents[vertex.Trains[num]].status is not \
                                RailAgentStatus.DONE_REMOVED:
                            try:
                                bitmap[num][item[0]:item[1] + 1] = np.ones(
                                    (item[1] - item[0] + 1))  # [1]*(item[1]-item[0]+1)
                            except:
                                print(item, item[1] - item[0] + 1)

                    occupancy = np.sum(bitmap, axis=0)

                    # we do not want a delay for the final edge if the number of matching cells in the trajectory is not just the final one
                    for number in range(vertex.TrainsTime[id][0],
                                        observation_builder.max_prediction_depth - (vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]) + 1):
                        if np.all(occupancy[number-1:number + (vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]) + 1] == 0):

                            if vertex.Type == "junction":
                                vertex.TrainsTime[id] = [number, number + 1]
                            elif vertex.Type == "edge":
                                vertex.TrainsTime[id] = [number, number + vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]]

                            observations = observation_builder.update_for_delay(observations, agent_id, node_type)
                            observations.setCosts()
                            inner_observation_structure.append([observations.CostTotalEnv, observations])

                            break


                    #if observations.CostTotalEnv < obs.CostTotalEnv:
                    #    obs = observations

            if len(inner_observation_structure) > 1:
                sorted_cost_list = sorted(inner_observation_structure, key=lambda x: x[0], reverse=False)
                obs = sorted_cost_list[0][1]
            elif len(inner_observation_structure) == 1:
                obs = inner_observation_structure[0][1]

        if obs.CostTotalEnv >= max_CostTotalEnv:
            break
        else:
            print(max_CostTotalEnv, obs.CostTotalEnv)
            max_CostTotalEnv = obs.CostTotalEnv
        """
            #min_cost = []
            if len(inner_observation_structure) > 1:
                sorted_cost_list = sorted(inner_observation_structure, key=lambda x: x[0], reverse=False)
                outer_observation_structure.append(sorted_cost_list[0])
            elif len(inner_observation_structure) == 1:
                outer_observation_structure.append(inner_observation_structure[0])
            #print("Here")

        if len(outer_observation_structure) > 1:
            sorted_outer_cost_list = sorted(inner_observation_structure, key=lambda x: x[0], reverse=False)
            obs = sorted_outer_cost_list[0][1]
        elif len(outer_observation_structure) == 1:
            obs = outer_observation_structure[0][1]
        """

        #obs = sorted(outer_observation_structure, key=lambda x: x[0], reverse=False)[0][1]
        #sorted_cost_list = sorted(inner_observation_structure, key=lambda x: x[0], reverse=False)
        #outermost_observation_structure.append([sorted_cost_list[0][1]])
        #print("Here")

    return obs




def find_non_link(observation_builder, observations, collision_entry_point, agent_id):
    """
    find timestamp of last section where train can be stopped without blocking multiple sections

    :param collision_entry_point:
    :param agent_id:
    :return: ts
    """
    agent_pos = [num for num, item in enumerate(observation_builder.cells_sequence[agent_id])
                 if item[0] == collision_entry_point[0] and item[1] == collision_entry_point[1]][0]

    while True:
        if agent_pos-1 >= 1:
            prev = observation_builder.cells_sequence[agent_id][agent_pos-1]
            prev_prev = observation_builder.cells_sequence[agent_id][agent_pos-2]

        elif agent_pos-1 >= 0:
            prev = observation_builder.cells_sequence[agent_id][agent_pos-1]
            prev_prev = prev

        for edge in observations.edge_ids:
            if prev in edge.Cells and prev_prev in edge.Cells and len(edge.Cells) > 2:
                return agent_pos
            elif prev in edge.Cells and prev_prev in edge.Cells and len(edge.Cells) == 2:
                agent_pos -= 1
                break




def get_action_dict(observation_builder, observations):
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

    for a in observation_builder.env.agents:
        if a.status == RailAgentStatus.ACTIVE or a.status == RailAgentStatus.READY_TO_DEPART:
            cur_pos[a.handle] = (a.position if a.position is not None else a.initial_position)


    next_pos = defaultdict()
    for a in cur_pos:
        data = [i for i, i_list in groupby(observation_builder.agent_position_data[a])]

        if 0 < len(data) < len(observation_builder.introduced_delay_time[a]):
            if observation_builder.introduced_delay_time[a][len(data)-1] > 0:
                observation_builder.introduced_delay_time[a][len(data)-1] -= 1
                #observation_builder.malfunction_time[a][len(data)] += 1
                next_pos[a] = [cur_pos[a], observation_builder.cells_sequence[a][len(data)-1]]
            elif len(observation_builder.cells_sequence[a]) > len(data)+1:
                next_pos[a] = [cur_pos[a], observation_builder.cells_sequence[a][len(data)]]


    #for a in cur_pos:
    #    agent_pos_on_traj = [num for num, item in enumerate(observation_builder.cells_sequence[a])
    #                         if item[0] == cur_pos[a][0] and item[1] == cur_pos[a][1]][0]#

    #    if agent_pos_on_traj < len(observation_builder.cells_sequence[a]):
    #        next_pos[a] = [cur_pos[a], observation_builder.cells_sequence[a][agent_pos_on_traj+1]]


    for a in next_pos:
        cur_position = next_pos[a][0]
        next_position = next_pos[a][1]

        if np.all(cur_position == next_position):
            actions[a] = 4
        else:
            cur_direction = observation_builder.env.agents[a].direction
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


def agent_clipping(observation_builder):

    priority = defaultdict()
    clipped_agents = defaultdict(list)

    for num, item in enumerate(observation_builder.cur_pos_list):
        clipped_agents[num] = []
        priority[num] = 0

        for num_other, item_other in enumerate(observation_builder.cur_pos_list):
            if num != num_other:
                # check if they have teh same conflict and opening,
                # and they are unit distance apart
                # clip them together
                if item[3] == item_other[3] and len(item[3])\
                        and pow((item[0][0]-item_other[0][0]), 2)\
                        + pow((item[0][1]-item_other[0][1]), 2) < 2:
                    priority[num] += 1
                    clipped_agents[num].append(num_other)

    # final_priority = defaultdict()
    # final_clipped_agents = defaultdict(list)
    #
    # for num, item in enumerate(observation_builder.cur_pos_list):
    #     if item[2]:
    #         final_priority[num] = priority[num]
    #         final_clipped_agents[num] = clipped_agents[num]
    #
    # for waiting_agent in final_clipped_agents:
    #     if len(final_clipped_agents[waiting_agent]):
    #         print("here")





    return priority, clipped_agents

def get_action_dict_safety(observation_builder, signal_timer):

    actions = defaultdict()
    blocked_edges = []

    # nothing should stop movement within the safe zone or unsafe zone
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            # if observation_builder.env.agents[a].status == RailAgentStatus.READY_TO_DEPART:
            #     continue
            #
            # if observation_builder.env.agents[a].position == None:
            #     continue
            #
            # if observation_builder.env.agents[a].position[0] == \
            #         observation_builder.env.agents[a].initial_position[0] and \
            #         observation_builder.env.agents[a].position[1] == \
            #         observation_builder.env.agents[a].initial_position[1]:
            #     continue

            # if clear priority
            if observation_builder.cur_pos_list[a][4]:

                # if next is safe
                # whether current is unsafe or safe
                cur_position = observation_builder.cur_pos_list[a][0]
                next_position = observation_builder.cur_pos_list[a][1]

                actions = get_valid_action(observation_builder,
                                           a,
                                           cur_position,
                                           next_position,
                                           actions)

                if len(observation_builder.cur_pos_list[a][3]):
                    # if not on safe edge already
                    # set all the remaining edges as blocked
                    if not observation_builder.cur_pos_list[a][5]:
                        # set claim on the exit cell
                        observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                        # add all unsafe edges on teh way to global block list
                        for edge in observation_builder.cur_pos_list[a][3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)

    # allow actions based on junctions data
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            # if entering unsafe zone
            if observation_builder.cur_pos_list[a][2]:

                # if the entry junction is already entered by any agent before
                if observation_builder.cur_pos_list[a][3][0].id in observation_builder.signal_time.keys():

                    # if the timer set by the agent hasn't expired
                    if observation_builder.signal_time[observation_builder.cur_pos_list[a][3][0].id] > 0:

                        # if their deadlock zone have same length
                        # if len(observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id]) \
                        #         == len(observation_builder.cur_pos_list[a][3]):
                        #     equal_status = [0 if item1.id == item2.id else 1 for item1,item2 in
                        #                     zip(observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id],
                        #                         observation_builder.cur_pos_list[a][3])]
                        #
                        #     # if they are all the same in the same order
                        #     if not np.count_nonzero(equal_status):
                        # if their deadlock zone have same length
                        if len(observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id]) \
                                and len(observation_builder.cur_pos_list[a][3]):

                            num = 0
                            blocked = False
                            while True:
                                if observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id][num].id != \
                                        observation_builder.cur_pos_list[a][3][num].id:
                                    blocked = True
                                    break

                                num += 1
                                if num >= len(observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id])-1\
                                        or num >= len(observation_builder.cur_pos_list[a][3])-1:
                                    break

                            # if they are all the same in the same order
                            if not blocked:

                                # if the capacity of exit vertex is not already full
                                # or it is a target vertex
                                if observation_builder.cur_pos_list[a][3][-1].extended_capacity - \
                                        observation_builder.cur_pos_list[a][3][-1].occupancy > 0 or \
                                        observation_builder.cur_pos_list[a][3][-1].TrainsTraversal[a][1] == None:
                                    cur_position = observation_builder.cur_pos_list[a][0]
                                    next_position = observation_builder.cur_pos_list[a][1]

                                    actions = get_valid_action(observation_builder,
                                                               a,
                                                               cur_position,
                                                               next_position,
                                                               actions)

                                    # set claim on exit cell
                                    observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                                    # add all unsafe edges on teh way to global block list
                                    for edge in observation_builder.cur_pos_list[a][3][:-1]:
                                        if edge not in blocked_edges:
                                            blocked_edges.append(edge)

                                    # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                                    observation_builder.cur_pos_list[a][3][0].is_signal_on = True
                                    # because the agent is allowed
                                    # it should set occupancy on the exit cell
                                    observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                                    # it should also set on the junction,
                                    # the list of vertices which are blocked for this agent
                                    observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id] = observation_builder.cur_pos_list[a][3]
                                    # and the number of timesteps it should wait and
                                    # see if another agent is going in the same direction
                                    observation_builder.signal_time[observation_builder.cur_pos_list[a][3][0].id] = signal_timer

    # decrement signal timers as the decision based on junction data is taken
    for item in observation_builder.signal_time:
        if observation_builder.signal_time[item] > 0:
            observation_builder.signal_time[item] -= 1
        else:
            observation_builder.observations.vertices[item].is_signal_on = False

    # Fresh entry in unsafe zone
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            # if entering unsafe zone
            if observation_builder.cur_pos_list[a][2]:

                # if no place left after real agents at the exit section or existing claims to exit section
                if observation_builder.cur_pos_list[a][3][-1].extended_capacity - \
                        observation_builder.cur_pos_list[a][3][-1].occupancy > 0:
                    blocked = False
                    for transit_edge in observation_builder.cur_pos_list[a][3][:-1]:

                        # if edge not blocked
                        if transit_edge in blocked_edges:
                            blocked = True
                            break
                        # if no signal is on in opposite direction
                        if transit_edge.is_signal_on:
                            blocked = True
                            break

                    if blocked:
                        actions[a] = 4
                    else:
                        cur_position = observation_builder.cur_pos_list[a][0]
                        next_position = observation_builder.cur_pos_list[a][1]

                        actions = get_valid_action(observation_builder,
                                                   a,
                                                   cur_position,
                                                   next_position,
                                                   actions)

                        # add all unsafe edges on teh way to global block list
                        for edge in observation_builder.cur_pos_list[a][3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)

                        # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                        observation_builder.cur_pos_list[a][3][0].is_signal_on = True
                        # because the agent is allowed
                        # it should set occupancy on the exit cell
                        observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                        # it should also set on the junction,
                        # the list of vertices which are blocked for this agent
                        observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id] = observation_builder.cur_pos_list[a][3]
                        # and the number of timesteps it should wait and
                        # see if another agent is going in the same direction
                        observation_builder.signal_time[observation_builder.cur_pos_list[a][3][0].id] = signal_timer
                else:
                    actions[a] = 4

    # fresh start
    """
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            initial_allowed = False

            if observation_builder.env.agents[a].status == RailAgentStatus.DONE_REMOVED:
                initial_allowed = False
            elif observation_builder.env.agents[a].position == None:
                initial_allowed = False
            elif observation_builder.env.agents[a].position[0] == \
                    observation_builder.env.agents[a].initial_position[0] and \
                    observation_builder.env.agents[a].position[1] == \
                    observation_builder.env.agents[a].initial_position[1]:
                initial_allowed = True

            # if entering unsafe zone
            if initial_allowed:

                # if no place left after real agents at the exit section or existing claims to exit section
                if observation_builder.cur_pos_list[a][3][-1].extended_capacity - \
                        observation_builder.cur_pos_list[a][3][-1].occupancy > 0:
                    blocked = False
                    for transit_edge in observation_builder.cur_pos_list[a][3][:-1]:

                        # if edge not blocked
                        if transit_edge in blocked_edges:
                            blocked = True
                            break
                        # if no signal is on in opposite direction
                        if transit_edge.is_signal_on:
                            blocked = True
                            break

                    if blocked:
                        actions[a] = 4
                    else:
                        cur_position = observation_builder.cur_pos_list[a][0]
                        next_position = observation_builder.cur_pos_list[a][1]

                        actions = get_valid_action(observation_builder,
                                                   a,
                                                   cur_position,
                                                   next_position,
                                                   actions)

                        # add all unsafe edges on teh way to global block list
                        for edge in observation_builder.cur_pos_list[a][3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)

                        # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                        observation_builder.cur_pos_list[a][3][0].is_signal_on = True
                        # because the agent is allowed
                        # it should set occupancy on the exit cell
                        observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                        # it should also set on the junction,
                        # the list of vertices which are blocked for this agent
                        observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id] = observation_builder.cur_pos_list[a][3]
                        # and the number of timesteps it should wait and
                        # see if another agent is going in the same direction
                        observation_builder.signal_time[observation_builder.cur_pos_list[a][3][0].id] = signal_timer
                else:
                    actions[a] = 4
    """

    # arrived from bay
    """
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            # if entering unsafe zone
            if observation_builder.env.agents[a].status == RailAgentStatus.READY_TO_DEPART:

                # if no place left after real agents at the exit section or existing claims to exit section
                if observation_builder.cur_pos_list[a][3][-1].extended_capacity - \
                        observation_builder.cur_pos_list[a][3][-1].occupancy > 0:
                    blocked = False
                    for transit_edge in observation_builder.cur_pos_list[a][3][:-1]:

                        # if edge not blocked
                        if transit_edge in blocked_edges:
                            blocked = True
                            break
                        # if no signal is on in opposite direction
                        if transit_edge.is_signal_on:
                            blocked = True
                            break

                    if blocked:
                        actions[a] = 4
                    else:
                        cur_position = observation_builder.cur_pos_list[a][0]
                        next_position = observation_builder.cur_pos_list[a][1]

                        actions = get_valid_action(observation_builder,
                                                   a,
                                                   cur_position,
                                                   next_position,
                                                   actions)

                        # add all unsafe edges on teh way to global block list
                        for edge in observation_builder.cur_pos_list[a][3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)

                        # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                        observation_builder.cur_pos_list[a][3][0].is_signal_on = True
                        # because the agent is allowed
                        # it should set occupancy on the exit cell
                        observation_builder.cur_pos_list[a][3][-1].occupancy += 1
                        # it should also set on the junction,
                        # the list of vertices which are blocked for this agent
                        observation_builder.signal_deadlocks[observation_builder.cur_pos_list[a][3][0].id] = observation_builder.cur_pos_list[a][3]
                        # and the number of timesteps it should wait and
                        # see if another agent is going in the same direction
                        observation_builder.signal_time[observation_builder.cur_pos_list[a][3][0].id] = signal_timer
                else:
                    actions[a] = 4
    """

    # set everything else to halt
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys():

            actions[a] = 4

    return actions


def get_action_dict_junc(observation_builder, obs):
    """
    Takes an agent handle and returns next action for that agent following shortest path:
    - if agent status == READY_TO_DEPART => agent moves forward;
    - if agent status == ACTIVE => pick action using shortest_path.py() fun available in prediction utils;
    - if agent status == DONE => agent does nothing.
    :param ts: timestep at which teh action is to be looked up from graph for all agents
    :return:
    """
    actions = defaultdict()

    for a, row in enumerate(observation_builder.cur_pos_list):
        cur_position = observation_builder.cur_pos_list[a][0]
        next_position = observation_builder.cur_pos_list[a][1]

        if next_position != tuple((0,0)):

            # if the next position is already occupied then no action
            next_pos_occupied = [num for num, item in enumerate(observation_builder.cur_pos_list)
                                 if next_position[0] == item[0][0]
                                 and next_position[1] == item[0][1]
                                 and next_position[0] != observation_builder.env.agents[a].target[0]
                                 and next_position[1] != observation_builder.env.agents[a].target[1]]

            if len(next_pos_occupied):
                actions[a] = 4

            elif observation_builder.cur_pos_list[a][2]:

                # check first if the agent is allowed to move to the junction
                current_vertex = [obs.vertices[item]
                                  for item in obs.vertices
                                  if cur_position in obs.vertices[item].Cells][0]
                target_vertex = [obs.vertices[item]
                                 for item in obs.vertices
                                 if next_position in obs.vertices[item].Cells][0]
                agent_pos_id = [num for num,item in enumerate(target_vertex.Trains) if item == a][0]
                target_edge_vertex = [item[1] for item in target_vertex.Links if a in item[1].Trains
                                      and item[1] != current_vertex]

                if not len(target_edge_vertex):
                    actions = get_valid_action(observation_builder,
                                               a,
                                               cur_position,
                                               next_position,
                                               actions)
                    continue


                # check if capacity on the next safe section
                if not target_edge_vertex[0].is_safe:
                    current_vertex = copy.deepcopy(target_vertex)
                    target_safe_vertex = copy.deepcopy(target_edge_vertex)

                    # next safe edge and if it is free
                    while True:
                        if not target_safe_vertex[0].is_safe:
                            # find next
                            temp = [item[1] for item in target_safe_vertex[0].Links if a in item[1].Trains
                                              and item[1] != current_vertex]
                            current_vertex = target_safe_vertex[0]
                            target_safe_vertex = temp
                        else:
                            break
                else:
                    target_safe_vertex = copy.deepcopy(target_edge_vertex)

                # if no capacity on next safe edge the send stop signal
                if target_safe_vertex[0].extended_capacity <= target_safe_vertex[0].occupancy:
                    actions[a] = 4
                    continue


                agent_pos_target_edge_vertex = [num for num, item in enumerate(target_edge_vertex[0].Trains)
                                                if item == a]

                dead_conflict_status = target_edge_vertex[0].DeadlockCostPerTrain[agent_pos_target_edge_vertex[0]]
                conflict_status = target_edge_vertex[0].CostPerTrain[agent_pos_target_edge_vertex[0]]
                max_dead_conflict_status = max(target_edge_vertex[0].DeadlockCostPerTrain)


                own_deadlocks = [item for num,item in enumerate(obs.Deadlocks)
                        if item[2] == target_edge_vertex[0].id
                             and item[0] == a]

                others_deadlocks = [num for num,item in enumerate(obs.Deadlocks)
                            if item[2] == target_edge_vertex[0].id
                                    and item[1] == a]

                all_deadlocks = [num for num,item in enumerate(obs.Deadlocks)
                            if item[2] == target_edge_vertex[0].id]



                # if theer are any deadlocks
                # Solution depends upon them
                if len(all_deadlocks):

                    if len(own_deadlocks) and dead_conflict_status < 100000:

                        actions = get_valid_action(observation_builder,
                                                   a,
                                                   cur_position,
                                                   next_position,
                                                   actions)

                    elif len(own_deadlocks) and max_dead_conflict_status > 100000:
                        actions[a] = 4

                    # if it is with some other agents
                    elif len(others_deadlocks):

                        found = False
                        for deadlock in others_deadlocks:
                            if obs.Deadlocks[deadlock][1] == a:
                                actions = get_valid_action(observation_builder,
                                                           a,
                                                           cur_position,
                                                           next_position,
                                                           actions)
                                found = True
                                break

                        if not found:
                            actions[a] = 4

                    else:

                        actions[a] = 4

                elif observation_builder.ts+1 >= target_vertex.TrainsTime[agent_pos_id][0]\
                        and conflict_status < 10000:
                    actions = get_valid_action(observation_builder,
                                               a,
                                               cur_position,
                                               next_position,
                                               actions)
                else:
                    actions, obs = optimize_and_find_action(actions,
                                                            obs,
                                                            observation_builder,
                                                            a)
                if actions[a] != 4 and len(own_deadlocks):
                    for deadlock in own_deadlocks:
                        if deadlock in obs.Deadlocks:
                            obs.Deadlocks.remove(deadlock)

            else:
                actions = get_valid_action(observation_builder,
                                           a,
                                           cur_position,
                                           next_position,
                                           actions)
        else:
            if observation_builder.env.agents[a].status != RailAgentStatus.DONE_REMOVED:
                actions[a] = 2



    for deadlock in obs.Deadlocks:
        if observation_builder.env.agents[deadlock[0]].status == RailAgentStatus.DONE_REMOVED \
                or observation_builder.env.agents[deadlock[1]].status == RailAgentStatus.DONE_REMOVED:
            obs.Deadlocks.remove(deadlock)


    return actions, obs


def optimize_and_find_action(actions, obs, observation_builder, a):

    if obs.LastUpdated < observation_builder.ts:
        obs = optimize(observation_builder, obs, "edge")
        obs = optimize(observation_builder, obs, "junction")
        obs.LastUpdated = observation_builder.ts

    cur_position = observation_builder.cur_pos_list[a][0]
    next_position = observation_builder.cur_pos_list[a][1]

    # check first if the agent is allowed to move to the junction
    current_vertex = [obs.vertices[item]
                      for item in obs.vertices
                      if cur_position in obs.vertices[item].Cells][0]
    target_vertex = [obs.vertices[item]
                     for item in obs.vertices
                     if next_position in obs.vertices[item].Cells][0]
    agent_pos_id = [num for num, item in enumerate(target_vertex.Trains) if item == a][0]
    target_edge_vertex = [item[1] for item in target_vertex.Links if a in item[1].Trains
                          and item[1] != current_vertex]

    agent_pos_target_edge_vertex = [num for num, item in enumerate(target_edge_vertex[0].Trains)
                                    if item == a]
    conflict_status = target_edge_vertex[0].CostPerTrain[agent_pos_target_edge_vertex[0]]

    if observation_builder.ts + 1 >= target_vertex.TrainsTime[agent_pos_id][0] \
            and conflict_status < 10000:
        actions = get_valid_action(observation_builder,
                                   a,
                                   cur_position,
                                   next_position,
                                   actions)
    else:
        actions[a] = 4

    return actions, obs

def get_valid_action(observation_builder, a, cur_position, next_position, actions):
    cur_direction = observation_builder.env.agents[a].direction
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
    else:
        print(cur_position, next_position)

    if (cur_direction + 1) % 4 == next_direction:
        actions[a] = 3
    elif (cur_direction - 1) % 4 == next_direction:
        actions[a] = 1
    elif next_direction == cur_direction:
        actions[a] = 2
    else:
        print("Bug")

    return actions