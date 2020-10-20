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



def optimize(observation_builder, observations):
    """

    :return: List of observations containing resulting graph after each optimization step
    """

    check_again = True
    check_again_counter = 0

    while check_again and check_again_counter < 100:
        # check if the cost is within limits
        check_again = False
        check_again_counter += 1

        sorted_list = sorted(observations.vertices, key=lambda x: observations.vertices[x].CostTotal, reverse=True)
        optimization_candidate = [observations.vertices[vertex] for vertex in sorted_list
                                  if observations.vertices[vertex].CostTotal > 10000]

        for vertex in optimization_candidate:

            edge_copy = copy.deepcopy(vertex.CostPerTrain)
            trains_count = len(vertex.CostPerTrain)
            opt_try = 0

            while True:

                check_again = True

                try:
                    # find the train to be stopped
                    id = np.argmax(edge_copy)
                    agent_id = vertex.Trains[id]
                except:
                    print("Train IDS not found during optimize")




                collision_entry_point = vertex.Links[vertex.TrainsDir[id]][0]

                # find a cell in the trajectory which is not a two point link to clear the region.
                #cell_seq_safe = find_non_link(observation_builder, observations, collision_entry_point, agent_id)

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

                    if vertex.TrainsTime[id][0] < observation_builder.ts:
                        print("seems like the optimization cannot be done ahead in time")

                    # we do not want a delay for the final edge if the number of matching cells in the trajectory is not just the final one
                    for number in range(vertex.TrainsTime[id][0],
                                        observation_builder.max_prediction_depth - (vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]) + 1):
                        if np.all(occupancy[number:number + (vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]) + 1] == 0):

                            # update here
                            # if the edge is a junction
                            # delay here
                            # if not then delay on teh previous junction

                            # find previous

                            if vertex.Type == "junction":
                                vertex.TrainsTime[id] = [number, number + vertex.TrainsTime[id][1] - vertex.TrainsTime[id][0]]
                                observations = observation_builder.update_for_traffic(observations)
                                observations.setCosts()
                            elif vertex.Type== "edge":

                                connected_vertices = [item[1] for item in vertex.Links if agent_id in item[1].Trains]

                                for connected_vertex in connected_vertices:

                                    id_on_connected_vertex = \
                                    [num for num, item in enumerate(connected_vertex.Trains) if item == agent_id][0]

                                    if connected_vertex.TrainsTime[id_on_connected_vertex][1] == vertex.TrainsTime[id][0]:

                                        difference = number - vertex.TrainsTime[id][0]

                                        connected_vertex.TrainsTime[id][0] += difference
                                        connected_vertex.TrainsTime[id][1] += difference
                                observations = observation_builder.update_for_traffic(observations)
                                observations.setCosts()


                            break
                    break

                elif opt_try < trains_count:
                    edge_copy[id] = 100
                else:
                    break

                opt_try += 1

    return observations



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


    """
    candidate_edges = [edge for edge in observations.edge_ids if len(edge.Trains) > 0]
    for edge in candidate_edges:
        for a in next_pos:
            if a in edge.Trains:
                if next_pos[a][0] in edge.Cells and next_pos[a][1] in edge.Cells:



    candidate_edges = [edge for edge in observations.edge_ids if len(edge.Trains) > 0]
    for edge in candidate_edges:
        for a in next_pos:
            if a in edge.Trains:
                if next_pos[a][0] in edge.Cells and next_pos[a][1] in edge.Cells:

                    cur_position = next_pos[a][0]
                    next_position = next_pos[a][1]

                    #id = [num for num, item in enumerate(edge.Trains) if item == a][0]

                    # now either the train is leaving
                    if (next_pos[a][1] == edge.Cells[0] or next_pos[a][1] == edge.Cells[-1]) and observation_builder.ts <= edge.TrainsTime[id][1]+2:
                        actions[a] = 4
                    # or it is entering or normally travelling
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

"""



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

            if observation_builder.cur_pos_list[a][2] or True:
                # check first if the agent is allowed to move to the junction
                #target_vertex = obs.vertices[str(next_position)[1:-1]]
                target_vertex = [obs.vertices[item] for item in obs.vertices if next_position in obs.vertices[item].Cells][0]
                agent_pos_id = [num for num,item in enumerate(target_vertex.Trains) if item == a][0]

                if observation_builder.ts+1 >= target_vertex.TrainsTime[agent_pos_id][0]:
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
                else:
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

        else:
            actions[a] = 4

    return actions

