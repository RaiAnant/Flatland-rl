import copy
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict
from itertools import groupby


def optimize(observation_builder, observations):
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
                        if num != id and edge.TrainsDir[id] != edge.TrainsDir[num]:
                            try:
                                bitmap[num][item[0]:item[1] + 1] = np.ones(
                                    (item[1] - item[0] + 1))  # [1]*(item[1]-item[0]+1)
                            except:
                                print(item, item[1] - item[0] + 1)

                    occupancy = np.sum(bitmap, axis=0)

                    for number in range(edge.TrainsTime[id][0],
                                        observation_builder.max_prediction_depth - (edge.TrainsTime[id][1] - edge.TrainsTime[id][0]) + 1):
                        if np.all(occupancy[number:number + (edge.TrainsTime[id][1] - edge.TrainsTime[id][0]) + 1] == 0):
                            observation_builder.introduced_delay_time[agent_id][cell_seq_current] += number - edge.TrainsTime[id][0]
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
            if observation_builder.introduced_delay_time[a][len(data)] > 0:
                observation_builder.introduced_delay_time[a][len(data)] -= 1
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
