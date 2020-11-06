import copy
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict
from itertools import groupby

def get_action_dict_safety(observation_builder, signal_timer):

    actions = defaultdict()
    blocked_edges = []

    # nothing should stop movement within the safe zone or unsafe zone
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys() and (row[0][0] !=0 or row[0][1] !=0):

            # if clear priority
            if row[4]:

                # if next is safe
                # whether current is unsafe or safe
                cur_position = row[0]
                next_position = row[1]

                actions = get_valid_action(observation_builder,
                                           a,
                                           cur_position,
                                           next_position,
                                           actions)

                if len(row[3]):
                    # if not on safe edge already
                    # set all the remaining edges as blocked
                    if not row[5]:
                        # set claim on the exit cell
                        row[3][-1].occupancy += 1
                        # add all unsafe edges on teh way to global block list
                        for edge in row[3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)

                        if not row[3][-1].is_safe:
                            if row[3][-1] not in blocked_edges:
                                blocked_edges.append(row[3][-1])

    # allow actions based on junctions data
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys() and (row[0][0] !=0 or row[0][1] !=0):

            # if entering unsafe zone
            if row[2]:

                # if the entry junction is already entered by any agent before
                if row[3][0].id in observation_builder.signal_time.keys():

                    # if the timer set by the agent hasn't expired
                    if observation_builder.signal_time[row[3][0].id] > 0:

                        # if their deadlock zone have same length
                        if len(observation_builder.signal_deadlocks[row[3][0].id]) and len(row[3]):

                            num = 0
                            blocked = False
                            while True:
                                if observation_builder.signal_deadlocks[row[3][0].id][num].id != row[3][num].id:
                                    blocked = True
                                    break

                                num += 1
                                if num >= len(observation_builder.signal_deadlocks[row[3][0].id]) or num >= len(row[3]):
                                    break


                            # if they are all the same in the same order
                            if not blocked:

                                # if the capacity of exit vertex is not already full
                                # or it is a target vertex

                                if row[3][-1].extended_capacity - row[3][-1].occupancy > 0 or \
                                        row[3][-1].TrainsTraversal[a][0][1] == None:
                                    cur_position = row[0]
                                    next_position = row[1]

                                    actions = get_valid_action(observation_builder,
                                                               a,
                                                               cur_position,
                                                               next_position,
                                                               actions)

                                    # set claim on exit cell
                                    row[3][-1].occupancy += 1
                                    # add all unsafe edges on teh way to global block list
                                    for edge in row[3][:-1]:
                                        if edge not in blocked_edges:
                                            blocked_edges.append(edge)

                                    if not row[3][-1].is_safe:
                                        if row[3][-1] not in blocked_edges:
                                            blocked_edges.append(row[3][-1])

                                    # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                                    row[3][0].is_signal_on = True
                                    # because the agent is allowed
                                    # it should set occupancy on the exit cell
                                    row[3][-1].occupancy += 1
                                    # it should also set on the junction,
                                    # the list of vertices which are blocked for this agent
                                    observation_builder.signal_deadlocks[row[3][0].id] = row[3]
                                    # and the number of timesteps it should wait and
                                    # see if another agent is going in the same direction
                                    observation_builder.signal_time[row[3][0].id] = signal_timer

    # decrement signal timers as the decision based on junction data is taken
    for item in observation_builder.signal_time:
        if observation_builder.signal_time[item] > 0:
            observation_builder.signal_time[item] -= 1
        else:
            observation_builder.observations.vertices[item].is_signal_on = False

    # Fresh entry in unsafe zone
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys() and (row[0][0] !=0 or row[0][1] !=0):

            # if entering unsafe zone
            if row[2]:

                # if no place left after real agents at the exit section or existing claims to exit section
                if row[3][-1].extended_capacity - row[3][-1].occupancy > 0:
                    blocked = False
                    for transit_edge in row[3][:-1]:

                        # if edge not blocked
                        if transit_edge in blocked_edges:
                            blocked = True
                            break
                        # if no signal is on in opposite direction
                        if transit_edge.is_signal_on:
                            blocked = True
                            break

                    if len(row[3]):
                        if not row[3][-1].is_safe:
                            if row[3][-1] in blocked_edges:
                                blocked = True

                    if blocked:
                        actions[a] = 4
                    else:
                        cur_position = row[0]
                        next_position = row[1]

                        actions = get_valid_action(observation_builder,
                                                   a,
                                                   cur_position,
                                                   next_position,
                                                   actions)

                        # add all unsafe edges on teh way to global block list
                        for edge in row[3][:-1]:
                            if edge not in blocked_edges:
                                blocked_edges.append(edge)
                        if not row[3][-1].is_safe:
                            if row[3][-1] not in blocked_edges:
                                blocked_edges.append(row[3][-1])

                        # set a bit to tell everyone trying for a simple entry that this junction is set on in one direction
                        row[3][0].is_signal_on = True
                        # because the agent is allowed
                        # it should set occupancy on the exit cell
                        row[3][-1].occupancy += 1
                        # it should also set on the junction,
                        # the list of vertices which are blocked for this agent
                        observation_builder.signal_deadlocks[row[3][0].id] = row[3]
                        # and the number of timesteps it should wait and
                        # see if another agent is going in the same direction
                        observation_builder.signal_time[row[3][0].id] = signal_timer
                else:
                    actions[a] = 4

    # set everything else to halt
    for a, row in enumerate(observation_builder.cur_pos_list):

        # not already decided
        if a not in actions.keys() and (row[0][0] !=0 or row[0][1] !=0):
            actions[a] = 4

    return actions


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