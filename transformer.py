import pickle
from CustomTreeObservation import CustomTreeObservation, collections_alt
import numpy as np
import collections

# RailSpan = collections.namedtuple('Span', 'cells '
#                                           'AB_future '
#                                           'BA_future '
#                                           'AB_now '
#                                           'BA_now '
#                                           'dir')

NewNode = collections.namedtuple('Node', 'cells '
                                         'AB_future '
                                         'BA_future '
                                         'AB_now '
                                         'BA_now '
                                         'dir '
                                         'node_type')


def get_node_str(cell):
    c1 = str(cell[0]) if len(str(cell[0])) == 2 else '0' + str(cell[0])
    c2 = str(cell[1]) if len(str(cell[1])) == 2 else '0' + str(cell[1])
    return c1 + c2


# class RailSpanW(RailSpan):
#
#     def __repr__(self):
#         string = ''
#         for n in self.cells:
#             string += get_node_str(n) + "_"
#         return string[:-1]
#
#     def __str__(self):
#         string = ''
#         for n in self.cells:
#             string += get_node_str(n) + "_"
#         return string[:-1]


class NewNodeW(NewNode):

    def __repr__(self):
        string = ''
        for n in self.cells:
            string += get_node_str(n) + "_"
        return string[:-1]

    def __str__(self):
        string = ''
        for n in self.cells:
            string += get_node_str(n) + "_"
        return string[:-1]


def namefy1(obs):
    keys = list(obs.keys())
    for k in keys:
        obs['all TL of agent_' + str(k)] = obs.pop(k)
        if type(obs['all TL of agent_' + str(k)]) is list:
            for obs_snaps in obs['all TL of agent_' + str(k)]:
                namefy2(obs_snaps[1])


def namefy2(obs):
    keys = list(obs.keys())
    for k in keys:
        obs['agent_' + str(k)] = obs.pop(k)


def clip_tree_for_shortest_path(root):
    if root is None:
        return
    keys = list(root.childs.keys())

    if len(keys) == 0:
        return

    for k in keys:
        if type(root.childs[k]) is float:
            root.childs.pop(k)
    min = 1000
    min_k = None

    keys = list(root.childs.keys())
    for k in keys:
        if root.childs[k][6] + len(root.childs[k].cells) < min:
            min = root.childs[k][6] + len(root.childs[k].cells)
            min_k = k
    for k in keys:
        if k != min_k:
            root.childs.pop(k)
    if len(keys) == 0:
        return
    clip_tree_for_shortest_path(root.childs[min_k])


def transform_agent_observation(agent_obs):
    nodes_in_path = []
    try:
        next_agent = agent_obs.childs[list(agent_obs.childs.keys())[0]]
    except:
        next_agent = None
    while next_agent is not None:
        try:
            temp = next_agent.childs[list(next_agent.childs.keys())[0]]
            del next_agent.childs[list(next_agent.childs.keys())[0]]
        except:
            temp = None

        nodes_in_path.append(next_agent)
        next_agent = temp

    return nodes_in_path


def transform_observation(obs):
    for agent1 in obs.keys():

        for _, tl in obs[agent1]:
            agents = list(tl.keys());
            for agent2 in agents:
                if agent2 not in agent1:
                    continue
                nodes_in_path = []
                try:
                    next_agent = tl[agent2].childs[list(tl[agent2].childs.keys())[0]]
                except:
                    next_agent = None
                while next_agent is not None:
                    try:
                        temp = next_agent.childs[list(next_agent.childs.keys())[0]]
                        del next_agent.childs[list(next_agent.childs.keys())[0]]
                    except:
                        temp = None

                    nodes_in_path.append(next_agent)
                    next_agent = temp

                tl[agent2] = nodes_in_path


def create_node_copy_with_changes(node, safe, AB_future):
    return collections_alt(dist_own_target_encountered=node.dist_own_target_encountered,
                           dist_other_target_encountered=node.dist_other_target_encountered,
                           dist_other_agent_encountered=node.dist_other_agent_encountered,
                           dist_potential_conflict=node.dist_potential_conflict,
                           dist_unusable_switch=node.dist_unusable_switch,
                           dist_to_next_branch=node.dist_to_next_branch,
                           dist_min_to_target=node.dist_min_to_target,
                           num_agents_same_direction=node.num_agents_same_direction,
                           num_agents_opposite_direction=node.num_agents_opposite_direction,
                           num_agents_malfunctioning=node.num_agents_malfunctioning,
                           speed_min_fractional=node.speed_min_fractional,
                           num_agents_ready_to_depart=node.num_agents_ready_to_depart,
                           other_target_location=node.other_target_location,
                           other_agent_location=node.other_agent_location,
                           location_of_potential_conflict=node.location_of_potential_conflict,
                           location_of_agents_in_same_direction=node.location_of_agents_in_same_direction,
                           location_of_agents_in_opposite_direction=node.location_of_agents_in_opposite_direction,
                           location_of_agents_malfunctioning=node.location_of_agents_malfunctioning,
                           childs={},
                           start=node.start,
                           end=node.end,
                           cells=node.cells,
                           dir=node.dir,
                           safe_zones=safe,
                           tl_pos=node.tl_pos,
                           AB_future=AB_future)


def split_node_by_safe_zones(node, branches):
    if len(node.safe_zones) == abs(node.start[0] - node.end[0]) + abs(
            node.start[1] - node.end[1]):
        node = create_node_copy_with_changes(node, not len(node.safe_zones) == 0, not node.AB_future == [])
        return [node]
    else:
        nodes = []
        safe = node.cells[0] in node.safe_zones
        cells = []
        start_idx = 0
        other_target_location = None
        other_agent_location = None
        location_of_potential_conflict = None
        location_of_agents_in_same_direction = []
        location_of_agents_in_opposite_direction = []

        if node.tl_pos in node.location_of_agents_in_same_direction:
            location_of_agents_in_same_direction.append(node.tl_pos)
        if node.tl_pos in node.location_of_agents_in_opposite_direction:
            location_of_agents_in_opposite_direction.append(node.tl_pos)

        location_of_agents_malfunctioning = None
        AB_future = True
        for idx, c in enumerate(node.cells):

            if ((c in node.safe_zones and safe) or (c not in node.safe_zones and not safe)) and len(cells) != len(
                    node.cells) - 1 and c not in branches:
                cells.append(c)
                if c == node.other_target_location:
                    other_target_location = c
                if c == node.other_agent_location:
                    other_agent_location = c
                if c == node.location_of_potential_conflict:
                    location_of_potential_conflict = c
                if c in node.location_of_agents_in_same_direction:
                    location_of_agents_in_same_direction.append(c)
                if c in node.location_of_agents_in_opposite_direction:
                    location_of_agents_in_opposite_direction.append(c)
                if c == node.location_of_agents_malfunctioning:
                    location_of_agents_malfunctioning = c
                if c in node.AB_future:
                    AB_future = True
            else:
                cells.append(c)
                new_node = collections_alt(dist_own_target_encountered=node.dist_own_target_encountered,
                                           dist_other_target_encountered=node.dist_other_target_encountered if other_target_location is not None else np.inf,
                                           dist_other_agent_encountered=node.dist_other_agent_encountered if other_agent_location is not None else np.inf,
                                           dist_potential_conflict=node.dist_potential_conflict if location_of_potential_conflict is not None else np.inf,
                                           dist_unusable_switch=node.dist_unusable_switch,
                                           dist_to_next_branch=node.dist_to_next_branch,
                                           dist_min_to_target=node.dist_min_to_target + len(
                                               node.cells[idx + 1:]),
                                           num_agents_same_direction=len(location_of_agents_in_same_direction),
                                           num_agents_opposite_direction=len(location_of_agents_in_opposite_direction),
                                           num_agents_malfunctioning=node.num_agents_malfunctioning,
                                           speed_min_fractional=node.speed_min_fractional,
                                           num_agents_ready_to_depart=node.num_agents_ready_to_depart,
                                           other_target_location=other_target_location,
                                           other_agent_location=node.other_agent_location,
                                           location_of_potential_conflict=location_of_potential_conflict,
                                           location_of_agents_in_same_direction=location_of_agents_in_same_direction,
                                           location_of_agents_in_opposite_direction=location_of_agents_in_opposite_direction,
                                           location_of_agents_malfunctioning=location_of_agents_malfunctioning,
                                           childs={},
                                           start=len(nodes) == 0,
                                           end=cells[-1],
                                           cells=cells[start_idx:],
                                           dir=node.dir,
                                           safe_zones=safe,
                                           tl_pos=nodes[-1].cells[-1] if nodes != [] else node.tl_pos,
                                           AB_future=AB_future)
                nodes.append(new_node)
                if len(node.cells) == idx + 1:
                    break
                safe = node.cells[idx + 1] in node.safe_zones
                start_idx = len(cells)
                other_target_location = None
                other_agent_location = None
                location_of_potential_conflict = None
                location_of_agents_in_same_direction = []
                location_of_agents_in_opposite_direction = []

                if nodes[-1].cells[-1] in node.location_of_agents_in_same_direction:
                    location_of_agents_in_same_direction.append(nodes[-1].cells[-1])
                if nodes[-1].cells[-1] in node.location_of_agents_in_opposite_direction:
                    location_of_agents_in_opposite_direction.append(nodes[-1].cells[-1])
                location_of_agents_malfunctioning = None

        return nodes


def split_node_list(node_list, branches):
    new_node_list = []
    for node in node_list:
        new_node_list = new_node_list + split_node_by_safe_zones(node, branches)

    return new_node_list


def filter_agent_obs(obs_dict):
    node_list = []
    for node in obs_dict:
        if node == obs_dict[0]:
            node_type = "span"
        elif node.start:
            node_type = "split"
        elif node.tl_pos in branches:
            node_type = "merge"
        else:
            node_type = "start"

        node_list.append(NewNodeW(AB_future=node.AB_future,
                                  BA_future=not node.safe_zones,
                                  AB_now=node.tl_pos in node.location_of_agents_in_same_direction,
                                  BA_now=node.tl_pos in node.location_of_agents_in_opposite_direction,
                                  dir=None,
                                  cells=[node.tl_pos],
                                  node_type=node_type))

        if node.cells[:-1]:
            node_list.append(NewNodeW(cells=node.cells[:-1],
                                      AB_future=node.AB_future,
                                      BA_future=not node.safe_zones,
                                      AB_now=node.num_agents_same_direction >= 1,
                                      BA_now=node.num_agents_opposite_direction > 0,
                                      dir=node.dir,
                                      node_type="span"))

    return node_list


def filter(obs_dict):
    temp = obs_dict.copy()
    for k1 in obs_dict.keys():
        temp[k1] = {}
        for idx, tl in enumerate(obs_dict[k1]):
            node_list = []
            for node in tl[1][k1[-7:]]:
                if idx == 0 and node == tl[1][k1[-7:]][0]:
                    node_type = "span"
                elif node.start:
                    node_type = "split"
                elif node.tl_pos in branches:
                    node_type = "merge"
                else:
                    node_type = "start"

                node_list.append(NewNodeW(AB_future=node.AB_future,
                                          BA_future=not node.safe_zones,
                                          AB_now=node.tl_pos in node.location_of_agents_in_same_direction,
                                          BA_now=node.tl_pos in node.location_of_agents_in_opposite_direction,
                                          dir=None,
                                          cells=[node.tl_pos],
                                          node_type=node_type))

                if node.cells[:-1]:
                    node_list.append(NewNodeW(cells=node.cells[:-1],
                                              AB_future=node.AB_future,
                                              BA_future=not node.safe_zones,
                                              AB_now=node.num_agents_same_direction >= 1,
                                              BA_now=node.num_agents_opposite_direction > 0,
                                              dir=node.dir,
                                              node_type="span"))

            if idx != len(obs_dict[k1]) - 1:
                outcome = "pass"
            elif len(node_list) == 1:
                outcome = "target"
            else:
                outcome = "deadlock"

            temp[k1][str(idx) + "_TL_" + get_node_str(node_list[0].cells[0])] = {"time_step": tl[0],
                                                                                 "Nodes": node_list,
                                                                                 "outcome": outcome}

    return temp


with open('observations_and_agents.pickle', 'rb') as f:
    ff = pickle.load(f)

obs_dict, unfinished_agents, branches, _ = ff
namefy1(obs_dict)
for agent in obs_dict.keys():
    for obs in obs_dict[agent]:
        for agent2 in obs[1].keys():
            clip_tree_for_shortest_path(obs[1][agent2])
transform_observation(obs_dict)
for k1 in obs_dict.keys():
    for tl in obs_dict[k1]:
        for k2 in tl[1].keys():
            if k2 in k1:
                tl[1][k2] = split_node_list(tl[1][k2], branches)
filtered_tl_obs = filter(obs_dict)
filtered_tl_obs, obs_dict
"""filtered_tl_obs contains the observations in the new format that was requested and obs_dict is the old format. 
obs_dict contains list of observations for each agent. So the key is agent number. 
In th dict, each key contains list of observations whose length is equal to number rof nodes the agents passes through. 
Each element of the list is the snapshot of observation at the particular instant. 
Each snapshot of the observation is basically treeObservation of all the agents at that instant.


unfinished_agents is the list of agent numbers which didnt reach their final distinction, which likely go stuck in collision"""
