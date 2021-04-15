import pickle
from CustomTreeObservation import CustomTreeObservation


def namefy(obs):
    keys = list(obs.keys())
    for k in keys:
        obs['agent_' + str(k)] = obs.pop(k)
        if type(obs['agent_' + str(k)]) is list:
            for obs_snaps in obs['agent_' + str(k)]:
                namefy(obs_snaps)


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
        if root.childs[k][6] < min:
            min = root.childs[k][6]
            min_k = k
    for k in keys:
        if k != min_k:
            root.childs.pop(k)
    if len(keys) == 0:
        return
    clip_tree_for_shortest_path(root.childs[min_k])


with open('observations_and_agents.pickle', 'rb') as f:
    ff = pickle.load(f)

obs_dict, unfinished_agents = ff
namefy(obs_dict)
for agent in obs_dict.keys():
    for obs in obs_dict[agent]:
        for agent2 in obs.keys():
            clip_tree_for_shortest_path(obs[agent2])
obs_dict

"""obs_dict contains list of observations for each agent. So the key is agent number. 
In th dict, each key contains list of observations whose length is equal to number rof nodes the agents passes through. 
Each element of the list is the snapshot of observation at the particular instant. 
Each snapshot of the observation is basically treeObservation of all the agents at that instant.


unfinished_agents is the list of agent numbers which didnt reach their final distinction, which likely go stuck in collision"""
