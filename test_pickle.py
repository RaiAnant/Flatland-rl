import pickle


with open('observations_and_agents.pickle', 'rb') as f:
    ff = pickle.load(f)

obs_dict, unfinished_agents = ff

"""obs_dict contains list of observations for each agent. So the key is agent number. 
In th dict, each key contains list of observations whose length is equal to number rof nodes the agents passes through. 
Each element of the list is the snapshot of observation at the particular instant. 
Each snapshot of the observation is basically treeObservation of all the agents at that instant.


unfinished_agents is the list of agent numbers which didnt reach their final distinction, which likely go stuck in collision"""