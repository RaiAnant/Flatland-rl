# Flatland-rl

The observation generated by the code is a graph which has nodes and edges.

Nodes are the forks in the environment and abny two forks are connected by an edge.

The code first creates a graph with no trains. Hence, this is an empty graph. 
It then uses a trajectory prediction and finds at what time which trains will be on a given section/Edge.
This is the observation that we see. The prediction is limited and hence graph shows only a limited future of collision on any given edge.

The edge has a total cost where the cost of an edge if there is a collision is 100. Hence three trains colliding with each other generate a cost of 300.
There is also a transition cost which is 1 unit per train per step.

The overall cost on an edge if greater than 100 mean a collision.

The attributes of the edge are:

A: Start (One end of edge)

B: End (Other end of edge)

Cells: Predicted cell sequence upto max prediction depth.

CollisionLocKMatrix: a matrix of # of trains X ' of trains with 1 to indicate a collision and 0 otherwise.

CostCollisionLockTotal: 100 * # of collision for the train on that edge.

CostPerTrain: per train transition time added with 100 * # of collision for the train on that edge.

CostTotal: sum of CostCollisionLockTotal and CostTransitionTimeTotal.

CostTransitionTimeTotal: The total time spent by all the trains on an edge.

Trains: The ID's of the trains on the edge at any time during the predicted future.

TrainsDir: The direction for each train (0 for A->B; 1 for B->A).

TrainsTime: The time when a given train enters and exits an edge during the predicted future.