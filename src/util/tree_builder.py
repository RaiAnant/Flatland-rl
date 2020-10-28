from flatland.core.grid.grid4_utils import get_new_position, get_direction
import numpy as np


class Node():
    def __init__(self, node_id, path=[]):
        self.node_id = node_id #cell value of the node/fork
        self.path = path   #the rest of the path till the next node
        self.children = []
        self.cost = len(self.path)
        self.dist = 0
        self.is_blocked = False

    def get_cost(self, node):
        return node.cost

    def add_cell_to_path(self, cell): #funciton to add path cells
        self.path.append(cell)
        self.cost = len(self.path)

    def add_child(self, node):  #adding children
        self.children.append(node)
        self.children.sort(key=self.get_cost) # sort them in the basis of cost
        self.cost = self.children[0].cost + len(self.path) + 1  #update the cost of the parent node from child node with min cost


def find_new_direction(direction, action): #returns new direction after taking action
    new_direction = direction
    if action == 1:
        new_direction = direction - 1


    elif action == 3:
        new_direction = direction + 1

    new_direction %= 4

    return new_direction


class Agent_Tree():


    tree_map = {}  #stores the tree  for given starting and ending positions so new tree need not be calculated for the same routes

    def __init__(self, agent_no, starting_pos):
        self.agent_no = agent_no
        self.starting_pos = starting_pos
        # self.current_pos = starting_pos

        self.root = None  #root of the tree
        self.node_maps = {}

    def build_tree(self, obs, env, pos=None, dir=None, node=None, hash=None, i=0):
        # print(i)

        agent = env.agents[self.agent_no]


        if pos is None:

            pos = agent.position

            if pos is None:
                pos = agent.initial_position

            dir = agent.direction

            self.starting_pos = pos
            self.target = agent.target
            self.starting_direction = dir

            node = Node(pos, [])
            self.root = node

            if (self.starting_pos, self.target, self.starting_direction) in Agent_Tree.tree_map:  #if a tree for the given route already exists, return it
                self.root = Agent_Tree.tree_map[(self.starting_pos, self.target, self.starting_direction)]
                return True
            else:
                Agent_Tree.tree_map[(self.starting_pos, self.target, self.starting_direction)] = self.root



            hash = {}  #to store visited postions for avioding loops in trees



        # self.root.dist =

        while (True):

            if agent.target == pos:   #if current position is target, leaf has been reached
                # node.add_cell_to_path(pos)
                return True

            possible_transitions = env.rail.get_transitions(*pos, dir)  #generate all possible transitions

            num_transitions = np.count_nonzero(possible_transitions)

            # Start from the current orientation, and see which transitions are available;
            # organize them as [left, forward, right], relative to the current orientation
            # If only one transition is possible, the forward branch is aligned with it.

            if num_transitions == 1:
                actions = [2]
            else:
                actions = []
                for idx, direction in enumerate([(dir + i) % 4 for i in range(-1, 2)]):
                    if possible_transitions[direction]:
                        actions.append((idx + 1, direction))  #action will be a list of (action, direction)

            if len(actions) == 1:      #if only one action is possible, it is to remain in the same node (new node not encountered)

                if node.node_id!=pos:   #add pos as a path to current node
                    node.add_cell_to_path(pos)

                #calculate new position and direction
                dir = np.argmax(possible_transitions)
                pos = get_new_position(pos, dir)

                continue

            elif (pos[0], pos[1], dir) not in hash.keys():  # a node/fork has been reached, if the pos and direction exisits in has, it means this node has already been visited and is a loops

                notLoop = False  #var to check for loops in tree
                hash[(pos[0], pos[1], dir)] = 1   #marking the point as visited to avoid loops later

                #transition is a list of (dist, action, direction) for all the possible routes in the current nod/fork
                transitions = [(env.distance_map.get()[self.agent_no, get_new_position(pos, direction)[0], get_new_position(pos, direction)[1], direction], action, direction) for action, direction in actions]
                transitions.sort(key = lambda x: x[0])  #sort the actions on the basis of their distance from final position


                for idx, params in enumerate(transitions):  #loop through all possible transitions

                    _, action, direction = params

                    if pos[0] == 12 and pos[1]==5 and dir ==0:
                        print("at the node for debug")

                    if transitions[idx][0] - transitions[0][0] > 51:  # if the current transition leads to a cost 40 greater than the first transtion (with the least cost), avoid it
                        break

                    new_node = Node(pos, [])  #node for the new transition
                    new_node.dist = transitions[idx][0]
                    new_position = get_new_position(pos, direction) #new position of the agnet after taking the transition
                    new_direction = find_new_direction(dir, action) #new direction of the agnet after taking the transition

                    if (pos[0], pos[1], dir, action) in self.node_maps:  #to see if the node for the given route already exists
                        node.add_child(self.node_maps[(pos[0], pos[1], dir, action)])  #if the node for the given route already exists, add the node as a child
                        notLoop = True #there is no loop

                    elif self.build_tree(obs, env, new_position, new_direction, new_node, hash, i + 1):  #calulate the path from the node and see if it exists
                        self.node_maps[(pos[0], pos[1], dir, action)] = new_node  #add the new node the the map
                        node.add_child(new_node) #add the node as child
                        notLoop = True


                del hash[(pos[0], pos[1], dir)]

                return notLoop

            return False
