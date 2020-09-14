import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail
'''
Function that assigns a priority to each agent in the environment, following a specified criterion
e.g. random with all distinct priorities 0..num_agents - 1
Input: number of agents in the current env
Output: np.array of priorities

Ideas:
    - give priority to faster agents
    - give priority to agents closer to target
    - give priority to agent that have the shortest path "free", aka with less possible conflicts

'''

def assign_random_priority(num_agents):
    """
    
    :param num_agents: 
    :return: 
    """
    priorities = np.random.choice(range(num_agents), num_agents, replace=False)

    return priorities

def assign_speed_priority(agent):
    """
    Priority is assigned according to agent speed and it is fixed.
    max_priority: 1 (fast passenger train)
    min_priority: 1/4 (slow freight train)
    :param agent: 
    :return: 
    """
    priority = agent.speed_data.speed
    return priority

def assign_id_priority(handle):
    """
    Assign priority according to agent id (lower id means higher priority).
    :param agent: 
    :return: 
    """
    return handle

# TODO Improve - this definition is too naive
def assign_priority(env, env_agent, conflict):
    """
    Assign priority in this way:
    - if agent is READY_TO_DEPART or DONE, return 0
    - if agent is ACTIVE:
        - if no conflict was predicted, return 0 (max prio)
        - if conflict was predicted, return priority as function of distance to target and speed
    :param env_agent: 
    :param conflict:
    :return: 
    """
    if env_agent.status is not RailAgentStatus.ACTIVE:
        return 0

    if not conflict:
        return 0
    else:
        max_distance = distance_on_rail((0,0), (env.height - 1, env.width - 1))
        min_distance = 0
        min_speed = 0.25
        priority = distance_on_rail(env_agent.position, env_agent.target) # Use Euclidean distance
        priority /= env_agent.speed_data['speed']
        # Normalize priority
        priority = np.around(priority / (max_distance / min_speed), decimals=3)
        return priority
    

