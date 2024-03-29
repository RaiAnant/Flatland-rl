import numpy as np
import time
from flatland.envs.rail_generators import complex_rail_generator

from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
import random
from flatland.core.grid.grid4_utils import get_new_position
import r2_solver
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv
import copy
from collections import defaultdict

from src.util.tree_builder import Agent_Tree

from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.optimizer import optimize, get_action_dict
from itertools import groupby

import cv2


class stoch_data:
    def __init__(self, malfunction_rate, malfunction_min_duration, malfunction_max_duration):
        self.malfunction_rate = malfunction_rate
        self.min_duration = malfunction_min_duration
        self.max_duration = malfunction_max_duration


def GetTestParams(tid):
    seed = tid * 19997 + 0
    random.seed(seed)
    width = 50  # + random.randint(0, 100)
    height = 50  # + random.randint(0, 100)
    nr_cities = 4 + random.randint(0, (width + height) // 10)
    nr_trains = min(nr_cities * 20, 100 + random.randint(0, 100))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, 5)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)
    return (seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate,
            malfunction_min_duration, malfunction_max_duration)


DEFAULT_SPEED_RATIO_MAP = {1.: 0.25,
                           1. / 2.: 0.25,
                           1. / 3.: 0.25,
                           1. / 4.: 0.25}


def naive_solver(env, obs):
    actions = {}
    for idx, agent in enumerate(env.agents):
        try:
            possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
        except:
            possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            actions[idx] = 2
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(env.distance_map.get()[idx, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            actions[idx] = np.argmin(min_distances) + 1

    return actions


def reroute_solver(cell_sequence, actions, env, agent_idx):
    for k in actions.keys():
        if 0 < actions[k] < 4 and env.agents[k].position and agent_idx[k]+1 < len(cell_sequence[k]):
            for idx, direction in enumerate([(env.agents[k].direction + i) % 4 for i in range(-1, 2)]):
                new_position = get_new_position(env.agents[k].position, direction)
                if new_position == cell_sequence[k][int(agent_idx[k])+1]:
                    actions[k] = idx + 1
                    # agent_idx[k]+=1
                    break
    return actions


if __name__ == "__main__":

    # TODO : Note there is an error for the given enviornment which needs to be resolved
    # NUMBER_OF_AGENTS = 7
    # width = 40
    # height = 40
    # max_prediction_depth = 100
    # NUM_CITIES = 4

    # TODO : collision not detected for the given case
    NUMBER_OF_AGENTS = 8
    width = 36
    height = 35
    max_prediction_depth = 80
    NUM_CITIES = 4

    # NUMBER_OF_AGENTS = 8
    # width = 30
    # height = 30
    # max_prediction_depth = 80
    # NUM_CITIES = 4

    rail_generator = sparse_rail_generator(max_num_cities=NUM_CITIES,
                                           grid_mode=True,
                                           max_rails_between_cities=2,
                                           max_rails_in_city=3,
                                           seed=1500)

    observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=max_prediction_depth),
                                             bfs_depth=200)

    env = RailEnv(
        width=width, height=height,
        rail_generator=rail_generator,
        obs_builder_object=observation_builder,
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=NUMBER_OF_AGENTS
    )

    env_renderer = RenderTool(env)
    obs, _ = env.reset()

    # status = observation_builder.optimize(obs)
    # print("Success") if status else print("fail")
    env_renderer.render_env(show=True,
                            show_inactive_agents=False,
                            show_predictions=True,
                            show_observations=True,
                            frames=True,
                            return_image=True)

    tree_dict = {}
    agent_idx = {}

    for idx, agent in enumerate(env.agents):
        tree = Agent_Tree(idx, agent.initial_position)
        tree.build_tree(obs, env)
        tree_dict[idx] = tree
        agent_idx[idx] = 0

    success, rout_list = observation_builder.rerouting(obs, tree_dict)  # returns list of re-routes
    if success:
        print("Success!")
    else:
        print("Failed!")

    for id, route in rout_list:  # viualising the re-routes
        env.dev_pred_dict[id] = route
        env_renderer.render_env(show=True,
                                show_inactive_agents=False,
                                show_predictions=True,
                                show_observations=True,
                                frames=True,
                                return_image=True)
    #     time.sleep(1.0)
    #
    cell_sequence = observation_builder.cells_sequence.copy()  # getting the final routes of the agents

    for step in range(8 * (width + height + 20)):

# <<<<<<< HEAD
#         _action = naive_solver(env, obs)
#
#         _action = reroute_solver(cell_sequence, _action, env,
#                                  agent_idx)  # checking the action against new routs to update accordingly
#
#         for k in _action.keys():
#
#             if env.agents[k].status == 1:
#                 # env.dev_pred_dict[k] = cell_sequence[k][agent_idx[k]:]
#                 agent_idx[k] += env.agents[k].speed_data['speed']
#             if env.agents[k].position is None:
#                 continue
#
#             pos = (env.agents[k].position[0], env.agents[k].position[1], env.agents[k].direction)
#             if _action[k] != 0 and pos in env.dev_pred_dict[k]:
#                 env.dev_pred_dict[k].remove(pos)
# =======
        obs = optimize(observation_builder, obs)
        _action = get_action_dict(observation_builder, obs)


        next_obs, all_rewards, done, _ = env.step(_action)

        for k in _action.keys():  # for changing predict_dict on the basis of rerouted path
            if len(cell_sequence[k][int(agent_idx[k]):]) > 1:
                env.dev_pred_dict[k] = cell_sequence[k][int(agent_idx[k]):]

        print("Rewards: {}, [done={}]".format(all_rewards, done))

        img = env_renderer.render_env(show=True,
                                      show_inactive_agents=False,
                                      show_predictions=True,
                                      show_observations=True,
                                      frames=True,
                                      return_image=True)


        #cv2.imwrite("./env_images/"+str(step).zfill(3)+".jpg", img)


        # time.sleep(1.0)

        obs = copy.deepcopy(next_obs)
        if obs is None or done['__all__']:
            break
