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

from src.junction_graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.optimizer import get_action_dict_junc, optimize

import cv2

if __name__ == "__main__":
    # NUMBER_OF_AGENTS = 10
    # width = 30
    # height = 30
    # max_prediction_depth = 200
    # NUM_CITIES = 3

    NUMBER_OF_AGENTS = 10
    width = 31
    height = 31
    max_prediction_depth = 200
    NUM_CITIES = 4

    # NUMBER_OF_AGENTS = 13
    # width = 46
    # height = 31
    # max_prediction_depth = 200
    # NUM_CITIES = 5

    find_alternate_paths = True
    rail_generator = sparse_rail_generator(max_num_cities=NUM_CITIES,
                                           grid_mode=False,
                                           max_rails_between_cities=3,
                                           max_rails_in_city=4,
                                           seed=1500)

    observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=max_prediction_depth),
                                             bfs_depth=200)

    env = RailEnv(
        width=width, height=height,
        rail_generator=rail_generator,
        obs_builder_object=observation_builder,
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=NUMBER_OF_AGENTS,
        remove_collisions=True
    )

    env_renderer = RenderTool(env)
    obs, _ = env.reset()

    obs_list = []

    conflict_data = []

    # obs_temp = copy.deepcopy(obs)

    tree_dict = {}
    agent_idx = {}

    img = env_renderer.render_env(show=True,
                                  show_inactive_agents=False,
                                  show_predictions=True,
                                  show_observations=True,
                                  frames=True,
                                  return_image=True)
    if find_alternate_paths:
        for idx, agent in enumerate(env.agents):
            tree = Agent_Tree(idx, agent.initial_position)
            tree.build_tree(obs, env)
            tree_dict[idx] = tree
            agent_idx[idx] = 0
            tree.optimize_path(obs)
            root = tree.root

            temp_node = root
            observation_builder.cells_sequence[idx] = []

            while temp_node:
                observation_builder.cells_sequence[idx].append(temp_node.node_id)
                observation_builder.cells_sequence[idx] += temp_node.path
                # TODO: will have to adjust this when cases with more than 2 children come
                if len(temp_node.children) > 1 and temp_node.children[0].min_flow_cost > temp_node.children[1].min_flow_cost:
                    print(idx, temp_node.children[0].node_id, temp_node.children[0].min_flow_cost,
                          temp_node.children[1].min_flow_cost)
                    temp_node = temp_node.children[1]
                else:
                    temp_node = temp_node.children[0] if temp_node.children != [] else None

            observation_builder.cells_sequence[idx].append(agent.target)
            observation_builder.cells_sequence[idx].append((0, 0))

        obs = observation_builder.get()

    for step in range(8 * (width + height + 20)):

        print("==================== ", step)

        obs.Deadlocks = conflict_data

        obs_temp = copy.deepcopy(obs)
        obs_list.append(obs_temp)
        observation_builder.setDeadLocks(obs)
        obs.setCosts()
        obs_list.append(obs_temp)
        # obs = optimize(observation_builder, obs, "edge")
        # obs = optimize(observation_builder, obs, "junction")

        # obs = optimize(observation_builder, obs, "edge")
        # obs_temp = copy.deepcopy(obs)
        # obs_list.append(obs_temp)
        # obs = optimize(observation_builder, obs, "junction")
        # obs_temp = copy.deepcopy(obs)
        # obs_list.append(obs_temp)

        conflict_data = obs.Deadlocks
        """
        obs = optimize(observation_builder, obs, "edge")
        obs_temp = copy.deepcopy(obs)
        obs_list.append(obs_temp)
        obs = optimize(observation_builder, obs, "junction")
        obs_temp = copy.deepcopy(obs)
        obs_list.append(obs_temp)
        """

        _action, obs = get_action_dict_junc(observation_builder, obs)

        obs_temp = copy.deepcopy(obs)
        obs_list.append(obs_temp)

        next_obs, all_rewards, done, _ = env.step(_action)

        # print("Rewards: {}, [done={}]".format(all_rewards, done))

        img = env_renderer.render_env(show=True,
                                      show_inactive_agents=False,
                                      show_predictions=True,
                                      show_observations=True,
                                      frames=True,
                                      return_image=True)

        cv2.imwrite("./env_images/" + str(step).zfill(3) + ".jpg", img)

        # time.sleep(1.0)

        obs = copy.deepcopy(next_obs)
        if obs is None or done['__all__']:
            break
