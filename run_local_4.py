import numpy as np
import time
from flatland.envs.rail_generators import complex_rail_generator

from flatland.envs.schedule_generators import complex_schedule_generator
from src.util.custom_rail_env import RailEnv
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

from src.optimizer import get_action_dict_safety, optimize
from src.util.tree_builder import optimize_all_agent_paths_for_min_flow_cost
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.malfunction_generators import malfunction_from_file

import cv2

if __name__ == "__main__":
    NUMBER_OF_AGENTS = 200
    width = 35
    height = 35
    max_prediction_depth = 300
    NUM_CITIES = 3
    SIGNAL_TIMER = 2
    # problem
    # NUMBER_OF_AGENTS = 100
    # width = 25
    # height = 25

    # NUMBER_OF_AGENTS = 10
    # width = 30
    # height = 30
    # max_prediction_depth = 200
    # NUM_CITIES = 3

    # NUMBER_OF_AGENTS = 10
    # width = 31
    # height = 31
    # max_prediction_depth = 200
    # NUM_CITIES = 4

    # NUMBER_OF_AGENTS = 13
    # width = 46
    # height = 31
    # max_prediction_depth = 200
    # NUM_CITIES = 5

    NUMBER_OF_AGENTS = 50
    width = 35
    height = 35
    max_prediction_depth = 200
    NUM_CITIES = 4

    SIGNAL_TIMER = 2

    test_env_file_path = '/home/anant/Projects/flatland-challenge-starter-kit-master/scratch/test-envs/Test_0/Level_0.pkl'

    find_alternate_paths = True

    observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=max_prediction_depth),
                                             bfs_depth=200)

    if test_env_file_path:
        env = RailEnv(width=1, height=1, rail_generator=rail_from_file(test_env_file_path),
                      schedule_generator=schedule_from_file(test_env_file_path),
                      malfunction_generator_and_process_data=malfunction_from_file(test_env_file_path),
                      obs_builder_object=observation_builder)

        obs, _ = env.reset()

        width = env.width
        height = env.height
        NUMBER_OF_AGENTS = env.number_of_agents
    else:
        rail_generator = sparse_rail_generator(max_num_cities=NUM_CITIES,
                                               grid_mode=False,
                                               max_rails_between_cities=3,
                                               max_rails_in_city=4,
                                               seed=1500)
        env = RailEnv(
            width=width, height=height,
            rail_generator=rail_generator,
            obs_builder_object=observation_builder,
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=NUMBER_OF_AGENTS,
            remove_collisions=True
        )
        obs, _ = env.reset()




    env_renderer = RenderTool(env)
    obs_list = []

    conflict_data = []

    # obs_temp = copy.deepcopy(obs)

    tree_dict = {}

    img = env_renderer.render_env(show=True,
                                  show_inactive_agents=False,
                                  show_predictions=True,
                                  show_observations=True,
                                  frames=True,
                                  return_image=True)
    if find_alternate_paths:
        obs = optimize_all_agent_paths_for_min_flow_cost(env, obs, tree_dict, observation_builder)

    for step in range(20 * (width + height + 20)):

        print("==================== ", step)

        _action = get_action_dict_safety(observation_builder, SIGNAL_TIMER)

        obs_temp = copy.deepcopy(obs)
        obs_list.append(obs_temp)

        next_obs, all_rewards, done, _ = env.step(_action)

        img = env_renderer.render_env(show=True,
                                      show_inactive_agents=False,
                                      show_predictions=True,
                                      show_observations=True,
                                      frames=True,
                                      return_image=True)

        cv2.imwrite("./env_images/" + str(step).zfill(3) + ".jpg", img)

        obs = copy.deepcopy(next_obs)
        if obs is None or done['__all__']:
            break
