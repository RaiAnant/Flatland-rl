from collections import defaultdict
import random
import copy
import numpy as np
import time
import os
import cv2

from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool
from flatland.evaluators.client import FlatlandRemoteClient

import r2_solver
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.optimizer import get_action_dict_safety


remote_client = FlatlandRemoteClient()

if __name__ == "__main__":
    max_prediction_depth = 400

    observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=max_prediction_depth),
                                             bfs_depth=max_prediction_depth)

    evaluation_number = 0
    while True:

        evaluation_number += 1

        time_start = time.time()
        obs, info = remote_client.env_create(
            obs_builder_object=observation_builder
        )
        env_creation_time = time.time() - time_start
        if not obs:
            break

        print("Evaluation Number : {}".format(evaluation_number))

        env = remote_client.env

        time_taken_by_controller = []
        time_taken_per_step = []
        steps = 0

        env_renderer = RenderTool(env)

        while True:

            time_start = time.time()
            _action = observation_builder.get_action_dict_safety(obs)
            time_taken = time.time() - time_start
            time_taken_by_controller.append(time_taken)

            time_start = time.time()
            next_obs, all_rewards, done, _ = remote_client.env_step(_action)

            time_taken = time.time() - time_start
            time_taken_per_step.append(time_taken)

            obs = next_obs
            steps += 1
            if obs is None or done['__all__']:
                break

        np_time_taken_by_controller = np.array(time_taken_by_controller)
        np_time_taken_per_step = np.array(time_taken_per_step)
        print("="*100)
        print("="*100)
        print("Done Status : ", done)
        print("Evaluation Number : ", evaluation_number)
        print("Current Env Path : ", remote_client.current_env_path)
        print("Env Creation Time : ", env_creation_time)
        print("Number of Steps : ", steps)
        print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
        print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
        print("="*100)

    print("Evaluation of all environments complete...")
    print(remote_client.submit())
