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
import os

from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from flatland.evaluators.client import FlatlandRemoteClient
import cv2

remote_client = FlatlandRemoteClient()

if __name__ == "__main__":
    #NUMBER_OF_AGENTS = 17
    #width = 25
    #height = 25
    max_prediction_depth = 200
    #NUM_CITIES = 2

    #rail_generator = sparse_rail_generator(max_num_cities=NUM_CITIES,
    #                                       grid_mode=False,
    #                                       max_rails_between_cities=2,
    #                                       max_rails_in_city=3,
    #                                       seed=15)

    observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=max_prediction_depth),
                                             bfs_depth=max_prediction_depth)


    os.makedirs("./env_images/", exist_ok=True)

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

        #obs, info = remote_client.env_create(
        #                obs_builder_object=observation_builder
        #            )
        #env = remote_client.env
        #env = RailEnv(
        #    width=width, height=height,
        #    rail_generator=rail_generator,
        #    obs_builder_object=observation_builder,
        #    schedule_generator=sparse_schedule_generator(),
        #    number_of_agents=NUMBER_OF_AGENTS
        #)
        env_renderer = RenderTool(env)
        #obs, _ = env.reset()
        #step = 0

        os.makedirs("./env_images/"+str(evaluation_number), exist_ok=True)
        while True:

            time_start = time.time()
            obs = observation_builder.optimize(obs)
            _action = observation_builder.get_action_dict(obs)
            time_taken = time.time() - time_start
            time_taken_by_controller.append(time_taken)

            time_start = time.time()
            next_obs, all_rewards, done, _ = remote_client.env_step(_action)

            time_taken = time.time() - time_start
            time_taken_per_step.append(time_taken)

            img = env_renderer.render_env(show=True,
                                          show_inactive_agents=False,
                                          show_predictions=True,
                                          show_observations=True,
                                          frames=True,
                                          return_image= True)

            cv2.imwrite("./env_images/"+str(evaluation_number)+"/"+str(steps).zfill(3)+".jpg", img)

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
