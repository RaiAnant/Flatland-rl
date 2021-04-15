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
import cv2
import pickle

from CustomTreeObservation import CustomTreeObservation
import transformer
import random
import copy


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


def improved_solver(path_state):
    return random.randint(0, 2)


def TL_detector(env, obs, actions):
    obs_paths = {}
    for idx, agent in enumerate(env.agents):

        if agent.position is None:
            continue
        new_direction, transition_valid = env.check_action(agent, actions[idx])
        new_position = get_new_position(agent.position, new_direction)
        transition_bit = bin(env.rail.get_full_transitions(*new_position))
        total_transitions = transition_bit.count("1")
        if total_transitions == 4:
            agent_obs_path = copy.deepcopy(obs[idx])
            transformer.clip_tree_for_shortest_path(agent_obs_path)
            agent_obs_path = transformer.transform_agent_observation(agent_obs_path)
            agent_obs_path = transformer.split_node_list(agent_obs_path, env.obs_builder.branches)
            agent_obs_path = transformer.filter_agent_obs(agent_obs_path)
        else:
            agent_obs_path = None

        obs_paths[idx] = agent_obs_path
        
    return obs_paths






def solve(env, width, height, naive, predictor):
    env_renderer = RenderTool(env)
    solver = r2_solver.Solver(1)

    obs, _ = env.reset()
    env.obs_builder.find_safe_edges(env)

    predictor.env = env
    predictor.get()
    for step in range(100):

        # print(obs)
        # print(obs.shape)

        if naive:
            _action = naive_solver(env, obs)
        else:
            _action = solver.GetMoves(env.agents, obs)

        for k in _action.keys():
            if env.agents[k].position is None:
                continue

            pos = (env.agents[k].position[0], env.agents[k].position[1], env.agents[k].direction)
            if _action[k] != 0 and pos in env.dev_pred_dict[k]:
                env.dev_pred_dict[k].remove(pos)



        TL_detector(env, obs, _action)


        # for k in _action.keys():
        #     if _action[k] != 2:
        #         print("fdasfds")

        next_obs, all_rewards, done, _ = env.step(_action)


        print("Rewards: {}, [done={}]".format(all_rewards, done))
        img = env_renderer.render_env(show=True, show_inactive_agents=False, show_predictions=True,
                                      show_observations=False,
                                      frames=True, return_image=True)
        cv2.imwrite("./env_images/" + str(step).zfill(3) + ".jpg", img)

        obs = next_obs.copy()
        if obs is None or done['__all__']:
            break

    unfinished_agents = []
    for k in done.keys():
        if not done[k] and type(k) is int:
            unfinished_agents.append(k)

    with open('observations_and_agents.pickle', 'wb') as f:
        pickle.dump((env.obs_builder.obs_dict, unfinished_agents, env.obs_builder.branches, env.obs_builder.safe_map),
                    f)
    return


if __name__ == "__main__":
    width = 25
    height = 25

    rail_generator = sparse_rail_generator(max_num_cities=2, grid_mode=False, max_rails_between_cities=2,
                                           max_rails_in_city=3, seed=1)

    shortestPred = ShortestPathPredictorForRailEnv(max_depth=10)
    env = RailEnv(
        width=width, height=height,
        rail_generator=rail_generator,
        schedule_generator=sparse_schedule_generator(),
        obs_builder_object=CustomTreeObservation(max_depth=8, predictor=shortestPred),
        number_of_agents=5
    )

    solve(env, width, height, True, shortestPred)
