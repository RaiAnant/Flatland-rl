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
from flatland.envs.observations import  TreeObsForRailEnv
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


def solve(env, width, height, naive):
    env_renderer = RenderTool(env)
    solver = r2_solver.Solver(1)
    obs, _ = env.reset()
    shortestPred = ShortestPathPredictorForRailEnv(max_depth=40)
    # shortestPred.get()
    shortestPred.env = env
    shortestPred.get()
    for step in range(8 * (width + height + 20)):

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

        next_obs, all_rewards, done, _ = env.step(_action)
        # print(next_obs[0]==next_obs[1])

        print("Rewards: {}, [done={}]".format(all_rewards, done))
        img = env_renderer.render_env(show=True, show_inactive_agents=False, show_predictions=True, show_observations=True,
                                frames=True, return_image= True)
        cv2.imwrite("./env_images/"+str(step).zfill(3)+".jpg", img)
        # render_env(self,
        #            show=False,  # whether to call matplotlib show() or equivalent after completion
        #            show_agents=True,  # whether to include agents
        #            show_inactive_agents=False,  # whether to show agents before they start
        #            show_observations=True,  # whether to include observations
        #            show_predictions=False,  # whether to include predictions
        #            frames=False,  # frame counter to show (intended since invocation)
        #            episode=None,  # int episode number to show
        #            step=None,  # int step number to show in image
        #            selected_agent=None,  # indicate which agent is "selected" in the editor):
        #            return_image=False):  # indicate if image is returned for use in monitor:
        time.sleep(1.0)
        obs = next_obs.copy()
        if obs is None or done['__all__']:
            break


def approach2(naive=False):
    # NUMBER_OF_AGENTS = 10
    width = 25
    height = 28

    rail_generator = sparse_rail_generator(max_num_cities=5, grid_mode=False, max_rails_between_cities=2,
                                           max_rails_in_city=3, seed=1)

    env = RailEnv(
        width=width, height=height,
        rail_generator=rail_generator,
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=7
    )

    solve(env, width, height, naive)


approach2(True)
# def approach1(tid = 9):
#     seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate, malfunction_min_duration, malfunction_max_duration = GetTestParams(tid)
#     rail_generator = sparse_rail_generator(max_num_cities=nr_cities,
#                                            seed=seed,
#                                            grid_mode=False,
#                                            max_rails_between_cities=max_rails_between_cities,
#                                            max_rails_in_city=max_rails_in_cities,
#                                            )
#     schedule_generator = sparse_schedule_generator(DEFAULT_SPEED_RATIO_MAP)
#
#     stochastic_data = {'malfunction_rate': malfunction_rate,
#                        'min_duration': malfunction_min_duration,
#                        'max_duration': malfunction_max_duration
#                        }
#     observation_builder = GlobalObsForRailEnv()
#     env = RailEnv(width=width,
#                   height=height,
#                   rail_generator=rail_generator,
#                   schedule_generator=schedule_generator,
#                   number_of_agents=nr_trains,
#                   malfunction_generator_and_process_data=malfunction_from_params(stoch_data(malfunction_rate, malfunction_min_duration, malfunction_max_duration)),
#                   obs_builder_object=observation_builder,
#                   remove_agents_at_target=True
#                   )
#
#     solve(env, width, height)


# def my_controller():
#     """
#     You are supposed to write this controller
#     """
#     _action = {}
#     for _idx in range(NUMBER_OF_AGENTS):
#         _action[_idx] = np.random.randint(0, 5)
#     return _action
