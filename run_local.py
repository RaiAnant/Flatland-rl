from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
import random
import r2_solver
from flatland.utils.rendertools import RenderTool

import sys
import time
class stoch_data:
    def __init__(self):
        self.malfunction_rate =  malfunction_rate
        self.min_duration =  malfunction_min_duration
        self.max_duration = malfunction_max_duration


def GetTestParams(tid):
  seed = tid * 19997 + 0
  random.seed(seed)
  width = 50 #+ random.randint(0, 100)
  height = 50 #+ random.randint(0, 100)
  nr_cities = 4 + random.randint(0, (width + height) // 10)
  nr_trains = min(nr_cities * 20, 100 + random.randint(0, 100))
  max_rails_between_cities = 2
  max_rails_in_cities = 3 + random.randint(0, 5)
  malfunction_rate = 30 + random.randint(0, 100)
  malfunction_min_duration = 3 + random.randint(0, 7)
  malfunction_max_duration = 20 + random.randint(0, 80)
  return (seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate, malfunction_min_duration, malfunction_max_duration)

def ShouldRunTest(tid):
  # return tid >= 7
  #return tid >= 3
  return True

DEFAULT_SPEED_RATIO_MAP = {1.: 0.25,
                           1. / 2.: 0.25,
                           1. / 3.: 0.25,
                           1. / 4.: 0.25}
                           
NUM_TESTS = 10

d_base = {}

f = open("scores.txt", "r")
for line in f.readlines():
  lsplit = line.split(" ")
  if len(lsplit) >= 4:
    test_id = int(lsplit[0])
    num_done_agents = int(lsplit[1])
    percentage_num_done_agents = float(lsplit[2])
    score = float(lsplit[3])
    d_base[test_id] = (num_done_agents, score)  
f.close()

f = open("tmp-scores.txt", "w")

total_percentage_num_done_agents = 0.0
total_score = 0.0
total_base_percentage_num_done_agents = 0.0
total_base_score = 0.0

num_tests = 0

for test_id in range(NUM_TESTS):
  seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate, malfunction_min_duration, malfunction_max_duration = GetTestParams(test_id)
  if not ShouldRunTest(test_id):
    continue

  rail_generator = sparse_rail_generator(max_num_cities=nr_cities,
                                         seed=seed,
                                         grid_mode=False,
                                         max_rails_between_cities=max_rails_between_cities,
                                         max_rails_in_city=max_rails_in_cities,
                                        )


  schedule_generator = sparse_schedule_generator(DEFAULT_SPEED_RATIO_MAP)

  stochastic_data = {'malfunction_rate': malfunction_rate,
                     'min_duration': malfunction_min_duration,
                     'max_duration': malfunction_max_duration
                    }

  observation_builder = GlobalObsForRailEnv()

  env = RailEnv(width=width,
                height=height,
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=nr_trains,
                malfunction_generator_and_process_data=malfunction_from_params(stoch_data()),
                obs_builder_object=observation_builder,
                remove_agents_at_target=True
               )
  obs = env.reset()

  env_renderer = RenderTool(env)

  solver = r2_solver.Solver(test_id)
  score = 0.0
  num_steps = 8 * (width + height + 20)
  print("test_id=%d seed=%d nr_trains=%d nr_cities=%d num_steps=%d" % (test_id, seed, nr_trains, nr_cities, num_steps))

  for step in range(num_steps):
    moves = solver.GetMoves(env.agents, obs[0])
    print(moves)
    next_obs, all_rewards, done, _ = env.step(moves)
    env_renderer.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.3)
    for a in range(env.get_num_agents()):
      score += float(all_rewards[a])

    obs = next_obs.copy()
    if done['__all__']:
      break

  num_done_agents = 0
  for aid, agent in enumerate(env.agents):
    if agent.status == RailAgentStatus.DONE_REMOVED:
      num_done_agents += 1
  percentage_num_done_agents = 100.0 * num_done_agents / len(env.agents)
  total_percentage_num_done_agents += percentage_num_done_agents
  total_score += score
  num_tests += 1

  base_num_done_agents = 0
  base_score = -1e9
  if test_id in d_base:
    base_num_done_agents, base_score = d_base[test_id]
  base_percentage_num_done_agents = 100.0 * base_num_done_agents / len(env.agents)
  total_base_percentage_num_done_agents += base_percentage_num_done_agents
  total_base_score += base_score

  avg_nda = total_percentage_num_done_agents / num_tests
  avg_nda_dif = (total_percentage_num_done_agents - total_base_percentage_num_done_agents) / num_tests

  print("\n### test_id=%d nda=%d(dif=%d) pnda=%.6f(dif=%.6f) score=%.6f(dif=%.6f) avg_nda=%.6f(dif=%.6f) avg_sc=%.6f(dif=%.6f)\n" % (test_id, num_done_agents, num_done_agents - base_num_done_agents, percentage_num_done_agents, percentage_num_done_agents - base_percentage_num_done_agents, score, score - base_score, avg_nda, avg_nda_dif, total_score / num_tests, (total_score - total_base_score) / num_tests))
  f.write("%d %d% .10f %.10f %d %.10f %.10f\n" % (test_id, num_done_agents, percentage_num_done_agents, score, num_done_agents - base_num_done_agents, percentage_num_done_agents - base_percentage_num_done_agents, avg_nda_dif))
  f.flush()

f.close()
