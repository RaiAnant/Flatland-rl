from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions
from enum import IntEnum
from itertools import islice
import r2sol
import os
import sys
import time

class Agent:
  def __init__(self):
    self.aid = -1
    self.poz = (-1, -1, -1)
    self.target = (-1, -1)
    self.speed = 0.0
    self.poz_frac = 0.0
    self.malfunc = 0
    self.nr_malfunc = 0
    self.status = RailAgentStatus.READY_TO_DEPART
  
  def GetDebugString(self):
    poz_row, poz_col, poz_o = self.poz
    target_row, target_col = self.target
    return "%d %d %d %d %d %d %.10f %.10f %d %d %d 0 0 0" % (self.aid, poz_row, poz_col, poz_o, target_row, target_col, self.speed, self.poz_frac, self.malfunc, self.nr_malfunc, self.status)
    #return "%d %d %d %d %d %d %.10f %.10f %d %d %d 0 0" % (self.aid, poz_row, poz_col, poz_o, target_row, target_col, self.speed, self.poz_frac, self.malfunc, self.nr_malfunc, self.status)

  def GetExtendedDebugString(self):
    poz_row, poz_col, poz_o = self.poz
    target_row, target_col = self.target
    return "aid=%d poz=(%d %d %d) target=(%d %d) speed=%.10f poz_frac=%.10f malf=%d nr_malf=%d status=%d" % (self.aid, poz_row, poz_col, poz_o, target_row, target_col, self.speed, self.poz_frac, self.malfunc, self.nr_malfunc, self.status)

class Solver:
  def __init__(self, testid):
    self.TESTID = str(testid)
    self.T = -1
    self.TSTART = 0.0
    self.time_in_solver = 0.0

  def WriteTransitionsMapToFile(self, f, TMAP):
    if self.T >= 1:
      f.write("0\n")
      return
    else:
      f.write("1\n")
    H = len(TMAP)
    W = len(TMAP[0])
    print("TESTID=%s H=%d W=%d" % (self.TESTID, H, W))
    f.write("%d %d\n" % (H, W))
    for i in range(H):
      for j in range(W):
        for o1 in range(4):
          for o2 in range(4):
            if TMAP[i][j][o1 * 4 + o2]:
              f.write("%d %d %d %d\n" % (i, j, o1, o2))
    f.write("-1 -1 -1 -1\n")

  def WriteAgentsDataToFile(self, f, AGENTS):
    N = len(AGENTS)
    f.write("%d %d\n" % (N, self.T))
    for aid in range(N):
      agent = AGENTS[aid]
      f.write(agent.GetDebugString() + "\n")

  def ReadMovesFromFile(self, f, N):
    line = f.readline()
    lsplit = line.split(" ")
    moves = {}
    for aid in range(N):
      moves[aid] = int(lsplit[aid])
    return moves

  def GetAgentsData(self, agents):
    agents_list = []
    for aid, agent in enumerate(agents):
      a = Agent()
      a.aid = aid
      a.target = agent.target
      a.status = agent.status
      if agent.position:
        row, col = agent.position
        a.poz = (row, col, agent.direction)
      elif a.status == RailAgentStatus.READY_TO_DEPART:
        row, col = agent.initial_position
        a.poz = (row, col, agent.direction)
      else:
        assert(a.status == RailAgentStatus.DONE_REMOVED)
        row, col = a.target
        a.poz = (row, col, agent.direction)     
      a.speed = agent.speed_data['speed']
      a.poz_frac = agent.speed_data['position_fraction']
      a.malfunc = agent.malfunction_data['malfunction']
      a.nr_malfunc = agent.malfunction_data['nr_malfunctions']
      agents_list.append(a)
    return agents_list

  def GetMoves(self, agents, obs):
    start_time = time.time()
    self.T += 1
    if self.T == 0: self.TSTART = start_time
    fname = "input-" + self.TESTID + ".txt"
    if os.path.exists(fname): os.remove(fname)
    f = open(fname, "w")
    TMAP = None
    # try:
    if self.T == 0:
      idx = 0
      while obs[idx]==None:
        idx+=1
      TMAP = obs[idx][0]
    # except:
    #   print(self.T)
    self.WriteTransitionsMapToFile(f, TMAP);
    self.WriteAgentsDataToFile(f, self.GetAgentsData(agents));
    f.close()
    fname_out = "output-" + self.TESTID + ".txt"
    if os.path.exists(fname_out): os.remove(fname_out)    
    r2sol.GetMoves(self.TESTID)
    assert(os.path.exists(fname_out))
    f = open(fname_out, "r")
    moves = self.ReadMovesFromFile(f, len(agents))
    f.close()
    if os.path.exists(fname): os.remove(fname)
    if os.path.exists(fname_out): os.remove(fname_out)
    self.time_in_solver += time.time() - start_time
    #print("[GetMoves] testid=%s T=%d ttime=%.3f tis=%.3f" % (self.TESTID, self.T, time.time() - self.TSTART, self.time_in_solver))
    return moves
