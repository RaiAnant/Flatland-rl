#include "r2sol.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>

#include <algorithm>
#include <vector>

using namespace std;

#define DEBUG_LEVEL 0//2

#define DBG(debug_level, ...)         \
  {                                   \
    if (debug_level <= DEBUG_LEVEL) { \
      fprintf(stderr, __VA_ARGS__);   \
      fflush(stderr);                 \
    }                                 \
  }

#define HMAX 151
#define NMAX 201
#define MAXNODES 3000
#define TMAX 2807//2567
#define MAX_NUM_THREADS 3//4
#define MAX_CTURNS 5
#define INF 10000
#define ZERO 0.000001
#define ONE 0.999999
#define USE_STRICT_SPACING_TO_AVOID_DEADLOCKS 0
#define MIN_TINIT_FOR_SAVE_DATA_FOR_REPLAY 1000000

class Xor128 {
 public:
  void reset(unsigned int seed) {
    xx = seed;
    yy = 362436069;
    zz = 521288629;
    uu = 786232308;
  }

  inline unsigned int rand() {
    unsigned int t = (xx ^ (xx << 11));
    xx = yy;
    yy = zz;
    zz = uu;
    return (uu = (uu ^ (uu >> 19)) ^ (t ^ (t >> 8)));
  }

 private:
  unsigned int xx, yy, zz, uu;
};

double GetTime() {
  return 1.0 * clock() / CLOCKS_PER_SEC;
}

const int DROW[4] = {-1, 0, +1, 0};
const int DCOL[4] = {0, +1, 0, -1};

namespace SOLVE {

FILE* fin;
char fname[128];
int H, W, T, TEST, N, TINIT;
double TSTART;

struct Node {
  int row, col;
} node[MAXNODES];

int cell_to_node[HMAX][HMAX], nnodes;
int next[MAXNODES][4][4];
int revnext[MAXNODES][4][4];

int target_node_agents[MAXNODES][NMAX], num_target_node_agents[MAXNODES];

void ReadTransitionsMap() {
  int has_transition_map;
  fscanf(fin, "%d", &has_transition_map);
  if (!has_transition_map) return;
  fscanf(fin, "%d %d", &H, &W);
  DBG(2, "[ReadTransitionsMap] has_transition_map=%d H=%d W=%d\n", has_transition_map, H, W);
  assert(1 <= H && H < HMAX && 1 <= W && W < HMAX);
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      cell_to_node[i][j] = -2;
    }
  }
  nnodes = 0;
  T = TMAX - 7;
  int row, col, o1, o2, num_transitions = 0;
  while (fscanf(fin, "%d %d %d %d", &row, &col, &o1, &o2) == 4) {
    if (row < 0 || col < 0 || o1 < 0 || o2 < 0) break;
    DBG(4, "row=%d/%d col=%d/%d o1=%d o2=%d nnodes=%d\n", row, H, col, W, o1, o2, nnodes);
    assert(0 <= row && row < H);
    assert(0 <= col && col < W);
    assert(0 <= o1 && o1 < 4);
    assert(0 <= o2 && o2 < 4);
    if (cell_to_node[row][col] < 0) {
      assert(nnodes < MAXNODES);
      auto& new_node = node[nnodes];
      new_node.row = row;
      new_node.col = col;
      auto& next_nnodes = next[nnodes];
      auto& revnext_nnodes = revnext[nnodes];
      for (int tmpo1 = 0; tmpo1 <= 3; ++tmpo1) {
        auto& next_nnodes_tmpo1 = next_nnodes[tmpo1];
        auto& revnext_nnodes_tmpo1 = revnext_nnodes[tmpo1];
        for (int tmpo2 = 0; tmpo2 <= 3; ++tmpo2) {
          next_nnodes_tmpo1[tmpo2] = revnext_nnodes_tmpo1[tmpo2] = -1;
        }
      }
      num_target_node_agents[nnodes] = 0;
      cell_to_node[row][col] = nnodes++;
    }
    const auto& v1 = cell_to_node[row][col];
    const int rnext = row + DROW[o2], cnext = col + DCOL[o2];
    assert(0 <= rnext && rnext < H);
    assert(0 <= cnext && cnext < W);
    if (cell_to_node[rnext][cnext] < 0) {
      assert(nnodes < MAXNODES);
      auto& new_node = node[nnodes];
      new_node.row = rnext;
      new_node.col = cnext;
      auto& next_nnodes = next[nnodes];
      auto& revnext_nnodes = revnext[nnodes];
      for (int tmpo1 = 0; tmpo1 <= 3; ++tmpo1) {
        auto& next_nnodes_tmpo1 = next_nnodes[tmpo1];
        auto& revnext_nnodes_tmpo1 = revnext_nnodes[tmpo1];
        for (int tmpo2 = 0; tmpo2 <= 3; ++tmpo2) {
          next_nnodes_tmpo1[tmpo2] = revnext_nnodes_tmpo1[tmpo2] = -1;
        }
      }
      num_target_node_agents[nnodes] = 0;
      cell_to_node[rnext][cnext] = nnodes++;
    }
    const auto& v2 = cell_to_node[rnext][cnext];
    next[v1][o1][o2] = v2;
    revnext[v2][o2][o1] = v1;
    ++num_transitions;
  }
  DBG(0, "[ReadTransitionsMap] num_transitions=%d nnodes=%d/%d\n", num_transitions, nnodes, H * W);
}

enum HowIGotHere {
  INVALID = -1,
  OUTSIDE_SRC = 0,
  ENTERED_SRC = 1,
  STARTED_MOVING = 2,
  CONTINUED_MOVING = 3,
  WAITED = 4,
  MALFUNCTIONED = 5,
};

enum RailAgentStatus {
  READY_TO_DEPART = 0,
  ACTIVE = 1,
  DONE = 2,
  DONE_REMOVED = 3,
};

enum RailEnvActions {
  DO_NOTHING = 0,
  MOVE_LEFT = 1,
  MOVE_FORWARD = 2,
  MOVE_RIGHT = 3,
  STOP_MOVING = 4,
};

struct PathElem {
  int node, o, moving_to_node, moving_to_o, num_partial_turns;
  HowIGotHere how_i_got_here;

  void PrintDebug(int t = -1) {
    DBG(0, "t=%d poz=(%d %d) moving_to=(%d %d) npturns=%d how=%d\n", t, node, o, moving_to_node, moving_to_o, num_partial_turns, how_i_got_here);
  }
};

struct Path {
  struct PathElem p[TMAX];
  int tmax;
};

void CopyPath(const Path& src, Path* dst) {
  dst->tmax = src.tmax;
  if (src.tmax >= TINIT) memcpy(&dst->p[TINIT], &src.p[TINIT], (src.tmax - TINIT + 1) * sizeof(PathElem));
}

struct Agent {
  int aid, poz_row, poz_col, poz_o, poz_node, target_row, target_col, target_node, malfunc, nr_malfunc, cturns;
  double speed, poz_frac;
  int inside_poz, fresh_malfunc, moving_to_node, moving_to_o;
  RailAgentStatus status;
} agent[NMAX], tmp_agent;

int cturns_agents[MAX_CTURNS][NMAX], num_cturns_agents[MAX_CTURNS];

int MAX_DONE_AGENTS;
double MIN_COST;
struct Path path[NMAX], ipath[NMAX];

int checkpoints[NMAX][TMAX], checkpoints_o[NMAX][TMAX], checkpoints_t[NMAX][TMAX], checkpoints_cnt[NMAX][TMAX], num_checkpoints[NMAX], tcheckpoints[NMAX][TMAX], next_checkpoint[NMAX];

vector<pair<int, int>> ipath_visiting_order[MAXNODES];

void RepopulateVisitingOrders() {
  for (int node = 0; node < nnodes; ++node) ipath_visiting_order[node].clear();
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;
    const auto& checkpoints_aid = checkpoints[aid];
    const auto& checkpoints_t_aid = checkpoints_t[aid];
    const auto& num_checkpoints_aid = num_checkpoints[aid];
    for (int idx = next_checkpoint[aid]; idx < num_checkpoints_aid; ++idx)
      ipath_visiting_order[checkpoints_aid[idx]].push_back({checkpoints_t_aid[idx], aid});
  }
  for (int node = 0; node < nnodes; ++node) {
    auto& ipath_visiting_order_node = ipath_visiting_order[node];
    sort(ipath_visiting_order_node.begin(), ipath_visiting_order_node.end());
  }
}

short int covered_by[MAX_NUM_THREADS][TMAX][MAXNODES];
int is_covered[MAX_NUM_THREADS][TMAX][MAXNODES], is_covered_idx[MAX_NUM_THREADS], can_reach_idx[MAX_NUM_THREADS], can_reach[MAX_NUM_THREADS][TMAX][MAXNODES][4];
short int can_reach_with_t1[MAX_NUM_THREADS][TMAX][MAXNODES][4];
short int tmax_at_poz_node[MAX_NUM_THREADS][NMAX];

void CheckAgent(int aid) {
  auto& agent_aid = agent[aid];
  const auto& path = ipath[aid];
  if (path.tmax < TINIT) {
    assert(agent_aid.status == DONE_REMOVED);
    return;
  }

  const auto& path_elem = path.p[TINIT];
  if (path_elem.how_i_got_here == OUTSIDE_SRC) {
    assert(!agent_aid.inside_poz);
    assert(agent_aid.status == READY_TO_DEPART);
    return;
  }

  if (agent_aid.inside_poz) {
    auto& next_checkpoint_aid = next_checkpoint[aid];
    assert(next_checkpoint_aid < num_checkpoints[aid]);
    if (checkpoints[aid][next_checkpoint_aid] == agent_aid.poz_node) {
      assert(checkpoints_o[aid][next_checkpoint_aid] == agent_aid.poz_o);
      ++next_checkpoint_aid;
    }
  }
  
  const auto& is_covered_idx_0 = is_covered_idx[0];
  auto& is_covered_0 = is_covered[0];
  
  if (agent_aid.fresh_malfunc) {
    if (path_elem.how_i_got_here == STARTED_MOVING || path_elem.how_i_got_here == CONTINUED_MOVING) {
      assert(0 <= agent_aid.moving_to_node && agent_aid.moving_to_node < nnodes);
      assert(0 <= agent_aid.moving_to_o && agent_aid.moving_to_o < nnodes);
      if (path_elem.num_partial_turns >= 1) {
        assert(agent_aid.poz_node == path_elem.node);
        assert(agent_aid.poz_o == path_elem.o);
        assert(fabs(agent_aid.poz_frac - (path_elem.num_partial_turns - 1) * agent_aid.speed) < ZERO);
      } else {
        assert(next[agent_aid.poz_node][agent_aid.poz_o][path_elem.o] == path_elem.node);
        assert(agent_aid.poz_frac >= (agent_aid.cturns - 1) * agent_aid.speed - ZERO);
      }
    } else {
      assert(agent_aid.poz_node == path_elem.node);
      assert(agent_aid.poz_o == path_elem.o);
      assert(fabs(agent_aid.poz_frac - path_elem.num_partial_turns * agent_aid.speed) < ZERO);
    }
  } else if (is_covered_0[TINIT][path_elem.node] == is_covered_idx_0) {
    assert((path_elem.how_i_got_here == STARTED_MOVING || path_elem.how_i_got_here == CONTINUED_MOVING || path_elem.how_i_got_here == ENTERED_SRC) && path_elem.num_partial_turns == 0);
    if (path_elem.how_i_got_here == ENTERED_SRC) assert(!agent_aid.inside_poz);
    else {
      assert(agent_aid.inside_poz);
      assert(agent_aid.poz_frac >= ONE);
      assert(agent_aid.moving_to_node == path_elem.node);
      assert(agent_aid.moving_to_o == path_elem.o);
    }
  } else if (path.tmax == TINIT) {
    assert(!agent_aid.inside_poz);
    assert(agent_aid.status == DONE_REMOVED);
    assert(agent_aid.moving_to_node == path_elem.node);
    agent_aid.moving_to_node = agent_aid.moving_to_o = -1;
  } else {
    assert(agent_aid.inside_poz);
    assert(agent_aid.poz_node == path_elem.node);
    assert(agent_aid.poz_o == path_elem.o);
    assert(fabs(agent_aid.poz_frac - path_elem.num_partial_turns * agent_aid.speed) < ZERO);
    if (0 <= agent_aid.moving_to_node && agent_aid.moving_to_node < nnodes && path_elem.num_partial_turns == 0 &&
        (path_elem.how_i_got_here == STARTED_MOVING || path_elem.how_i_got_here == CONTINUED_MOVING)) {
      assert(agent_aid.moving_to_node == path_elem.node);
      assert(agent_aid.moving_to_o == path_elem.o);
      agent_aid.moving_to_node = agent_aid.moving_to_o = -1;
    }
    if (path_elem.num_partial_turns == 0 && (path_elem.how_i_got_here == STARTED_MOVING || path_elem.how_i_got_here == CONTINUED_MOVING))
      assert(agent_aid.moving_to_node < 0);
  }

  if (agent_aid.inside_poz) is_covered_0[TINIT][agent_aid.poz_node] = is_covered_idx_0;
}

bool reschedule;
int num_done_agents, num_planned, num_reschedules, num_adjust_ipaths;
int num_adjust_ipaths_without_full_plan_regeneration;

int important_row[2 * NMAX], important_col[2 * NMAX];
char visited[2 * NMAX];

void DFSMarkCities(int aid) {
  visited[aid] = 1;
  static const int kMaxDiff = 2;
  for (int aid2 = 0; aid2 < 2 * N; ++aid2) {
    if (visited[aid2]) continue;
    if (abs(important_row[aid] - important_row[aid2]) <= kMaxDiff && abs(important_col[aid] - important_col[aid2]) <= kMaxDiff)
      DFSMarkCities(aid2);
  }
}

void EstimateT() {
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    important_row[2 * aid] = agent_aid.poz_row;
    important_col[2 * aid] = agent_aid.poz_col;
    important_row[2 * aid + 1] = agent_aid.target_row;
    important_col[2 * aid + 1] = agent_aid.target_col;
    visited[2 * aid] = visited[2 * aid + 1] = 0;
  }
  int num_cities = 0;
  for (int aid = 0; aid < 2 * N; ++aid) {
    if (visited[aid]) continue;
    DFSMarkCities(aid);
    ++num_cities;
  }
  if (1.0 * N / num_cities < 20.0 - 1e-6)
    TEST = (int)(8.0 * (H + W + 1.0 * N / num_cities));
  else
    TEST = 8 * (H + W + 20);
  DBG(0, "(estimated) num_cities=%d TEST=%d\n", num_cities, TEST);
}

void ReadAgentsData(bool replay_mode = false) {
  fscanf(fin, "%d %d", &N, &TINIT);
  DBG(2, "[ReadAgentsData] N=%d TINIT=%d/%d\n", N, TINIT, T);
  assert(1 <= N && N < NMAX);
  if (replay_mode) {
    fscanf(fin, "%d %d %d", &num_reschedules, &num_planned, &num_adjust_ipaths_without_full_plan_regeneration);
    reschedule = TINIT == 0;
  } else if (TINIT == 0) {
    reschedule = true;
    num_reschedules = 0;
    num_adjust_ipaths = 0;
    num_adjust_ipaths_without_full_plan_regeneration = 0;
    num_planned = N;
  } else reschedule = false;
  if (num_planned == 0) return;
  for (int cturns = 0; cturns < MAX_CTURNS; ++cturns) num_cturns_agents[cturns] = 0;
  num_done_agents = 0;
  num_planned = TINIT == 0 ? N : 0;
  ++is_covered_idx[0];
  for (int aid = 0; aid < N; ++aid) {
    int has_path_data = 0, has_moving_to_data = 0, status = -1;
    fscanf(fin, "%d %d %d %d %d %d %lf %lf %d %d %d %d %d %d", &tmp_agent.aid, &tmp_agent.poz_row, &tmp_agent.poz_col, &tmp_agent.poz_o, &tmp_agent.target_row, &tmp_agent.target_col, &tmp_agent.speed, &tmp_agent.poz_frac, &tmp_agent.malfunc, &tmp_agent.nr_malfunc, &status, &tmp_agent.fresh_malfunc, &has_moving_to_data, &has_path_data);
    assert(aid == tmp_agent.aid);
    assert(status == READY_TO_DEPART || status == ACTIVE || status == DONE || status == DONE_REMOVED);
    tmp_agent.status = (RailAgentStatus) status;
    tmp_agent.poz_node = cell_to_node[tmp_agent.poz_row][tmp_agent.poz_col];
    tmp_agent.target_node = cell_to_node[tmp_agent.target_row][tmp_agent.target_col];
    tmp_agent.cturns = 0;
    while (tmp_agent.cturns * tmp_agent.speed < ONE) ++tmp_agent.cturns;
    tmp_agent.inside_poz = tmp_agent.status != READY_TO_DEPART && tmp_agent.status != DONE_REMOVED;

    auto& agent_aid = agent[aid];
    if (has_moving_to_data) {
      int row, col;
      fscanf(fin, "%d %d %d", &row, &col, &tmp_agent.moving_to_o);
      if (row < 0 && col < 0) {
        tmp_agent.moving_to_node = -1;
        assert(tmp_agent.moving_to_o < 0);
      } else {
        tmp_agent.moving_to_node = cell_to_node[row][col];
        assert(0 <= tmp_agent.moving_to_node && tmp_agent.moving_to_node < nnodes);
        assert(0 <= tmp_agent.moving_to_o && tmp_agent.moving_to_o <= 3);
      }
    } else if (TINIT == 0) tmp_agent.moving_to_node = tmp_agent.moving_to_o = -1;
    else {
      tmp_agent.moving_to_node = agent_aid.moving_to_node;
      tmp_agent.moving_to_o = agent_aid.moving_to_o;
    }
    
    if (TINIT >= 1) {
      if (!replay_mode) tmp_agent.fresh_malfunc |= (tmp_agent.nr_malfunc > agent_aid.nr_malfunc);
      reschedule |= tmp_agent.fresh_malfunc;
      if (tmp_agent.status == DONE_REMOVED) ++num_done_agents;
    }
    memcpy(&agent_aid, &tmp_agent, sizeof(Agent));
    
    if (replay_mode || TINIT == 0) target_node_agents[agent_aid.target_node][num_target_node_agents[agent_aid.target_node]++] = aid;

    auto& ipath_aid = ipath[aid];
    if (has_path_data) {
      fscanf(fin, "%d", &ipath_aid.tmax);
      for (int t = TINIT; t <= ipath_aid.tmax; ++t) {
        auto& new_path_elem = ipath_aid.p[t];
        int poz_row, poz_col, moving_to_row, moving_to_col, how_i_got_here = -1;
        fscanf(fin, "%d %d %d %d %d %d %d %d", &poz_row, &poz_col, &new_path_elem.o, &moving_to_row, &moving_to_col, &new_path_elem.moving_to_o, &new_path_elem.num_partial_turns, &how_i_got_here);
        assert(0 <= poz_row && poz_row < H && 0 <= poz_col && poz_col < W);
        new_path_elem.node = cell_to_node[poz_row][poz_col];
        assert(0 <= new_path_elem.node && new_path_elem.node < nnodes);
        if (moving_to_row >= 0 && moving_to_col >= 0)
          new_path_elem.moving_to_node = cell_to_node[moving_to_row][moving_to_col];
        else new_path_elem.moving_to_node = -1;
        assert(how_i_got_here == OUTSIDE_SRC || how_i_got_here == ENTERED_SRC || how_i_got_here == STARTED_MOVING || how_i_got_here == CONTINUED_MOVING || how_i_got_here == WAITED || how_i_got_here == MALFUNCTIONED);
        new_path_elem.how_i_got_here = (HowIGotHere) how_i_got_here;
      }
      auto& num_checkpoints_aid = num_checkpoints[aid];
      fscanf(fin, "%d", &num_checkpoints_aid);
      ++num_checkpoints_aid;
      auto& checkpoints_aid = checkpoints[aid];
      auto& checkpoints_o_aid = checkpoints_o[aid];
      auto& checkpoints_t_aid = checkpoints_t[aid];
      next_checkpoint[aid] = 1;
      for (int cid = 1; cid < num_checkpoints_aid; ++cid) {
        int row, col;
        fscanf(fin, "%d %d %d %d", &row, &col, &checkpoints_o_aid[cid], &checkpoints_t_aid[cid]);
        checkpoints_aid[cid] = cell_to_node[row][col];
        assert(0 <= checkpoints_aid[cid] && checkpoints_aid[cid] < nnodes);
      }
      if (agent_aid.inside_poz) {
        checkpoints_aid[0] = agent_aid.poz_node;
        checkpoints_o_aid[0] = agent_aid.poz_o;
      } else {
        checkpoints_aid[0] = checkpoints_o_aid[0] = -1;
      }
      checkpoints_t_aid[0] = TINIT;
    } else if (TINIT == 0) {
      // Initialize the path with fully waiting outside.
      ipath_aid.tmax = T;
      for (int t = 0; t <= T; ++t) {
        auto& new_path_elem = ipath_aid.p[t];
        new_path_elem.node = agent_aid.poz_node;
        new_path_elem.o = agent_aid.poz_o;
        new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
        new_path_elem.num_partial_turns = 0;
        new_path_elem.how_i_got_here = OUTSIDE_SRC;
      }
      num_checkpoints[aid] = next_checkpoint[aid] = 1;
      checkpoints[aid][0] = checkpoints_o[aid][0] = -1;
    }

    if (ipath_aid.tmax >= TINIT && ipath_aid.p[ipath_aid.tmax].node == agent_aid.target_node && agent_aid.status != DONE_REMOVED)
      ++num_planned;
    
    DBG(2, "\nTINIT=%d aid=%d: poz=(%d %d %d):%d inside_poz=%d target=(%d %d):%d moving_to_node=%d:(%d %d) moving_to_o=%d cturns=%d speed=%.6lf poz_frac=%.6lf malf=%d nr_malf=%d status=%d fresh_malf=%d\n", TINIT, agent_aid.aid, agent_aid.poz_row, agent_aid.poz_col, agent_aid.poz_o, agent_aid.poz_node, agent_aid.inside_poz, agent_aid.target_row, agent_aid.target_col, agent_aid.target_node, agent_aid.moving_to_node, node[agent_aid.moving_to_node].row, node[agent_aid.moving_to_node].col, agent_aid.moving_to_o, agent_aid.cturns, agent_aid.speed, agent_aid.poz_frac, agent_aid.malfunc, agent_aid.nr_malfunc, agent_aid.status, agent_aid.fresh_malfunc);

    assert(0 <= agent_aid.poz_row && agent_aid.poz_row < H);
    assert(0 <= agent_aid.poz_col && agent_aid.poz_col < W);
    assert(0 <= agent_aid.poz_node && agent_aid.poz_node < nnodes);
    assert(0 <= agent_aid.poz_o && agent_aid.poz_o <= 3);
    assert(0 <= agent_aid.target_row && agent_aid.target_row < H);
    assert(0 <= agent_aid.target_col && agent_aid.target_col < W);
    assert(0 <= agent_aid.target_node && agent_aid.target_node < nnodes);
    assert(agent_aid.cturns < MAX_CTURNS);
    cturns_agents[agent_aid.cturns][num_cturns_agents[agent_aid.cturns]++] = aid;
    
    if (TINIT >= 1) {
      CheckAgent(aid);
      DBG(2, " still moving_to_node=%d:(%d %d) moving_to_o=%d\n", agent_aid.moving_to_node, node[agent_aid.moving_to_node].row, node[agent_aid.moving_to_node].col, agent_aid.moving_to_o);
    }
  }
  
  if (TINIT == 0 || replay_mode) {
    if (TINIT == 0) EstimateT();
    else fscanf(fin, "%d", &T);
  }
  
  //exit(1);
}

struct ShortestPathQueueElement {
  int node, o;
} qshpath[MAXNODES * 4];

int dmin[NMAX][MAXNODES][4], prev_node[NMAX][MAXNODES][4], prev_o[NMAX][MAXNODES][4];

void ComputeShortestPaths(int aid) {
  auto& dmin_aid = dmin[aid];
  auto& prev_node_aid = prev_node[aid];
  auto& prev_o_aid = prev_o[aid];
  for (int node = 0; node < nnodes; ++node) {
    auto& dmin_aid_node = dmin_aid[node];
    for (int o = 0; o <= 3; ++o) dmin_aid_node[o] = INF;
  }
  int qli = 0, qls = -1;
  const auto& agent_aid = agent[aid];
  for (int o = 0; o <= 3; ++o) {
    auto& new_queue_elem = qshpath[++qls];
    new_queue_elem.node = agent_aid.target_node;
    new_queue_elem.o = o;
    dmin_aid[agent_aid.target_node][o] = 0;
  }
  while (qli <= qls) {
    const auto& qelem = qshpath[qli++];
    const auto& node1 = qelem.node;
    const auto& o1 = qelem.o;
    const int dnew = dmin_aid[node1][o1] + agent_aid.cturns;
    const auto& revnext_node1_o1 = revnext[node1][o1];
    for (int o2 = 0; o2 <= 3; ++o2) {
      const auto& node2 = revnext_node1_o1[o2];
      if (node2 < 0) continue;
      if (dnew < dmin_aid[node2][o2]) {
        dmin_aid[node2][o2] = dnew;
        auto& new_queue_elem = qshpath[++qls];
        new_queue_elem.node = node2;
        new_queue_elem.o = o2;
        prev_node_aid[node2][o2] = node1;
        prev_o_aid[node2][o2] = o1;
      }
    }
  }
}

void ComputeShortestPaths() {
  for (int aid = 0; aid < N; ++aid) ComputeShortestPaths(aid);
}

Xor128 xor128[MAX_NUM_THREADS];
short int perm[MAX_NUM_THREADS][NMAX];
char pused[MAX_NUM_THREADS][NMAX];

#define MAX_HEAP_SIZE 100000

struct HeapElement {
  short int t, node, t1, est_tmin;
  char o;

  bool IsSmaller(const HeapElement& other) const {
    return est_tmin < other.est_tmin || (est_tmin == other.est_tmin && t1 > other.t1);
  }
} heap[MAX_NUM_THREADS][MAX_HEAP_SIZE];

int heap_size[MAX_NUM_THREADS];

inline void Swap(HeapElement h[], int poza, int pozb) {
  auto& a = h[poza];
  auto& b = h[pozb];
  const HeapElement tmp = a;
  a = b;
  b = tmp;
}

inline void PushUp(HeapElement h[], int poz) {
  while (poz > 1) {
    const int parent = poz >> 1;
    auto& h_poz = h[poz];
    auto& h_parent = h[parent];
    if (h_poz.IsSmaller(h_parent)) {
      Swap(h, poz, parent);
      poz = parent;
    } else
      break;
  }
}

inline void PushDown(HeapElement h[], int hsize, int poz) {
  while (1) {
    const int lchild = poz << 1;
    if (lchild > hsize) break;
    const auto& h_lchild = h[lchild];
    const auto& h_poz = h[poz];
    const int rchild = lchild + 1;
    if (rchild <= hsize) {
      const auto& h_rchild = h[rchild];
      if (h_lchild.IsSmaller(h_rchild)) {
        if (h_lchild.IsSmaller(h_poz)) {
          Swap(h, lchild, poz);
          poz = lchild;
        } else
          break;
      } else {
        if (h_rchild.IsSmaller(h_poz)) {
          Swap(h, rchild, poz);
          poz = rchild;
        } else
          break;
      }
    } else {
      if (h_lchild.IsSmaller(h_poz)) {
        Swap(h, lchild, poz);
        poz = lchild;
      } else
        break;
    }
  }
}

inline void InsertIntoHeap(HeapElement h[], int& hsize, int t, int node, int o, int t1, int est_tmin) {
  assert(hsize + 1 < MAX_HEAP_SIZE);
  auto& new_element = h[++hsize];
  new_element.t = t;
  new_element.node = node;
  new_element.o = o;
  new_element.t1 = t1;
  new_element.est_tmin = est_tmin;
  PushUp(h, hsize);
}

inline void ExtractMinFromHeap(HeapElement h[], int& hsize, int& t, int& node, int& o, int& t1) {
  assert(hsize >= 1);
  auto& min_element = h[1];
  t = min_element.t;
  node = min_element.node;
  o = min_element.o;
  t1 = min_element.t1;
  min_element = h[hsize--];
  if (hsize >= 1) PushDown(h, hsize, 1);
}

inline void GetMinFromHeap(HeapElement h[], int& hsize, int& t, int& node, int& o, int& t1) {
  assert(hsize >= 1);
  auto& min_element = h[1];
  t = min_element.t;
  node = min_element.node;
  o = min_element.o;
  t1 = min_element.t1;
}

struct Path tmp_path[MAX_NUM_THREADS][NMAX], tmp_path2[MAX_NUM_THREADS][NMAX];
mutex m;

struct PrevState {
  short int node, t;
  char o;
  HowIGotHere type;
} prev[MAX_NUM_THREADS][TMAX][MAXNODES][4];


void CopyTmpPathToPath(int tid) {
  // The mutex must be held before calling this function.
  const auto& tmp_path_tid = tmp_path[tid];
  for (int aid = 0; aid < N; ++aid) {
    auto& path_aid = path[aid];
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) {
      path_aid.tmax = -1;
      continue;
    }
    const auto& tmp_path_tid_aid = tmp_path_tid[aid];
    CopyPath(tmp_path_tid_aid, &path_aid);
  }
}

void RecomputeCheckpoints() {
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    auto& num_checkpoints_aid = num_checkpoints[aid];

    if (agent_aid.status == DONE_REMOVED) {
      num_checkpoints_aid = next_checkpoint[aid] = 0;
      continue;
    }
    
    const auto& ipath_aid = ipath[aid];
    assert(ipath_aid.tmax > TINIT);
    if (ipath_aid.p[ipath_aid.tmax].node != agent_aid.target_node) {
      // This path hasn't been updated. We should keep the previously computed checkpoints.
      continue;
    }

    num_checkpoints_aid = next_checkpoint[aid] = 1;
    
    auto& checkpoints_aid = checkpoints[aid];
    auto& checkpoints_o_aid = checkpoints_o[aid];
    auto& checkpoints_t_aid = checkpoints_t[aid];

    if (agent_aid.inside_poz) {
      checkpoints_aid[0] = agent_aid.poz_node;
      checkpoints_o_aid[0] = agent_aid.poz_o;
    } else {
      checkpoints_aid[0] = checkpoints_o_aid[0] = -1;
    }
    
    for (int t = TINIT + 1; t <= ipath_aid.tmax; ++t) {
      const auto& path_elem = ipath_aid.p[t];
      if (path_elem.how_i_got_here == OUTSIDE_SRC) continue;
      if (path_elem.how_i_got_here == ENTERED_SRC || path_elem.node != checkpoints_aid[num_checkpoints_aid - 1]) {
        checkpoints_aid[num_checkpoints_aid] = path_elem.node;
        checkpoints_o_aid[num_checkpoints_aid] = path_elem.o;
        checkpoints_t_aid[num_checkpoints_aid] = t;
        ++num_checkpoints_aid;
      }
    }
  }
}

bool CanEnterCell(int aid, int t, int from, int to, const short int covered_by[][MAXNODES], const int is_covered[][MAXNODES], int is_covered_idx, const Path tmp_path[]) {
  assert(t > TINIT);
  assert(0 <= to && to < nnodes);
  const auto& agent_aid = agent[aid];
  const auto& is_covered1 = is_covered[t][to]; 
  const auto& is_covered2 = is_covered[t - 1][to]; 
  const auto& aid1 = covered_by[t][to];
  const auto& aid2 = covered_by[t - 1][to];

  if (agent_aid.target_node == to) {
    if (is_covered1 == is_covered_idx && aid1 < aid) return false;
    if (is_covered2 == is_covered_idx && aid2 > aid) return false;
  } else {
    if (is_covered1 == is_covered_idx) return false;
    if (is_covered2 == is_covered_idx && aid2 > aid) return false;
    if (from >= 0) {
      const auto& is_covered3 = is_covered[t][from];
      const auto& aid3 = covered_by[t][from];
      if (is_covered2 == is_covered_idx && is_covered3 == is_covered_idx && aid2 == aid3) return false;
      if (is_covered3 == is_covered_idx && aid3 < aid) return false;
    }
    if (t + 1 < T) {
      const auto& is_covered4 = is_covered[t + 1][to];
      const auto& aid4 = covered_by[t + 1][to];
      if (is_covered4 == is_covered_idx && aid4 < aid) return false;
    }
    const auto& target_node_agents_to = target_node_agents[to];
    const auto& num_target_node_agents_to = num_target_node_agents[to];
    for (int idx = num_target_node_agents_to - 1; idx >= 0; --idx) {
      const auto& aid5 = target_node_agents_to[idx];
      const auto& agent_aid5 = agent[aid5];
      if (agent_aid5.status == DONE_REMOVED) continue;
      const auto& tmp_path_aid5 = tmp_path[aid5];
      assert(tmp_path_aid5.tmax > TINIT);
      if (aid5 > aid && tmp_path_aid5.tmax == t && tmp_path_aid5.p[t].node == to) return false;
      if (aid5 < aid && tmp_path_aid5.tmax == t + 1 && tmp_path_aid5.p[t + 1].node == to) return false;
    }
  }

  return true;
}

inline bool IsFreeTimeWindow(int aid, int t1, int t2, int node, const short int covered_by[][MAXNODES], const int is_covered[][MAXNODES], int is_covered_idx, const Path tmp_path[]) {
  for (int t = t1; t <= t2; ++t) if (!CanEnterCell(aid, t, node, node, covered_by, is_covered, is_covered_idx, tmp_path)) return false;
  return true;
}

int tend_ongoing_move[NMAX];

bool OverlapsOngoingMove(int t1, int t2, int node, const short int covered_by[][MAXNODES], const int is_covered[][MAXNODES], int is_covered_idx, const Path tmp_path[], short int tmax_at_poz_node[]) {

  //if (t1 < tend_ongoing_move[node]) return true;
  if (t2 <= tend_ongoing_move[node]) return true;

  //return false;

  //const int min_tstart = t2;//USE_STRICT_SPACING_TO_AVOID_DEADLOCKS ? t2 : t1;
  const int min_tstart = t1;//USE_STRICT_SPACING_TO_AVOID_DEADLOCKS ? t2 : t1;

  const int aid_t1 = is_covered[t1][node] == is_covered_idx ? covered_by[t1][node] : -1;

  for (int tend = t2 + 1; tend <= t2 + 3 && tend <= T; ++tend) {
    if (is_covered[tend][node] == is_covered_idx) {
      const auto& aid = covered_by[tend][node];
      const auto& agent_aid = agent[aid];
      if (tend - agent_aid.cturns < min_tstart && (aid != aid_t1 || is_covered[tend - 1][node] != is_covered_idx)) return true;
    }
  }

  if (USE_STRICT_SPACING_TO_AVOID_DEADLOCKS) {
    for (int tend = t1 + 1; tend <= t2; ++tend) {
      if (is_covered[tend][node] == is_covered_idx) {
        const auto& aid_tend = covered_by[tend][node];
        if (aid_tend != aid_t1 || is_covered[tend - 1][node] != is_covered_idx) return true;
      }
    }
  }

  return false;
}

bool FindBestPath(int aid, const short int covered_by[][MAXNODES], const int is_covered[][MAXNODES], int is_covered_idx, int can_reach[][MAXNODES][4], int& can_reach_idx, short int can_reach_with_t1[][MAXNODES][4], HeapElement h[], int& hsize, PrevState prev[][MAXNODES][4], Path tmp_path[], Path* tmp_path2, short int tmax_at_poz_node[]) {
  const auto& agent_aid = agent[aid];
  auto& tmp_path_aid = tmp_path[aid];
    
  hsize = 0;

  ++can_reach_idx;
  const auto& target_node = agent_aid.target_node;
  const auto& cturns = agent_aid.cturns;
  const auto& dmin_aid = dmin[aid];

  int t1 = TINIT, t2 = TINIT, rnode = -1, ro = -1, tstart = T;

  const int MIN_HSIZE = 1;//5;

  if (!agent_aid.inside_poz) {
    rnode = agent_aid.poz_node;
    ro = agent_aid.poz_o;
    t1 = TINIT + max(agent_aid.malfunc - 1, 0);    
    for (tstart = t1; tstart < T; ++tstart) {
      if (!CanEnterCell(aid, tstart + 1, -1, rnode, covered_by, is_covered, is_covered_idx, tmp_path)) continue;
      if (hsize >= MIN_HSIZE) break;
      int t2 = tstart + 1;
      const int est_tmin = t2 + dmin_aid[rnode][ro];
      if (est_tmin > T) {
        tstart = T;
        break;
      }
      can_reach[t2][rnode][ro] = can_reach_idx;
      can_reach_with_t1[t2][rnode][ro] = tstart + 1;
      auto& new_prev = prev[t2][rnode][ro];
      new_prev.t = new_prev.node = new_prev.o = -1;
      InsertIntoHeap(h, hsize, t2, rnode, ro, tstart + 1, est_tmin);
    }
  } else {
    t1 = TINIT;
    rnode = agent_aid.poz_node;
    ro = agent_aid.poz_o;
    int moving_to_node = agent_aid.moving_to_node, moving_to_o = agent_aid.moving_to_o;
    while (t1 + 1 <= tmp_path_aid.tmax && tmp_path_aid.p[t1 + 1].how_i_got_here == MALFUNCTIONED) {
      ++t1;
      if (!CanEnterCell(aid, t1, rnode, rnode, covered_by, is_covered, is_covered_idx, tmp_path)) return false;
    }
    if (t1 >= T) return false;
    if (t1 != TINIT + agent_aid.malfunc) {
      DBG(0, "!!! [FindBestPath] aid=%d TINIT=%d malfunc=%d t1=%d/%d/%d\n", aid, TINIT, agent_aid.malfunc, t1, TINIT + agent_aid.malfunc, T);
      exit(1);
    }
    memcpy(&tmp_path2->p[TINIT], &tmp_path_aid.p[TINIT], (t1 - TINIT + 1) * sizeof(PathElem));
    t2 = t1;
    if (moving_to_node >= 0) {
      bool finished_moving = false;
      while (t2 < T && rnode == agent_aid.poz_node) {
        const auto& prev_path_elem = tmp_path2->p[t2];
        auto& next_path_elem = tmp_path2->p[t2 + 1];
        if (prev_path_elem.num_partial_turns + 1 < agent_aid.cturns || !CanEnterCell(aid, t2 + 1, rnode, moving_to_node, covered_by, is_covered, is_covered_idx, tmp_path)) {
          memcpy(&next_path_elem, &prev_path_elem, sizeof(PathElem));
          ++next_path_elem.num_partial_turns;
          next_path_elem.how_i_got_here = CONTINUED_MOVING;
        } else {
          next_path_elem.node = moving_to_node;
          next_path_elem.o = moving_to_o;
          next_path_elem.moving_to_node = next_path_elem.moving_to_o = -1;
          next_path_elem.num_partial_turns = 0;
          next_path_elem.how_i_got_here = CONTINUED_MOVING;
          finished_moving = true;
        }
        ++t2;
      }
      if (finished_moving) {
        rnode = moving_to_node;
        ro = moving_to_o;
      }
      if (OverlapsOngoingMove(TINIT, t2, moving_to_node, covered_by, is_covered, is_covered_idx, tmp_path, tmax_at_poz_node))
        return false;
    }
    const int est_tmin = t2 + dmin_aid[rnode][ro];
    if (est_tmin <= T) {
      can_reach[t2][rnode][ro] = can_reach_idx;
      can_reach_with_t1[t2][rnode][ro] = t1;
      auto& new_prev = prev[t2][rnode][ro];
      new_prev.t = new_prev.node = new_prev.o = -1;
      InsertIntoHeap(h, hsize, t2, rnode, ro, t1, est_tmin);
    }
  }

  int TMIN = T + 1, best_o = -1, best_node = -1, best_t1 = -1;
    
  if (tmp_path_aid.tmax < TINIT || tmp_path_aid.p[tmp_path_aid.tmax].node == agent_aid.target_node) TMIN = tmp_path_aid.tmax;

  while (hsize >= 1 || tstart < T) {
    int t, node, o, ct1;

    //DBG(0, "hsize=%d tstart=%d\n", hsize, tstart);
    bool first_iteration = true;
    for (; tstart < T; ++tstart) {
      if (!first_iteration && !CanEnterCell(aid, tstart + 1, -1, rnode, covered_by, is_covered, is_covered_idx, tmp_path)) continue;
      //if (first_iteration) assert(CanEnterCell(aid, tstart + 1, -1, rnode, covered_by, is_covered, is_covered_idx, tmp_path));
      first_iteration = false;
      int t2 = tstart + 1;
      const int est_tmin = t2 + dmin_aid[rnode][ro];
      if (est_tmin > T || est_tmin > TMIN) {
        tstart = T;
        break;
      }
      bool do_not_insert = false;
      while (hsize > MIN_HSIZE) {
        GetMinFromHeap(h, hsize, t, node, o, ct1);
        assert(0 <= t && t <= T);
        assert(can_reach[t][node][o] == can_reach_idx);
        if (ct1 != can_reach_with_t1[t][node][o]) {
          ExtractMinFromHeap(h, hsize, t, node, o, ct1);
          continue;
        }
        const auto& curr_dmin_aid = dmin_aid[node][o];
        //DBG(0, "min=%d (t=%d node=%d o=%d ct1=%d) est_tmin=%d\n", t + curr_dmin_aid, t, node, o, ct1, est_tmin);
        if (t + curr_dmin_aid < est_tmin) do_not_insert = true;
        break;
      }
      if (do_not_insert) {
        //DBG(0, "do_not_insert tstart=%d\n", tstart);
        //for (int hidx = 1; hidx <= 10 && hidx <= hsize; ++hidx) DBG(0, " hidx=%d: t=%d\n", hidx, h[hidx].t);
        break;
      }
      //DBG(0, "insert tstart=%d t2=%d rnode=%d ro=%d est_tmin=%d\n", tstart, t2, rnode, ro, est_tmin);
      can_reach[t2][rnode][ro] = can_reach_idx;
      can_reach_with_t1[t2][rnode][ro] = tstart + 1;
      auto& new_prev = prev[t2][rnode][ro];
      new_prev.t = new_prev.node = new_prev.o = -1;
      InsertIntoHeap(h, hsize, t2, rnode, ro, tstart + 1, est_tmin);
      //for (int hidx = 1; hidx <= 10 && hidx <= hsize; ++hidx) DBG(0, " hidx=%d: t=%d\n", hidx, h[hidx].t);
    }
    if (hsize == 0) break;
  
    ExtractMinFromHeap(h, hsize, t, node, o, ct1);
    //DBG(0, "[exmin] t=%d node=%d o=%d ct1=%d\n", t, node, o, ct1);
    //for (int hidx = 1; hidx <= 10 && hidx <= hsize; ++hidx) DBG(0, " hidx=%d: t=%d\n", hidx, h[hidx].t);

    assert(0 <= t && t <= T);
    assert(can_reach[t][node][o] == can_reach_idx);
    if (ct1 != can_reach_with_t1[t][node][o]) continue;
    const auto& curr_dmin_aid = dmin_aid[node][o];
    if (t + curr_dmin_aid > TMIN) break;

    // Case 1: Wait.
    if (t + 1 <= TMIN && t + 1 <= T && CanEnterCell(aid, t + 1, node, node, covered_by, is_covered, is_covered_idx, tmp_path) &&
        (can_reach[t + 1][node][o] != can_reach_idx || can_reach_with_t1[t + 1][node][o] < ct1) &&
          (agent_aid.inside_poz || t > ct1)) {
      can_reach[t + 1][node][o] = can_reach_idx;
      can_reach_with_t1[t + 1][node][o] = ct1;
      auto& new_prev = prev[t + 1][node][o];
      new_prev.t = t;
      new_prev.node = node;
      new_prev.o = o;
      new_prev.type = WAITED;
      const int est_tmin = t + 1 + dmin_aid[node][o];
      if (best_node != target_node || est_tmin <= TMIN) InsertIntoHeap(h, hsize, t + 1, node, o, ct1, est_tmin);
    }

    const int tarrive_node2 = t + cturns;
    if (tarrive_node2 > TMIN || tarrive_node2 > T) continue;
    if (!IsFreeTimeWindow(aid, t + 1, tarrive_node2 - 1, node, covered_by, is_covered, is_covered_idx, tmp_path)) continue;
    
    // Case 2: Move.
    const auto& next_node_o = next[node][o];
    for (int onext = 0; onext <= 3; ++onext) {
      const auto& node2 = next_node_o[onext];
      if (node2 >= 0 && node2 < nnodes && CanEnterCell(aid, tarrive_node2, node, node2, covered_by, is_covered, is_covered_idx, tmp_path) &&
          (can_reach[tarrive_node2][node2][onext] != can_reach_idx || can_reach_with_t1[tarrive_node2][node2][onext] < ct1)) {       
        if (OverlapsOngoingMove(t, tarrive_node2, node2, covered_by, is_covered, is_covered_idx, tmp_path, tmax_at_poz_node))
          continue;
        if (node2 != target_node) {
          can_reach[tarrive_node2][node2][onext] = can_reach_idx;
          can_reach_with_t1[tarrive_node2][node2][onext] = ct1;
          auto& new_prev = prev[tarrive_node2][node2][onext];
          new_prev.t = t;
          new_prev.node = node;
          new_prev.o = o;
          new_prev.type = STARTED_MOVING;
          const int est_tmin = tarrive_node2 + dmin_aid[node2][onext];
          if (best_node != target_node || est_tmin <= TMIN) InsertIntoHeap(h, hsize, tarrive_node2, node2, onext, ct1, est_tmin);
        } else {
          can_reach[tarrive_node2][node2][onext] = can_reach_idx;
          can_reach_with_t1[tarrive_node2][node2][onext] = ct1;
          auto& new_prev = prev[tarrive_node2][node2][onext];
          new_prev.t = t;
          new_prev.node = node;
          new_prev.o = o;
          new_prev.type = STARTED_MOVING;
          if (best_node != target_node || tarrive_node2 < TMIN || (tarrive_node2 == TMIN && best_t1 < ct1)) {
            TMIN = tarrive_node2;
            best_node = target_node;
            best_o = onext;
            best_t1 = ct1;
          }
        }
      }
    }
  }
  
  //DBG(0, "aid=%d TMIN=%d/%d tstart=%d hsize=%d\n", aid, TMIN, T, tstart, hsize);
  //if (TMIN > T) exit(1);

  //if (agent_aid.inside_poz) assert(best_o >= 0);
  if (best_o < 0) return false;
  if (best_node != target_node && !agent_aid.inside_poz) {
    tmp_path_aid.tmax = T;
    for (int t = TINIT; t <= T; ++t) {
      auto& new_path_elem = tmp_path_aid.p[t];
      new_path_elem.node = agent_aid.poz_node;
      new_path_elem.o = agent_aid.poz_o;
      new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
      new_path_elem.num_partial_turns = 0;
      new_path_elem.how_i_got_here = OUTSIDE_SRC;
    }   
    return true;
  }

  //assert(best_node == target_node);

  tmp_path_aid.tmax = TMIN;
  int ct = TMIN, cnode = best_node, co = best_o;
  while (1) {
    assert(can_reach[ct][cnode][co] == can_reach_idx);
    assert(ct >= can_reach_with_t1[ct][cnode][co]);
    const auto& cprev = prev[ct][cnode][co];
    if (cprev.t < 0) break;
    if (cprev.type == WAITED || cprev.type == MALFUNCTIONED) {
      // Wait 1 unit.
      auto& new_path_elem = tmp_path_aid.p[ct];
      new_path_elem.node = cnode;
      new_path_elem.o = co;
      new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
      new_path_elem.num_partial_turns = 0;
      new_path_elem.how_i_got_here = cprev.type;
    } else {
      assert(cprev.type == STARTED_MOVING);
      // Move cturns units.
      int num_partial_turns = 0;

      for (int tmove = cprev.t; tmove < ct; ++tmove) {
        auto& new_path_elem = tmp_path_aid.p[tmove + 1];
        ++num_partial_turns;
        if (num_partial_turns < agent_aid.cturns) {
          new_path_elem.node = cprev.node;
          new_path_elem.o = cprev.o;
          new_path_elem.moving_to_node = cnode;
          new_path_elem.moving_to_o = co;
          new_path_elem.num_partial_turns = num_partial_turns;
          new_path_elem.how_i_got_here = num_partial_turns == 1 ? STARTED_MOVING : CONTINUED_MOVING;
        } else {
          new_path_elem.node = cnode;
          new_path_elem.o = co;
          new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
          new_path_elem.num_partial_turns = 0;
          new_path_elem.how_i_got_here = num_partial_turns == 1 ? STARTED_MOVING : CONTINUED_MOVING;
        }
      }
    }
    ct = cprev.t;
    cnode = cprev.node;
    co = cprev.o;
  }
  
  if (!agent_aid.inside_poz) {
    for (int t = TINIT + 1; t < ct; ++t) {
      auto& new_path_elem = tmp_path_aid.p[t];
      new_path_elem.node = rnode;
      new_path_elem.o = ro;
      new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
      new_path_elem.num_partial_turns = 0;
      new_path_elem.how_i_got_here = OUTSIDE_SRC;
    }  
    auto& new_path_elem = tmp_path_aid.p[ct];
    new_path_elem.node = rnode;
    new_path_elem.o = ro;
    new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
    new_path_elem.num_partial_turns = 0;
    new_path_elem.how_i_got_here = ENTERED_SRC;
  } else {
    memcpy(&tmp_path_aid.p[TINIT], &tmp_path2->p[TINIT], (t2 - TINIT + 1) * sizeof(PathElem));
  }

  return true;
}

bool updated_best_solution;
int rerun;

vector<pair<int, int>> shpaths_sorted;

void CoverPath(int aid, const Path& path, short int covered_by[][MAXNODES], int is_covered[][MAXNODES], int is_covered_idx) {
  const auto& agent_aid = agent[aid];
  for (int t = TINIT; t <= path.tmax; ++t) {
    const auto& path_elem = path.p[t];
    if (path_elem.how_i_got_here != OUTSIDE_SRC && path_elem.node != agent_aid.target_node) {
      assert(is_covered[t][path_elem.node] != is_covered_idx);
      is_covered[t][path_elem.node] = is_covered_idx;
      covered_by[t][path_elem.node] = aid;
    }
  }
}

void UncoverPath(int aid, const Path& path, short int covered_by[][MAXNODES], int is_covered[][MAXNODES], int is_covered_idx) {
  const auto& agent_aid = agent[aid];
  for (int t = TINIT; t <= path.tmax; ++t) {
    const auto& path_elem = path.p[t];
    if (path_elem.how_i_got_here != OUTSIDE_SRC && path_elem.node != agent_aid.target_node) {
      assert(is_covered[t][path_elem.node] == is_covered_idx && covered_by[t][path_elem.node] == aid);
      is_covered[t][path_elem.node] = 0;
    }
  }
}

void CoverPath1(int aid, const Path& path, short int covered_by[][MAXNODES], int is_covered[][MAXNODES], int is_covered_idx) {
  const auto& agent_aid = agent[aid];
  if (!agent_aid.inside_poz) return;
  for (int t = TINIT; t <= T; ++t) {
    assert(is_covered[t][agent_aid.poz_node] != is_covered_idx);
    is_covered[t][agent_aid.poz_node] = is_covered_idx;
    covered_by[t][agent_aid.poz_node] = aid;
  }
}

void UncoverPath1(int aid, const Path& path, short int covered_by[][MAXNODES], int is_covered[][MAXNODES], int is_covered_idx) {
  const auto& agent_aid = agent[aid];
  if (!agent_aid.inside_poz) return;
  for (int t = TINIT; t <= T; ++t) {
    assert(is_covered[t][agent_aid.poz_node] == is_covered_idx && covered_by[t][agent_aid.poz_node] == aid);
    is_covered[t][agent_aid.poz_node] = 0;
  }
}

bool RunConsistencyChecks(Path path[], const short int covered_by[][MAXNODES], const int is_covered[][MAXNODES], int is_covered_idx, bool crash_on_error = true) {
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;
    const auto& path_aid = path[aid];
    for (int t = TINIT + 1; t <= path_aid.tmax; ++t) {
      const auto& path_elem = path_aid.p[t];
      if (path_elem.how_i_got_here == OUTSIDE_SRC) continue;
      const auto& is_covered1 = is_covered[t][path_elem.node];
      const auto& is_covered2 = is_covered[t - 1][path_elem.node];
      const auto& aid1 = covered_by[t][path_elem.node];
      const auto& aid2 = covered_by[t - 1][path_elem.node];     
      if (path_elem.node == agent_aid.target_node) {
        if (is_covered1 == is_covered_idx) {
          const bool ok = aid1 > aid;
          if (!ok) {
            if (crash_on_error) assert(ok);
            return false;
          }
        }
        if (is_covered2 == is_covered_idx) {
          const bool ok = aid2 < aid;
          if (!ok) {
            if (crash_on_error) assert(ok);
            return false;
          }
        }
      } else {
        Agent* agent_aid1 = is_covered1 != is_covered_idx ? nullptr : &agent[aid1];
        if (is_covered1 == is_covered_idx && aid1 != aid && agent_aid1->target_node != path_elem.node) {
          DBG(0, "!!! [RunConsistencyChecks] aid=%d t=%d node=%d cov=%d how=%d\n", aid, t, path_elem.node, covered_by[t][path_elem.node], path_elem.how_i_got_here);
          if (crash_on_error) exit(1);
          return false;
        }

        const auto& target_node_agents_node = target_node_agents[path_elem.node];
        const auto& num_target_node_agents_node = num_target_node_agents[path_elem.node];
        for (int idx = num_target_node_agents_node - 1; idx >= 0; --idx) {
          const auto& aid3 = target_node_agents_node[idx];
          const auto& agent_aid3 = agent[aid3];
          if (agent_aid3.status == DONE_REMOVED) continue;
          const auto& path_aid3 = path[aid3];
          bool ok = path_aid3.tmax > TINIT;
          if (!ok) {
            if (crash_on_error) assert(ok);
            return false;
          }
          if (path_aid3.tmax == t && path_aid3.p[t].node == path_elem.node) {
            ok = aid3 < aid;
            if (!ok) {
              if (crash_on_error) assert(ok);
              return false;
            }
          }
          if (path_aid3.tmax == t + 1 && path_aid3.p[t].node == path_elem.node) {
            ok = aid3 > aid;
            if (!ok) {
              if (crash_on_error) assert(ok);
              return false;
            }
          }
        }
        
        Agent* agent_aid2 = is_covered2 != is_covered_idx ? nullptr : &agent[aid2];      
        if (t > TINIT && is_covered2 == is_covered_idx && aid2 > aid && agent_aid2->target_node != path_elem.node) {
          DBG(0, "!!! [RunConsistencyChecks] aid=%d t=%d node=%d cov_t-1=%d(target=%d)\n", aid, t, path_elem.node, aid2, agent_aid2->target_node);
          if (crash_on_error) exit(1);
          return false;
        }
      }

      if (path_elem.num_partial_turns >= agent_aid.cturns && (path_elem.how_i_got_here == STARTED_MOVING || path_elem.how_i_got_here == CONTINUED_MOVING)) {
        bool ok = path_elem.moving_to_node >= 0;
        if (!ok) {
          if (crash_on_error) assert(ok);
          return false;
        }
        if ((is_covered[t][path_elem.moving_to_node] != is_covered_idx &&                  
             (is_covered[t - 1][path_elem.moving_to_node] != is_covered_idx ||
              covered_by[t - 1][path_elem.moving_to_node] <= aid)) ||
            (is_covered[t][path_elem.moving_to_node] == is_covered_idx &&
             covered_by[t][path_elem.moving_to_node] >= aid &&
             (is_covered[t - 1][path_elem.moving_to_node] != is_covered_idx ||
              covered_by[t - 1][path_elem.moving_to_node] != covered_by[t][path_elem.moving_to_node]))) {
          DBG(0, "!!! [RunConsistencyChecks] Agent in motion can enter next cell, but doesn't: aid=%d t=%d node=%d->%d npt=%d/%d cov_t=%d cov_t-1=%d\n", aid, t, path_elem.node, path_elem.moving_to_node, path_elem.num_partial_turns, agent_aid.cturns, is_covered[t][path_elem.moving_to_node] != is_covered_idx ? -1 : covered_by[t][path_elem.moving_to_node], is_covered[t - 1][path_elem.moving_to_node] != is_covered_idx ? -1 : covered_by[t - 1][path_elem.moving_to_node]);
          if (crash_on_error) exit(1);
          return false;
        }
      }
    }
  }
  return true;
}

vector<pair<pair<int, int>, int>> tmoves[MAXNODES];

void CheckNonDeadlockPaths() {
  return;
  for (int node = 0; node < nnodes; ++node) tmoves[node].clear();
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;
    const auto& ipath_aid = ipath[aid];
    
    int curr_node = agent_aid.poz_node;
    int next_node = agent_aid.moving_to_node;
    int tdeparture = TINIT, tarrival = -1;
    
    for (int t = TINIT + 1; t <= ipath_aid.tmax; ++t) {
      const auto& path_elem = ipath_aid.p[t];
      if (next_node < 0) {
        if (path_elem.node != curr_node) {
          next_node = path_elem.node;
          tdeparture = t - 1;
        } else if (path_elem.moving_to_node >= 0) {
          next_node = path_elem.moving_to_node;
          tdeparture = t - 1;
        }
      }
      if (path_elem.node == next_node) {
        tarrival = t;
        if (next_node != agent_aid.target_node) {
          tmoves[next_node].push_back({{tarrival, tdeparture}, aid});
        }
        curr_node = next_node;
        next_node = -1;
      }
    }
  }
  for (int node = 0; node < nnodes; ++node) {
    auto& tmoves_node = tmoves[node];
    if (tmoves_node.empty()) continue;
    sort(tmoves_node.begin(), tmoves_node.end());
    int prev_tarrival = -1, prev_tdeparture = -1, prev_aid = -1;
    for (const auto& tuple : tmoves_node) {
      const auto& tarrival = tuple.first.first;
      const auto& tdeparture = tuple.first.second;
      const auto& aid = tuple.second;      
      if (prev_tarrival > tdeparture) {
        DBG(0, "!!! [CheckNonDeadlockPaths] node=%d: aid=%d ct=%d tdep=%d tarr=%d target=%d | aid2=%d ct2=%d tdep2=%d tarr2=%d target2=%d\n", node, aid, agent[aid].cturns, tdeparture, tarrival, agent[aid].target_node, prev_aid, agent[prev_aid].cturns, prev_tdeparture, prev_tarrival, agent[prev_aid].target_node);
        DBG(0, "  tend_ongoing_move[node]=%d\n", tend_ongoing_move[node]);
        exit(1);
      }
      prev_tarrival = tarrival;
      prev_tdeparture = tdeparture;
      prev_aid = aid;
    }
  }
}

double SCORE_EXPONENT1, SCORE_EXPONENT2;
double MAX_TMAX_WEIGHT;

#define GetScore(t) (t <= TEST ? pow(1.0 * t / TEST, SCORE_EXPONENT1) : pow(1.0 * t / TEST, SCORE_EXPONENT2))

void RandomPermutations(int tid, int ntries) {
  auto& pused_tid = pused[tid];
  auto& perm_tid = perm[tid];
  auto& xor128_tid = xor128[tid];
  auto& covered_by_tid = covered_by[tid];
  auto& is_covered_tid = is_covered[tid];
  auto& is_covered_idx_tid = is_covered_idx[tid];
  auto& can_reach_tid = can_reach[tid];
  auto& can_reach_idx_tid = can_reach_idx[tid];
  auto& can_reach_with_t1_tid = can_reach_with_t1[tid];
  auto& heap_tid = heap[tid];
  auto& heap_size_tid = heap_size[tid];
  auto& prev_tid = prev[tid];
  auto& tmp_path_tid = tmp_path[tid];
  auto& tmp_path2_tid = tmp_path2[tid];
  auto& tmax_at_poz_node_tid = tmax_at_poz_node[tid];

  if (0&&tid == 0) {
    shpaths_sorted.resize(N);
    for (int aid = 0; aid < N; ++aid) {
      shpaths_sorted[aid].second = aid;
      const auto& agent_aid = agent[aid];
      shpaths_sorted[aid].first = agent_aid.malfunc + dmin[aid][agent_aid.poz_node][agent_aid.poz_o];
    }
    sort(shpaths_sorted.begin(), shpaths_sorted.end());
  }

  for (int trial = 1; trial <= ntries; ++trial) {
    if (0&&tid == 0 && trial <= 1) {
      for (int i = 0; i < N; ++i) perm_tid[i] = shpaths_sorted[i].second;
      reverse(shpaths_sorted.begin(), shpaths_sorted.end());
    } else {
      for (int i = 0; i < N; ++i) pused_tid[i] = 0;
      int idx = 0;
      if (0&&(trial & 1) == 1) {
        for (int cturns = MAX_CTURNS - 1; cturns >= 0; --cturns) {
          const auto& cturns_agents_cturns = cturns_agents[cturns];
          const auto& num_cturns_agents_cturns = num_cturns_agents[cturns];
          for (int i = 0; i < num_cturns_agents_cturns; ++i) {
            do {
              perm_tid[idx] = cturns_agents_cturns[xor128_tid.rand() % num_cturns_agents_cturns];
            } while (pused_tid[perm_tid[idx]]);
            pused_tid[perm_tid[idx++]] = 1;
          }
        }
      } else {
        for (int cturns = 0; cturns < MAX_CTURNS; ++cturns) {
          const auto& cturns_agents_cturns = cturns_agents[cturns];
          const auto& num_cturns_agents_cturns = num_cturns_agents[cturns];
          for (int i = 0; i < num_cturns_agents_cturns; ++i) {
            do {
              perm_tid[idx] = cturns_agents_cturns[xor128_tid.rand() % num_cturns_agents_cturns];
            } while (pused_tid[perm_tid[idx]]);
            pused_tid[perm_tid[idx++]] = 1;
          }
        }
      }
      /*for (int i = 0; i < N; ++i) {
        do {
          perm_tid[idx] = xor128_tid.rand() % N;
        } while (pused_tid[perm_tid[idx]]);
        pused_tid[perm_tid[idx++]] = 1;
      }*/    
    }
    ++is_covered_idx_tid;
    for (int aid = 0; aid < N; ++aid) {
      auto& tmax_at_poz_node_tid_aid = tmax_at_poz_node_tid[aid];
      tmax_at_poz_node_tid_aid = -1;
      auto& tmp_path_tid_aid = tmp_path_tid[aid];
      CopyPath(ipath[aid], &tmp_path_tid_aid);
      const auto& agent_aid = agent[aid];
      if (agent_aid.status == DONE_REMOVED) continue;
      CoverPath(aid, tmp_path_tid_aid, covered_by_tid, is_covered_tid, is_covered_idx_tid);
      if (agent_aid.inside_poz) {
        tmax_at_poz_node_tid_aid = TINIT;
        for (int t = TINIT + 1; t <= tmp_path_tid_aid.tmax; ++t) {
          const auto& path_elem = tmp_path_tid_aid.p[t];
          if (path_elem.node != agent_aid.poz_node) break;
          tmax_at_poz_node_tid_aid = t;
        }
      }
    }
    for (int idx = 0; idx < N; ++idx) {
      const auto& aid = perm_tid[idx];
      const auto& agent_aid = agent[aid];
      if (agent_aid.status == DONE_REMOVED) continue;
      auto& tmp_path_tid_aid = tmp_path_tid[aid];
      UncoverPath(aid, tmp_path_tid_aid, covered_by_tid, is_covered_tid, is_covered_idx_tid);
      FindBestPath(aid, covered_by_tid, is_covered_tid, is_covered_idx_tid, can_reach_tid, can_reach_idx_tid, can_reach_with_t1_tid, heap_tid, heap_size_tid, prev_tid, tmp_path_tid, &tmp_path2_tid[aid], tmax_at_poz_node_tid);
      CoverPath(aid, tmp_path_tid_aid, covered_by_tid, is_covered_tid, is_covered_idx_tid);
    }
    if (RunConsistencyChecks(tmp_path_tid, covered_by_tid, is_covered_tid, is_covered_idx_tid, false)) {   
      int num_done_agents = 0;
      double cost = 0.0;
      int max_tmax = 0;
      for (int aid = 0; aid < N; ++aid) {
        const auto& agent_aid = agent[aid];
        if (agent_aid.status == DONE_REMOVED) continue;      
        const auto& tmp_path_tid_aid = tmp_path_tid[aid];
        assert(tmp_path_tid_aid.tmax > TINIT);
        if (tmp_path_tid_aid.tmax > TINIT && tmp_path_tid_aid.tmax <= T && tmp_path_tid_aid.p[tmp_path_tid_aid.tmax].node == agent_aid.target_node) {
          ++num_done_agents;
          cost += GetScore(tmp_path_tid_aid.tmax);
          if (tmp_path_tid_aid.tmax > max_tmax) max_tmax = tmp_path_tid_aid.tmax;
        }
      }
      if (num_done_agents >= 1) cost /= num_done_agents;

      {
        lock_guard<mutex> guard(m);
        if (num_done_agents > MAX_DONE_AGENTS || (num_done_agents == MAX_DONE_AGENTS && cost < MIN_COST - 1e-6)) {
          MAX_DONE_AGENTS = num_done_agents;
          MIN_COST = cost;
          updated_best_solution = true;        
          CopyTmpPathToPath(tid);
          DBG(0, "[RandomPermutations] rerun=%d tid=%d trial=%d/%d maxda=%d/%d minc=%.6lf time=%.3lf\n", rerun, tid, trial, ntries, MAX_DONE_AGENTS, num_planned, MIN_COST, GetTime() - TSTART);
        }
      }
    }
  }
}

bool any_best_solution_updates;

void RegenerateFullPlan(int max_reruns, int num_permutations) {
  ++num_reschedules;
  any_best_solution_updates = false;
  updated_best_solution = true;
  rerun = 0;

  while (updated_best_solution && rerun < max_reruns) {
    updated_best_solution = false;
    ++rerun;

    for (int node = 0; node < nnodes; ++node) tend_ongoing_move[node] = TINIT;
    for (int aid = 0; aid < N; ++aid) {
      const auto& agent_aid = agent[aid];
      if (!agent_aid.inside_poz || agent_aid.status == DONE_REMOVED) continue;
      if (agent_aid.moving_to_node >= 0) {
        const auto& ipath_aid = ipath[aid];
        const auto& first_path_elem = ipath_aid.p[TINIT];
        assert(ipath_aid.tmax >= TINIT);
        tend_ongoing_move[agent_aid.moving_to_node] = TINIT + agent_aid.malfunc + max(0, agent_aid.cturns - first_path_elem.num_partial_turns);
      }
    }
    
    int max_threads = thread::hardware_concurrency();
    if (max_threads > MAX_NUM_THREADS) max_threads = MAX_NUM_THREADS;
    DBG(2, "max_threads=%d\n", max_threads);

    if (N >= 2) {
      thread** th = nullptr;
      if (max_threads >= 2) th = new thread*[max_threads - 1];

      // Random Permutations.
      if (max_threads >= 2) {
        for (int tid = 0; tid + 1 < max_threads; ++tid) th[tid] = new thread([tid, num_permutations]{
          RandomPermutations(tid, num_permutations);
        });
      }
      RandomPermutations(max_threads - 1, num_permutations);
      if (max_threads >= 2) {
        for (int tid = 0; tid + 1 < max_threads; ++tid) {
          auto& th_tid = th[tid];
          th_tid->join();
          delete th_tid;
        }
        delete th;
      }
    } else RandomPermutations(0, 1);

    if (updated_best_solution) {
      for (int aid = 0; aid < N; ++aid) CopyPath(path[aid], &ipath[aid]);
      any_best_solution_updates = true;
    }
  }

  if (any_best_solution_updates) RecomputeCheckpoints();
}

int nodecnt[MAXNODES], nodecnt_marked[MAXNODES], nodecnt_marked_idx;
char is_free[MAXNODES];

int next_aidx[NMAX], next_vidx[MAXNODES];

void SwapVisitingOrder(int aid1, int idx1, int aid2) {
  const auto& checkpoints_aid1 = checkpoints[aid1];
  const auto& checkpoints_aid2 = checkpoints[aid2];
  const int node = checkpoints_aid1[idx1];
  int idx2 = next_aidx[aid2] - 1;
  while (idx2 >= 0 && checkpoints_aid2[idx2] != node) --idx2;
  assert(idx2 >= 0);
  const auto& num_checkpoints_aid1 = num_checkpoints[aid1];
  const auto& num_checkpoints_aid2 = num_checkpoints[aid2];
  const auto& checkpoints_cnt_aid1 = checkpoints_cnt[aid1];
  const auto& checkpoints_cnt_aid2 = checkpoints_cnt[aid2];

  int num_swaps = 0;
  while (idx1 < num_checkpoints_aid1 && idx2 < num_checkpoints_aid2 && checkpoints_aid1[idx1] == checkpoints_aid2[idx2]) {
    int vidx1 = -1, cnt1 = 0, vidx2 = -1, cnt2 = 0;
    auto& ipath_visiting_order_node = ipath_visiting_order[checkpoints_aid1[idx1]];
    for (int idx = 0; idx < ipath_visiting_order_node.size() && (vidx1 < 0 || vidx2 < 0); ++idx) {
      const auto& elem_pair = ipath_visiting_order_node[idx];
      if (elem_pair.second == aid1) {
        ++cnt1;
        if (cnt1 == checkpoints_cnt_aid1[idx1]) vidx1 = idx;
      }
      if (elem_pair.second == aid2) {
        ++cnt2;
        if (cnt2 == checkpoints_cnt_aid2[idx2]) vidx2 = idx;
      }
    }
    assert(vidx1 >= 0 && vidx2 >= 0);
    assert(vidx1 > vidx2);
    ipath_visiting_order_node[vidx1].second = aid2;
    ipath_visiting_order_node[vidx2].second = aid1;
    ++idx1;
    ++idx2;
    ++num_swaps;
  }
  
  DBG(2, "[SwapVisitingOrder] aid1=%d aid2=%d idx1=%d num_swaps=%d\n", aid1, aid2, idx1, num_swaps);
}

#define QMOD 255
int qaid[QMOD + 1];

bool AdjustIPaths() {
  ++num_adjust_ipaths;

  nodecnt_marked_idx = 0;
  for (int node = 0; node < nnodes; ++node) {
    ipath_visiting_order[node].clear();
    nodecnt_marked[node] = 0;
  }

  int max_tmax = 0;

  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    const auto& ipath_aid = ipath[aid];
    auto& path_aid = path[aid];
    if (agent_aid.status == DONE_REMOVED) {
      CopyPath(ipath_aid, &path_aid);
      continue;
    }
    
    if (ipath_aid.p[ipath_aid.tmax].node == agent_aid.target_node && ipath_aid.tmax > max_tmax) max_tmax = ipath_aid.tmax;

    auto& new_path_elem = path_aid.p[TINIT];
    new_path_elem.node = agent_aid.poz_node;
    new_path_elem.o = agent_aid.poz_o;
    new_path_elem.num_partial_turns = 0;
    while (fabs(agent_aid.poz_frac - new_path_elem.num_partial_turns * agent_aid.speed) >= ZERO)
      ++new_path_elem.num_partial_turns;
    new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
    if (agent_aid.inside_poz) {
      const auto& first_expected_step = ipath_aid.p[TINIT];
      if (new_path_elem.num_partial_turns >= 1) {
        if (first_expected_step.num_partial_turns >= 1) {
          if (agent_aid.fresh_malfunc)
            assert(first_expected_step.num_partial_turns >= new_path_elem.num_partial_turns);
          assert(first_expected_step.node == agent_aid.poz_node);
          assert(first_expected_step.o == agent_aid.poz_o);
          new_path_elem.moving_to_node = first_expected_step.moving_to_node;
          new_path_elem.moving_to_o = first_expected_step.moving_to_o;
          assert(0 <= new_path_elem.moving_to_node && new_path_elem.moving_to_node < nnodes);
          assert(0 <= new_path_elem.moving_to_o && new_path_elem.moving_to_o < nnodes);            
        } else {
          assert(new_path_elem.num_partial_turns >= agent_aid.cturns - 1);
          assert(first_expected_step.node != agent_aid.poz_node || first_expected_step.o != agent_aid.poz_o);
          assert(first_expected_step.moving_to_node < 0);
          assert(first_expected_step.moving_to_o < 0);
          new_path_elem.moving_to_node = first_expected_step.node;
          new_path_elem.moving_to_o = first_expected_step.o;
        }
      }
      if (agent_aid.fresh_malfunc) new_path_elem.how_i_got_here = MALFUNCTIONED;      
      else new_path_elem.how_i_got_here = first_expected_step.how_i_got_here;
    } else new_path_elem.how_i_got_here = OUTSIDE_SRC;
    
    const int tmin = TINIT + max(0, agent_aid.malfunc - (agent_aid.inside_poz ? 0 : 1));
    for (int t = TINIT + 1; t <= tmin && t <= T; ++t) {
      auto& new_path_elem = path_aid.p[t];
      memcpy(&new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
      new_path_elem.how_i_got_here = agent_aid.inside_poz ? MALFUNCTIONED : OUTSIDE_SRC;
    }

    const auto& next_checkpoint_aid = next_checkpoint[aid];
    const auto& num_checkpoints_aid = num_checkpoints[aid];
    
    if (next_checkpoint_aid == num_checkpoints_aid) {
      assert(!agent_aid.inside_poz);
      continue;
    }
    
    assert(1 <= next_checkpoint_aid && next_checkpoint_aid < num_checkpoints_aid);

    auto& tcheckpoints_aid = tcheckpoints[aid];
    tcheckpoints_aid[next_checkpoint_aid - 1] = path_aid.tmax = tmin;

    const auto& checkpoints_aid = checkpoints[aid];
    const auto& checkpoints_o_aid = checkpoints_o[aid];
    auto& checkpoints_cnt_aid = checkpoints_cnt[aid];

    ++nodecnt_marked_idx;
   
    if (agent_aid.inside_poz) {
      nodecnt_marked[agent_aid.poz_node] = nodecnt_marked_idx;
      nodecnt[agent_aid.poz_node] = checkpoints_cnt_aid[next_checkpoint_aid - 1] = 1;
      assert(checkpoints_aid[next_checkpoint_aid - 1] == agent_aid.poz_node);
    }
    
    for (int cid = next_checkpoint_aid; cid < num_checkpoints_aid; ++cid) {
      const auto& node = checkpoints_aid[cid];
      if (nodecnt_marked[node] != nodecnt_marked_idx) {
        nodecnt_marked[node] = nodecnt_marked_idx;
        nodecnt[node] = 0;
      }
      checkpoints_cnt_aid[cid] = ++nodecnt[node];
    }
  }

  RepopulateVisitingOrders();

  auto& is_covered_0 = is_covered[0];
  auto& is_covered_idx_0 = is_covered_idx[0];
  auto& covered_by_0 = covered_by[0];

  bool updated_visiting_order = true;
  int nelems = 0;

  while (updated_visiting_order) {  
    updated_visiting_order = false;
    ++is_covered_idx_0;

    for (int node = 0; node < nnodes; ++node) is_free[node] = 1;

    for (int aid = 0; aid < N; ++aid) {
      const auto& agent_aid = agent[aid];
      if (agent_aid.status == DONE_REMOVED) continue;
      next_aidx[aid] = next_checkpoint[aid];
      auto& path_aid = path[aid];
      const int tmin = TINIT + max(0, agent_aid.malfunc - (agent_aid.inside_poz ? /*1*/ 0 : /*2*/ 1));
      for (int t = TINIT + 1; t <= tmin && t <= T; ++t) {
        auto& new_path_elem = path_aid.p[t];
        memcpy(&new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
        new_path_elem.how_i_got_here = agent_aid.inside_poz ? MALFUNCTIONED : OUTSIDE_SRC;
        if (agent_aid.inside_poz) {
          assert(is_covered_0[t][new_path_elem.node] != is_covered_idx_0);
          is_covered_0[t][new_path_elem.node] = is_covered_idx_0;
          covered_by_0[t][new_path_elem.node] = aid;
        }
      }

      path_aid.tmax = tmin;
      
      if (agent_aid.inside_poz) {
        is_free[agent_aid.poz_node] = 0;
        is_covered_0[TINIT][agent_aid.poz_node] = is_covered_idx_0;
        covered_by_0[TINIT][agent_aid.poz_node] = aid;
      }
    }

    int qli = 0, qls = 0;

    for (int node = 0; node < nnodes; ++node) {
      next_vidx[node] = 0;
      auto& ipath_visiting_order_node = ipath_visiting_order[node];
      if (ipath_visiting_order_node.empty()) continue;
      for (auto& elem_pair : ipath_visiting_order_node) elem_pair.first = TINIT;
      const auto& aid_0 = ipath_visiting_order_node[0].second;
      if (is_free[node] && checkpoints[aid_0][next_aidx[aid_0]] == node) {
        qaid[qls] = aid_0;
        qls = (qls + 1) & QMOD;
      }
    }

    nelems = 0;

    int tmin_incomplete = -1, aid_incomplete = -1;
    
    while (qli != qls) {
      ++nelems;

      const auto& aid = qaid[qli];
      const auto& agent_aid = agent[aid];
      const auto& checkpoints_aid = checkpoints[aid];
      const auto& checkpoints_o_aid = checkpoints_o[aid];
      auto& tcheckpoints_aid = tcheckpoints[aid];
      auto& next_aidx_aid = next_aidx[aid];
      const auto& next_checkpoint_aid = next_checkpoint[aid];
      assert(next_checkpoint_aid <= next_aidx_aid && next_aidx_aid < num_checkpoints[aid]);
      const auto& curr_node = checkpoints_aid[next_aidx_aid - 1];
      const auto& next_node = checkpoints_aid[next_aidx_aid];

      qli = (qli + 1) & QMOD;

      assert(0 <= next_node && next_node < nnodes);
      assert(is_free[next_node]);
      if (curr_node >= 0) assert(!is_free[curr_node]);

      const auto& t_curr_node = tcheckpoints_aid[next_aidx_aid - 1];
      const auto& next_o = checkpoints_o_aid[next_aidx_aid];

      auto& next_vidx_next_node = next_vidx[next_node];
      auto& ipath_visiting_order_next_node = ipath_visiting_order[next_node];
      assert(next_vidx_next_node < ipath_visiting_order_next_node.size());
      assert(ipath_visiting_order_next_node[next_vidx_next_node].second == aid);

      int tend_move = ipath_visiting_order_next_node[next_vidx_next_node].first;

      //const bool print_debug = (aid == 76 || aid == 24) && (curr_node == 117 || next_node == 117);
      const bool print_debug = false;      

      if (print_debug) DBG(0, "\n[AdjustIPaths-Debug-A] aid=%d next_aidx_aid=%d next_node=%d:(%d %d) next_vidx_next_node=%d curr_node=%d:(%d %d) t_curr_node=%d tend_move=%d(init) \n", aid, next_aidx_aid, next_node, node[next_node].row, node[next_node].col, next_vidx_next_node, curr_node, node[curr_node].row, node[curr_node].col, t_curr_node, tend_move);

      auto& path_aid = path[aid];

      if (next_aidx_aid > next_checkpoint_aid && path_aid.tmax <= T) assert(path_aid.p[path_aid.tmax].num_partial_turns == 0);

      const int move_duration = max(1, curr_node < 0 ? 1 : (next_aidx_aid == next_checkpoint_aid ? agent_aid.cturns - path_aid.p[TINIT].num_partial_turns : agent_aid.cturns));
      int tstart_move = tend_move - move_duration;
      if (tstart_move < t_curr_node) tstart_move = t_curr_node;

      if (print_debug) DBG(0, "\n[AdjustIPaths-Debug-B] aid=%d next_aidx_aid=%d next_node=%d:(%d %d) next_vidx_next_node=%d curr_node=%d:(%d %d) t_curr_node=%d tstart_move=%d tend_move=%d move_duration=%d\n", aid, next_aidx_aid, next_node, node[next_node].row, node[next_node].col, next_vidx_next_node, curr_node, node[curr_node].row, node[curr_node].col, t_curr_node, tstart_move, tend_move, move_duration);

      assert(path_aid.tmax == t_curr_node);
      if (t_curr_node <= T && curr_node >= 0) assert(path_aid.p[t_curr_node].node == curr_node);
      int t = t_curr_node + 1;

      tend_move = tstart_move + move_duration;
    
      if (!agent_aid.inside_poz && next_aidx_aid == next_checkpoint_aid) {
        for (; t <= tstart_move; ++t) {
          assert(path_aid.tmax == t - 1);
          path_aid.tmax = t;
          if (t <= T) {
            auto& new_path_elem = path_aid.p[t];
            memcpy(&new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
            new_path_elem.how_i_got_here = OUTSIDE_SRC;
          }
        }
        assert(t == tend_move);
        path_aid.tmax = t;
        if (t <= T) {
          auto& new_path_elem = path_aid.p[t];
          memcpy(&new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
          new_path_elem.how_i_got_here = ENTERED_SRC;
          is_covered_0[t][new_path_elem.node] = is_covered_idx_0;
          covered_by_0[t][new_path_elem.node] = aid;
          assert(t == tend_move);
        }
        ++t;
      }

      if (curr_node >= 0) {
        if (path_aid.p[TINIT].num_partial_turns >= 1) assert(agent_aid.moving_to_node >= 0);
        if (next_aidx_aid > next_checkpoint_aid || /*path_aid.p[TINIT].num_partial_turns == 0*/ agent_aid.moving_to_node < 0) {
          for (; t <= tstart_move; ++t) {
            assert(path_aid.tmax == t - 1);
            path_aid.tmax = t;
            if (t <= T) {
              auto& new_path_elem = path_aid.p[t];
              memcpy(&new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
              new_path_elem.how_i_got_here = WAITED;
              assert(new_path_elem.num_partial_turns == 0);
              is_covered_0[t][new_path_elem.node] = is_covered_idx_0;
              covered_by_0[t][new_path_elem.node] = aid;
            }
          }
        }
        int num_partial_turns = next_aidx_aid > next_checkpoint_aid ? 0 : path_aid.p[TINIT].num_partial_turns;
        for (; 1; ++t) {
          assert(path_aid.tmax == t - 1);
          assert(t > TINIT);
          path_aid.tmax = t;
          PathElem* new_path_elem = nullptr;
          if (t <= T) new_path_elem = &path_aid.p[t];
          if (new_path_elem != nullptr) memcpy(new_path_elem, &path_aid.p[t - 1], sizeof(PathElem));
          ++num_partial_turns;
          if (print_debug) DBG(0, " aid=%d t=%d/%d npt=%d\n", aid, t, tend_move, num_partial_turns);
          if (new_path_elem != nullptr) {
            new_path_elem->num_partial_turns = num_partial_turns;
            if (new_path_elem->num_partial_turns == 1) new_path_elem->how_i_got_here = STARTED_MOVING;
            else new_path_elem->how_i_got_here = CONTINUED_MOVING;
            new_path_elem->moving_to_node = next_node;
            new_path_elem->moving_to_o = next_o;
            if (new_path_elem->num_partial_turns >= 2) {
              if (print_debug) DBG(0, " prev_moving_to=(%d %d)\n", path_aid.p[t - 1].moving_to_node, path_aid.p[t - 1].moving_to_o);
              assert(path_aid.p[t - 1].moving_to_node == next_node);
              assert(path_aid.p[t - 1].moving_to_o == next_o);
            }
          }
          if (num_partial_turns >= agent_aid.cturns) {
            if (t >= tend_move) {
              if (new_path_elem != nullptr) {
                new_path_elem->node = next_node;
                new_path_elem->o = next_o;
                new_path_elem->moving_to_node = new_path_elem->moving_to_o = -1;
                new_path_elem->num_partial_turns = 0;
                if (new_path_elem->node != agent_aid.target_node) {
                  is_covered_0[t][new_path_elem->node] = is_covered_idx_0;
                  covered_by_0[t][new_path_elem->node] = aid;
                }
              }
              tend_move = t;
              ++t;
              break;
            } else if (new_path_elem != nullptr) {
              if (print_debug) DBG(0, " >>> is_cov_next_node=%d cov_by=%d\n", is_covered_0[t][next_node] == is_covered_idx_0, covered_by_0[t][next_node]);
              if ((is_covered_0[t][next_node] != is_covered_idx_0 &&                  
                  (is_covered_0[t - 1][next_node] != is_covered_idx_0 ||
                   covered_by_0[t - 1][next_node] < aid)) ||
                  (is_covered_0[t][next_node] == is_covered_idx_0 && covered_by_0[t][next_node] > aid &&
                   (is_covered_0[t - 1][next_node] != is_covered_idx_0 ||
                    covered_by_0[t - 1][next_node] != covered_by_0[t][next_node]))) {
                DBG(0, "||| [AdjustIPaths] Agent in motion cannot wait for delayed agent: aid=%d next_aidx_aid=%d nextcp=%d tend=%d t=%d curr_node=%d:(%d %d) next_node=%d:(%d %d) next_vidx_next_node=%d target_node=%d:(%d %d) nelems=%d\n", aid, next_aidx_aid, next_checkpoint[aid], tend_move, t, curr_node, node[curr_node].row, node[curr_node].col, next_node, node[next_node].row, node[next_node].col, next_vidx_next_node, agent_aid.target_node, node[agent_aid.target_node].row, node[agent_aid.target_node].col, nelems);
                assert(next_vidx_next_node >= 1);
                const int prev_aid = ipath_visiting_order_next_node[next_vidx_next_node - 1].second;
                DBG(0,  "  prev_aid=%d\n", prev_aid);
                SwapVisitingOrder(aid, next_aidx_aid, prev_aid);
                updated_visiting_order = true;
                break;
              }
              if (new_path_elem->node != agent_aid.target_node) {
                is_covered_0[t][new_path_elem->node] = is_covered_idx_0;
                covered_by_0[t][new_path_elem->node] = aid;
              }            
            }
          } else if (new_path_elem != nullptr) {
            is_covered_0[t][new_path_elem->node] = is_covered_idx_0;
            covered_by_0[t][new_path_elem->node] = aid;
            assert(num_partial_turns >= 1);
            assert(0 <= new_path_elem->moving_to_node && new_path_elem->moving_to_node < nnodes);            
          }
        }
      }
    
      if (updated_visiting_order) break;

      if (print_debug) DBG(0, "\n[AdjustIPaths-Debug-C] aid=%d next_aidx_aid=%d next_node=%d:(%d %d) next_vidx_next_node=%d curr_node=%d:(%d %d) t_curr_node=%d tstart_move=%d tend_move=%d move_duration=%d\n", aid, next_aidx_aid, next_node, node[next_node].row, node[next_node].col, next_vidx_next_node, curr_node, node[curr_node].row, node[curr_node].col, t_curr_node, tstart_move, tend_move, move_duration);

      if (t != tend_move + 1) {
        DBG(0, "!!!A aid=%d next_aidx_aid=%d curr_node=%d curr_npt=%d curr_how=%d next_node=%d tstart_move=%d tend_move=%d t=%d: last_path_elem=(node=%d npt=%d/%d how=%d)\n", aid, next_aidx_aid, curr_node, path_aid.p[t_curr_node].num_partial_turns, path_aid.p[t_curr_node].how_i_got_here, next_node, tstart_move, tend_move, t, path_aid.p[path_aid.tmax].node, path_aid.p[path_aid.tmax].num_partial_turns, agent_aid.cturns, path_aid.p[path_aid.tmax].how_i_got_here);
        exit(1);
      }

      assert(path_aid.tmax == tend_move);
      if (tend_move <= T && path_aid.p[tend_move].node != next_node) {
        DBG(0, "!!!B aid=%d next_aidx_aid=%d curr_node=%d curr_npt=%d curr_how=%d next_node=%d tstart_move=%d tend_move=%d: last_path_elem=(node=%d npt=%d/%d how=%d)\n", aid, next_aidx_aid, curr_node, path_aid.p[t_curr_node].num_partial_turns, path_aid.p[t_curr_node].how_i_got_here, next_node, tstart_move, tend_move, path_aid.p[path_aid.tmax].node, path_aid.p[path_aid.tmax].num_partial_turns, agent_aid.cturns, path_aid.p[path_aid.tmax].how_i_got_here);
        exit(1);
      }

      tcheckpoints_aid[next_aidx_aid] = tend_move;
      if (curr_node >= 0) is_free[curr_node] = 1;
      is_free[next_node] = 0;
      ++next_aidx_aid;
      ++next_vidx_next_node;

      if (next_aidx_aid < num_checkpoints[aid]) {
        const auto& next2_node = checkpoints_aid[next_aidx_aid];
        auto& next_vidx2_next2_node = next_vidx[next2_node];
        auto& ipath_visiting_order_next2_node = ipath_visiting_order[next2_node];
        if (next2_node != curr_node && ipath_visiting_order_next2_node[next_vidx2_next2_node].second == aid && is_free[next2_node]) {
          qaid[qls] = aid;
          qls = (qls + 1) & QMOD;
        }
      } else if (next_node == agent_aid.target_node) {
        // Free up next_node right away.
        is_free[next_node] = 1;
        if (next_vidx_next_node < ipath_visiting_order_next_node.size()) {
          const auto& aid1 = ipath_visiting_order_next_node[next_vidx_next_node].second;
          const auto& path_aid1 = path[aid1];
          const auto& last_path_elem = path_aid1.p[path_aid1.tmax];
          int tend_move_aid1 = tend_move + (aid1 < aid && agent[aid1].target_node != next_node ? 1 : 0);
          const auto& agent_aid1 = agent[aid1];
          if (last_path_elem.num_partial_turns == 0 && (next_aidx[aid1] > next_checkpoint[aid1] || agent[aid1].moving_to_node < 0)) {
            const int min_tstart = tend_move;
            assert(t_curr_node < tend_move);
            if (tend_move_aid1 - agent_aid1.cturns < min_tstart) tend_move_aid1 = min_tstart + agent_aid1.cturns;
          }
          ipath_visiting_order_next_node[next_vidx_next_node].first = tend_move_aid1;
          if (checkpoints[aid1][next_aidx[aid1]] == next_node) {
            qaid[qls] = aid1;
            qls = (qls + 1) & QMOD;
          }
        }
      }

      if (curr_node >= 0) {
        const auto& next_vidx_curr_node = next_vidx[curr_node];
        auto& ipath_visiting_order_curr_node = ipath_visiting_order[curr_node];
        if (next_vidx_curr_node < ipath_visiting_order_curr_node.size()) {
          const auto& aid1 = ipath_visiting_order_curr_node[next_vidx_curr_node].second;
          const auto& path_aid1 = path[aid1];
          const auto& last_path_elem = path_aid1.p[path_aid1.tmax];
          int tend_move_aid1 = tend_move + (aid1 < aid ? 1 : 0);
          const auto& agent_aid1 = agent[aid1];
          if (last_path_elem.num_partial_turns == 0 && (next_aidx[aid1] > next_checkpoint[aid1] || agent[aid1].moving_to_node < 0)) {
            const int min_tstart = t_curr_node;//USE_STRICT_SPACING_TO_AVOID_DEADLOCKS ? tend_move : t_curr_node;
            assert(t_curr_node < tend_move);
            if (tend_move_aid1 - agent_aid1.cturns < min_tstart) tend_move_aid1 = min_tstart + agent_aid1.cturns;
          }
          ipath_visiting_order_curr_node[next_vidx_curr_node].first = tend_move_aid1;
          if (checkpoints[aid1][next_aidx[aid1]] == curr_node) {
            qaid[qls] = aid1;
            qls = (qls + 1) & QMOD;
          }
        }
      }
    }
    
    DBG(2, "[AdjustIPaths] nelems=%d updvis=%d\n", nelems, updated_visiting_order);
    if (updated_visiting_order) continue;
    
    if (aid_incomplete < 0) {
      for (int node = 0; node < nnodes; ++node) {
        const auto& next_vidx_node = next_vidx[node];
        const auto& ipath_visiting_order_node = ipath_visiting_order[node];
        if (next_vidx_node < ipath_visiting_order_node.size()) {
          const auto& aid = ipath_visiting_order_node[next_vidx_node].second;
          const auto& next_aidx_aid = next_aidx[aid];
          DBG(0, "[AdjustIPaths] incomplete: node=%d is_free=%d next_vidx=%d/%d: aid=%d next_aidx_aid=%d/%d:node=%d t=%d nextcp=%d\n", node, is_free[node], next_vidx_node, ipath_visiting_order_node.size(), aid, next_aidx_aid, num_checkpoints[aid], checkpoints[aid][next_aidx_aid], checkpoints_t[aid][next_aidx_aid], next_checkpoint[aid]);
          assert(next_aidx_aid < num_checkpoints[aid]);
          assert(!is_free[node] || checkpoints[aid][next_aidx_aid] != node);
          const auto& t_inc = max(ipath_visiting_order_node[next_vidx_node].first, tcheckpoints[aid][next_aidx_aid - 1]);
          if (aid_incomplete < 0 || t_inc < tmin_incomplete) {
            aid_incomplete = aid;
            tmin_incomplete = t_inc;
          }
        }
      }

      assert(aid_incomplete < 0);
    }
  }

  ++is_covered_idx_0;
  
  for (int aid = 0; aid < N; ++aid) {
    auto& ipath_aid = ipath[aid];
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;

    const auto& num_checkpoints_aid = num_checkpoints[aid];
    const auto& checkpoints_aid = checkpoints[aid];
    const auto& tcheckpoints_aid = tcheckpoints[aid];
    const auto& next_aidx_aid = next_aidx[aid];
    
    if (next_aidx_aid < num_checkpoints_aid) {
      const auto& v = checkpoints_aid[next_aidx_aid];
      const auto& ipath_visiting_order_v = ipath_visiting_order[v];
      const auto& next_vidx_v = next_vidx[v];
      assert(next_vidx_v < ipath_visiting_order_v.size());
      DBG(0, "!!! [AdjustIPaths] aid=%d next_aidx_aid=%d/%d: v=%d(%d) next_vidx_v=%d/%d: t=%d vaid=%d\n", aid, next_aidx_aid, num_checkpoints_aid, v, is_free[v], next_vidx_v, ipath_visiting_order_v.size(), ipath_visiting_order_v[next_vidx_v].first, ipath_visiting_order_v[next_vidx_v].second);
      exit(1);
    }

    // Construct the path.
    auto& path_aid = path[aid];
    if (path_aid.tmax > T) path_aid.tmax = T;

    if (path_aid.p[path_aid.tmax].how_i_got_here == OUTSIDE_SRC) {
      for (int t = path_aid.tmax + 1; t <= T; ++t) {
        auto& new_path_elem = path_aid.p[t];
        new_path_elem.node = agent_aid.poz_node;
        new_path_elem.o = agent_aid.poz_o;
        new_path_elem.moving_to_node = new_path_elem.moving_to_o = -1;
        new_path_elem.num_partial_turns = 0;
        new_path_elem.how_i_got_here = OUTSIDE_SRC;      
      }
      path_aid.tmax = T;
    } else if (path_aid.tmax < T && path_aid.p[path_aid.tmax].node != agent_aid.target_node) {
      const auto& last_num_partial_turns = path_aid.p[path_aid.tmax].num_partial_turns;
      const auto& last_how_i_got_here = path_aid.p[path_aid.tmax].how_i_got_here;
      DBG(3, "!!! [AdjustIPaths] aid=%d next_aidx_aid=%d/%d tmax=%d/%d last_node=%d:(%d %d) npt=%d/%d target_node=%d:(%d %d)\n", aid, next_aidx[aid], num_checkpoints[aid], path_aid.tmax, T, path_aid.p[path_aid.tmax].node, node[path_aid.p[path_aid.tmax].node].row, node[path_aid.p[path_aid.tmax].node].col, last_num_partial_turns, agent_aid.cturns, agent_aid.target_node, node[agent_aid.target_node].row, node[agent_aid.target_node].col);
      while (path_aid.tmax < T) {
        ++path_aid.tmax;
        auto& new_path_elem = path_aid.p[path_aid.tmax];
        memcpy(&new_path_elem, &path_aid.p[path_aid.tmax - 1], sizeof(PathElem));
        if (last_num_partial_turns == 0)
          new_path_elem.how_i_got_here = last_how_i_got_here == OUTSIDE_SRC ? OUTSIDE_SRC : WAITED;
        else {
          ++new_path_elem.num_partial_turns;
          new_path_elem.how_i_got_here = CONTINUED_MOVING;
        }
      }
    }

    assert(path_aid.tmax <= T);
    for (int t = TINIT; t <= path_aid.tmax; ++t) {
      const auto& path_elem = path_aid.p[t];
      if (path_elem.how_i_got_here == OUTSIDE_SRC) continue;
      if (path_elem.node == agent_aid.target_node) {
        assert(t == path_aid.tmax);
        continue;
      }
      assert(0 <= path_elem.node && path_elem.node < nnodes);
      if (is_covered_0[t][path_elem.node] == is_covered_idx_0) {
        DBG(0, "!!! [AdjustIPaths] aid=%d t=%d node=%d already covered by %d\n", aid, t, path_elem.node, covered_by_0[t][path_elem.node]);
        exit(1);
      }
      is_covered_0[t][path_elem.node] = is_covered_idx_0;
      covered_by_0[t][path_elem.node] = aid;
    }

    if (path_aid.tmax != ipath_aid.tmax || path_aid.p[path_aid.tmax].node != ipath_aid.p[path_aid.tmax].node) {
      DBG(3, "[AdjustIPaths] aid=%d diff: prev:(tmax=%d node=%d) curr:(tmax=%d node=%d)\n", aid, path_aid.tmax, path_aid.p[path_aid.tmax].node, ipath_aid.tmax, ipath_aid.p[ipath_aid.tmax].node);
    }

    CopyPath(path_aid, &ipath_aid);
  }
  
  RunConsistencyChecks(path, covered_by_0, is_covered_0, is_covered_idx_0);
  
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;
    auto& checkpoints_t_aid = checkpoints_t[aid];
    const auto& tcheckpoints_aid = tcheckpoints[aid];
    const auto& num_checkpoints_aid = num_checkpoints[aid];
    for (int cid = next_checkpoint[aid]; cid < num_checkpoints_aid; ++cid)
      checkpoints_t_aid[cid] = tcheckpoints_aid[cid];
  }
  
  MAX_DONE_AGENTS = 0;
  MIN_COST = 0.0;
  int new_max_tmax = 0;
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    if (agent_aid.status == DONE_REMOVED) continue;
    const auto& path_aid = path[aid];
    assert(path_aid.tmax > TINIT);
    if (path_aid.p[path_aid.tmax].node == agent_aid.target_node) {
      ++MAX_DONE_AGENTS;
      MIN_COST += GetScore(path_aid.tmax);
      if (path_aid.tmax > new_max_tmax) new_max_tmax = path_aid.tmax;
    } else {
      assert(path_aid.tmax == T);
    }
  }
  if (MAX_DONE_AGENTS >= 1) MIN_COST /= MAX_DONE_AGENTS;

  const bool changed_important_data = MAX_DONE_AGENTS != num_planned || new_max_tmax > max_tmax;
  if (changed_important_data) {
    DBG(0, ">>> [AdjustIPaths] mda=%d/%d minc=%.6lf new_max_tmax=%d/%d\n", MAX_DONE_AGENTS, num_planned, MIN_COST, new_max_tmax, max_tmax);
  }

  CheckNonDeadlockPaths();

  return !changed_important_data;
}

void ReinitDataStructures() {
  MAX_DONE_AGENTS = 0;
  MIN_COST = 1e10;
  for (int aid = 0; aid < N; ++aid) path[aid].tmax = -1000;
  xor128[0].reset(13U);
  for (int tid = 1; tid < MAX_NUM_THREADS; ++tid) xor128[tid].reset(19997U * tid + 997U);
  for (int tid = 0; tid < MAX_NUM_THREADS; ++tid) {
    can_reach_idx[tid] = is_covered_idx[tid] = 0;
    auto& can_reach_tid = can_reach[tid];
    auto& is_covered_tid = is_covered[tid];
    for (int t = 0; t <= T + 2; ++t) {
      auto& can_reach_tid_t = can_reach_tid[t];
      auto& is_covered_tid_t = is_covered_tid[t];
      for (int i = 0; i < nnodes; ++i) {
        is_covered_tid_t[i] = 0;
        auto& can_reach_tid_t_i = can_reach_tid_t[i];
        for (int o = 0; o <= 3; ++o) can_reach_tid_t_i[o] = 0;
      }
    }
  }
}

int GetMove(int aid) {
  if (num_planned == 0) return DO_NOTHING;
  const auto& path_aid = path[aid];
  auto& agent_aid = agent[aid];
  if (path_aid.tmax < TINIT + 1) return DO_NOTHING;
  const auto& path_elem = path_aid.p[TINIT];
  const auto& path_elem_end_turn = path_aid.p[TINIT + 1];
  int action = DO_NOTHING;
  DBG(2, " GetMove for aid=%d: tmax=%d\n", aid, path_aid.tmax);
  if (path_elem_end_turn.how_i_got_here == ENTERED_SRC) {
    assert(path_elem.node == path_elem_end_turn.node);
    assert(path_elem.o == path_elem_end_turn.o);
    assert(path_elem.moving_to_node < 0);
    assert(path_elem.moving_to_o < 0);
    assert(path_elem_end_turn.num_partial_turns == 0);
    action = MOVE_FORWARD;
    assert(agent_aid.moving_to_node < 0);
  } else if (path_elem_end_turn.how_i_got_here == WAITED) {
    assert(path_elem.node == path_elem_end_turn.node);
    assert(path_elem.o == path_elem_end_turn.o);
    assert(path_elem.moving_to_node < 0);
    assert(path_elem.moving_to_o < 0);
    assert(path_elem.num_partial_turns == 0);
    assert(path_elem_end_turn.num_partial_turns == 0);
    action = STOP_MOVING;
    assert(agent_aid.moving_to_node < 0);
  } else if (path_elem_end_turn.how_i_got_here == STARTED_MOVING) {
    assert(path_elem_end_turn.num_partial_turns <= 1);
    int dst_node = -1, dst_o = -1;
    if (path_elem_end_turn.moving_to_node >= 0 && path_elem_end_turn.moving_to_o >= 0) {
      assert(path_elem_end_turn.num_partial_turns == 1);
      dst_node = path_elem_end_turn.moving_to_node;
      dst_o = path_elem_end_turn.moving_to_o;
    } else {
      assert(path_elem_end_turn.num_partial_turns == 0);
      dst_node = path_elem_end_turn.node;
      dst_o = path_elem_end_turn.o;
    }
    assert(path_elem.node != dst_node);
    assert(next[path_elem.node][path_elem.o][dst_o] == dst_node);
    if (dst_o == path_elem.o || dst_o == ((path_elem.o + 2) & 3))
      action = MOVE_FORWARD;
    else if (dst_o == ((path_elem.o + 1) & 3))
      action = MOVE_RIGHT;
    else if (dst_o == ((path_elem.o + 3) & 3))
      action = MOVE_LEFT;
    else {
      DBG(0, "Incorrect move!!!\n");
      exit(1);
    }
    assert(agent_aid.moving_to_node < 0 || (agent_aid.moving_to_node == dst_node && agent_aid.moving_to_o == dst_o));
    agent_aid.moving_to_node = dst_node;
    agent_aid.moving_to_o = dst_o;
  } else if (path_elem_end_turn.how_i_got_here == CONTINUED_MOVING) {
    if (path_elem.node == path_elem_end_turn.node) {
      assert(path_elem.o == path_elem_end_turn.o);
      assert(path_elem.moving_to_node == path_elem_end_turn.moving_to_node);
      assert(path_elem_end_turn.num_partial_turns == path_elem.num_partial_turns + 1);
    } else {
      assert(path_elem_end_turn.moving_to_node < 0);
      assert(path_elem_end_turn.moving_to_o < 0);
      assert(path_elem_end_turn.num_partial_turns == 0);
    }
  } else if (path_elem_end_turn.how_i_got_here == MALFUNCTIONED) {
    assert(path_elem.node == path_elem_end_turn.node);
    assert(path_elem.o == path_elem_end_turn.o);
    assert(path_elem.moving_to_node == path_elem_end_turn.moving_to_node);
    assert(path_elem.moving_to_o == path_elem_end_turn.moving_to_o);
    assert(path_elem.num_partial_turns == path_elem_end_turn.num_partial_turns);
  }
  return action;
}

void WriteMoves(const char* testid) {
  num_planned = 0;
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    const auto& ipath_aid = ipath[aid];
    if (ipath_aid.tmax > TINIT && ipath_aid.p[ipath_aid.tmax].node == agent_aid.target_node) ++num_planned;
  }
  sprintf(fname, "output-%s.txt", testid);
  FILE* f = fopen(fname, "w");
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    const int action = GetMove(aid);
    fprintf(f, "%d ", action);
    if (action != DO_NOTHING) DBG(2, "    Move aid=%d: %d\n", aid, action);
  }
  fprintf(f, "\n");
  fclose(f);
}

void SaveDataForReplay(const char* testid) {
  sprintf(fname, "saved-%s-%d.txt", testid, TINIT);
  FILE* f = fopen(fname, "w");
  fprintf(f, "1\n%d %d\n", H, W);
  for (int v = 0; v < nnodes; ++v) {
    const auto& node_v = node[v];
    const auto& next_v = next[v];
    for (int o1 = 0; o1 <= 3; ++o1) for (int o2 = 0; o2 <= 3; ++o2)
      if (next_v[o1][o2] >= 0) fprintf(f, "%d %d %d %d\n", node_v.row, node_v.col, o1, o2);
  }
  fprintf(f, "-1 -1 -1 -1\n%d %d\n%d %d %d\n", N, TINIT, num_reschedules, num_planned, num_adjust_ipaths_without_full_plan_regeneration);
  for (int aid = 0; aid < N; ++aid) {
    const auto& agent_aid = agent[aid];
    fprintf(f, "%d %d %d %d %d %d %.10lf %.10lf %d %d %d %d 1 1 %d %d %d\n", agent_aid.aid, agent_aid.poz_row, agent_aid.poz_col, agent_aid.poz_o, agent_aid.target_row, agent_aid.target_col, agent_aid.speed, agent_aid.poz_frac, agent_aid.malfunc, agent_aid.nr_malfunc, agent_aid.status, agent_aid.fresh_malfunc, agent_aid.moving_to_node >= 0 ? node[agent_aid.moving_to_node].row : -1, agent_aid.moving_to_node >= 0 ? node[agent_aid.moving_to_node].col : -1, agent_aid.moving_to_o);
    const auto& ipath_aid = ipath[aid];
    fprintf(f, "%d\n", ipath_aid.tmax);
    for (int t = TINIT; t <= ipath_aid.tmax; ++t) {
      const auto& path_elem = ipath_aid.p[t];
      fprintf(f, "%d %d %d %d %d %d %d %d\n", node[path_elem.node].row, node[path_elem.node].col, path_elem.o, path_elem.moving_to_node < 0 ? -1 : node[path_elem.moving_to_node].row, path_elem.moving_to_node < 0 ? -1 : node[path_elem.moving_to_node].col, path_elem.moving_to_o, path_elem.num_partial_turns, path_elem.how_i_got_here);
    }
    const auto& num_checkpoints_aid = num_checkpoints[aid];
    const auto& next_checkpoint_aid = next_checkpoint[aid];
    fprintf(f, "%d\n", num_checkpoints_aid - next_checkpoint_aid);
    const auto& checkpoints_aid = checkpoints[aid];
    const auto& checkpoints_o_aid = checkpoints_o[aid];
    const auto& checkpoints_t_aid = checkpoints_t[aid];
    for (int cid = next_checkpoint_aid; cid < num_checkpoints_aid; ++cid) {
      fprintf(f, "%d %d %d %d\n", node[checkpoints_aid[cid]].row, node[checkpoints_aid[cid]].col, checkpoints_o_aid[cid], checkpoints_t_aid[cid]);
    }
  }
  fprintf(f, "%d\n", T);
  fclose(f);
}

double total_time;

void GetMoves(const char* testid, bool replay_mode = false) {
  TSTART = GetTime();
  sprintf(fname, "input-%s.txt", testid);
  fin = fopen(fname, "r");
  ReadTransitionsMap();
  ReadAgentsData(replay_mode);
  fclose(fin);
  if (TINIT == 0 || replay_mode) {
    ReinitDataStructures();
    ComputeShortestPaths();
  }
  SCORE_EXPONENT1 = 1.0;
  SCORE_EXPONENT2 = 1.0;
  DBG(2, "testid=%s TINIT=%d: resc=%d nda=%d npl=%d\n", testid, TINIT, reschedule, num_done_agents, num_planned);
  if (reschedule) {
    if (TINIT >= MIN_TINIT_FOR_SAVE_DATA_FOR_REPLAY) SaveDataForReplay(testid);
    bool updated_paths_ok = false;
    if (TINIT >= 1) {
      updated_paths_ok = AdjustIPaths();
      ++num_adjust_ipaths_without_full_plan_regeneration;
    }
    const int kMaxNumAdjustIPathsWithoutFullPlanRegenartion = 3;//3;
    const bool full_regenerate_paths = !updated_paths_ok || num_adjust_ipaths_without_full_plan_regeneration > kMaxNumAdjustIPathsWithoutFullPlanRegenartion;
    if (full_regenerate_paths) {
      RegenerateFullPlan(4, TINIT == 0 ? 20 : 10);
      num_adjust_ipaths_without_full_plan_regeneration = 0;
    } else {
      assert(TINIT >= 1);
      RegenerateFullPlan(2, 2);
    }
    if (TINIT >= 1 && any_best_solution_updates) AdjustIPaths();
  }
  WriteMoves(testid);
  if (TINIT == 0) total_time = 0.0;
  total_time += GetTime() - TSTART;
  DBG(0, "[GetMoves] testid=%s TINIT=%d/%d ttime=%.3lf nresc=%d nadjip=%d nadjipwofpr=%d nda=%d npl=%d sum=%d/%d(%.2lf)\n", testid, TINIT, TEST, total_time, num_reschedules, num_adjust_ipaths, num_adjust_ipaths_without_full_plan_regeneration, num_done_agents, num_planned, num_done_agents + num_planned, N, 100.0 * (num_done_agents + num_planned) / N);
}

}

void GetMoves(const char* testid) {
  SOLVE::GetMoves(testid, false);
}

int main() {
  SOLVE::GetMoves("1", true);
  return 0;
}

