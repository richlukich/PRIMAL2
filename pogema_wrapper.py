import copy
from operator import sub, add
import gym
import numpy as np
import math
import warnings
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
from GroupLock import Lock
from matplotlib.colors import *
from gym.envs.classic_control import rendering
import imageio
from gym import spaces
from pogema.animation import AnimationMonitor, AnimationConfig
from pogema import pogema_v0, GridConfig
import time
from Map_Generator import maze_generator
from ACNet import ACNet
import tensorflow as tf
import os
from pogema.generator import generate_new_target
from copy import deepcopy

class BatchAgent():
    
    def __init__(self,master_network,rnn_state,):

        self.master_network = master_network
        self.rnn_state = rnn_state
            
    def act(self,observ):
        obs = observ[0]
        validActions = observ[1]
        actions = {}
        for key in obs.keys():
            s = obs[key]
            a_dist, v, rnn_state = sess.run([self.master_network.policy,
                                            self.master_network.value,
                                            self.master_network.state_out],
                                            feed_dict={self.master_network.inputs     : [s[0]],  # state
                                                    self.master_network.goal_pos   : [s[1]],  # goal vector
                                                    self.master_network.state_in[0]: self.rnn_state[0],
                                                    self.master_network.state_in[1]: self.rnn_state[1]})
            #action = np.argmax(a_dist.flatten())
            valid_dist = np.array([a_dist[0, validActions[key]]])
            valid_dist /= np.sum(valid_dist)
            #print (validActions[key])
            action = validActions[key][np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
            actions[key] = action
            #self.rnn_state = rnn_state
        return actions

def make_gif(images, fname):
    gif = imageio.mimwrite(fname, images, subrectangles=True)
    print("wrote gif")
    return gif


def opposite_actions(action, isDiagonal=False):
    if isDiagonal:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
        raise NotImplemented
    else:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
    return checking_table[action]
def action2dir(action):
    checking_table = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    return checking_table[action]


def dir2action(direction):
    checking_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    return checking_table[direction]
'''
def action2dir(action):
    checking_table = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    return checking_table[action]


def dir2action(direction):
    checking_table = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
    return checking_table[direction]
'''

def tuple_plus(a, b):
    """ a + b """
    return tuple(map(add, a, b))


def tuple_minus(a, b):
    """ a - b """
    return tuple(map(sub, a, b))


def _heap(ls, max_length):
    while True:
        if len(ls) > max_length:
            ls.pop(0)
        else:
            return ls


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def getAstarDistanceMap(map: np.array, start: tuple, goal: tuple, isDiagonal: bool = False):
    """
    returns a numpy array of same dims as map with the distance to the goal from each coord
    :param map: a n by m np array, where -1 denotes obstacle
    :param start: start_position
    :param goal: goal_position
    :return: optimal distance map
    """

    def lowestF(fScore, openSet):
        # find entry in openSet with lowest fScore
        assert (len(openSet) > 0)
        minF = 2 ** 31 - 1
        minNode = None
        for (i, j) in openSet:
            if (i, j) not in fScore: continue
            if fScore[(i, j)] < minF:
                minF = fScore[(i, j)]
                minNode = (i, j)
        return minNode

    def getNeighbors(node):
        # return set of neighbors to the given node
        n_moves = 9 if isDiagonal else 5
        neighbors = set()
        for move in range(1, n_moves):  # we dont want to include 0 or it will include itself
            direction = action2dir(move)
            dx = direction[0]
            dy = direction[1]
            ax = node[0]
            ay = node[1]
            if (ax + dx >= map.shape[0] or ax + dx < 0 or ay + dy >= map.shape[
                1] or ay + dy < 0):  # out of bounds
                continue
            if map[ax + dx, ay + dy] == -1:  # collide with static obstacle
                continue
            neighbors.add((ax + dx, ay + dy))
        return neighbors

    # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
    start, goal = goal, start
    start, goal = tuple(start), tuple(goal)
    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = set()
    openSet.add(start)

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = dict()

    # For each node, the cost of getting from the start node to that node.
    gScore = dict()  # default value infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = dict()  # default infinity

    # our heuristic is euclidean distance to goal
    heuristic_cost_estimate = lambda x, y: math.hypot(x[0] - y[0], x[1] - y[1])

    # For the first node, that value is completely heuristic.
    fScore[start] = heuristic_cost_estimate(start, goal)

    while len(openSet) != 0:
        # current = the node in openSet having the lowest fScore value
        current = lowestF(fScore, openSet)

        openSet.remove(current)
        closedSet.add(current)
        for neighbor in getNeighbors(current):
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)

            # The distance from start to a neighbor
            # in our case the distance between is always 1
            tentative_gScore = gScore[current] + 1
            if tentative_gScore >= gScore.get(neighbor, 2 ** 31 - 1):
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)

            # parse through the gScores
    Astar_map = map.copy()
    for (i, j) in gScore:
        Astar_map[i, j] = gScore[i, j]
    return Astar_map


class Agent:
    """
    The agent object that contains agent's position, direction dict and position dict,
    currently only supporting 4-connected region.
    self.distance_map is None here. Assign values in upper class.
    ###########
    WARNING: direction_history[i] means the action taking from i-1 step, resulting in the state of step i,
    such that len(direction_history) == len(position_history)
    ###########
    """

    def __init__(self, isDiagonal=False):
        self._path_count = -1
        self.IsDiagonal = isDiagonal
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None

    def reset(self):
        self._path_count = -1
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None

    def move(self, pos, status=None):
        if pos is None:
            pos = self.position
        if self.position is not None:
            assert pos in [self.position,
                           tuple_plus(self.position, (-1, 0)), tuple_plus(self.position, (0, -1)),
                           tuple_plus(self.position, (1, 0)), tuple_plus(self.position, (0, 1)), ], \
                "only 1 step 1 cell allowed. Previous pos:" + str(self.position)
        self.add_history(pos, status)

    def add_history(self, position, status):
        assert len(position) == 2
        self.status = status
        self._path_count += 1
        self.position = tuple(position)
        if self._path_count != 0:
            direction = tuple_minus(position, self.position_history[-1])
            action = dir2action(direction)
            assert action in list(range(4 + 1)), \
                "direction not in actionDir, something going wrong"
            self.direction_history.append(direction)
            self.action_history.append(action)
        self.position_history.append(tuple(position))

        self.position_history = _heap(self.position_history, 30)
        self.direction_history = _heap(self.direction_history, 30)
        self.action_history = _heap(self.action_history, 30)


class PogemaWrapper():
    def __init__(self,env):
        self.env = AnimationMonitor(env)
        self.num_future_steps = 3
        self.printTime = False
        #obs, _ = self.env.reset()
        self.env.reset()
        self.primal2pogema  = {0:0,1:4,2:2, 3:3,4:1}
        self.state = self.env.grid_config.map
        self.IsDiagonal = False
        self.observation_size = self.env.grid_config.obs_radius * 2 + 1
        self.num_agents = self.env.grid_config.num_agents
        self.state = self.pogema2primal(self.state) * -1
        self.goal_generate_distance = 2
        #self.env.save_animation("render.svg")
        self.agents_init_pos, self.goals_init_pos = None, None
        if self.env.grid_config.on_target == 'restart':
            self.generators = self.get_generators()
        #self.reset()

    def get_generators(self):
        generators = deepcopy(self.env.random_generators)
        return generators

    def pogema2primal(self, state):
        h,w = state.shape[0], state.shape[1]
        state1 = np.ones((h+2,w+2))
        state1[1:h+1,1:w+1] = state
        return state1
        #return obs['global_obstacles'][self.observation_size // 2 - 1:h - self.observation_size // 2  + 1,self.observation_size // 2  - 1:w - self.observation_size // 2  + 1]

    def reset(self):
        self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        self.env.reset()
    
        
        if self.agents_init_pos is None:
            self.agents_init_pos = {}
            #for na, pos in enumerate(self.env.grid_config.agents_xy):
            for na, pos in enumerate(self.env.get_agents_xy(ignore_borders=False)):
                self.agents[na+1].position = (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1)
                #self.agents[na+1].position = (pos[0]+1,pos[1]+1)
                self.state[self.agents[na+1].position] = na + 1
                self.agents[na+1].move(self.agents[na+1].position)
                self.agents_init_pos[na+1] = self.agents[na+1].position
        else:
            #for na, pos in enumerate(self.env.grid_config.agents_xy):
            for na, pos in enumerate(self.env.get_agents_xy(ignore_borders=False)):
                self.state[self.agents[na+1].position] = 0
                self.agents[na+1].position = (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1)
                #self.agents[na+1].position = (pos[0]+1,pos[1]+1)
                self.state[self.agents[na+1].position] = na + 1
                self.agents[na+1].move(self.agents[na+1].position)

        if self.goals_init_pos is None:
            self.goals_init_pos = {}
            self.goals_map = np.zeros((self.state.shape[0],self.state.shape[1]))
            #for na, pos in enumerate(self.env.grid_config.targets_xy):
            poses = self.env.get_targets_xy(ignore_borders=False)
            for na, pos in enumerate(poses):
                self.agents[na+1].goal_pos = (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1) 
                #self.agents[na+1].goal_pos = (pos[0]+1,pos[1]+1)   
                self.goals_map[self.agents[na+1].goal_pos] = na + 1
                self.goals_init_pos[na+1] = self.agents[na+1].goal_pos
                if self.env.grid_config.on_target == 'restart':
                    new_goal = generate_new_target(self.generators[na],
                              self.env.grid.point_to_component,
                              self.env.grid.component_to_points,
                              self.env.grid.positions_xy[na])
                else:
                    new_goal = poses[(na + 1) % (len(poses) - 1)]
                 
                self.agents[na+1].next_goal = (new_goal[0]- self.observation_size // 2 + 1, new_goal[1]- self.observation_size // 2 + 1)
        else:
            #for na, pos in enumerate(self.env.grid_config.targets_xy):
            for na, pos in enumerate(self.env.get_targets_xy(ignore_borders=False)):
                self.goals_map[self.agents[na+1].goal_pos] = 0
                self.agents[na+1].goal_pos = (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1)
                #self.agents[na+1].goal_pos = (pos[0]+1,pos[1]+1) 
                self.goals_map[self.agents[na+1].goal_pos] = na + 1

        for agentID in range(1,self.num_agents + 1):
            self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)
            
            self.agents[agentID].next_distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)
        
      
        self.corridor_map = {}
        self.restrict_init_corridor = True
        self.visited = []
        self.corridors = {}
        self.get_corridors()

        obs =  self.get_many()
        
        validactions = {}
        for na in range(self.num_agents):
            listValidActions = self.listValidActions(na+1,obs[na+1])
            validactions[na+1] = listValidActions
        return (obs,validactions)
    
    def step(self, actions):
        actions = list(actions.values())

        if self.env.grid_config.on_target == 'nothing':
            goals = self.env.get_targets_xy(ignore_borders=False)
            for na, pos in enumerate(self.env.get_agents_xy(ignore_borders=False)):
                if goals[na] == pos:
                    actions[na] = 0
                    
        actions = [self.primal2pogema[action] for action in actions]

        _, _, terminated, truncated, _=  self.env.step(actions)
        goals = self.env.get_targets_xy(ignore_borders=False)
        
        for i, pos in enumerate(goals):
            goal =  (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1)
            if self.agents[i+1].goal_pos != goal:
                self.goals_map[self.agents[i+1].goal_pos] = 0
                self.agents[i+1].goal_pos = goal
                self.goals_map[goal] = i + 1

                if self.env.grid_config.on_target == 'restart':
                    new_goal = generate_new_target(self.generators[i],
                              self.env.grid.point_to_component,
                              self.env.grid.component_to_points,
                              self.env.grid.positions_xy[i])
                else:
                    new_goal = goals[(i + 1) % (len(goals) - 1)]
                self.agents[i+1].next_goal = (new_goal[0]- self.observation_size // 2 + 1, new_goal[1]- self.observation_size // 2 + 1)

        for na, pos in enumerate(self.env.get_agents_xy(ignore_borders=False)):
            self.state[self.agents[na+1].position] = 0
            self.agents[na+1].position = (pos[0] - self.observation_size // 2 + 1, pos[1] - self.observation_size // 2 + 1)
            self.state[self.agents[na+1].position] = na + 1
            self.agents[na+1].move(self.agents[na+1].position)
        
        for agentID in range(1,self.num_agents + 1):
            self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)
            self.agents[agentID].next_distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)

        obs =  self.get_many()
        validactions = {}
        for na in range(self.num_agents):
            listValidActions = self.listValidActions(na+1,obs[na+1])
            validactions[na+1] = listValidActions

        return (obs,validactions), terminated, truncated
    
    def save_animation(self):
        self.env.save_animation("render1.svg")
        print ('Render saved as render1.svg')
        return 
    def init_agents_and_goals(self):
        """
        place all agents and goals in the blank env. If turning on corridor population restriction, only 1 agent is
        allowed to be born in each corridor.
        """

        def corridor_restricted_init_poss(state_map, corridor_map, goal_map, id_list=None):
            """
            generate agent init positions when corridor init population is restricted
            return a dict of positions {agentID:(x,y), ...}
            """
            if id_list is None:
                id_list = list(range(1, self.num_agents + 1))

            free_space1 = list(np.argwhere(state_map == 0))
            free_space1 = [tuple(pos) for pos in free_space1]
            corridors_visited = []
            manual_positions = {}
            break_completely = False
            for idx in id_list:
                if break_completely:
                    return None
                pos_set = False
                agentID = idx
                while not pos_set:
                    try:
                        assert (len(free_space1) > 1)
                        random_pos = np.random.choice(len(free_space1))
                    except AssertionError or ValueError:
                        print('wrong agent')
                        self.reset()
                        self.init_agents_and_goals()
                        break_completely = True
                        if idx == id_list[-1]:
                            return None
                        break
                    position = free_space1[random_pos]
                    cell_info = corridor_map[position[0], position[1]][1]
                    if cell_info in [0, 2]:
                        if goal_map[position[0], position[1]] != agentID:
                            manual_positions.update({idx: (position[0], position[1])})
                            free_space1.remove(position)
                            pos_set = True
                    elif cell_info == 1:
                        corridor_id = corridor_map[position[0], position[1]][0]
                        if corridor_id not in corridors_visited:
                            if goal_map[position[0], position[1]] != agentID:
                                manual_positions.update({idx: (position[0], position[1])})
                                corridors_visited.append(corridor_id)
                                free_space1.remove(position)
                                pos_set = True
                        else:
                            free_space1.remove(position)
                    else:
                        print("Very Weird")
                        # print('Manual Positions' ,manual_positions)
            return manual_positions

        # no corridor population restriction
        
        if not self.restrict_init_corridor or (self.restrict_init_corridor and self.manual_world):
            self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            self._put_agents(list(range(1, self.num_agents + 1)), self.agents_init_pos)
        # has corridor population restriction
        else:
            check = self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            if check is not None:
                manual_positions = corridor_restricted_init_poss(self.state, self.corridor_map, self.goals_map)
                if manual_positions is not None:
                    self._put_agents(list(range(1, self.num_agents + 1)), manual_positions)

    def _put_agents(self, id_list, manual_pos=None):
        """
        put some agents in the blank env, saved history data in self.agents and self.state
        get distance map for the agents
        :param id_list: a list of agent_id
                manual_pos: a dict of manual positions {agentID: (x,y),...}
        """
        if manual_pos is None:
            # randomly init agents everywhere
            free_space = np.argwhere(np.logical_or(self.state == 0, self.goals_map == 0) == 1)
            new_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
            init_poss = [free_space[idx] for idx in new_idx]
        else:
            assert len(manual_pos.keys()) == len(id_list)
            init_poss = [manual_pos[agentID] for agentID in id_list]
        assert len(init_poss) == len(id_list)
        self.agents_init_pos = {}
        for idx, agentID in enumerate(id_list):
            self.agents[agentID].ID = agentID
            if self.state[init_poss[idx][0], init_poss[idx][1]] in [0, agentID] \
                    and self.goals_map[init_poss[idx][0], init_poss[idx][1]] != agentID:
                self.state[init_poss[idx][0], init_poss[idx][1]] = agentID
                self.agents_init_pos.update({agentID: (init_poss[idx][0], init_poss[idx][1])})
            else:
                print(self.state)
                print(init_poss)
                raise ValueError('invalid manual_pos for agent' + str(agentID) + ' at: ' + str(init_poss[idx]))
            self.agents[agentID].move(init_poss[idx])
            self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)
            
    def put_goals(self, id_list, manual_pos=None):
        """
        put a goal of single agent in the env, if the goal already exists, remove that goal and put a new one
        :param manual_pos: a dict of manual_pos {agentID: (x, y)}
        :param id_list: a list of agentID
        :return: an Agent object
        """

        def random_goal_pos(previous_goals=None, distance=None):
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            # print(previous_goals)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    if distance > 0:
                        previous_x, previous_y = previous_goals[agentID]
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                        init_pos = free_spaces_for_previous_goal[init_idx]
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print('wrong goal')
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        previous_goals = {agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is None:
            new_goals = random_goal_pos(previous_goals, distance=self.goal_generate_distance)
        else:
            new_goals = manual_pos
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                if self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0:
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0], new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        new_next_goals = random_goal_pos(new_goals, distance=self.goal_generate_distance)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        self.goals_map[self.agents[agentID].next_goal[0], self.agents[agentID].next_goal[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0], new_goals[agentID][1])  # store new goal into next_goal
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                else:
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError('invalid manual_pos for goal' + str(agentID) + ' at: ', str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    if previous_goals[agentID] != self.agents[agentID].position:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError("agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                self.agents[agentID].next_distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)
                if refresh_distance_map:
                    self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                           self.agents[agentID].goal_pos)
            return 1
        else:
            return None

    def getPos(self, agent_id):
        return tuple(self.agents[agent_id].position)

    def getDone(self, agentID):
        # get the number of goals that an agent has finished
        return self.agents[agentID].dones
    
    def getGoal(self, agent_id):
        return tuple(self.agents[agent_id].goal_pos)

    def get_corridors(self):
        """
        in corridor_map , output = list:
            list[0] : if In corridor, corridor id , else -1 
            list[1] : If Inside Corridor = 1
                      If Corridor Endpoint = 2
                      If Free Cell Outside Corridor = 0   
                      If Obstacle = -1 
        """
        corridor_count = 1
        # Initialize corridor map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        # Compute All Corridors and End-points, store them in self.corridors , update corridor_map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]['Positions'] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]['EndPoints'] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        if positions[num] is not None:
                            self.visit(positions[num][0], positions[num][1], corridor_count)
                    corridor_count += 1
        # Get Delta X , Delta Y for the computed corridors ( Delta= Displacement to corridor exit)       
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]['EndPoints']) == 2:
                    self.corridors[k]['DeltaX'] = {}
                    self.corridors[k]['DeltaY'] = {}
                    pos_a = self.corridors[k]['EndPoints'][0]
                    pos_b = self.corridors[k]['EndPoints'][1]
                    self.corridors[k]['DeltaX'][pos_a] = (pos_a[0] - pos_b[0])  # / (max(1, abs(pos_a[0] - pos_b[0])))
                    self.corridors[k]['DeltaX'][pos_b] = -1 * self.corridors[k]['DeltaX'][pos_a]
                    self.corridors[k]['DeltaY'][pos_a] = (pos_a[1] - pos_b[1])  # / (max(1, abs(pos_a[1] - pos_b[1])))
                    self.corridors[k]['DeltaY'][pos_b] = -1 * self.corridors[k]['DeltaY'][pos_a]
            else:
                print('Weird2')

                # Rearrange the computed corridor list such that it becomes easier to iterate over the structure
        # Basically, sort the self.corridors['Positions'] list in a way that the first element of the list is
        # adjacent to Endpoint[0] and the last element of the list is adjacent to EndPoint[1] 
        # If there is only 1 endpoint, the sorting doesn't matter since blocking is easy to compute
        for t in range(1, corridor_count):
            positions = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][0][0],
                                                      self.corridors[t]['EndPoints'][0][1])
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t:
                    break
            index = self.corridors[t]['Positions'].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]['Positions']) - 1:
                temp_list = self.corridors[t]['Positions'][0:index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)

            elif index == len(self.corridors[t]['Positions']) - 1 and len(self.corridors[t]['EndPoints']) == 2:
                positions2 = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][1][0],
                                                           self.corridors[t]['EndPoints'][1][1])
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t:
                        break
                index2 = self.corridors[t]['Positions'].index(position2)
                temp_list = self.corridors[t]['Positions'][0:index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index2 + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)
                self.corridors[t]['Positions'].reverse()
            else:
                if len(self.corridors[t]['EndPoints']) == 2:
                    print("Weird3")

            self.corridors[t]['StoppingPoints'] = []
            if len(self.corridors[t]['EndPoints']) == 2:
                position_first = self.corridors[t]['Positions'][0]
                position_last = self.corridors[t]['Positions'][-1]
                self.corridors[t]['StoppingPoints'].append([position_first[0], position_first[1]])
                self.corridors[t]['StoppingPoints'].append([position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]['Positions'][0]
                self.corridors[t]['StoppingPoints'].append([position[0], position[1]])
                self.corridors[t]['StoppingPoints'].append(None)
        return

    def check_for_singular_state(self, positions):
        counter = 0
        for num in range(4):
            if positions[num] is not None:
                new_positions = self.blank_env_valid_neighbor(positions[num][0], positions[num][1])
                if new_positions.count(None) in [2, 3]:
                    counter += 1
        return counter > 0

    def visit(self, i, j, corridor_id):
        positions = self.blank_env_valid_neighbor(i, j)
        if positions.count(None) in [0, 1]:
            self.corridors[corridor_id]['EndPoints'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 2]
            return
        elif positions.count(None) in [2, 3]:
            self.visited.append((i, j))
            self.corridors[corridor_id]['Positions'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 1]
            for num in range(4):
                if positions[num] is not None and positions[num] not in self.visited:
                    self.visit(positions[num][0], positions[num][1], corridor_id)
        else:
            print('Weird')

    def blank_env_valid_neighbor(self, i, j):
        possible_positions = [None, None, None, None]
        move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        if self.state[i, j] == -1:
            return possible_positions
        else:
            for num in range(4):
                x = i + move[num][0]
                y = j + move[num][1]
                if 0 <= x < self.state.shape[0] and 0 <= y < self.state.shape[1]:
                    if self.state[x, y] != -1:
                        possible_positions[num] = (x, y)
                        continue
        return possible_positions
    

    def get_next_positions(self, agent_id):
        agent_pos = self.getPos(agent_id)
        positions = []
        current_pos = [agent_pos[0], agent_pos[1]]
        next_positions = self.blank_env_valid_neighbor(current_pos[0], current_pos[1])
        for position in next_positions:
            if position is not None and position != agent_pos:
                positions.append([position[0], position[1]])
                next_next_positions = self.blank_env_valid_neighbor(position[0], position[1])
                for pos in next_next_positions:
                    if pos is not None and pos not in positions and pos != agent_pos:
                        positions.append([pos[0], pos[1]])

        return positions
    
    def _get(self, agent_id, all_astar_maps):

        start_time = time.time()

        assert (agent_id > 0)
        agent_pos = self.getPos(agent_id)
        top_left = (agent_pos[0] - self.observation_size // 2,
                    agent_pos[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        centre = (self.observation_size - 1) / 2
        obs_shape = (self.observation_size, self.observation_size)
        
        goal_map = np.zeros(obs_shape)
        poss_map = np.zeros(obs_shape)
        goals_map = np.zeros(obs_shape)
        obs_map = np.zeros(obs_shape)
        astar_map = np.zeros([self.num_future_steps, self.observation_size, self.observation_size])
        astar_map_unpadded = np.zeros([self.num_future_steps, self.state.shape[0], self.state.shape[1]])
        pathlength_map = np.zeros(obs_shape)
        deltax_map = np.zeros(obs_shape)
        deltay_map = np.zeros(obs_shape)
        blocking_map = np.zeros(obs_shape)

        time1 = time.time() - start_time
        start_time = time.time()

        # concatenate all_astar maps
        other_agents = list(range(self.num_agents))  # needs to be 0-indexed for numpy magic below
        other_agents.remove(agent_id - 1)  # 0-indexing again
        astar_map_unpadded = np.zeros([self.num_future_steps, self.state.shape[0], self.state.shape[1]])
        astar_map_unpadded[:self.num_future_steps, max(0, top_left[0]):min(bottom_right[0], self.state.shape[0]),
        max(0, top_left[1]):min(bottom_right[1], self.state.shape[1])] = \
            np.sum(all_astar_maps[other_agents, :self.num_future_steps,
                   max(0, top_left[0]):min(bottom_right[0], self.state.shape[0]),
                   max(0, top_left[1]):min(bottom_right[1], self.state.shape[1])], axis=0)

        time2 = time.time() - start_time
        start_time = time.time()

        # original layers from PRIMAL1
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.state.shape[0] or i < 0 or j >= self.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    pathlength_map[i - top_left[0], j - top_left[1]] = -1
                    continue

                astar_map[:self.num_future_steps, i - top_left[0], j - top_left[1]] = astar_map_unpadded[
                                                                                      :self.num_future_steps, i, j]
                if self.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.state[i, j] == agent_id:
                    # agent's position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.goals_map[i, j] == agent_id:
                    # agent's goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.state[i, j] > 0 and self.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                
                # we can keep this map even if on goal,
                # since observation is computed after the refresh of new distance map
                pathlength_map[i - top_left[0], j - top_left[1]] = self.agents[agent_id].distanceMap[i, j]
        
        time3 = time.time() - start_time
        start_time = time.time()

        for agent in visible_agents:
            x, y = self.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.getGoal(agent_id)[0] - agent_pos[0]
        dy = self.getGoal(agent_id)[1] - agent_pos[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag > 60:
            mag = 60

        time4 = time.time() - start_time
        start_time = time.time()

        current_corridor_id = -1
        current_corridor = self.corridor_map[self.getPos(agent_id)[0], self.getPos(agent_id)[1]][1]
        if current_corridor == 1:
            current_corridor_id = \
                self.corridor_map[self.getPos(agent_id)[0], self.getPos(agent_id)[1]][0]

        positions = self.get_next_positions(agent_id)
        for position in positions:
            cell_info = self.corridor_map[position[0], position[1]]
            if cell_info[1] == 1:
                corridor_id = cell_info[0]
                if corridor_id != current_corridor_id:
                    if len(self.corridors[corridor_id]['EndPoints']) == 1:
                        if [position[0], position[1]] == self.corridors[corridor_id]['StoppingPoints'][0]:
                            blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                                corridor_id,
                                0, agent_id,
                                1)
                    elif [position[0], position[1]] == self.corridors[corridor_id]['StoppingPoints'][0]:
                        end_point_pos = self.corridors[corridor_id]['EndPoints'][0]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            0, agent_id,
                            2)
                    elif [position[0], position[1]] == self.corridors[corridor_id]['StoppingPoints'][1]:
                        end_point_pos = self.corridors[corridor_id]['EndPoints'][1]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            1, agent_id,
                            2)
                    else:
                        pass

        time5 = time.time() - start_time
        start_time = time.time()

        free_spaces = list(np.argwhere(pathlength_map > 0))
        distance_list = []
        for arg in free_spaces:
            dist = pathlength_map[arg[0], arg[1]]
            if dist not in distance_list:
                distance_list.append(dist)
        distance_list.sort()
        step_size = (1 / len(distance_list))
        for i in range(self.observation_size):
            for j in range(self.observation_size):
                dist_mag = pathlength_map[i, j]
                if dist_mag > 0:
                    index = distance_list.index(dist_mag)
                    pathlength_map[i, j] = (index + 1) * step_size
        state = np.array([poss_map, goal_map, goals_map, obs_map, pathlength_map, blocking_map, deltax_map,
                          deltay_map])
        state = np.concatenate((state, astar_map), axis=0)
        
        time6 = time.time() - start_time
        start_time = time.time()

        return state, [dx, dy, mag], np.array([time1, time2, time3, time4, time5, time6])
    
    def get_many(self, handles=None):
        observations = {}
        all_astar_maps = self.get_astar_map()
        if handles is None:
            handles = list(range(1, self.num_agents + 1))

        times = np.zeros((1, 6))
        
        for h in handles:
            state, vector, time = self._get(h, all_astar_maps)
            observations[h] = [state, vector]
            times += time
        if self.printTime:
            print(times)
        
        return observations
    
    def get_astar_map(self):
        """

        :return: a dict of 3D np arrays. Each astar_maps[agentID] is a num_future_steps * obs_size * obs_size matrix.
        """

        def get_single_astar_path(distance_map, start_position, path_len):
            """
            :param distance_map:
            :param start_position:
            :param path_len:
            :return: [[(x,y), ...],..] a list of lists of positions from start_position, the length of the return can be
            smaller than num_future_steps. Index of the return: list[step][0-n] = tuple(x, y)
            """

            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.state.shape[0]
                w = self.state.shape[1]
                for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    # print(position, direction)
                    new_pos = tuple_plus(position, direction)
                    if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                        if distance_map[new_pos] == distance_map[position] - 1 \
                                and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:  # no next step, should be standing on goal
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))  # remove repeated positions
                astar_list.append(next_step_cells)
                path_counter += 1
            astar_list.pop(0)
            return astar_list

        astar_maps = {}
        for agentID in range(1, self.num_agents + 1):
            astar_maps.update(
                {agentID: np.zeros([self.num_future_steps, self.state.shape[0], self.state.shape[1]])})

            distance_map0, start_pos0 = self.agents[agentID].distanceMap, self.agents[agentID].position
            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)
            if not len(astar_path) == self.num_future_steps:  # this agent reaches its goal during future steps
                distance_map1, start_pos1 = self.agents[agentID].next_distanceMap, \
                                            self.agents[agentID].goal_pos
                astar_path.extend(
                    get_single_astar_path(distance_map1, start_pos1, self.num_future_steps - len(astar_path)))

            for i in range(self.num_future_steps - len(astar_path)):  # only happen when min_distance not sufficient
                astar_path.extend([[astar_path[-1][-1]]])  # stay at the last pos

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = 1

        return np.asarray([astar_maps[i] for i in range(1, self.num_agents + 1)])

    def get_blocking(self, corridor_id, reverse, agent_id, dead_end):
        def get_last_pos(agentID, position):
            history_list = copy.deepcopy(self.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        positions_to_check = copy.deepcopy(self.corridors[corridor_id]['Positions'])
        if reverse:
            positions_to_check.reverse()
        idx = -1
        for position in positions_to_check:
            idx += 1
            state = self.state[position[0], position[1]]
            if state > 0 and state != agent_id:
                if dead_end == 1:
                    return 1
                if idx == 0:
                    return 1
                last_pos = get_last_pos(state, position)
                if last_pos == None:
                    return 1
                if idx == len(positions_to_check) - 1:
                    if last_pos != positions_to_check[idx - 1]:
                        return 1
                    break
                if last_pos == positions_to_check[idx + 1]:
                    return 1
        if dead_end == 2:
            if not reverse:
                other_endpoint = self.corridors[corridor_id]['EndPoints'][1]
            else:
                other_endpoint = self.corridors[corridor_id]['EndPoints'][0]
            state_endpoint = self.state[other_endpoint[0], other_endpoint[1]]
            if state_endpoint > 0 and state_endpoint != agent_id:
                return -1
        return 0

    def CheckCollideStatus(self, movement_dict):
        """
        WARNING: ONLY NON-DIAGONAL IS IMPLEMENTED
        return collision status and predicted next positions, do not move agent directly
        return:
         2: (only in oneShot mode) action not executed, agents has done its target and has been removed from the env.
         1: action executed, and agents standing on its goal.
         0: action executed
        -1: collision with env (obstacles, out of bound)
        -2: collision with robot, swap
        -3: collision with robot, cell-wise
        """

        Assumed_newPos_dict = {}
        newPos_dict = {}
        status_dict = {agentID: None for agentID in range(1, self.num_agents + 1)}
        not_checked_list = list(range(1, self.num_agents + 1))

        # detect env collision
        for agentID in range(1, self.num_agents + 1):
            direction_vector = action2dir(movement_dict[agentID])
            newPos = tuple_plus(self.getPos(agentID), direction_vector)
            Assumed_newPos_dict.update({agentID: newPos})
            if newPos[0] < 0 or newPos[0] > self.state.shape[0] or newPos[1] < 0 \
                    or newPos[1] > self.state.shape[1] or self.state[newPos] == -1:
                status_dict[agentID] = -1
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
                # collide, stay at the same place

        # detect swap collision

        for agentID in copy.deepcopy(not_checked_list):
            collided_ID = self.state[Assumed_newPos_dict[agentID]]
            if collided_ID != 0 and Assumed_newPos_dict[agentID] != self.getGoal(
                    agentID):  # some one is standing on the assumed pos
                if Assumed_newPos_dict[collided_ID] == self.getPos(agentID):  # he wants to swap
                    if status_dict[agentID] is None:
                        status_dict[agentID] = -2
                        newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                        Assumed_newPos_dict[agentID] = self.getPos(agentID)
                        not_checked_list.remove(agentID)
                    if status_dict[collided_ID] is None:
                        status_dict[collided_ID] = -2
                        newPos_dict.update({collided_ID: self.getPos(collided_ID)})  # stand still
                        Assumed_newPos_dict[collided_ID] = self.getPos(collided_ID)
                        not_checked_list.remove(collided_ID)

        # detect cell-wise collision
        for agentID in copy.deepcopy(not_checked_list):
            other_agents_dict = copy.deepcopy(Assumed_newPos_dict)
            other_agents_dict.pop(agentID)
            ignore_goal_agents_dict = copy.deepcopy(newPos_dict)
            for agent in range(1, self.num_agents + 1):
                if agent != agentID:
                    if Assumed_newPos_dict[agent] == self.getGoal(agent):
                        other_agents_dict.pop(agent)
                        try:
                            ignore_goal_agents_dict.pop(agent)
                        except:
                            pass
            if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos:
                continue
            if Assumed_newPos_dict[agentID] in ignore_goal_agents_dict.values():
                status_dict[agentID] = -3
                newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
            elif Assumed_newPos_dict[agentID] in other_agents_dict.values():
                other_coming_agents = get_key(Assumed_newPos_dict, Assumed_newPos_dict[agentID])
                other_coming_agents.remove(agentID)
                # if the agentID is the biggest among all other coming agents,
                # allow it to move. Else, let it stand still
                if agentID < min(other_coming_agents):
                    status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                    newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
                    not_checked_list.remove(agentID)
                else:
                    status_dict[agentID] = -3
                    newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                    Assumed_newPos_dict[agentID] = self.getPos(agentID)
                    not_checked_list.remove(agentID)

        # the rest are valid actions
        for agentID in copy.deepcopy(not_checked_list):
            status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
            newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
            not_checked_list.remove(agentID)
        assert not not_checked_list

        return status_dict, newPos_dict
    
    def listValidActions(self, agent_ID, agent_obs):
        """
        :return: action:int, pos:(int,int)
        in non-corridor states:
            return all valid actions
        in corridor states:
            if standing on goal: Only going 'forward' allowed
            if not standing on goal: only going 'forward' allowed
        """

        def get_last_pos(agentID, position):
            """
            get the last different position of an agent
            """
            history_list = copy.deepcopy(self.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        VANILLA_VALID_ACTIONS = False   

        if VANILLA_VALID_ACTIONS== True :
            available_actions = []
            pos = self.getPos(agent_ID)
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                try:
                    lastpos = self.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

            return available_actions

        available_actions = []
        pos = self.getPos(agent_ID)
        # if the agent is inside a corridor
        if self.corridor_map[pos[0], pos[1]][1] == 1:
            corridor_id = self.corridor_map[pos[0], pos[1]][0]
            if [pos[0], pos[1]] not in self.corridors[corridor_id]['StoppingPoints']:
                possible_moves = self.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                for possible_position in possible_moves:
                    if possible_position is not None and possible_position != last_position \
                            and self.state[possible_position[0], possible_position[1]] == 0:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                    elif len(self.corridors[corridor_id]['EndPoints']) == 1 and possible_position is not None \
                            and possible_moves.count(None) == 3:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                if not available_actions:
                    available_actions.append(0)
            else:
                possible_moves = self.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                if last_position in self.corridors[corridor_id]['Positions']:
                    available_actions.append(0)
                    for possible_position in possible_moves:
                        if possible_position is not None and possible_position != last_position \
                                and self.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                else:
                    for possible_position in possible_moves:
                        if possible_position is not None \
                                and self.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                    if not available_actions:
                        available_actions.append(0)
        else:
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                blocking_valid = self.get_blocking_validity(agent_obs, agent_ID, new_pos)
                if not blocking_valid:
                    continue
                try:
                    lastpos = self.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.corridor_map[new_pos[0], new_pos[1]][1] == 1:
                    valid = self.get_convention_validity(agent_obs, agent_ID, new_pos)
                    if not valid:
                        continue
                if self.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

        return available_actions
    
    def get_blocking_validity(self, observation, agent_ID, pos):
        top_left = (self.getPos(agent_ID)[0] - self.observation_size // 2,
                    self.getPos(agent_ID)[1] - self.observation_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 1:
            return 0
        return 1
    
    def get_convention_validity(self, observation, agent_ID, pos):
        top_left = (self.getPos(agent_ID)[0] - self.observation_size // 2,
                    self.getPos(agent_ID)[1] - self.observation_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == -1:
            deltay_map = observation[0][7]
            if deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                return 1
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 0:
                deltax_map = observation[0][6]
                if deltax_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                    return 1
                else:
                    return 0
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] < 0:
                return 0
            else:
                print('Weird')
        else:
            return 1



gamma                   = .95  # discount rate for advantage estimation and reward discounting
LR_Q                    = 2.e-5  # 8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR                = True
ADAPT_COEFF             = 5.e-5  # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
EXPERIENCE_BUFFER_SIZE  = 256 
max_episode_length      = 256
episode_count           = 0

# observer parameters
OBS_SIZE                = 11   # the size of the FOV grid to apply to each agent
NUM_FUTURE_STEPS        = 3

# environment parameters
ENVIRONMENT_SIZE        = (48, 48)  # the total size of the environment (length of one side)
WALL_COMPONENTS         = (10, 21)
OBSTACLE_DENSITY        = (0.2, 0.7)  # range of densities
CHANGE_FREQUENCY        = 5000       # Frequency of Changing environment params  
DIAG_MVMT               = False  # Diagonal movements allowed?
a_size                  = 5 + int(DIAG_MVMT) * 4
NUM_META_AGENTS         = 1
NUM_THREADS             = 4 # int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS             = 1  # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)

# training parameters
SUMMARY_WINDOW          = 10
load_model              = True
RESET_TRAINER           = False
training_version        = 'primal2_continuous' 
model_path              = 'model_' + training_version
gifs_path               = 'gifs_' + training_version
train_path              = 'train_' + training_version
OUTPUT_GIFS             = False  
GIF_FREQUENCY           = 512
SAVE_IL_GIF             = False   
IL_GIF_PROB             = 0  
IS_ONESHOT              = True

# Imitation options
PRIMING_LENGTH          = 0    # number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB      = 0.5   # probability of training on a demonstration per episode
IL_DECAY_RATE           = 0

# observation variables
NUM_CHANNEL             = 8 + NUM_FUTURE_STEPS

# others
EPISODE_START           = episode_count
TRAINING                = True
EPISODE_SAMPLES         = EXPERIENCE_BUFFER_SIZE  # 64
GLOBAL_NET_SCOPE        = 'global'
swarm_reward            = [0] * NUM_META_AGENTS
swarm_targets           = [0] * NUM_META_AGENTS

# Shared arrays for tensorboard

IL_agents_done         = []
episode_rewards         = [[] for _ in range(NUM_META_AGENTS)] 
episode_finishes        = [[] for _ in range(NUM_META_AGENTS)]
episode_lengths         = [[] for _ in range(NUM_META_AGENTS)]
episode_mean_values     = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops     = [[] for _ in range(NUM_META_AGENTS)]
episode_stop_ops        = [[] for _ in range(NUM_META_AGENTS)]
episode_wrong_blocking  = [[] for _ in range(NUM_META_AGENTS)]
rollouts                = [None for _ in range(NUM_META_AGENTS)]
demon_probs             = [np.random.rand() for _ in range(NUM_META_AGENTS)]
GIF_frames              = []

# Joint variables 
joint_actions           = [{} for _ in range(NUM_META_AGENTS)]
joint_env               = [None for _ in range(NUM_META_AGENTS)]
joint_observations      =[{} for _ in range(NUM_META_AGENTS)]
joint_rewards           = [{} for _ in range(NUM_META_AGENTS)]
joint_done              = [{} for _ in range(NUM_META_AGENTS)]

map_generator = maze_generator(env_size = ENVIRONMENT_SIZE)
state, goals = map_generator()
'''
state = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                                [-1,  0, -1,  0, -1,  0, -1,  0, -1],
                                [-1,  0, -1,  0, -1,  0, -1,  0, -1],
                                [-1,  0,  0,  0, -1,  0, -1,  0, -1],
                                [-1,  0, -1, -1, -1,  0, -1,  0, -1],
                                [-1,  0,  0,  0, -1,  0,  0,  0, -1],
                                [-1,  0,  0,  0, -1, -1, -1,  0, -1],
                                [-1,  0,  0,  0,  0,  0,  0,  0, -1],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1]])
'''
grid_config = GridConfig()   
w,h = state.shape[0], state.shape[1]
state[state >= 0] = 0 
grid_map = state[1:w-1,1:h-1]
grid_map[grid_map >= 0] = 0 
grid_config.map = -grid_map

grid_config.num_agents = 16

grid_config.on_target="nothing"
grid_config.obs_radius = 5
grid_config.observation_type = 'MAPF'
grid_config.max_episode_steps = 128
#grid_config.agents_xy = [(4,6),(6,2),(2,6),(3,4)]
#grid_config.targets_xy = [(6,3),(6,0),(4,4),(4,6)]
env = pogema_v0(grid_config=grid_config)
pogema = PogemaWrapper(env)
tf.reset_default_graph()
print("Hello World")
for path in [train_path, model_path, gifs_path]:
    if not os.path.exists(path):
        os.makedirs(path)
config = tf.ConfigProto(allow_soft_placement=True)

config.gpu_options.allow_growth = True
with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE, a_size, None, False, NUM_CHANNEL, OBS_SIZE, GLOBAL_NET_SCOPE)

    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        # we need the +1 so that lr at step 0 is defined
        lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
    else:
        lr = tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')

            ckpt = tf.train.get_checkpoint_state(model_path)
            p = ckpt.model_checkpoint_path
            p = p[p.find('-') + 1:]
            p = p[:p.find('.')]
            episode_count = int(p)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("episode_count set to ", episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
    
            actions = []

            rnn_state = master_network.state_init
            rnn_state0 = rnn_state

            agent = BatchAgent(master_network,rnn_state)

            obs = pogema.reset()
            
            while True:
                
                actions = agent.act(obs)
                obs,terminated, truncated = pogema.step(actions)
                if all(terminated) or all(truncated):
                    break   
            
            
            #name = f'render{goals[0]}.svg'
            pogema.save_animation()




