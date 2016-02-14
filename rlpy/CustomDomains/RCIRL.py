"""
This class implements a GridWorld with
consumable rewards
"""
import os,sys,inspect
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
from rlpy.Domains.Domain import Domain
import numpy as np
from rlpy.Tools import plt, FONTSIZE, linearMap

class RCIRL(Domain): 
    # #__metaclass__ = Domain
    # #default paths
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # default_map_dir = os.path.join(currentdir,"ConsumableGridWorldMaps")

    # map = start_state = goal = None
    # # Used for graphics to show the domain
    # agent_fig = upArrows_fig = downArrows_fig = leftArrows_fig = None
    # rightArrows_fig = domain_fig = valueFunction_fig = None
    # #: Number of rows and columns of the map
    # ROWS = COLS = 0
    # #: Reward constants
    # GOAL_REWARD = +1
    # PIT_REWARD = -1
    # STEP_REWARD = -.001
    # #: Set by the domain = min(100,rows*cols)
    # episodeCap = None
    # #: Movement Noise
    # NOISE = 0
    # # Used for graphical normalization
    # MAX_RETURN = 1
    # RMAX = MAX_RETURN
    # # Used for graphical normalization
    # MIN_RETURN = -1
    # # Used for graphical shifting of arrows
    # SHIFT = .1
    # #MAX_STEPS = min(100,rows*cols)

    # actions_num = 4
    # # Constants in the map
    # EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6)
    # #: Up, Down, Left, Right
    # ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])


    # #an encoding function maps a set of previous states to a fixed
    # #width encoding
    # """
    # A reward function is a function over ALL of the previous states, 
    # the set of goal states,
    # the step reward constant, 
    # and the goal reward constant.
    # """
    # def __init__(self, 
    #              goalArray, 
    # 			 mapname=os.path.join(default_map_dir, "4x5.txt"),
    #              encodingFunction=None,
    #              rewardFunction=None,
    #              noise=.1, 
    #              episodeCap=None):
       
    #     #setup consumable rewards
    #     self.map = np.loadtxt(mapname, dtype=np.uint8)
    #     self.goalArray0 = np.array(goalArray)
    #     self.goalArray = np.array(goalArray)
    #     self.prev_states = []

    #     if encodingFunction == None:
    #         self.encodingFunction = ConsumableGridWorldIRL.allMarkovEncoding
    #     else:
    #         self.encodingFunction = encodingFunction

    #     self.rewardFunction = rewardFunction

    #     if self.map.ndim == 1:
    #         self.map = self.map[np.newaxis, :]

    #     self.start_state = np.concatenate((np.argwhere(self.map == self.START)[0], 
    #                                          self.encodingFunction(self.prev_states)))

    #     self.ROWS, self.COLS = np.shape(self.map)

    #     #remove goals for existing maps
    #     for r in range(0,self.ROWS):
    #     	for c in range(0,self.COLS):
    #     		if self.map[r,c] == self.GOAL:
    #     			self.map[r,c] = self.EMPTY

    #     for g in goalArray:
    #     	self.map[g[0],g[1]] = self.GOAL

    #     encodingLimits = []
    #     for i in range(0,len(self.encodingFunction(self.prev_states))):
    #         encodingLimits.append([0,1])

    #     sslimits = [[0, self.ROWS - 1], [0, self.COLS - 1]]
    #     sslimits.extend(encodingLimits)
    #     #print np.array(sslimits)[:,0]

    #     self.statespace_limits = np.array(sslimits)

    #     self.continuous_dims = []
    #     self.NOISE = noise
    #     self.DimNames = ["Dim: "+str(k) for k in range(0,2+len(self.encodingFunction(self.prev_states)))]
    #     # 2*self.ROWS*self.COLS, small values can cause problem for some
    #     # planning techniques
    #     self.episodeCap = 2*self.ROWS*self.COLS
    #     super(RCIRL, self).__init__()

    # def showDomain(self, a=0, s=None):
    # 	raise NotImplementedError

    # def step(self, a):
    #     r = self.STEP_REWARD
    #     ns = self.state.copy()
    #     ga = self.goalArray.copy()

    #     self.prev_states.append(ns[0:2])

    #     if self.random_state.random_sample() < self.NOISE:
    #         # Random Move
    #         a = self.random_state.choice(self.possibleActions())

    #     # Take action
    #     statesize = np.shape(self.state)
    #     actionsize = np.shape(self.ACTIONS[a])
    #    # print statesize[0]-actionsize[0]


    #     ns = np.concatenate((self.state[0:2] + self.ACTIONS[a],
    #                          self.encodingFunction(self.prev_states)))
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]
    #     # Check bounds on state values
    #     if (ns[0] < 0 or ns[0] == self.ROWS or
    #             ns[1] < 0 or ns[1] == self.COLS or
    #             self.map[ns[0], ns[1]] == self.BLOCKED):
    #         ns = self.state.copy()
    #     else:
    #         # If in bounds, update the current state
    #         self.state = ns.copy()

    #     terminal = self.isTerminal()
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]
    #     # Compute the reward and enforce ordering
    #     if terminal:
    #         pass
    #     elif ga[0][0] == ns[0] and  ga[0][1] == ns[1]:
    #         r = self.GOAL_REWARD
    #         ga = ga[1:]
    #         self.goalArray = ga
    #         #print "Goal!", ns

    #     if self.map[ns[0], ns[1]] == self.PIT:
    #         r = self.PIT_REWARD

    #     if self.rewardFunction != None:
    #         r = self.rewardFunction(self.prev_states, 
    #                                 self.goalArray, 
    #                                 self.STEP_REWARD, 
    #                                 self.GOAL_REWARD)

    #     return r, ns, terminal, self.possibleActions()

    # def s0(self):
    #     self.prev_states = []
    #     self.state = self.start_state.copy()
    #     self.goalArray = self.goalArray0
        
    #     return self.state, self.isTerminal(), self.possibleActions()

    # def isTerminal(self, s=None):
    #     if s is None:
    #         s = self.state
    #     if len(self.goalArray) == 0:
    #         return True
    #     if self.map[s[0], s[1]] == self.PIT:
    #         return True
    #     #if len(self.prev_states) > self.MAX_STEPS:
    #     #    return True

    #     return False

    # def possibleActions(self, s=None):
    #     if s is None:
    #         s = self.state
    #     possibleA = np.array([], np.uint8)
    #     for a in xrange(self.actions_num):
    #         ns = s[0:2] + self.ACTIONS[a]
    #         #print s[0:1],ns[0], ns[1]
    #         if (
    #                 ns[0] < 0 or ns[0] == self.ROWS or
    #                 ns[1] < 0 or ns[1] == self.COLS or
    #                 self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
    #             continue
    #         possibleA = np.append(possibleA, [a])
    #     return possibleA

    # def expectedStep(self, s, a):
    #     # Returns k possible outcomes
    #     #  p: k-by-1    probability of each transition
    #     #  r: k-by-1    rewards
    #     # ns: k-by-|s|  next state
    #     #  t: k-by-1    terminal values
    #     # pa: k-by-??   possible actions for each next state
    #     actions = self.possibleActions(s)
    #     k = len(actions)
    #     # Make Probabilities
    #     intended_action_index = findElemArray1D(a, actions)
    #     p = np.ones((k, 1)) * self.NOISE / (k * 1.)
    #     p[intended_action_index, 0] += 1 - self.NOISE
    #     # Make next states
    #     ns = np.tile(s, (k, 1)).astype(int)
    #     actions = self.ACTIONS[actions]
    #     ns += actions
    #     # Make next possible actions
    #     pa = np.array([self.possibleActions(sn) for sn in ns])
    #     # Make rewards
    #     r = np.ones((k, 1)) * self.STEP_REWARD
    #     goal = self.map[ns[:, 0], ns[:, 1]] == self.GOAL
    #     pit = self.map[ns[:, 0], ns[:, 1]] == self.PIT
    #     r[goal] = self.GOAL_REWARD
    #     r[pit] = self.PIT_REWARD
    #     # Make terminals
    #     t = np.zeros((k, 1), bool)
    #     t[goal] = True
    #     t[pit] = True
    #     return p, r, ns, t, pa

    # def allStates(self):
    #     if self.continuous_dims == []:
    #         # Recall that discrete dimensions are assumed to be integer
    #         return (
    #             perms(
    #                 self.discrete_statespace_limits[:,
    #                                                 1] - self.discrete_statespace_limits[:,
    #                                                                                      0] + 1) + self.discrete_statespace_limits[
    #                 :,
    #                 0]
    #         )
