"""
This class implements a GridWorld with
consumable rewards
"""
import os,sys,inspect
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
from rlpy.Domains.Domain import Domain
# from rlpy.Domains import RCCar
import numpy as np
from rlpy.Tools import plt, FONTSIZE, linearMap

class RCIRL(Domain): 
    # #default paths
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # default_map_dir = os.path.join(currentdir,"ConsumableGridWorldMaps")

    actions_num = 9
    state_space_dims = 4
    continuous_dims = np.arange(state_space_dims)

    ROOM_WIDTH = 3  # in meters
    ROOM_HEIGHT = 2  # in meters
    XMIN = -ROOM_WIDTH / 2.0
    XMAX = ROOM_WIDTH / 2.0
    YMIN = -ROOM_HEIGHT / 2.0
    YMAX = ROOM_HEIGHT / 2.0
    ACCELERATION = .1
    TURN_ANGLE = np.pi / 6
    SPEEDMIN = -.3
    SPEEDMAX = .3
    HEADINGMIN = -np.pi
    HEADINGMAX = np.pi
    INIT_STATE = np.array([0.0, 0.0, 0.0, 0.0])
    STEP_REWARD = -1
    GOAL_REWARD = 0
    GOAL = [.5, .5]
    GOAL_RADIUS = .1
    actions = np.outer([-1, 0, 1], [-1, 0, 1])
    discount_factor = .9
    episodeCap = 10000
    delta_t = .1  # time between steps
    CAR_LENGTH = .3  # L on the webpage
    CAR_WIDTH = .15
    # The location of rear wheels if the car facing right with heading 0
    REAR_WHEEL_RELATIVE_LOC = .05
    # Used for visual stuff:
    domain_fig = None
    X_discretization = 20
    Y_discretization = 20
    SPEED_discretization = 5
    HEADING_discretization = 3
    ARROW_LENGTH = .2
    car_fig = None


    # #an encoding function maps a set of previous states to a fixed
    # #width encoding
    # """
    # A reward function is a function over ALL of the previous states, 
    # the set of goal states,
    # the step reward constant, 
    # and the goal reward constant.
    # """
    def __init__(self, 
                 goalArray, 
    			 mapname=os.path.join(default_map_dir, "4x5.txt"),
                 encodingFunction=ConsumableGridWorldIRL.allMarkovEncoding,
                 rewardFunction=None,
                 noise=.1, 
                 episodeCap=None):
       
        #setup consumable rewards
        self.statespace_limits = np.array(
            [[self.XMIN,
              self.XMAX],
             [self.YMIN,
              self.YMAX],
                [self.SPEEDMIN,
                 self.SPEEDMAX],
                [self.HEADINGMIN,
                 self.HEADINGMAX]])

        self.goalArray0 = np.array(goalArray)
        self.goalArray = np.array(goalArray)
        self.prev_states = []

        self.encodingFunction = encodingFunction

        self.rewardFunction = rewardFunction

        # should convert to bins? or leave discrete
        self.start_state = np.concatenate((np.argwhere(self.map == self.START)[0], 
                                             self.encodingFunction(self.prev_states)))

        #remove goals for existing maps

        # set given goals

        encodingLimits = []
        for i in range(0,len(self.encodingFunction(self.prev_states))):
            encodingLimits.append([0,1])

        self.statespace_limits.extend(encodingLimits)

        self.NOISE = noise
        self.DimNames = ["Dim: "+str(k) for k in range(0,2+len(self.encodingFunction(self.prev_states)))]
        # 2*self.ROWS*self.COLS, small values can cause problem for some
        # planning techniques
        self.episodeCap = 2*self.ROWS*self.COLS # TODO
        super(RCIRL, self).__init__()

    # def showDomain(self, a=0, s=None):
    # 	raise NotImplementedError

    def step(self, a):
        r = self.STEP_REWARD
        ns = self.state.copy()
        ga = self.goalArray.copy()

        self.prev_states.append(ns[0:2])

        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

    #     # Take action
    #     statesize = np.shape(self.state)
    #     actionsize = np.shape(self.ACTIONS[a])
    #    # print statesize[0]-actionsize[0]

        ns = np.concatenate(self.simulate_step(self.state[:-1], a), 
                            self.encodingFunction(self.prev_states))
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]

        terminal = self.isTerminal()
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]
    #     # Compute the reward and enforce ordering
        if terminal:
            pass
        elif ga[0][0] == ns[0] and  ga[0][1] == ns[1]:
            r = self.GOAL_REWARD
            self.goalArray = ga[1:]
            #print "Goal!", ns

        if self.rewardFunction != None:
            r = self.rewardFunction(self.prev_states, 
                                    self.goalArray, 
                                    self.STEP_REWARD, 
                                    self.GOAL_REWARD)

        return r, ns, terminal, self.possibleActions()


    def simulate_step(self, state, a):
        x, y, speed, heading = state
        # Map a number between [0,8] to a pair. The first element is
        # acceleration direction. The second one is the indicator for the wheel
        acc, turn = id2vec(a, [3, 3])
        acc -= 1                # Mapping acc to [-1, 0 1]
        turn -= 1                # Mapping turn to [-1, 0 1]

        # Calculate next state
        nx = x + speed * np.cos(heading) * self.delta_t
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        # Bound values
        nx = bound(nx, self.XMIN, self.XMAX)
        ny = bound(ny, self.YMIN, self.YMAX)
        nspeed = bound(nspeed, self.SPEEDMIN, self.SPEEDMAX)
        nheading = wrap(nheading, self.HEADINGMIN, self.HEADINGMAX)

        # Collision to wall => set the speed to zero
        if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
            nspeed = 0

        return np.array([nx, ny, nspeed, nheading])

    def s0(self):
        self.prev_states = []
        self.state = self.INIT_STATE.copy()
        self.goalArray = np.array(self.goalArray0)
        
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        # TODO
        # termination condition should include max_steps?
    #     if s is None:
    #         s = self.state
    #     if len(self.goalArray) == 0:
    #         return True
    #     #if len(self.prev_states) > self.MAX_STEPS:
    #     #    return True

    #     return False
        return self.at_goal() 

    def at_goal(self):
        """Check if current state is at goal"""
        return (np.linalg.norm(self.state[:2] - self.GOAL[:2]) < self.GOAL_RADIUS
            and abs(self.state[3] - self.GOAL[3]) < self.GOAL_ORIENT_BOUND)