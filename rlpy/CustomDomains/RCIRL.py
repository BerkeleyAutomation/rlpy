"""
This class implements a GridWorld with
consumable rewards
"""
import os,sys,inspect
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
from rlpy.Domains.Domain import Domain
from DomainMethods import allMarkovEncoding
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from copy import deepcopy
import numpy as np
import pickle
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
    GOAL_REWARD = 1
    # GOAL = [.5, .5]
    GOAL_RADIUS = .1
    HEADBOUND = 0.10000
    SPEEDBOUND = 0.02

    actions = np.outer([-1, 0, 1], [-1, 0, 1])
    discount_factor = .9
    episodeCap = 10000
    delta_t = .3  # time between steps
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
                 encodingFunction=allMarkovEncoding, # TODO
                 rewardFunction=None, 
                 episodeCap=None,
                 goal_radius=None, 
                 headbound=None,
                 noise=.1,
                 step_reward=None):
        self.debug = False
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
        self.start_state = self.augment_state(self.INIT_STATE)

        if noise:
            self.NOISE = noise
        if step_reward:
            self.STEP_REWARD = step_reward
        if episodeCap:
            self.episodeCap = episodeCap
        if goal_radius:
            self.GOAL_RADIUS = goal_radius
        if headbound:
            self.HEADBOUND = headbound
        if step_reward:
            self.STEP_REWARD = step_reward
        #remove goals for existing maps

        # set given goals

        encodingLimits = []
        for i in range(0,len(self.encodingFunction(self.prev_states))):
            encodingLimits.append([0,1])

        self.statespace_limits = np.vstack((self.statespace_limits, encodingLimits))
        self.state_space_dims = len(self.statespace_limits)
        continuous_dims = np.arange(self.state_space_dims)

        self.DimNames = ["Dim: "+str(k) for k in range(0,2+len(self.encodingFunction(self.prev_states)))]

        super(RCIRL, self).__init__()

    def step(self, a):
        r = self.STEP_REWARD
        ga = self.goalArray.copy()

        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

    #     # Take action
    #     statesize = np.shape(self.state)
    #     actionsize = np.shape(self.ACTIONS[a])
    #    # print statesize[0]-actionsize[0]

        ns = self.augment_state(self.simulate_step(self.state[:4], a))
        # print ns
        # if any(ns):
            # import ipdb; ipdb.set_trace()

        self.prev_states.append(ns[:4])
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]

        terminal = self.isTerminal() # TODO: Check - get terminal after?

        if self.collided(self.state):
            # r = (self.episodeCap - len(self.prev_states)) * self.STEP_REWARD
            r = -5
            terminal = True
            if True or self.debug:
                print "====================== BAM ====================="


    #     # Compute the reward and enforce ordering
        if not terminal and self.at_goal(state=ns, goal=ga[0]): 
            r = self.GOAL_REWARD * (len(self.goalArray0) - len(self.goalArray)) * 2
            self.goalArray = ga[1:]
            if self.debug:
                print "New Goal Reached!", ns, r, len(self.prev_states)

        if not terminal and self.rewardFunction != None:
            r = self.rewardFunction(self.prev_states, 
                                    self.goalArray, 
                                    self.STEP_REWARD, 
                                    self.GOAL_REWARD)

        self.state = ns.copy()
        return r, ns, terminal, self.possibleActions()

    def augment_state(self, state):
        return np.concatenate((state, 
                            self.encodingFunction(self.prev_states)))

    def collided(self, state):
        x, y = state[:2]
        return x == self.XMIN or x == self.XMAX or y == self.YMIN or y == self.YMAX


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

        if self.debug:
            print "#############################"
            print "######## NEW TRIAL ##########"
            print "#############################"

        self.prev_states = []
        self.state = self.augment_state(self.INIT_STATE.copy())
        self.goalArray = np.array(self.goalArray0)
        
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        # TODO
        # termination condition should include max_steps?
        if s is None:
            s = self.state
        if len(self.goalArray) == 0:
            return True

        return False

    def at_goal(self, state=None, goal=None):
        """Check if current state is at goal"""
        state = state if state is not None else self.state
        goal = goal if goal is not None else self.GOAL
        return (np.linalg.norm(state[:2] - goal[:2]) < self.GOAL_RADIUS
            and (abs(state[2] - goal[2]) < self.SPEEDBOUND)
            and abs(state[3] - goal[3]) < self.HEADBOUND)

    def showDomain(self, a):
        s = self.state
        # Plot the car
        x, y, speed, heading = s[:4]
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure()
            # Goal
            for goal in self.goalArray:
                plt.gca(
                ).add_patch(
                    plt.Circle(
                        goal[:2],
                        radius=self.GOAL_RADIUS,
                        color='g',
                        alpha=.4))
                plt.xlim([self.XMIN, self.XMAX])
                plt.ylim([self.YMIN, self.YMAX])
                plt.gca().set_aspect('1')
        # Car
        if self.car_fig is not None:
            plt.gca().patches.remove(self.car_fig)

        self.car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            self.CAR_LENGTH,
            self.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + plt.gca().transData
        self.car_fig.set_transform(rotation)
        plt.gca().add_patch(self.car_fig)

        plt.draw()
