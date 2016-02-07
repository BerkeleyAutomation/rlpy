"""RC-Car domain"""

from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from .Domain import Domain
from copy import deepcopy
import numpy as np
import pickle

__author__ = "Alborz Geramifard"


class RCSegment(Domain):

    """
    This is a simple simulation of Remote Controlled Car in a room with no obstacle.

    **STATE:** 4 continuous dimensions:

    * x, y: (center point on the line connecting the back wheels),
    * speed (S on the webpage)
    * heading (theta on the webpage) w.r.t. body frame.
        positive values => turning right, negative values => turning left

    **ACTIONS:** Two action dimensions:

    * accel [forward, coast, backward]
    * phi [turn left, straight, turn Right]

    This leads to 3 x 3 = 9 possible actions.

    **REWARD:** -1 per step, 100 at goal.

    **REFERENCE:**

    .. seealso::
        http://planning.cs.uiuc.edu/node658.html

    """

    actions_num = 9
    state_space_dims = 4
    continuous_dims = np.arange(state_space_dims)
    discretization = 20

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

    XBIN = float(XMAX - XMIN) / discretization
    YBIN = float(YMAX - YMIN) / discretization

    #REWARDS
    STEP_REWARD = -0.1
    GOAL_REWARD = 5
    GOAL = [0, .4, 0, np.pi]
    GOAL_RADIUS = .1

    SEG_REWARD = 0.1

    GOAL_ORIENT_BOUND = 0.2

    rewards = []
    cur_rewards = None

    COLLIDE_REWARD = -1

    actions = np.outer([-1, 0, 1], [-1, 0, 1])
    discount_factor = .9
    episodeCap = 100
    delta_t = .3  # time between steps #TODO MAKE VARIABLE
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

    EMPTY, BLOCKED = range(2)

    def __init__(self, goal=[0.5, 0.3], 
                    noise=0, discretize=20, with_collision=False, 
                    mapname=None, episodeCap=200, rewardfile=None,
                    goal_radius=None, orientation_bound=None,
                    goal_reward=None, collide_reward=None, step_reward=None):
        self.map = None
        self.statespace_limits = np.array(
            [[self.XMIN,
              self.XMAX],
             [self.YMIN,
              self.YMAX],
            [self.SPEEDMIN,
             self.SPEEDMAX],
            [self.HEADINGMIN,
             self.HEADINGMAX]])
        self.episodeCap = episodeCap
        self.noise = noise
        self.GOAL = np.array(goal) # currently no bound on acceleration

        if goal_radius:
            self.GOAL_RADIUS = goal_radius
        if orientation_bound:
            self.GOAL_ORIENT_BOUND = orientation_bound
        if goal_reward:
            self.GOAL_REWARD = goal_reward
        if collide_reward:
            self.COLLIDE_REWARD = collide_reward
        if step_reward:
            self.STEP_REWARD = step_reward

        if mapname:
            self.map = np.loadtxt(mapname, dtype=np.uint8)
            assert self.map.shape == (20, 20) # no access to the discretization parameter?
            assert self.get_bin(self.GOAL) not in np.argwhere(self.map == self.BLOCKED)
            self.calculate_protrusions()

        if rewardfile:
            with open(rewardfile, "r") as f:
                self.rewards = pickle.load(f)

        self.with_segment = (rewardfile is not None)
        self.with_collision = with_collision
        super(RCSegment, self).__init__()

    def step(self, a):
        ns = self.simulate_step(self.state, a)
        r = self.STEP_REWARD
        self.state = ns.copy()

        if self.with_segment:
            if self.get_segmented_reward():
                r += self.SEG_REWARD * (50 - len(self.cur_rewards)) # need to make this portable

        terminal = self.isTerminal()
        if (self.with_collision and self.collided(self.state)):
            r += self.COLLIDE_REWARD
            terminal = True
        elif terminal:
            r += self.GOAL_REWARD
            # terminal = False #terminal debug
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

        # TODO: Collisions should represent full car body, rather than X,Y state

        ##simulate real car - bind by car body, not by x-y
        if False and self.with_collision:
            ns = np.array([nx, ny, nspeed, nheading])
            if self.collided(ns):
                nspeed = 0

                #bind nx, ny, heading to negate movement
            elif False and self.hit_protrusion(ns):
                nspeed = 0
                #bind nx, ny to negate movement

        return np.array([nx, ny, nspeed, nheading])


    def closeness(self, state):
        head = state[3]
        cosine_sim = lambda a, b: (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        orient = np.array([np.cos(head), np.sin(head)])
        dgoal = np.subtract(self.GOAL[:2], state[:2])
        return cosine_sim(orient, dgoal)

    def s0(self):
        self.state = self.INIT_STATE.copy()
        if len(self.rewards):
            self.reset_rewards()
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        return self.at_goal() 

    def get_segmented_reward(self): # may want to make this into a trajectory rather than 
        # currently only takes one solution; could technically start on reward path in the middle of exectuion
        if len(self.cur_rewards) == 0:
            return False
        seg = self.cur_rewards[0]
        if (np.linalg.norm(self.state[:2] - seg[:2]) < self.GOAL_RADIUS
            and abs(self.state[3] - seg[3]) < self.GOAL_ORIENT_BOUND): # may need to constrict angle too
            reward = self.cur_rewards[0]
            self.cur_rewards = self.cur_rewards[1:]
            return True
        return False

    def at_goal(self):
        """Check if current state is at goal"""
        return (np.linalg.norm(self.state[:2] - self.GOAL[:2]) < self.GOAL_RADIUS
            and abs(self.state[3] - self.GOAL[3]) < self.GOAL_ORIENT_BOUND)

    def collided(self, state):
        """Given a car state, check if collided with walls"""
        corners = self.get_car_corners(state)
        return any( not self.within_boundaries(c)
                    #or (self.map[self.get_bin(c)] == self.BLOCKED) 
                    for c in corners)

    def get_car_corners(self, state):
        """Given a car state, return position of 4 corners"""
        # http://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation
        x, y, _, theta = state
        translate = lambda ln, wd: ((ln*np.cos(theta) - wd*np.sin(theta) + x), (ln*np.sin(theta) + wd*np.cos(theta) + y))
        return [translate(a, b) for a, b in [(self.CAR_LENGTH - self.REAR_WHEEL_RELATIVE_LOC, self.CAR_WIDTH/2), 
                                             (self.CAR_LENGTH - self.REAR_WHEEL_RELATIVE_LOC, -self.CAR_WIDTH/2),
                                             (-self.REAR_WHEEL_RELATIVE_LOC, -self.CAR_WIDTH/2), 
                                             (-self.REAR_WHEEL_RELATIVE_LOC, self.CAR_WIDTH/2)]]

    def hit_protrusion(self, state):
        """Check if car walls have hit any protrusions
        :param state: state/orientation of car
        """
        corners = self.get_car_corners(state)
        car_walls = ((corners[x-1], corners[x]) for x in range(len(corners)))
        def same_side(wall, p2): # corners of the car
            cor1, cor2 = wall
            v1 = np.array([cor2[0] - cor1[0], cor2[1] - cor1[1]]) # origin centered base vector
            v2 = np.array([state[0] - cor1[0] , state[1] - cor1[1]]) # 
            v3 = np.array([p2[0] - cor1[0],  p2[1] - cor1[1]])
            return (np.dot(np.cross(v1, v2), np.cross(v1, v3)) >= 0)
        
        return any(same_side(wall, obstacle) for wall in car_walls for obstacle in self.protrusions)

    def within_boundaries(self, pos):
        """
        Returns true if pos is within the domain limits
        :param pos: x, y coordinate
        """
        x, y = pos
        return ((self.XMIN <= x <= self.XMAX) and (self.YMIN <= y <= self.YMAX))

    def is_near_obstacle(self, state):
        """Returns true if state has corners that are in bins next to obstacles"""
        pass

    def get_bin(self, pos):
        """Calculate which map-bin the position is in"""
        x, y = pos
        dis = self.discretization
        return (x//self.XBIN + dis/2, y//self.YBIN + dis/2) 

    def get_bin_corner(self, x, y):
        """
        Given a bin in map[y][x], return the corners
        """
        assert hasattr(self, "map") # not necessary
        assert 0 <= x < self.discretization
        def convert_to_real(x, y):
            return (x*self.XBIN + self.XMIN, x*self.YBIN + self.YMIN)
        return [convert_to_real(x + dx, y + dy) for dx in range(2) for dy in range(2)]

    def calculate_protrusions(self):
        """Need to test"""
        assert hasattr(self, "map")
        self.protrusions = set()
        dis = self.discretization
        for y in range(dis): 
            for x in range(dis):
                if self.map[y][x] == self.BLOCKED:
                    bottom = self.map[y+1][x]
                    top = self.map[y-1][x]
                    right = self.map[y][x+1]
                    left = self.map[y][x-1]
                    bincorners = self.get_bin_corner(x, y)
                    if bottom and left:
                        self.protrusions.add(bincorners[0])
                    if top and left:
                        self.protrusions.add(bincorners[1])
                    if bottom and right:
                        self.protrusions.add(bincorners[2])
                    if top and right:
                        self.protrusions.add(bincorners[3])
        return len(self.protrusions)

    def reset_rewards(self):
        self.cur_rewards = deepcopy(self.rewards)

    def show_inline(self):
        s = self.state
        # Plot the car
        x, y, speed, heading = s
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        import matplotlib.pyplot as _plt
        self.domain_fig = _plt.figure()
        # Goal
        _plt.gca(
        ).add_patch(
            _plt.Circle(
                self.GOAL,
                radius=self.GOAL_RADIUS,
                color='g',
                alpha=.4))

        for reward in self.rewards:
            plt.gca(
            ).add_patch(
                plt.Circle(
                    reward[:2],
                    radius=self.GOAL_RADIUS,
                    color='y',
                    alpha=.4))

        _plt.xlim([self.XMIN, self.XMAX])
        _plt.ylim([self.YMIN, self.YMAX])
        _plt.gca().set_aspect('1')
        # try:
        #     _plt.gca().patches.remove(self.car_fig)
        car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            self.CAR_LENGTH,
            self.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + _plt.gca().transData
        car_fig.set_transform(rotation)
        _plt.gca().add_patch(car_fig)
        _plt.draw()


    def showDomain(self, a):
        s = self.state
        # Plot the car
        x, y, speed, heading = s
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure()
            # Goal
            plt.gca(
            ).add_patch(
                plt.Circle(
                    self.GOAL,
                    radius=self.GOAL_RADIUS,
                    color='g',
                    alpha=.4))

            for reward in self.rewards:
                plt.gca(
                ).add_patch(
                    plt.Circle(
                        reward[:2],
                        radius=self.GOAL_RADIUS,
                        color='y',
                        alpha=.4))

            for block in np.argwhere(self.map == self.BLOCKED):
                wall_xmin, wall_ymin = block
                plt.gca().add_patch( 
                    mpatches.Rectangle(
                        [wall_xmin,
                         wall_ymin],
                        self.XBIN,
                        self.YBIN,
                        alpha=.4)
                    )

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
