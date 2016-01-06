"""Standard Experiment for Learning Control in RL."""

import logging
from rlpy.Tools import plt
import numpy as np
from copy import deepcopy
import re
import argparse
from rlpy.Tools import deltaT, clock, hhmmss
from rlpy.Tools import className, checkNCreateDirectory
from rlpy.Tools import printClass
import rlpy.Tools.results
from rlpy.Tools import lower
import os
import rlpy.Tools.ipshell
import json
from collections import defaultdict
from Experiment import Experiment
from rlpy.CustomDomains import GridWorldTime

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class ExperimentSegment(Experiment):

    """
    The Experiment controls the training, testing, and evaluation of the
    agent. Reinforcement learning is based around
    the concept of training an :py:class:`~Agents.Agent.Agent` to solve a task,
    and later testing its ability to do so based on what it has learned.
    This cycle forms a loop that the experiment defines and controls. First
    the agent is repeatedly tasked with solving a problem determined by the
    :py:class:`~Domains.Domain.Domain`, restarting after some termination
    condition is reached.
    (The sequence of steps between terminations is known as an *episode*.)

    This class is specifically framed around the GridWorldTime class.

    Each time the Agent attempts to solve the task, it learns more about how
    to accomplish its goal. The experiment controls this loop of "training
    sessions", iterating over each step in which the Agent and Domain interact.
    After a set number of training sessions defined by the experiment, the
    agent's current policy is tested for its performance on the task.
    The experiment collects data on the agent's performance and then puts the
    agent through more training sessions. After a set number of loops, training
    sessions followed by an evaluation, the experiment is complete and the
    gathered data is printed and saved. For each section, training and
    evaluation, the experiment determines whether or not the visualization
    of the step should generated.

    The Experiment class is a base class that provides
    the basic framework for all RL experiments. It provides the methods and
    attributes that allow child classes to interact with the Agent
    and Domain classes within the RLPy library.

    .. note::
        All experiment implementations should inherit from this class.
    """

    #: The Main Random Seed used to generate other random seeds (we use a different seed for each experiment id)
    mainSeed = 999999999
    #: Maximum number of runs used for averaging, specified so that enough
    #: random seeds are generated
    maxRuns = 1000
    # Array of random seeds. This is used to make sure all jobs start with
    # the same random seed
    randomSeeds = np.random.RandomState(mainSeed).randint(1, mainSeed, maxRuns)

    #: ID of the current experiment (main seed used for calls to np.rand)
    exp_id = 1

    # The domain to be tested on
    domain = None
    # The agent to be tested
    agent = None

    #: A 2-d numpy array that stores all generated results.The purpose of a run
    #: is to fill this array. Size is stats_num x num_policy_checks.
    result = None
    #: The name of the file used to store the data
    output_filename = ''
    # A simple object that records the prints in a file
    logger = None

    max_eps = 0  # Total number of interactions
    # Number of Performance Checks uniformly scattered along timesteps of the
    # experiment
    num_policy_checks = 0
    log_interval = 0  # Number of seconds between log prints to console

    log_template = '{total_steps: >6}: Eps {episode_number}: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'
    performance_log_template = '{total_steps: >6}: >>>Eps {episode_number}: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'

    def __init__(self, agent, domain, eval_domain, exp_id=1, max_eps=1000,
                 config_logging=True, num_policy_checks=10, log_interval=1,
                 path='Results/Temp',
                 checks_per_policy=1, stat_bins_per_state_dim=0, **kwargs):
        """
        :param agent: the :py:class:`~Agents.Agent.Agent` to use for learning the task.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param eval_domain: the original MDP that we should evaluate the agent on
        :param exp_id: ID of this experiment (main seed used for calls to np.rand)
        :param max_eps: Total number of policy rollouts to be done

        :param num_policy_checks: Number of Performance Checks uniformly
            scattered along timesteps of the experiment
        :param log_interval: Number of seconds between log prints to console
        :param path: Path to the directory to be used for results storage
            (Results are stored in ``path/output_filename``)
        :param checks_per_policy: defines how many episodes should be run to
            estimate the performance of a single policy

        """
        self.exp_id = exp_id
        assert exp_id > 0
        self.agent = agent
        self.checks_per_policy = checks_per_policy

        assert np.array_equal(np.argwhere(domain.map == domain.GOAL), 
                            np.argwhere(eval_domain.map == eval_domain.GOAL))

        self.domain = domain
        self.eval_domain = eval_domain

        self.max_eps = max_eps
        self.num_policy_checks = num_policy_checks
        self.logger = logging.getLogger("rlpy.Experiments.Experiment")

        logging.disable(40)

        self.log_interval = log_interval
        self.config_logging = config_logging
        self.path = path
        if stat_bins_per_state_dim > 0:
            self.state_counts_learn = np.zeros(
                (domain.statespace_limits.shape[0],
                 stat_bins_per_state_dim), dtype=np.long)
            self.state_counts_perf = np.zeros(
                (domain.statespace_limits.shape[0],
                 stat_bins_per_state_dim), dtype=np.long)

        self.perform_trajs = defaultdict(list)

    def dcperformanceRun(self, visualize=False, saveTrajectories=False, current_steps=0):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        :param saveTrajectories: boolean, optional
            defines whether to save the trajectory of the run 
        :param current_steps: int, optional
            the number of learned steps that have passed before this evaluation step
        """

        # Set Exploration to zero and sample one episode from the domain
        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.agent.policy.turnOffExploration()
        temp_performance_domain = deepcopy(self.performance_domain) ##Modification here

        # #### CHECK that steps are reset to 0
        # import ipdb; ipdb.set_trace()
        # ##########
        s, eps_term, p_actions = temp_performance_domain.s0()

        while not eps_term and eps_length < self.domain.episodeCap:
            if saveTrajectories:
                self.perform_trajs[current_steps].append(s) #TODO: move somewhere else
            a = self.agent.policy.pi(s, eps_term, p_actions)
            if visualize:
                temp_performance_domain.showDomain(a)

            r, ns, eps_term, p_actions = temp_performance_domain.step(a)
                
            self._gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += temp_performance_domain.discount_factor ** eps_length * r
            eps_length += 1
        if visualize:
            temp_performance_domain.showDomain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying
        # to accomodate them as an MDP

        return eps_return, eps_length, eps_term, eps_discount_return


    def performanceRun(self, total_steps, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.agent.policy.turnOffExploration()

        s, eps_term, p_actions = self.performance_domain.s0()

        while not eps_term and eps_length < self.domain.episodeCap:
            a = self.agent.policy.pi(s, eps_term, p_actions)
            if visualize:
                self.performance_domain.showDomain(a)

            r, ns, eps_term, p_actions = self.performance_domain.step(a)
            self._gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += self.performance_domain.discount_factor ** eps_length * \
                r
            eps_length += 1
        if visualize:
            self.performance_domain.showDomain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying
        # to accomodate them as an MDP


        return eps_return, eps_length, eps_term, eps_discount_return

    def run(self, visualize_performance=0, visualize_learning=False,
            visualize_steps=False, debug_on_sigurg=False, saveTrajectories=False):
        """
        Run the experiment and collect statistics / generate the results

        :param visualize_performance: (int)
            determines whether a visualization of the steps taken in
            performance runs are shown. 0 means no visualization is shown.
            A value n > 0 means that only the first n performance runs for a
            specific policy are shown (i.e., for n < checks_per_policy, not all
            performance runs are shown)
        :param visualize_learning: (boolean)
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        :param visualize_steps: (boolean)
            visualize all steps taken during learning
        :param debug_on_sigurg: (boolean)
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """
        if debug_on_sigurg:
            rlpy.Tools.ipshell.ipdb_on_SIGURG()

        self.train_domain = deepcopy(self.domain)
        self.performance_domain = deepcopy(self.eval_domain) #TODO: Redundant copying

        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        total_steps = 0
        eps_steps = 0
        eps_return = 0
        episode_number = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.agent.representation)

        # Used to bound the number of logs in the file
        start_log_time = clock()
        # Used to show the total time took the process
        self.start_time = clock()
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        # self.evaluate(total_steps, episode_number, visualize_performance)
        self.total_eval_time = 0.
        terminal = True
        while episode_number <= self.max_eps:
            if terminal or eps_steps >= self.domain.episodeCap:
                # Check Performance
                if self.num_policy_checks > 0 and episode_number % (self.max_eps / self.num_policy_checks) == 0:
                    # show policy or value function
                    if visualize_learning: #or episode_number == 400:
                        self.domain.showLearning(self.agent.representation)

                    self.evaluate(
                        total_steps,
                        episode_number,
                        visualize_performance,
                        saveTrajectories)

                self.domain.map = deepcopy(self.train_domain.map) # get a copy of the original map

                if isinstance(self.domain, GridWorldTime):
                    self.domain.reset_steps()
                s, terminal, p_actions = self.domain.s0()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been
                # passed
                eps_return = 0
                eps_steps = 0
                episode_number += 1

                # if episode_number == 4000 and self.domain.map[1, 7] < 10:
                #     import ipdb; ipdb.set_trace()
                #     self.agent.policy.epsilon += 0.05
            # Act,Step
            r, ns, terminal, np_actions = self.domain.step(a)

            self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps += 1
            eps_steps += 1
            eps_return += r

            # learning
            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions = ns, na, np_actions

            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)

        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.info("Total Experiment Duration %s" % (hhmmss(deltaT(self.start_time))))

        if saveTrajectories:
            ###debug purposes
            import json
            results_fn = os.path.join(self.full_path, "trajectories.json")
            print results_fn
            if not os.path.exists(self.full_path):
                os.makedirs(self.full_path)
            with open(results_fn, "w") as f:
                json.dump(self.perform_trajs, f)

    def cur_v(self, performance=True):
        ''':param performance: Will only be evaluated on performance domain (not on the training environment); should not change though'''
        
        d = deepcopy(self.performance_domain if performance else self.train_domain)
        representation = self.agent.representation
        V = np.zeros((d.ROWS, d.COLS))
        for r in xrange(d.ROWS):
            for c in xrange(d.COLS):
                if d.map[r, c] == d.BLOCKED:
                    V[r, c] = 0
                if d.map[r, c] == d.GOAL:
                    V[r, c] = d.MAX_RETURN
                if d.map[r, c] == d.PIT:
                    V[r, c] = d.MIN_RETURN
                if d.map[r, c] == d.EMPTY or d.map[r, c] == d.START or (hasattr(d, "DOOR") and d.map[r, c] == d.DOOR):
                    s = np.array([r, c])
                    As = d.possibleActions(s)
                    terminal = d.isTerminal(s)
                    Qs = representation.Qs(s, terminal)
                    bestA = representation.bestActions(s, terminal, As)
                    V[r, c] = max(Qs[As])
        return V

                
 
    def evaluate(self, total_steps, episode_number, visualize=0, saveTrajectories=False):
        """
        Evaluate the current agent within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """
        # TODO resolve this hack
        if className(self.agent) == 'PolicyEvaluation':
            # Policy Evaluation Case
            self.result = self.agent.STATS
            return

        # print "EPISODE NUMBER {}".format(episode_number)
        # print (self.cur_v()).astype(np.int64)
        # print "Learn rate: {}".format(self.agent.learn_rate)

        random_state = np.random.get_state()
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        performance_discounted_return = 0.
        for j in xrange(self.checks_per_policy):
            if saveTrajectories and j == 0:
                p_ret, p_step, p_term, p_dret = self.dcperformanceRun(
                    visualize=visualize > j, saveTrajectories=True, current_steps=total_steps)
            else:
                p_ret, p_step, p_term, p_dret = self.dcperformanceRun(
                    visualize=visualize > j)

            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
            performance_discounted_return += p_dret

        performance_return /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        performance_discounted_return /= self.checks_per_policy

        self.result["learning_steps"].append(total_steps)
        self.result["return"].append(performance_return)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.agent.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["learning_episode"].append(episode_number)
        self.result["discounted_return"].append(performance_discounted_return)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        self.logger.info(
            self.performance_log_template.format(episode_number=episode_number,
                                                 total_steps=total_steps,
                                                 totreturn=performance_return,
                                                 steps=performance_steps,
                                                 num_feat=self.agent.representation.features_num))

        np.random.set_state(random_state)
        #self.domain.rand_state = random_state_domain

    def saveWeights(self, path=None):
        """Saves the weights of the representation to be transferred to another MDP"""
        import pickle
        results_fn = os.path.join(self.full_path, "weights.p")
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            pickle.dump(self.agent.representation.weight_vec, f)

    def plot(self, y="return", x="learning_steps", save=False):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider
        :py:class:`Tools.Merger.Merger`.
        """
        labels = rlpy.Tools.results.default_labels
        performance_fig = plt.figure("Performance")
        res = self.result
        plt.plot(res[x], res[y], '-bo', lw=3, markersize=10)
        plt.xlim(0, res[x][-1] * 1.01)
        y_arr = np.array(res[y])
        m = y_arr.min()
        M = y_arr.max()
        delta = M - m
        if delta > 0:
            plt.ylim(m - .1 * delta - .1, M + .1 * delta + .1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(
                self.full_path,
                "{:3}-performance.png".format(self.exp_id))
            performance_fig.savefig(path, transparent=True, pad_inches=.1)
        plt.ioff()
        plt.show()

