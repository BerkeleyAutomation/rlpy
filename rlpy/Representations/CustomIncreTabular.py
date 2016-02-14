"""Incrementally expanded Tabular Representation"""

from .Representation import Representation
from rlpy.Tools import className, addNewElementForAllActions
import numpy as np
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class CustomIncreTabular(Representation):
    """
    Identical to Tabular representation (ie assigns a binary feature function 
    f_{d}() to each possible discrete state *d* in the domain, with
    f_{d}(s) = 1 when d=s, 0 elsewhere.
    HOWEVER, unlike *Tabular*, feature functions are only created for *s* which
    have been encountered in the domain, not instantiated for every single 
    state at the outset.

    """
    hash = None

    def __init__(self, domain, discretization=20):
        self.hash = {}
        self.features_num = 0
        self.isDynamic = True
        super(
            CustomIncreTabular,
            self).__init__(
            domain,
            discretization)

    def phi_nonTerminal(self, s):
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        F_s = np.zeros(self.features_num, bool)
        if hashVal is not None:
            F_s[hashVal] = 1
        return F_s

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return self._add_state(s) + self._add_state(sn)

    def _add_state(self, s):
        """
        :param s: the (possibly un-cached) state to hash.
        
        Accepts state ``s``; if it has been cached already, do nothing and 
        return 0; if not, add it to the hash table and return 1.
        
        """
        
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        if hashVal is None:
            # New State
            self.features_num += 1
            # New id = feature_num - 1
            hashVal = self.features_num - 1
            self.hash[hash_id] = hashVal
            # Add a new element to the feature weight vector, theta
            self.addNewWeight(s)
            return 1
        return 0

    def __deepcopy__(self, memo):
        new_copy = CustomIncreTabular(
            self.domain,
            self.discretization)
        new_copy.hash = deepcopy(self.hash)
        return new_copy

    def featureType(self):
        return bool

    def initial_value(self, state):
        def calculate(state):
            delt = 1./np.linalg.norm(state[:2] - self.domain.GOAL[:2]) ** 2 # may need to make goal full
            orient = 0 #1./(self.domain.GOAL[3] - state[3]) 
            trajdist = self.domain.closeness(state) #hi = closer
            return (delt + orient + 2 * trajdist) * 0.01

        return np.array([calculate(self.domain.simulate_step(state, i)) 
                for i in range(self.actions_num)])

    def Qs(self, s, terminal, phi_s=None):
        """
        Returns an array of actions available at a state and their
        associated values.

        :param s: The queried state
        :param terminal: Whether or not *s* is a terminal state
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        :return: The tuple (Q,A) where:
            - Q: an array of Q(s,a), the values of each action at *s*. \n
            - A: the corresponding array of actionIDs (integers)

        .. note::
            This function is distinct
            from :py:meth:`~rlpy.Representations.Representation.Representation.Q`,
            which computes the Q function for an (s,a) pair. \n
            Instead, this function ``Qs()`` computes all Q function values
            (for all possible actions) at a given state *s*.

        """

        if phi_s is None:
            phi_s = self.phi(s, terminal)
        if len(phi_s) == 0:
            return self.initial_value(s)
        weight_vec_prime = self.weight_vec.reshape(-1, self.features_num)
        if self._phi_sa_cache.shape != (self.actions_num, self.features_num):
            self._phi_sa_cache = np.empty(
                (self.actions_num, self.features_num))
        Q = np.multiply(weight_vec_prime, phi_s,
                        out=self._phi_sa_cache).sum(axis=1)
        # stacks phi_s in cache
        return Q

    def addNewWeight(self, state): # TODO: OVERRIDE
        """
        Add a new zero weight, corresponding to a newly added feature,
        to all actions.
        """
        # import ipdb; ipdb.set_trace()
        self.weight_vec = addNewElementForAllActions(
            self.weight_vec,
            self.actions_num,
            self.initial_value(state).reshape(-1, 1))
