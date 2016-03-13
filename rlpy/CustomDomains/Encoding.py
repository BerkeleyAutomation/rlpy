import numpy as np

class Encoding:

    def __init__(self, waypoints, goalfn):
        self.ps_idx = 0
        self.waypoint_count = 0 # number of waypoints acquired
        self.waypoints = waypoints
        self.goalfn = goalfn

    @staticmethod
    def allMarkovEncoding(ps):
        return [0]

    def strict_encoding(self, ps):
        if len(ps) == 0:
            self.ps_idx = 0
            self.waypoint_count = 0

        result = []
        wpc = self.waypoint_count # waypointindex
        if wpc == len(self.waypoints):
            # print "All Waypoints Reached"
            return np.ones(wpc)
        for i, s in list(enumerate(ps))[self.ps_idx:]:
            # check if achieved next waypoint
                # increment waypoint
            while self.goalfn(s, goal=self.waypoints[wpc]):
                print "New waypoint reached - Got em {} + 1".format(wpc)
                wpc += 1 
                if wpc >= len(self.waypoints):
                    break

        self.waypoint_count = wpc
        self.ps_idx = len(ps)
        return [1 if i < wpc else 0 
                    for i in range(len(self.waypoints))]

    def weak_encoding(self, ps):
        result = []
        waypoints = self.waypoints
        for w in waypoints:
            k = -1
            pl = [tuple(p) for p in ps]
            try:
                k = next((i for i, state in enumerate(ps) 
                            if self.goalfn(state, goal=w)))
            except StopIteration:
                pass

            result.append(k)

        result_hash = []

        if len(waypoints) == 1 and result[0] == -1:
            return [0]
        elif len(waypoints) == 1 and result[0] != -1:
            return [1]
        
        if result[0] != -1:
            result_hash.append(1)
        else:
            result_hash.append(0)

        for i in range(1,len(waypoints)):
            if result[i] != -1 and result[i] > result[i-1]: # false alarm if previous state = -1
                result_hash.append(1)
            else:
                result_hash.append(0)
        return result_hash