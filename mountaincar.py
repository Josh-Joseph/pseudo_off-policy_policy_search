import numpy as np
import scipy.stats
import rl_tools
from copy import deepcopy

reload(rl_tools)

XMIN = -1.2
XMAX = 0.5
XDOTMIN = -0.07
XDOTMAX = 0.07
INITSTATE = np.array([-np.pi / 2.0 / 3.0, 0.0])

class Mountaincar(rl_tools.Domain):
    def __init__(self, input_pars):
        self.input_pars = input_pars
        self.N_MC_eval_samples = 200
        self.episode_length = 500
        self.data_columns = ('x','xdot','u','r') # assume the 2nd to last is u and the last is r
        self.n_dim = 2
        self.bounds = np.array([[XMIN, XMAX],[XDOTMIN, XDOTMAX]]).transpose()
        self.goal = np.array([[-np.inf, XMAX],[-np.inf, np.inf]]).transpose()
        self.initstate = INITSTATE
        self.action_centers = np.array([-1, 1])
        self.n_x_centers = 300 #275
        self.n_xdot_centers = 150
        self.true_pars = (-0.0025, 3)
        self.initial_par_search_space = [[p1, p2] for p1 in np.linspace(-0.003, -.002, 5) for p2 in np.linspace(2, 4, 5)]
        self.noise = input_pars
        self.value_iteration_threshold = 1e-3
        self.state_centers = self.construct_discrete_policy_centers()
        self.dim_centers = rl_tools.split_states_on_dim(self.state_centers)

    def at_goal(self, s):
        return s[0] == XMAX

    def reward(self, s):
        return -1 if s[0] < XMAX else 0

    def approx_dynamics(self, s, u, pars):
        s = s.copy()
        x = s[0]
        xdot = s[1]
        s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
        s[1] = min(max(xdot+0.001*u+(pars[0]*np.cos(pars[1]*x)), self.bounds[0,1]), self.bounds[1,1])
        return s

    def true_dynamics(self, s, u):
        s = s.copy()
        x = s[0]
        xdot = s[1]
        s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
        if self.noise[1]:
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + np.random.normal(loc=-self.noise[0]*(xdot**2), scale=max(1e-4, self.noise[1]*(xdot**2))), self.bounds[0,1]), self.bounds[1,1])
        else:
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) - np.sign(xdot)*self.noise[0]*(xdot**2), self.bounds[0,1]), self.bounds[1,1])
        return s

    def true_dynamics_pmf(self, s_i, a_i):
        # returns a vector of probability masses the same size as state_centers
        s = self.state_centers[s_i,:].copy()
        u = self.action_centers[a_i]
        x = s[0]
        xdot = s[1]
        s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
        s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) - np.sign(xdot)*self.noise[0]*(xdot**2), self.bounds[0,1]), self.bounds[1,1])
        pmf = np.zeros(self.state_centers.shape[0])
        s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
        if self.noise[1]:
            supported_s = self.state_centers[:,0] == self.state_centers[s_next_i,0]
            tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,1], loc=s[1], scale=max(1e-4, self.noise[1]*(xdot**2)))
            pmf[supported_s] = tmp_pmf
            pmf /= np.sum(pmf)
        else:
            pmf[s_next_i] = 1.0
        return pmf







