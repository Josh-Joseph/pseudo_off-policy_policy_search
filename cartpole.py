import numpy as np
import scipy.stats
import rl_tools

reload(rl_tools)

XBOUND = 90
THETABOUND = np.pi/15
XDOTBOUND = 2
THETADOTBOUND = 2
INITSTATE = np.array([0, 0, 0, 0])

class Cartpole(rl_tools.Domain):
    def __init__(self, input_pars):
        self.input_pars = input_pars
        self.N_MC_eval_samples = 250
        self.episode_length = 500
        self.data_columns = ('x','theta','xdot','thetadot','u','r') # assume the 2nd to last is u and the last is r
        self.n_dim = 4
        self.bounds = np.array([[-XBOUND, XBOUND],[-THETABOUND, THETABOUND],[-XDOTBOUND, XDOTBOUND],[-THETADOTBOUND, THETADOTBOUND]]).transpose()
        self.goal = np.array([[-np.inf, np.inf], [-THETABOUND, THETABOUND], [-np.inf, np.inf], [-np.inf, np.inf]]).transpose()
        self.initstate = INITSTATE
        self.action_centers = np.array([-5, 5])
        self.n_x_centers = 1
        self.n_theta_centers = 50
        self.n_xdot_centers = 30
        self.n_thetadot_centers = 30
        self.true_pars = (1, 1, 1)
        self.optimization_pars = {'initial step size':np.array([.95, .95, .95]),
                                  'start':np.array([1, 1, 1]),
                                  'maximum evaluations':25}
        #self.initial_par_search_space = [[p1, p2] for p1 in np.linspace(-0.003, -.002, 5) for p2 in np.linspace(2, 4, 5)] # TODO
        self.noise = input_pars
        self.value_iteration_threshold = 1e-5
        self.state_centers = self.construct_discrete_policy_centers()
        self.dim_centers = rl_tools.split_states_on_dim(self.state_centers)
        self.pi_init = 1-np.int8((np.sign(self.state_centers[:,1]) + 1)/2)

    def distance_fn(self, x1, u1, x2, u2):
        return 1e5*(u1 != u2) + np.sum(((x1-x2)/np.array([1e5, 0.4189, 10, 10]))**2, axis=1)

    def at_goal(self, s):
        return np.abs(s[1]) == THETABOUND

    def reward(self, s):
        #return 1 if np.abs(s[1]) < THETABOUND else 0
        #return 0 if np.abs(s[1]) < THETABOUND else -1
        return 1-np.abs(s[1]/THETABOUND) if np.abs(s[1]) < THETABOUND else 0

    def approx_dynamics(self, s, u, pars):
        mc = pars[0]
        mp = pars[1]
        l = pars[2]
        g = 9.8
        dt = .02

        x = s[0]
        theta = s[1]
        xdot = s[2]
        thetadot = s[3]

        s = -np.sin(theta)
        s2 = s**2
        c = -np.cos(theta)

        den = mc + mp * s2
        xdotdot = (u + (mp * s * (l * (thetadot**2) + g * c))) / den
        thetadotdot = (-u * c - mp * l * (thetadot**2) * c * s - (mp + mc) * g * s) / (l * den)

        xdot = min(max(xdot + xdotdot * dt, self.bounds[0,2]), self.bounds[1,2])
        thetadot = min(max(thetadot + thetadotdot * dt, self.bounds[0,3]), self.bounds[1,3])

        x = min(max(x + xdot * dt, self.bounds[0,0]), self.bounds[1,0])
        theta = min(max(theta + thetadot * dt, self.bounds[0,1]), self.bounds[1,1])
        return np.array([x, theta, xdot, thetadot])

    def true_dynamics(self, s, u):
        mc = self.true_pars[0]
        mp = self.true_pars[1]
        l = self.true_pars[2]
        g = 9.8
        dt = .02

        x = s[0]
        theta = s[1]
        xdot = s[2]
        thetadot = s[3]

        s = -np.sin(theta)
        s2 = s**2
        c = -np.cos(theta)

        den = mc + mp * s2
        xdotdot = (u + (mp * s * (l * (thetadot**2) + g * c))) / den
        thetadotdot = (-u * c - mp * l * (thetadot**2) * c * s - (mp + mc) * g * s) / (l * den)

        xdot = min(max(xdot + xdotdot * dt, self.bounds[0,2]), self.bounds[1,2])
        thetadot = min(max(thetadot + thetadotdot * dt, self.bounds[0,3]), self.bounds[1,3])

        x = min(max(x + xdot * dt, self.bounds[0,0]), self.bounds[1,0])
        theta = min(max(theta + thetadot * dt + (-s*np.sign(theta)*self.input_pars[0]) + np.random.normal(loc=0, scale=self.input_pars[1]), self.bounds[0,1]), self.bounds[1,1])
        return np.array([x, theta, xdot, thetadot])

    def true_dynamics_pmf(self, s_i, a_i):
        # returns a vector of probability masses the same size as state_centers
        s = self.state_centers[s_i,:].copy()
        u = self.action_centers[a_i]
        pmf = np.zeros(self.state_centers.shape[0])

        mc = self.true_pars[0]
        mp = self.true_pars[1]
        l = self.true_pars[2]
        g = 9.8
        dt = .02

        x = s[0]
        theta = s[1]
        xdot = s[2]
        thetadot = s[3]

        s = -np.sin(theta)
        s2 = s**2
        c = -np.cos(theta)

        den = mc + mp * s2
        xdotdot = (u + (mp * s * (l * (thetadot**2) + g * c))) / den
        thetadotdot = (-u * c - mp * l * (thetadot**2) * c * s - (mp + mc) * g * s) / (l * den)

        xdot = min(max(xdot + xdotdot * dt, self.bounds[0,2]), self.bounds[1,2])
        thetadot = min(max(thetadot + thetadotdot * dt, self.bounds[0,3]), self.bounds[1,3])

        x = min(max(x + xdot * dt, self.bounds[0,0]), self.bounds[1,0])
        theta = min(max(theta + thetadot * dt + (-s*np.sign(theta)*self.input_pars[0]), self.bounds[0,1]), self.bounds[1,1])

        s = np.array([x, theta, xdot, thetadot])

        s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
        if self.noise[1]:
            supported_s = np.all(np.equal(self.state_centers[:,2:], self.state_centers[s_next_i,2:]),axis=1)
            tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,1], loc=s[1], scale=self.noise[1])
            pmf[supported_s] = tmp_pmf
            pmf /= np.sum(pmf)
        else:
            pmf[s_next_i] = 1.0

        return pmf







