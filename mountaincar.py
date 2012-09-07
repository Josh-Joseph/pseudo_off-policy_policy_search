import numpy as np
import scipy.stats
import rl_tools

reload(rl_tools)

XMIN = -1.2
XMAX = 0.5
XDOTMIN = -0.07
XDOTMAX = 0.07
INITSTATE = np.array([-np.pi / 2.0 / 3.0, 0.0])

rnd_start = True
print "[mountaincar]: Training random start is " + ("On" if rnd_start else "Off")

class Mountaincar(rl_tools.Domain):
    def __init__(self, input_pars):
        self.input_pars = input_pars
        self.N_MC_eval_samples = 250
        self.episode_length = 500
        self.data_columns = ('x','xdot','u','r') # assume the 2nd to last is u and the last is r
        self.n_dim = 2
        self.bounds = np.array([[XMIN, XMAX],[XDOTMIN, XDOTMAX]]).transpose()
        self.goal = np.array([[-np.inf, XMAX],[-np.inf, np.inf]]).transpose()
        self.initstate = INITSTATE
        self.action_centers = np.array([-1, 1])
        self.n_x_centers = 300
        self.n_xdot_centers = 300
        self.true_pars = (-0.0025, 3)
        self.initial_par_search_space = [[p1, p2] for p1 in np.linspace(-0.003, -.002, 5) for p2 in np.linspace(2, 4, 5)]
        self.noise = input_pars
        self.value_iteration_threshold = 1e-5
        self.optimization_pars = {'initial step size':np.array([.0024, 1]),
                                  'start':np.array([-0.0025, 3]),
                                  'maximum evaluations':50,
                                  'only positive':False}
        self.state_centers = self.construct_discrete_policy_centers()
        self.dim_centers = rl_tools.split_states_on_dim(self.state_centers)
        self.pi_init = None
        self.training_data_random_start = rnd_start

    def distance_fn(self, x1, x2):
        return np.sum(((x1-x2)/np.array([1.7, .14]))**2, axis=1)

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

        if 1:
            if self.noise[1]:
                s[0] = min(max(x+xdot + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,0]), self.bounds[1,0])
            else:
                s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            slip = 0
            #if x < .25 <= s[0]:
            if (np.sign(x-.25) != np.sign(s[0]-.25)) or (np.sign(x-.75) != np.sign(s[0]-.75)): # rocks at -.25, .25, .75
                if xdot > 0:
                    slip = max(-self.noise[0], -xdot)
                else:
                    slip = min(self.noise[0], -xdot)
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])
        elif 0:
            slip = 0
            if .25 < x < .5:
                if xdot > 0:
                    slip = max(-self.noise[0]*xdot, -xdot)
            if self.noise[1]:
                s[0] = min(max(x+xdot + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,0]), self.bounds[1,0])
            else:
                s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])
        elif 0:
            #noise on x
            if self.noise[1]:
                s[0] = min(max(x+xdot + self.noise[0] + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,0]), self.bounds[1,0])
            else:
                s[0] = min(max(x+xdot + self.noise[0], self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)), self.bounds[0,1]), self.bounds[1,1])
        elif 0:
            #noise on xdot
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            if self.noise[1]:
                s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + self.noise[0] + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,1]), self.bounds[1,1])
            else:
                s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + self.noise[0], self.bounds[0,1]), self.bounds[1,1])
        elif 0:
            #noise and slip on x
            slip = 0
            if -.5 < x < -.75:
                if xdot > 0:
                    slip = -self.noise[0]
                else:
                    slip = self.noise[0]
            if self.noise[1]:
                s[0] = min(max(x+xdot + slip + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,0]), self.bounds[1,0])
            else:
                s[0] = min(max(x+xdot + slip, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)), self.bounds[0,1]), self.bounds[1,1])
        elif 0:
            #noise and constant slip on xdot
            slip = 0 if x < -0.5235987755982988 else -np.sign(xdot)*self.noise[0]
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            if self.noise[1]:
                s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip + np.random.normal(loc=0, scale=self.noise[1]), self.bounds[0,1]), self.bounds[1,1])
            else:
                s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])

        return s

    def true_dynamics_pmf(self, s_i, a_i):
        # returns a vector of probability masses the same size as state_centers
        s = self.state_centers[s_i,:].copy()
        u = self.action_centers[a_i]
        x = s[0]
        xdot = s[1]
        pmf = np.zeros(self.state_centers.shape[0])

        if 1:
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            slip = 0
            #if x < .25 <= s[0]:
            if (np.sign(x-.25) != np.sign(s[0]-.25)) or (np.sign(x-.75) != np.sign(s[0]-.75)): # rocks at -.25, .25, .75
                if xdot > 0:
                    slip = max(-self.noise[0], -xdot)
                else:
                    slip = min(self.noise[0], -xdot)
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,1] == self.state_centers[s_next_i,1]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,0], loc=s[0], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0
        elif 0:
            slip = 0 if x < .25 else -np.sign(xdot)*self.noise[0]
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,1] == self.state_centers[s_next_i,1]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,0], loc=s[0], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0
        elif 0:
            #noise on x
            s[0] = min(max(x+xdot + self.noise[0], self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)), self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,1] == self.state_centers[s_next_i,1]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,0], loc=s[0], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0
        elif 0:
            #noise on xdot
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + self.noise[0], self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,0] == self.state_centers[s_next_i,0]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,1], loc=s[1], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0
        elif 0:
            #noise and slip on x
            slip = 0 if x < -0.5235987755982988 else self.noise[0]*(self.true_pars[0]*np.cos(self.true_pars[1]*x))
            s[0] = min(max(x+xdot + slip, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)), self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,1] == self.state_centers[s_next_i,1]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,0], loc=s[0], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0
        elif 0:
            #noise and constant slip on xdot
            slip = 0 if x < -0.5235987755982988 else -np.sign(xdot)*self.noise[0]
            s[0] = min(max(x+xdot, self.bounds[0,0]), self.bounds[1,0])
            s[1] = min(max(xdot+0.001*u+(self.true_pars[0]*np.cos(self.true_pars[1]*x)) + slip, self.bounds[0,1]), self.bounds[1,1])
            s_next_i = rl_tools.find_nearest_index_fast(self.dim_centers, s)
            if self.noise[1]:
                supported_s = self.state_centers[:,0] == self.state_centers[s_next_i,0]
                tmp_pmf = scipy.stats.norm.pdf(self.state_centers[supported_s,1], loc=s[1], scale=self.noise[1])
                pmf[supported_s] = tmp_pmf
                pmf /= np.sum(pmf)
            else:
                pmf[s_next_i] = 1.0

        return pmf

#print "[mountaincar]: only noise on x !!!!!!!!!!!!!!"
#print "[mountaincar]: only noise on xdot !!!!!!!!!!!!!!"
#print "[mountaincar]: noise and slip on x !!!!!!!!!!!!!!"
#print "[mountaincar]: noise and constant slip on xdot !!!!!!!!!!!!!!"
#print "[mountaincar]: noise on x, slip on xdot !!!!!!!!!!!!!!"

