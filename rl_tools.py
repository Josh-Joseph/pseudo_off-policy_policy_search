import itertools
import numpy as np
import pandas
import time
from scipy.optimize import leastsq, fmin
import scipy.spatial
import scipy.sparse
import policy_representations
import parallel
import mfmc

reload(policy_representations)

def evaluate_methods(method, domains, all_trials, all_n):
    columns = pandas.MultiIndex.from_tuples([(n, trial) for n in all_n for trial in all_trials])
    results = pandas.DataFrame(index=[domain.input_pars for domain in domains], columns=columns)
    for domain in domains:
        for trial in all_trials:
            data = domain.generate_batch_data(N=np.max(all_n))
            for n in all_n:
                #print "[rl_tools]: On " + str(n) + " of " + str(all_n) + " episodes of data..."
                policy = learn_policy(domain, method, data.ix[:n-1]) # the DataFrame indexing with labels is inclusive
                results[n, trial][domain.input_pars] = domain.evaluate_policy(policy)
        print "---------------------------- method = " + method
        print results
    return [method, results]

def find_nearest_index(array, value):
    return np.nanargmin(np.sum(np.abs(array-value), axis=1))

def find_nearest_index_fast(list_of_arrays, value):
    indx = 0
    for i in range(len(list_of_arrays)):
        indx *= len(list_of_arrays[i][:-1])
        indx += np.argmin(np.abs(list_of_arrays[i][:-1]-value[i]))
    return indx

def state_space_meshgrid(list_of_binned_states):
    # note -- this function appends an absorbing state to the end of the set of states
    n_dims = len(list_of_binned_states)
    state_centers = np.array(list(itertools.product(*list_of_binned_states)))
    nanarray = np.empty((1,n_dims))
    nanarray.fill(np.nan)
    state_centers = np.append(state_centers, nanarray, axis=0)
    return state_centers

gamma = .99
#print "[rl_tools]: A gamma of " + str(gamma) + " is used for value iteration."
def value_iteration(T, state_centers, reward_fn, threshold=1e-5, pi_init=None):
    if scipy.sparse.issparse(T[0]):
        n_actions = len(T)
        n_states = T[0].shape[0]
    else:
        n_states, n_actions = T.shape
    R = np.empty(n_states)
    for s in range(n_states):
        R[s] = reward_fn(state_centers[s])
    V_old = R.copy()
    if pi_init is None:
        pi = np.zeros(n_states, dtype=np.int)
    else:
        pi = pi_init.copy()
    V = 2*V_old
    Q = np.empty((n_states, n_actions))
    while np.nanmax(np.abs((V-V_old)/(V_old+1e-50)))>=threshold:
        #print np.nanmax(np.abs((V-V_old)/(V_old+1e-50)))
        V_old = V.copy()
        if scipy.sparse.issparse(T[0]):
            for a in range(n_actions):
                Q[:,a] = R + gamma*(T[a] * V_old)
        else:
            for a in range(n_actions):
                Q[:,a] = R + gamma*V_old[T[:,a]]
        pi = Q.argmax(axis=1)
        V = np.array([Q[s,pi[s]] for s in range(n_states)])
    return pi, V

def split_states_on_dim(state_centers):
    dim_centers = []
    for i in range(state_centers.shape[1]):
        dim_centers.append(np.sort(np.array(list(set(state_centers[:,i])))))
    return dim_centers

def fit_T(data, state_centers, action_centers, goal_check):
    kdtree = scipy.spatial.KDTree(state_centers[:-1])
    n_states, n_dims = state_centers.shape
    n_actions, = action_centers.shape
    T = []
    for a in range(n_actions):
        T.append(scipy.sparse.lil_matrix((n_states, n_states)))
    all_s = []
    all_s_next = []
    for n in range(len(data)):
        nan_inds = np.where(np.isnan(data[n].values[:,0]))[0]
        first_nan = len(data[n].values[:,0]) if len(nan_inds) == 0 else nan_inds[0]
        trash, this_s = kdtree.query(data[n].values[:first_nan,:-2])
        all_s.append(this_s[:-1])
        all_s_next.append(this_s[1:])
    for n in range(len(data)):
        for t in range(len(data[n])-1):
            if np.isnan(data[n]['x'][t+1]):
                break
            u = data[n]['u'][t]
            a = np.argmin(np.abs(u - action_centers))
            T[a][all_s[n][t], all_s_next[n][t]] += 1
    for a in range(n_actions):
        for s in range(n_states-1):
            if goal_check(state_centers[s]):
                # transitions to absorbing state
                T[a][s,n_states-1] = 1e10
            else:
                # small probability mass on self transition
                T[a][s,s] += 1e-5
        T[a][n_states-1,n_states-1] = 1e10
        T[a] = scipy.sparse.spdiags(1./T[a].sum(axis=1).T, 0, *T[a].shape) * T[a]
    return T

def deterministic_continuous_to_discrete_model(dynamics, state_centers, action_centers, goal_check):
    dim_centers = split_states_on_dim(state_centers)
    n_states, n_dims = state_centers.shape
    n_actions, = action_centers.shape
    T = np.empty((n_states, n_actions), dtype=np.int)
    for a_i in range(n_actions):
        u = action_centers[a_i]
        for s_i in range(n_states-1):
            state = state_centers[s_i].copy()
            if goal_check(state):
                T[s_i][a_i] = n_states-1 # absorbing state
            else:
                s_next = find_nearest_index_fast(dim_centers, dynamics(state, u))
                T[s_i][a_i] = s_next
        T[n_states-1][a_i] = n_states-1 # absorbing state
    return T

def stochastic_continuous_to_discrete_model(domain):
    n_states, n_dims = domain.state_centers.shape
    n_actions, = domain.action_centers.shape
    T = [scipy.sparse.lil_matrix((n_states, n_states)) for a in domain.action_centers]
    for a_i in range(n_actions):
        for s_i in range(n_states-1):
            if domain.at_goal(domain.state_centers[s_i]):
                T[a_i][s_i,n_states-1] = 1.0 # absorbing state
            else:
                tmp_pmf = domain.true_dynamics_pmf(s_i, a_i)
                inds = np.where(tmp_pmf > 0.0)[0]
                T[a_i][s_i,inds] = tmp_pmf[inds]
        T[a_i][n_states-1,n_states-1] = 1.0 # absorbing state
    return T

def discrete_policy(domain, states_to_actions=None):
    if states_to_actions is None: # if None, create a random policy
        states_to_actions = np.random.randint(len(domain.action_centers), size=domain.state_centers.shape[0])
    return policy_representations.Discrete_policy(domain.state_centers, domain.action_centers, states_to_actions)

def policy_wrt_approx_model(domain, pars):
    dynamics = lambda s, u: domain.approx_dynamics(s, u, pars)
    T = deterministic_continuous_to_discrete_model(dynamics, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold)
    return discrete_policy(domain, states_to_actions)

def policy_wrt_true_model(domain):
    T = stochastic_continuous_to_discrete_model(domain)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold)
    return discrete_policy(domain, states_to_actions)

def err_array(domain, pars, data):
    err = []
    for i in range(len(data)): # for i in set([i for i,j in data.index]):
        episode_values = data[i].values.copy() # episode_values = data.ix[i].values.copy()
        for j in range(domain.episode_length):
            if not np.isnan(episode_values[j,0]) and j != domain.episode_length-1 and not np.isnan(episode_values[j+1,0]):
                err.append(np.sum(np.array([domain.approx_dynamics(episode_values[j,:domain.n_dim], episode_values[j,domain.n_dim], pars)-episode_values[j+1,:domain.n_dim]])**2))
    return err

def hill_climb(fn, optimization_pars, ml_start=None):
    # fn must return [input, fn(input)] with input as a tuple
    step = optimization_pars['initial step size']
    eval_record = {}
    new_inputs = []
    new_inputs.append(optimization_pars['start'].copy())
    if ml_start is not None:
        new_inputs.append(ml_start)
    all_out = parallel.parmap(fn, [np.around(input, decimals=10) for input in new_inputs])
    for out in all_out:
        eval_record[out[0]] = out[1]
    center = eval_record.keys()[np.argmax(eval_record.values())]
    #print eval_record
    while len(eval_record.keys()) < optimization_pars['maximum evaluations'] and np.max(step) > 1e-6:
        #print "-----------"
        #print "[rl_tools.hill_climb]: step = " + str(step)
        #print center
        new_inputs = []
        for i in range(len(step)):
            if optimization_pars['only positive']:
                new_inputs.append([max(center[i] - step[i],.000001), max(center[i].copy(),.000001), max(center[i] + step[i],.000001)])
            else:
                new_inputs.append([center[i] - step[i], center[i].copy(), center[i] + step[i]])
        new_inputs = list(itertools.product(*new_inputs))
        all_out = parallel.parmap(fn, [np.around(input, decimals=10) for input in new_inputs if input not in eval_record.keys()])
        for out in all_out:
            eval_record[out[0]] = out[1]
        new_center = eval_record.keys()[np.argmax(eval_record.values())]
        if np.all(new_center == center):
            step /= 2.0
        else:
            center = new_center
    return np.array(eval_record.keys()[np.argmax(eval_record.values())])


def best_policy(domain):
    f = lambda pars: [tuple(pars), domain.evaluate_policy(policy_wrt_approx_model(domain, pars))]
    pars = hill_climb(f, domain.optimization_pars)
    #print pars
    #if 1:
    #    raw_returns = parallel.largeparmap(f, domain.initial_par_search_space)
    #else:
    #    raw_returns = map(f, domain.initial_par_search_space)
    #ind = np.argmax([raw[1] for raw in raw_returns])
    #pars = raw_returns[ind][0]
    #if 0:
    #    f = lambda pars: -domain.evaluate_policy(policy_wrt_approx_model(domain, pars))
    #    pars = fmin(f, pars)
    return policy_wrt_approx_model(domain, pars)

class Domain:
    def __init__(self, input_pars=np.nan):
        self.input_pars = input_pars
        self.N_MC_eval_samples = np.nan
        self.episode_length = np.nan
        self.data_columns = []
        self.n_dim = np.nan
        self.bounds = np.nan
        self.goal = np.nan
        self.initstate = np.nan
        self.action_centers = np.nan
        self.n_x_centers = np.nan
        self.n_theta_centers = np.nan
        self.n_xdot_centers = np.nan
        self.n_thetadot_centers = np.nan
        self.true_pars = np.nan
        self.noise = np.nan
        self.value_iteration_threshold = np.nan
        self.distance_fn = np.nan
        self.optimization_pars = {'initial step size':np.array([]), 'start':np.array([]), 'maximum evaluations':np.nan, 'only positive':False}
        self.state_centers = self.construct_discrete_policy_centers()
        self.dim_centers = split_states_on_dim(self.state_centers)
        self.pi_init = None
        self.training_data_random_start = True

    def construct_discrete_policy_centers(self):
        if self.n_dim == 2:
            x_centers = np.linspace(self.bounds[0,0], self.bounds[1,0], self.n_x_centers)
            xdot_centers = np.linspace(self.bounds[0,1], self.bounds[1,1], self.n_xdot_centers)
            state_centers = state_space_meshgrid((x_centers, xdot_centers))
        elif self.n_dim == 4:
            x_centers = np.array([0]) # np.linspace(self.bounds[0,0], self.bounds[1,0], self.n_x_centers)
            theta_centers = np.linspace(self.bounds[0,1], self.bounds[1,1], self.n_theta_centers)
            xdot_centers = np.linspace(self.bounds[0,2], self.bounds[1,2], self.n_xdot_centers)
            thetadot_centers = np.linspace(self.bounds[0,3], self.bounds[1,3], self.n_thetadot_centers)
            state_centers = state_space_meshgrid((x_centers, theta_centers, xdot_centers, thetadot_centers))
        else:
            raise Exception('construct_discrete_policy_centers is only implemented for cartpole and mountaincar')
        return state_centers

    def evaluate_policy(self, policy):
        data = self.generate_batch_data(N=self.N_MC_eval_samples, policy=policy)
        episode_return = 0
        for n in range(len(data)):
            episode_return += data[n]['r'].sum()
        episode_return *= 1.0/self.N_MC_eval_samples
        return episode_return

    def generate_batch_data(self, N, policy=None):
        #print "[rl_tools]: Generating " + str(N) + " episodes of " + ("batch training data..." if policy is None else "evaluation data...")
        #index = pandas.MultiIndex.from_tuples([(n, t) for n in range(N) for t in range(self.episode_length) ], names=['episode','time'])
        #data = pandas.DataFrame(index=index, columns=self.data_columns)
        f = lambda trash: self.simulate_episode(policy)
        if 1:
            raw_data = parallel.largeparmap(f, range(N))
        else:
            raw_data = map(f, range(N))
        #for n in range(N):
        #    data.ix[n] = raw_data[n]
        data = raw_data
        return data

    def simulate_episode(self, policy=None):
        np.random.seed(int(1e6*time.time()))
        episode_data = pandas.DataFrame(index=range(self.episode_length), columns=self.data_columns)
        s = self.initstate.copy()
        if policy is None: # create batch training data
            policy = discrete_policy(self)
            if self.training_data_random_start:
                s = np.random.random(self.n_dim) * np.diff(self.bounds, axis=0)[0] + self.bounds[0,:]
        for t in range(self.episode_length):
            u = policy.get_action(s)
            for col_i in range(self.n_dim):
                episode_data[self.data_columns[col_i]][t] = s[col_i]
            episode_data['u'][t] = u
            episode_data['r'][t] = self.reward(s)
            if self.at_goal(s):
                break
            s = self.true_dynamics(s, u)
        return episode_data

    def optimal_policy(self):
        T = stochastic_continuous_to_discrete_model(self)
        states_to_actions, V = value_iteration(T, self.state_centers, self.reward, threshold=self.value_iteration_threshold)
        return discrete_policy(self, states_to_actions)

    def baseline_policy(self):
        return discrete_policy(self)

    def reward(self, s):
        raise Exception('reward(s) must be implemented')

    def approx_dynamics(self, s, u, pars=np.nan):
        raise Exception('dynamics(s, u, pars) must be implemented')

    def true_dynamics(self, s, u):
        raise Exception('true_dynamics(s, u) must be implemented')

    def true_dynamics_pmf(self, s_i, a_i):
        raise Exception('true_dynamics_pmf(s_i, a_i) must be implemented')

    def at_goal(self, s):
        raise Exception('at_goal(s) must be implemeneted')

    #def distance_fn(self, x1, u1, x2, u2):
    #    raise Exception('distance_fn(x1, u1, x2, u2) must be implemeneted')

    def distance_fn(self, x1, x2):
        # for this distance fn we're assuming u1==u2
        raise Exception('distance_fn(x1, u1, x2, u2) must be implemeneted')