from rl_tools import deterministic_continuous_to_discrete_model,value_iteration, discrete_policy, err_array, fit_T
from scipy.optimize import leastsq
import numpy as np


def approx_model_policy(domain, data):
    f = lambda pars: err_array(domain, pars, data)
    pars0 = domain.true_pars
    out = leastsq(f, pars0)
    pars = out[0]
    std = out[1]
    print err_array(domain, pars, data)
    print pars
    print std
    raise Exception('hi')
    dynamics = lambda s, u: domain.approx_dynamics(s, u, pars)
    T = deterministic_continuous_to_discrete_model(dynamics, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold, pi_init=domain.pi_init)
    return discrete_policy(domain, states_to_actions)

def discrete_model_policy(domain, data):
    T = fit_T(data, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold)
    return discrete_policy(domain, states_to_actions)

def compute_x_std(domain, pars, data):
    err = []
    for i in range(len(data)): # for i in set([i for i,j in data.index]):
        episode_values = data[i].values.copy() # episode_values = data.ix[i].values.copy()
        for j in range(domain.episode_length):
            if not np.isnan(episode_values[j,0]) and j != domain.episode_length-1 and not np.isnan(episode_values[j+1,0]):
                print "------"
                print domain.approx_dynamics(episode_values[j,:domain.n_dim], episode_values[j,domain.n_dim], pars)
                print episode_values[j+1,:domain.n_dim]
                err.append((domain.approx_dynamics(episode_values[j,:domain.n_dim], episode_values[j,domain.n_dim], pars)[0]-episode_values[j+1,:domain.n_dim][0])**2)
    return err