from rl_tools import policy_wrt_approx_model, deterministic_continuous_to_discrete_model,value_iteration, discrete_policy, hill_climb, err_array
from scipy.optimize import leastsq
import pandas
import numpy as np
import parallel
#import scipy.spatial

p = 10

def best_policy(domain, data):
    f = lambda pars: err_array(domain, pars, data)
    pars0 = domain.true_pars
    ml_start = leastsq(f, pars0)[0]
    if 1:
        f = lambda pars: [tuple(pars),  mfmc_evaluation(policy_wrt_approx_model(domain, pars), data, domain.distance_fn, domain.initstate, domain.episode_length, domain.at_goal)]
        pars = hill_climb(f, domain.optimization_pars, ml_start)
    else:
        f = lambda pars: [pars, mfmc_evaluation(policy_wrt_approx_model(domain, pars), data, domain.distance_fn, domain.initstate, domain.episode_length, domain.at_goal)]
        raw_returns = parallel.largeparmap(f, domain.initial_par_search_space)
        ind = np.argmax([raw[1] for raw in raw_returns])
        pars = raw_returns[ind][0]
    #print pars
    dynamics = lambda s, u: domain.approx_dynamics(s, u, pars)
    T = deterministic_continuous_to_discrete_model(dynamics, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold)
    return discrete_policy(domain, states_to_actions)

def mfmc_evaluation(policy, data, distance_fn, initstate, episode_length, goal_check):
    n_states, n_dims = policy.state_centers.shape
    #statekdtree = scipy.spatial.KDTree(policy.state_centers[:-1])
    x_array = np.zeros((0, n_dims))
    xnext_array = np.zeros((0, n_dims))
    u_array = np.zeros(0)
    r_array = np.zeros(0)
    for n in range(len(data)):
        nan_inds = np.where(np.isnan(data[n]['r']))[0]
        if len(nan_inds):
            first_nan_ind = nan_inds[0]
        else:
            first_nan_ind = len(data[n])
        x_array = np.append(x_array, data[n].values[:first_nan_ind-1,:-2], axis=0)
        xnext_array = np.append(xnext_array, data[n].values[1:first_nan_ind,:-2], axis=0)
        u_array = np.append(u_array, data[n].values[:first_nan_ind-1,-2])
        r_array = np.append(r_array, data[n].values[:first_nan_ind-1,-1])
    reconstructed_data = []
    reconstructed_returns = np.zeros(p)
    for i in range(p):
        reconstructed_data.append(pandas.DataFrame(index=range(episode_length), columns=data[0].columns))
        #print "[MFMC]: Piecing together episode " + str(i+1) + " of " + str(p)
        x = initstate.copy()
        r = data[0]['r'][0]
        for t in range(episode_length):
            #s = nearest_index(states, x)
            #trash, s = statekdtree.query(x) # this can be sped up by using rl_tools.find_nearst_index_fast
            u = policy.get_action(x)
            reconstructed_data[i].ix[t,:-2] = x.copy()
            reconstructed_data[i].ix[t,-2] = u
            reconstructed_data[i].ix[t,-1] = r
            #data_index = find_closest_segment(distance_fn, x[0], x[1], u, x_array[:,0], x_array[:,1], u_array)
            data_index = np.argmin(distance_fn(x, u, x_array, u_array))
            x_array[data_index,:] = np.inf
            x = xnext_array[data_index]
            r = r_array[data_index]
            if goal_check(x):
                break
        reconstructed_returns[i] = reconstructed_data[i]['r'].sum()
    return np.sum(reconstructed_returns)/p

def find_closest_segment(distance_fn, x_desired, xdot_desired, u_desired, x_array, xdot_array, u_array):
    return np.nanargmin(distance_fn(x_desired, xdot_desired, u_desired, x_array, xdot_array, u_array))