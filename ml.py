from rl_tools import deterministic_continuous_to_discrete_model,value_iteration, discrete_policy, err_array, fit_T
from scipy.optimize import leastsq


def approx_model_policy(domain, data, trim=False):
    f = lambda pars: err_array(domain, pars, data, trim)
    pars0 = domain.true_pars
    pars = leastsq(f, pars0)[0]
    #print pars
    dynamics = lambda s, u: domain.approx_dynamics(s, u, pars)
    T = deterministic_continuous_to_discrete_model(dynamics, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold, pi_init=domain.pi_init)
    return discrete_policy(domain, states_to_actions)

def discrete_model_policy(domain, data):
    T = fit_T(data, domain.state_centers, domain.action_centers, domain.at_goal)
    states_to_actions, V = value_iteration(T, domain.state_centers, domain.reward, threshold=domain.value_iteration_threshold)
    return discrete_policy(domain, states_to_actions)