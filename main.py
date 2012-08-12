import numpy as np
import pandas
from datetime import datetime
import matplotlib.pyplot as plt
import parallel
import rl_tools
import mountaincar
import cartpole
import pops
import ml

reload(ml)
reload(rl_tools)
reload(mountaincar)
reload(cartpole)


def evaluate_approach(method, problem, analysis, save_it=False):

    all_trials = range(5)
    if method == 'true_model':
        all_trials = [0]

    if problem == 'cartpole':
        if analysis == 'misspecification':
            all_wind = np.arange(0,2.1,.1)
            all_sig = [.01]
            all_n = [10000]
        elif analysis == 'sample_complexity':
            all_wind = [.2]
            all_sig = [.01]
            all_n = [10, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
        domains = [cartpole.Cartpole((wind, sig)) for sig in all_sig for wind in all_wind]
    elif problem == 'mountaincar':
        if analysis == 'misspecification':
            # drag and noise on xdot
            #all_drag_mu = [0, .001, .005, .01, .05, .1, .5]
            all_drag_mu = [.01, .05, .1, .5]
            all_drag_sig = [.001]
            all_n = [1000]
        elif analysis == 'sample_complexity':
            # drag and noise on xdot
            all_drag_mu = [.1] #np.arange(0,1.1,.1)
            all_drag_sig = [1]
            all_n = [10, 50, 100, 150, 200, 250, 500, 1000, 1500, 2000]
        domains = [mountaincar.Mountaincar((drag_mu, drag_sig)) for drag_sig in all_drag_sig for drag_mu in all_drag_mu]

    print "[main.evaluate_approach]: Evaluating the performance of " + method + " ..."
    index = pandas.MultiIndex.from_tuples([(n, trial) for n in all_n for trial in all_trials])
    results = pandas.DataFrame(index=index, columns=[domain.input_pars for domain in domains])
    for trial in all_trials:
        for domain in domains:
            if method != 'true_model':
                data = domain.generate_batch_data(N=np.max(all_n))
            for n in all_n:
                if method == 'true_model':
                    policy = domain.optimal_policy()
                elif method == 'best_model':
                    policy = rl_tools.best_policy(domain)
                elif method == 'random':
                    policy = domain.baseline_policy()
                elif method == 'pops':
                    policy = pops.best_policy(domain, data[:n])
                elif method == 'max_likelihood_approx':
                    policy = ml.approx_model_policy(domain, data[:n])
                elif method == 'max_likelihood_big_discrete':
                    policy = ml.discrete_model_policy(domain, data[:n])
                else:
                    raise Exception('Unknown policy learning method: ' + method)
                results[domain.input_pars][n, trial] = domain.evaluate_policy(policy)
                print str(domain.input_pars) + " - " + str(n) + " - " + str(results[domain.input_pars][n, trial])
    if save_it:
        save_results(method, problem, analysis, results)
    return results

def save_results(method, problem, analysis, results):
    store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')
    key = method # + ", " + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    store[key] = results
    store.close()

def plot_results(problem, analysis):
    store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')
    for par in list(set([pars[1] for pars in store['true_model']])):
        plt.figure()
        for key in store.keys():
            x = [pars[0] for pars in store['true_model'] if pars[1] == par]
            y = [store[key][(xx, par)].mean() for xx in x]
            if key == 'true_model':
                plt.plot(x, y, linewidth=2, label=key)
            else:
                yerr = np.array([2*store[key][(xx, par)].std()/np.sqrt(len(store[key][(xx, par)])) for xx in x])
                plt.errorbar(x, y, yerr=yerr, linewidth=2, label=key)
        plt.legend()
        plt.title("par[1] = " + str(par) + ", 95% confidence interval of the mean")
        plt.xlabel('par[0]')
        plt.ylabel('expected total reward')
        if problem == 'mountaincar':
            plt.ylim((-500,0))
        else:
            plt.xlim((0,1))
            plt.ylim((0,500))

    store.close()