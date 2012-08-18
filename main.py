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

# I'm pretty sure the following works:
# 0:.005:.05 drag mu
# .01 drag sig
# .0025 mud drag constant in mountaincar


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
            all_n = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000]
        domains = [cartpole.Cartpole((wind, sig)) for sig in all_sig for wind in all_wind]
    elif problem == 'mountaincar':
        if analysis == 'misspecification':
            # drag and noise on xdot
            #all_drag_mu = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            #all_drag_mu = [0, .0001, .0005, .001, .005]
            #all_drag_mu = [0, .0001, .0005, .001, .006, .007, .008] $ muddy everywhere
            #all_drag_mu = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1] # muddy circle
            #all_drag_mu = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1] # muddy top of the right hill on x
            #all_drag_mu = [0, .001, .002, .003, .004, .005, .006, .007, .008, .009, .01] # muddy top of the right hill on xdot
            all_drag_mu = [0, .005, .01, .015, .02, .025, .03, .035, .04, .045, .05] # % slip on xdot
            #all_drag_mu = [0, .001, .005, .01,.05, .1, .15, .175, .2, .25, .3] # % slip on xdot
            all_drag_sig = [.005]
            all_n = [500]
        elif analysis == 'sample_complexity':
            # drag and noise on xdot
            all_drag_mu = [.03] #np.arange(0,1.1,.1)
            all_drag_sig = [.005]
            all_n = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
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
                    #smaller_data = [d.copy(deep=True) for d in data[:n]]
                    #policy = pops.best_policy(domain, smaller_data)
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

    if analysis == 'misspecification':
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
    elif analysis == 'sample_complexity':
        plt.figure()
        for key in store.keys():
            x = np.sort(list(set([pars[0] for pars in store[key].index])))
            if key == 'true_model':
                y = [np.mean(store[key].values) for i in x]
                plt.plot(x, y, linewidth=2, label=key)
            else:
                y = [np.mean(store[key].ix[i].values) for i in x]
                yerr = [np.std(store[key].ix[i].values)/np.sqrt(len(store[key].ix[i].values)) for i in x]
                plt.errorbar(x, y, yerr=yerr, linewidth=2, label=key)
            plt.legend()
            plt.xlabel('training data size')
            plt.ylabel('expected total reward')
            if problem == 'mountaincar':
                plt.ylim((-500,0))
            else:
                plt.ylim((0,500))

    store.close()