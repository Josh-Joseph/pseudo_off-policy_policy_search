import numpy as np
import pandas
import os
from datetime import datetime
import matplotlib.pyplot as plt
import parallel
import rl_tools
import mountaincar
import cartpole
import pops
import ml
from copy import deepcopy
import csv

reload(ml)
reload(rl_tools)
reload(mountaincar)
reload(cartpole)

# I'm pretty sure the following works:
# 0:.005:.05 drag mu
# .01 drag sig
# .0025 mud drag constant in mountaincar


def evaluate_approach(method, problem, analysis, save_it=False, trial_start=0):

    all_trials = range(5)
    #if method == 'true_model':
        #all_trials = [0]

    if problem == 'cartpole':
        if analysis == 'misspecification':
            all_wind = np.arange(0,2.1,.1)
            #all_wind = [.2]
            all_sig = [.01]
            all_n = [10000]
        elif analysis == 'sample_complexity':
            all_wind = [.2]
            all_sig = [.01]
            #all_n = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000]
            all_n = [50, 100, 250, 500, 1000, 1500, 2500, 5000, 7500, 10000, 15000, 20000]
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
            all_drag_sig = [0]
            all_n = [2000]
        elif analysis == 'sample_complexity':
            # drag and noise on xdot
            all_drag_mu = [.02] #np.arange(0,1.1,.1)
            all_drag_sig = [.005]
            all_n = [50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
        domains = [mountaincar.Mountaincar((drag_mu, drag_sig)) for drag_sig in all_drag_sig for drag_mu in all_drag_mu]

    print "[main.evaluate_approach]: Evaluating the performance of " + method + " ..."
    for trial in all_trials:
        if trial < trial_start:
            continue
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
                result = domain.evaluate_policy(policy)
                if problem + "_" + analysis + '_store.h5' in os.listdir('.'):
                    results = load_results(method, problem, analysis, domains, all_n, all_trials)
                results[domain.input_pars][n, trial] = result
                print str(domain.input_pars) + " - " + str(n) + " - " + str(results[domain.input_pars][n, trial])
                if save_it:
                    save_results(method, problem, analysis, results)
    return results

def save_results(method, problem, analysis, results):
    store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')
    key = method # + ", " + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    store[key] = results
    store.close()

def load_results(method, problem, analysis, domains, all_n, all_trials):
    store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')
    key = method # + ", " + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if key in store.keys():
        results = store[key]
    else:
        index = pandas.MultiIndex.from_tuples([(n, trial) for n in all_n for trial in all_trials])
        results = pandas.DataFrame(index=index, columns=[domain.input_pars for domain in domains])
    store.close()
    return results

def write_data_to_csv():
    all_analysis = {'misspecification','sample_complexity'}
    all_problems = {'mountaincar','cartpole'}
    for analysis in all_analysis:
        for problem in all_problems:
            if problem == 'mountaincar':
                pretty_labels = {'true_model': 'True Model', 'pops' : 'POPS Standard Mountain Car', 'max_likelihood_approx' : 'ML Standard Mountain Car', 'max_likelihood_big_discrete' : 'ML Tabular'}
            else:
                pretty_labels = {'true_model': 'True Model', 'pops' : 'POPS Standard Cart-pole', 'max_likelihood_approx' : 'ML Standard Cart-pole', 'max_likelihood_big_discrete' : 'ML Tabular'}
            store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')
            if analysis == 'misspecification':
                for par in list(set([pars[1] for pars in store['true_model']])):
                    data = {'n' : [str(pars[0]) for pars in store['true_model'] if pars[1] == par]}
                    for key in store.keys():
                        x = [pars[0] for pars in store['true_model'] if pars[1] == par]
                        data[pretty_labels[key] + '_mu'] = [str(np.around(store[key][(xx, par)].mean(), decimals=2)) for xx in x]
                        data[pretty_labels[key] + '_stderr'] = [str(np.around(store[key][(xx, par)].std()/np.sqrt(len(store[key][(xx, par)])), decimals=2)) for xx in x]
                    keys = data.keys()
                    keys.sort()
                    print keys
                    keys[1:] = keys[:-1]
                    keys[0] = 'n'
                    print keys
                    file = open(problem + "_" + analysis + '.csv', 'w')
                    wr = csv.writer(file, delimiter=',')
                    row = deepcopy(keys)
                    wr.writerow(row)
                    file.close()
                    for i in range(len(data[keys[0]])):
                        row = []
                        for key in keys:
                            row.append(data[key][i])
                        file = open(problem + "_" + analysis + '.csv', 'a')
                        wr = csv.writer(file, delimiter=',')
                        print row
                        wr.writerow(row)
                        file.close()
            elif analysis == 'sample_complexity':
                data = {'n' : [str(n) for n in np.sort(list(set([pars[0] for pars in store['true_model'].index])))]}
                for key in store.keys():
                    x = np.sort(list(set([pars[0] for pars in store[key].index])))
                    data[pretty_labels[key] + '_mu'] = [str(np.around(store[key].ix[i].mean().values[0], decimals=2)) for i in x]
                    data[pretty_labels[key] + '_stderr'] = [str(np.around(store[key].ix[i].std().values[0]/np.sqrt(len(store[key].ix[i].values)), decimals=2)) for i in x]
                    keys = data.keys()
                    keys.sort()
                    print keys
                    keys[1:] = keys[:-1]
                    keys[0] = 'n'
                    print keys
                    file = open(problem + "_" + analysis + '.csv', 'w')
                    wr = csv.writer(file, delimiter=',')
                    row = deepcopy(keys)
                    wr.writerow(row)
                    file.close()
                    for i in range(len(data[keys[0]])):
                        row = []
                        for key in keys:
                            row.append(data[key][i])
                        file = open(problem + "_" + analysis + '.csv', 'a')
                        wr = csv.writer(file, delimiter=',')
                        print row
                        wr.writerow(row)
                        file.close()


def plot_results(problem, analysis):
    if problem == 'mountaincar':
        pretty_labels = {'true_model': 'True Model', 'pops' : 'POPS Standard Mountain Car', 'max_likelihood_approx' : 'ML Standard Mountain Car', 'max_likelihood_big_discrete' : 'ML Tabular'}
    else:
        pretty_labels = {'true_model': 'True Model', 'pops' : 'POPS Standard Cart-pole', 'max_likelihood_approx' : 'ML Standard Cart-pole', 'max_likelihood_big_discrete' : 'ML Tabular'}
    colors = {'true_model': 'g', 'pops' : 'b', 'max_likelihood_approx' : 'r', 'max_likelihood_big_discrete' : 'c'}
    store = pandas.HDFStore(problem + "_" + analysis + '_store.h5')

    if analysis == 'misspecification':
        for par in list(set([pars[1] for pars in store['true_model']])):
            plt.figure()
            for key in store.keys():
                x = [pars[0] for pars in store['true_model'] if pars[1] == par]
                y = [store[key][(xx, par)].mean() for xx in x]
                #if key == 'true_model':
                #    plt.plot(x, y, linewidth=2, label=key)
                #else:
                yerr = np.array([store[key][(xx, par)].std()/np.sqrt(len(store[key][(xx, par)])) for xx in x])
                plt.errorbar(x, y, yerr=yerr, linewidth=2, color=colors[key], label=pretty_labels[key])
            #plt.title("Performance vs Model Misspecification")
            plt.ylabel('Average Total Reward')
            if problem == 'mountaincar':
                plt.xlabel('Influence of the Rock (c)')
                plt.legend()
                plt.xlim((0,.05))
                plt.ylim((-500,0))
            else:
                plt.xlabel('Influence of the Wind (f)')
                plt.legend()
                plt.xlim((0,2))
                plt.ylim((0,500))
    elif analysis == 'sample_complexity':
        plt.figure()
        for key in store.keys():
            x = np.sort(list(set([pars[0] for pars in store[key].index])))
            #if key == 'true_model':
            #    y = [np.mean(store[key].values) for i in x]
            #    plt.plot(x, y, linewidth=2, label=key)
            #else:
            y = [store[key].ix[i].mean().values[0] for i in x]
            yerr = [store[key].ix[i].std().values[0]/np.sqrt(len(store[key].ix[i].values)) for i in x]
            plt.errorbar(x, y, yerr=yerr, linewidth=2, color=colors[key], label=pretty_labels[key])
            #plt.title("Performance vs Training Data Size")
            plt.xlabel('Episodes of Training Data')
            plt.ylabel('Average Total Reward')
            if problem == 'mountaincar':
                plt.legend(loc=4)
                plt.ylim((-500,-100))
            else:
                plt.legend(loc=4)
                plt.ylim((0,500))
                #plt.xlim((0,20000))
                #print "XLIM WAS MODIFIED!!!!"

    store.close()