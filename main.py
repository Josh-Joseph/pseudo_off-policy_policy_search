import matplotlib.pyplot as plt
import parallel
import rl_tools
import mountaincar
import cartpole

reload(rl_tools)
reload(mountaincar)
reload(cartpole)

#all_mu_wind = (0, -0.0001, -0.0002, -0.0003, -0.0004, -0.0005, -0.0006, -0.0007, -0.0008,  -0.0009, -0.001)
#all_sig_wind = (0, .00025, .0005, .00075, .001, .002, .003, .004, .005, .01)

#all_mu_wind = (0, -0.0001, -0.0002, -0.0003, -0.0004, -0.0005, -0.0006, -0.0007)
#all_sig_wind = (.0001, .005, .01)

#all_drag_mu = (0, .1, .2, .3, .4, .5, .6, .7, .8)
#all_drag_sig = (0, .25, .5, .75, 1, 1.5, 2)

all_drag_mu = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)
all_drag_sig = (0, .2, 2)

methods = ('Best_approx' ,'True_model','MB-ME')

all_trials = range(10)

all_n = [200]

def go(plow_it_with_all_those_cores=True):
    domains = [mountaincar.Mountaincar((drag_mu, drag_sig)) for drag_sig in all_drag_sig for drag_mu in all_drag_mu]
    f = lambda method: rl_tools.evaluate_methods(method=method, domains=domains, all_trials=all_trials, all_n=all_n)
    if plow_it_with_all_those_cores:
        raw_results = parallel.parmap(f, methods)
    else:
        raw_results = map(f, methods)
    results = {}
    for raw in raw_results:
        results[raw[0]] = raw[1]
    return results

def plot_results(results):
    for drag_sig in all_drag_sig:
        plt.figure()
        for key in results.keys():
            x = all_drag_mu
            y = [results[key][all_n[0]].mean(axis=1)[(drag_mu, drag_sig)] for drag_mu in all_drag_mu]
            plt.plot(x, y, linewidth=2, label=key)
        plt.legend()
        plt.title("drag_sig = " + str(drag_sig))
        plt.ylim((-500,0))
        plt.xlabel('drag_mu')
        plt.ylabel('expected total reward')
