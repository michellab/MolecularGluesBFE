"""This file contains analysis scripts for the unrestrained MD 
simulations"""

import pickle as pkl
import yaml
import numpy as np
import matplotlib.pyplot as plt

def obtain_Boresch_dof(system, run_number, dof):
    """
    Read in the Boresch distribution sampled from the unrestrained
    MD simulation.
    """

    boreschfile = f'{system}/Unrestrained_MD/results/run{run_number}/{dof}.pkl'

    with open(boreschfile, 'rb') as f:
        loaded_data = pkl.load(f)

    frames = loaded_data['Frames']
    time = loaded_data['Time (ns)']
    vals = loaded_data['DOF values']

    return time, vals

def plot_Boresch_distribution(system, replicas=[1,2,3]):
    """
    Plot the histogram of Boresch distributions for a given set of
    triplicate unrestrained MD simulations
    """
    labels = {'thetaA':'$\Theta _A$', 'thetaB':'$\Theta _B$', 'phiA':'$\phi _A$', 'phiB':'$\phi _B$', 'phiC':'$\phi _C$'}

    # Extract equilibration time (identified using RED)
    with open(f'{system}/US/US_config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    equil_time = 0

    for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:

        vals_all = []

        for run_number in replicas:

            time, vals = obtain_Boresch_dof(system, run_number, dof)

            start_idx = np.searchsorted(time, equil_time, side='left')

            time_slice = time[start_idx:]
            vals_slice = vals[start_idx:]

            # collect values across runs
            vals_all.append(vals_slice)

        if len(vals_all) == 0:
            raise RuntimeError("No data found...")

        vals_all = np.concatenate(vals_all)
        
        positive = np.mean(vals_all) > 0

        if positive:
            vals_all[vals_all < 0] += 2*np.pi
        else:
            vals_all[vals_all > 0] -= 2*np.pi

        vals_all = np.degrees(vals_all)

        # histogram
        counts, bins = np.histogram(vals_all, bins=50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        peak_center = bin_centers[counts.argmax()]

        plt.hist(vals_all, bins=60, label=labels[dof], alpha=0.7)

    plt.title(system)
    plt.ylabel('Counts')
    plt.xlabel('Degrees')
    plt.legend()
    plt.show()


def calc_R_squared(x,y):

    coeffs = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(coeffs)
    y_fit = fit_fn(x)

    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

def calc_R_sq_uncertainty(x, x_err, y, y_err, n_mc=10000, random_state=None):
    """
    Calculate the R squared uncertainty based on n_mc Monte Carlo trials.
    Return the R squared standard deviation, and the 95 % confidence intervals
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)
    x_err = np.asarray(x_err)
    y_err = np.asarray(y_err)

    # Regenerate random distributions
    x_s = rng.normal(x, x_err, size=(n_mc, x.size))
    y_s = rng.normal(y, y_err, size=(n_mc, x.size))

    r2 = np.array([calc_R_squared(x_s[i], y_s[i]) for i in range(n_mc)])
    ci_low, ci_high = np.percentile(r2, [2.5, 97.5])

    return np.std(r2, ddof=1), ci_low, ci_high