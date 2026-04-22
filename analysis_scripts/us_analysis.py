"""
This script contains functions used to analyse the US simulations

Note that the executable 'wham' must be added to the user's PATH,
see http://membrane.urmc.rochester.edu/?page_id=126 for installation 
instructions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import red
import shutil
import warnings
import yaml
import subprocess
from tqdm import tqdm

"""Defining constants"""

temperature = 298.15 # K
boltzmann = 0.0019872041 # kcal/mol K
beta = 1.0/(boltzmann*temperature)
k_boresch = 100 # kcal/mol rad**2
k_rmsd = 500 # kcal/mol nm**-2
standard_volume = 1660 # angstroms^3
standard_volume_nm = standard_volume*0.001 # nm^3
# radius of sphere whose volume is equal to the standard volume in nm
radius_sphere = (3*standard_volume_nm/(4*np.pi))**(1.0/3.0) 


def obtain_dirpath(system, free_energy_step, dof, equilibration, run_number):
    """
    Helper function to obtain the results directory for a given set of results

    system : str
        The name of the complex of interest
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)      
    run_number : int, default=1
        Specify the replicate

    Returns
    -------
    dirpath : str
        Directory for desired set of results
    """
    
    dirpath = f"{os.getcwd()}/{system}/US/{free_energy_step}/results/"

    if free_energy_step in ('Boresch', 'RMSD'):
        dirpath+=f"{dof}"

    dirpath+=f"/run{run_number}/"

    if equilibration == None:
        pass
    
    elif equilibration == 'RED':
        dirpath+=f"RED"

    elif isinstance(equilibration, int):
        dirpath+=f"{equilibration}ns_equil"

    return dirpath

def obtain_sampling_time(system, free_energy_step, dof, run_number):
    """
    Obtain the total sampling time for the full set of CV samples.
    This assumes that all files for a given run have the same sampling
    interval of 125 timesteps."""

    dirpath = obtain_dirpath(system, free_energy_step, dof, equilibration=None, run_number=run_number)
    filenames = [file for file in os.listdir(dirpath) if file.endswith('.txt')]
    
    # Filter any text files that don't contain CV samples
    CV_samples = []

    for filename in filenames: 
        parts = filename.split('.txt')

        try:
            CV =  float(parts[0])
            CV_samples.append(filename)

        except:
            pass

    sample_file = dirpath + '/' + filenames[0]

    # Assume we have 0.5 ps sampling interval
    n_samples = len(np.loadtxt(sample_file))

    return 0.5E-3*n_samples

def apply_RED(system, free_energy_step, dof=None, run_number=1, plot=False):
    """
    Apply the RED package (https://github.com/fjclark/red) to a 
    set of CVs (specified by free_energy_step + dof + run_number)
    """
    
    try:
        # Assign the sampling time
        total_time = obtain_sampling_time(system, free_energy_step, dof, run_number)
    except Exception as e:
        raise ValueError('Select one of the following free energy steps: "separation", "RMSD", "Boresch"!') from e

    # Directory path  
    inputpath = obtain_dirpath(system, free_energy_step, dof, None, run_number)

    savedir = "RED"

    os.makedirs(f"{inputpath}/{savedir}", exist_ok=True)

    # Array of all possible values over RMSD, sep and Boresch US stages
    values = np.append(np.arange(-10, 0, 0.05), np.arange(0.0, 40.0, 0.05))

    # Identify CV values 
    CV_values = []
    for value in values:
        value = np.round(value, 3)
        inputname = f"{inputpath}/{value}.txt"
        if os.path.exists(inputname):
            CV_values.append(value)

    for CV_value in tqdm(CV_values, desc='Applying RED', total=len(CV_values)):
        CV_value = np.round(CV_value, 3)
        inputname = f"{inputpath}/{CV_value}.txt"
        outputname = f"{inputpath}/{savedir}/{CV_value}.txt"

        try:
            if not os.path.exists(outputname):
                shutil.copy(inputname, outputname)

                # Truncate the copied file
                data = np.loadtxt(outputname)
                try:
                    # Subsample to provide RED with manageable amount of data
                    idx_start, g, ess = red.detect_equilibration_window(data[:20000:2, 1], method="min_sse", plot=plot)
                    idx_start*=2 # Multiply by 2 to account for subsampling
                except Exception as e:
                    print(f'Equilibration not detected in the first 10 ns, applying default 7 ns truncation for {CV_value}')
                    idx_start = int((3.5 / total_time) * len(data))

                np.savetxt(outputname, data[idx_start:]) 

        except Exception as e:
            print(f"Error processing {CV_value}: {e}")

def plot_timeseries(CV_value, system, free_energy_step, dof=None, equilibration=0, run_number=1):
    """
    Plot the timeseries for a specific set of CV values
    """

    # Assign the sampling time
    try:
        total_time = obtain_sampling_time(system, free_energy_step, dof, run_number)
    except:
        raise ValueError('Select one of the following free energy steps: "separation", "RMSD", "Boresch"!')

    # Reading in files
    dirpath = obtain_dirpath(system, free_energy_step, dof, None, run_number)

    full_data = np.loadtxt(f"{dirpath}/{CV_value}.txt")[:,1]
    
    time = np.linspace(0, total_time, len(full_data))

    plt.figure()
    plt.plot(time, full_data, label='Full sampling')

    if equilibration == 'RED':
        RED_data = np.loadtxt(f"{dirpath}/RED/{CV_value}.txt")[:,1]
        RED_idx = len(full_data) - len(RED_data)
        plt.vlines(time[RED_idx], ymin=0.5*np.min(full_data), ymax=1.1*np.max(full_data), colors='r', linestyle='dashed', label='RED truncation')
    
    elif isinstance(equilibration, int) and equilibration>0:
        truncation_idx = int((equilibration/total_time)*len(full_data))
        plt.vlines(time[truncation_idx], ymin=0.5*np.min(full_data), ymax=1.1*np.max(full_data), colors='k', linestyle='dotted', label=f'{equilibration} ns truncation')
    
    plt.legend(loc='lower right')
    plt.xlabel('Time (ns)')
    plt.ylabel(f"{free_energy_step}")
    # plt.xlim(0,5)
    plt.show()

    return time, full_data

def generate_metafile(system, free_energy_step, dof=None, equilibration=None, run_number=1, k_CV=None, ignore_values=[], plot=True):
    """
    Generate the metadata file required for WHAM implementation for a specific umbrella sampling run
    """

    dirpath = obtain_dirpath(system, free_energy_step, dof, equilibration, run_number)
    
    if plot == True:
        print(dirpath)
        plt.figure()
        plt.title(f"{system} {free_energy_step} {dof} US, run {run_number} sample distribution")

    metafilelines = []

    # Perform equilibration detection if required
    if equilibration=='RED':
        apply_RED(system, free_energy_step, dof, run_number, plot=False)

    # All possible CV vals
    values = np.append(np.arange(-10, 0, 0.05), np.arange(0.0, 40.0, 0.05))

    # Identify CV values 
    CV_values = []
    for value in values:
        value = np.round(value, 3)
        inputname = f"{dirpath}/{value}.txt"
        if os.path.exists(inputname) and value not in ignore_values:
            CV_values.append(value)

    for CV_value in CV_values: 
        CV_value = np.round(CV_value,4)
        filename = f"{CV_value}.txt"
        data = np.loadtxt(f'{dirpath}/{filename}')

        if free_energy_step == 'RMSD':
            metafileline = f'{dirpath}/{filename} {np.round(CV_value*0.1, 3)} {k_CV}\n'
        else:
            metafileline = f'{dirpath}/{filename} {CV_value} {k_CV}\n'

        if CV_value not in ignore_values:
            metafilelines.append(metafileline)                

        if plot==True:
            if free_energy_step != 'Boresch':
                plt.hist(10*data[:,1], bins=30, alpha=0.6, label=f"{CV_value}")
            else:
                plt.hist(data[:,1], bins=30, alpha=0.6, label=f"{CV_value}")

    if plot == True:
        plt.legend(fontsize='xx-small', ncol=6)
        plt.ylabel('Counts')
        plt.xlabel('CV value')

    with open(f"{dirpath}/metafile.txt", "w") as f:    
        f.writelines(metafilelines)         

def perform_WHAM(wham_params, system, free_energy_step, dof=None, equilibration=0, run_number=1):
    """
    Perform WHAM to generate the PMF for a given US run

    Parameters
    ----------
    wham_params : list
        hist_min hist_max num_bins tol temperature numpad [num_MC_trials randSeed]
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate

    Returns
    -------
    None
    """
    dirpath = obtain_dirpath(system, free_energy_step, dof, equilibration, run_number)

    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"Directory does not exist: {dirpath}")

    if len(wham_params) == 6:
        wham_list = ['wham'] + wham_params + [f"{dirpath}/metafile.txt", f"{dirpath}/pmf.txt", f"> {dirpath}/wham.log"]
    elif len(wham_params) == 8:
        wham_list = ['wham'] + wham_params[:-2] + [f"{dirpath}/metafile.txt", f"{dirpath}/pmf.txt"] + wham_params[:-2] + [f"> {dirpath}/wham.log"]
    else:
        raise ValueError("Specify the following values in a list: hist_min hist_max num_bins tol temperature numpad [num_MC_trials randSeed]")

    wham_list = list(map(str, wham_list))

    command = ''
    for item in wham_list:
        command+=f"{str(item)} "

    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def obtain_PMF(system, free_energy_step, dof=None, equilibration=0, run_number=1, plot=True):
    """
    Generate the PMF plot for a specific US simulation
    Parameters
    ----------
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate


    Returns
    -------
    x : array
        CV values
    y : array
        PMF 
    """

    dirpath = obtain_dirpath(system, free_energy_step, dof, equilibration, run_number)

    pmf = np.loadtxt(f'{dirpath}/pmf.txt')
    x=pmf[:,0]
    y=pmf[:,1]

    if plot == True:
        if free_energy_step != 'Boresch':
            plt.plot(10*x,y) #Units of Angstrom 
        else:
            plt.plot(x,y)
        plt.xlabel(f"{free_energy_step}")
        plt.ylabel('PMF (kcal/mol)')

    return x,y

def obtain_av_PMF(runs, system, free_energy_step, dof=None,
                  equilibration=0, plot_indiv=False, plot_av=False):
    """
    Generate average PMF and corresponding errors

    Parameters
    ----------
    runs : list[int]
        List of run indices (e.g. [1,2,3]) to combine into an average
    free_energy_step : str
        Section of the thermodynamic cycle where US simulation is performed
    dof : str, optional
        Degree of freedom for which to generate metafile (required if free_energy_step in ('Boresch','RMSD'))
    equilibration : 'RED' or int or 0, optional
        Type / amount of equilibration truncation ('RED' or integer number of ns). Default 0 (no equilibration folder)
    plot_indiv : bool, default=False
        Plot the individual PMFs for all replicas
    plot_av : bool, default=True
        Plot the average PMFs with the corresponding standard error

    Returns
    -------
    x : ndarray
        CV values (from pmf.txt first column)
    PMF_av : ndarray
        Average PMF (kcal/mol)
    PMF_err : ndarray
        Standard error in the mean PMF
    """
    # Basic validation
    if free_energy_step in ('Boresch', 'RMSD') and not dof:
        raise ValueError("dof must be provided when free_energy_step is 'Boresch' or 'RMSD'")

    resultspath = os.path.join(os.getcwd(), system, 'US', free_energy_step, 'results')
    if free_energy_step in ('Boresch', 'RMSD'):
        resultspath = os.path.join(resultspath, dof)

    if not os.path.isdir(resultspath):
        raise FileNotFoundError(f"Result path does not exist: {resultspath}")

    pmfs = []
    x = None

    if plot_indiv:

        # Set title and axes
        if free_energy_step == 'separation':
            plt.title('Separation stage individual PMFs')
            plt.xlabel('Separation (nm)')
        elif free_energy_step == 'RMSD':
            plt.title('RMSD stage individual PMFs')
            plt.xlabel('RMSD (nm)')
        else:
            plt.title('Boresch individual PMFs')
            plt.xlabel('Radians')

    for n_run in runs:
        dirpath = os.path.join(resultspath, f"run{n_run}")

        # Build equilibration/sampling subfolder if requested
        if equilibration == 'RED':
            dirpath = os.path.join(dirpath, 'RED')

        pmf_file = os.path.join(dirpath, 'pmf.txt')
        if not os.path.isfile(pmf_file):
            warnings.warn(f"pmf.txt not found for run {n_run} at {pmf_file}; skipping this run.")
            continue

        try:
            pmf = np.loadtxt(pmf_file)
        except Exception as exc:
            warnings.warn(f"Failed to load {pmf_file} for run {n_run}: {exc}; skipping this run.")
            continue

        if pmf.ndim != 2 or pmf.shape[1] < 2:
            warnings.warn(f"Unexpected pmf.txt shape for run {n_run} ({pmf.shape}); skipping.")
            continue

        pmfs.append(pmf[:,1])
        x = pmf[:,0]

        if plot_indiv:
            plt.plot(x, pmf[:,1], label=f"Run {n_run}")

    # Savepath for PMF figures
    if free_energy_step == 'separation':
        figpath = f"{system}/US/{free_energy_step}/results"
    else:
        figpath = f"{system}/US/{free_energy_step}/results/{dof}"

    # After loop: show individual plot if requested
    if plot_indiv:
        plt.ylabel('PMF (kcal/mol)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figpath}/pmfs.png")
        plt.show()

    if len(pmfs) == 0:
        raise RuntimeError("No PMF data was loaded for any run. Check paths and run identifiers.")

    pmfs_arr = np.vstack(pmfs)            # shape (n_loaded_runs, n_points)
    n_loaded = pmfs_arr.shape[0]

    av = np.mean(pmfs_arr, axis=0)
    err = np.std(pmfs_arr, axis=0, ddof=1) / np.sqrt(n_loaded)

    if plot_av:

        # Set title and axes
        if free_energy_step == 'separation':
            plt.title('Separation stage average PMF')
            plt.xlabel('Separation (nm)')
        elif free_energy_step == 'RMSD':
            plt.title('RMSD stage average PMF')
            plt.xlabel('RMSD (nm)')
        else:
            plt.title('Boresch average PMF')
            plt.xlabel('Radians')

        plt.plot(x, av, color='k')
        plt.fill_between(x, av - err, av + err, color='grey', alpha=0.4)
        plt.ylabel('PMF (kcal/mol)')
        plt.tight_layout()
        plt.show()

    return x, av, err

def RMSDContribution(x, pmf, k_rmsd, unbound=False):
    """
    Calculate the contribution of applying/removing a particular RMSD restraint
    Make sure units of k_rmsd are consistent
    """
    pmf = np.array([x,pmf])

    # mask = np.isfinite(pmf[:,1]) # Boolean mask to remove inf values 
    # pmf = pmf[:,0][mask], pmf[:,1][mask]

    width = pmf[0][1] - pmf[0][0]

    restraintCenter = 0

    # integration
    numerator = 0
    denominator = 0
    for x, y in zip(pmf[0], pmf[1]):
        numerator += math.exp(-beta * y)
        denominator += math.exp((-beta) * (y + 0.5 * k_rmsd * ((x - restraintCenter)**2)))
    
    contribution = math.log(numerator / denominator) / beta
    
    if unbound:
        return contribution
    else:
        return -contribution

def BoreschContribution(x, pmf, theta_0, k_restraint):
    """
    Calculate the free energy contribution from applying a Boresch 
    restraint in the bound state. 
    theta_0 is the equilibrium value"""

    pmf = np.array([x,pmf])
    
    width = pmf[0][1] - pmf[0][0]

    restraint_center = theta_0

    numerator = 0
    denominator = 0
    for x, y in zip(pmf[0], pmf[1]):
        numerator += width*math.exp(-beta*y)
        denominator += width*math.exp((-beta)*(y+0.5*k_restraint*((x-restraint_center)**2)))
    
    contribution = math.log(numerator/denominator)/beta

    return -contribution # bound state contribution always negative

def standard_state_correction(r_star, theta_a_min, theta_b_min, k_boresch):
    """
    Calculate the free energy contribution for release of the separated proteins to the standard state 
    """
    
    corr = (r_star**2)*math.sin(theta_a_min)*math.sin(theta_b_min)*(2*np.pi/beta)**2.5/(8*(np.pi**2)*(4*np.pi*radius_sphere**2)*(k_boresch)**2.5)

    return -1/(beta)*math.log(corr)

def SepContribution(x, pmf, r_star):
    """
    Integrate the separation PMF to obtain the free energy contribution of separating the restrained proteins
    """

    pmf = np.array([x,pmf])
    
    w_r_star = pmf[1][0]
    for x, y in zip(pmf[0], pmf[1]):
        if x >= r_star:
            w_r_star = y
            break
        
    width = pmf[0][1] - pmf[0][0]
    I = 0
    for x, y in zip(pmf[0], pmf[1]):
        I += width*math.exp(-beta*(y-w_r_star))
        if x >= r_star:
            break

    return -1/(beta)*math.log(3*I/radius_sphere)

def analyse_RMSDContribution(x, pmf, k_rmsd, unbound=False):
    """
    Analyse the free energy contribution as a function of the RMSD CV.
    This function returns the normalised numerator and denominator functions,
    which are integrated to calculate the total free energy contribution.
    """

    restraintCenter = 0

    # Convert x to np array if necessary
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Convert pmf to np array if necessary
    if not isinstance(pmf, np.ndarray):
        pmf = np.array(pmf)

    numerator = np.exp(-beta * pmf)
    denominator = np.exp((-beta) * (pmf + 0.5 * k_rmsd * (np.square(x - restraintCenter))))
    
    # Normalise for plotting purposes
    return numerator/np.max(numerator), denominator/np.max(denominator)

def analyse_BoreschContribution(x, pmf, theta_0, k_restraint):
    """
    Analyse the free energy contribution as a function of the Boresch DOF.
    This function returns the normalised numerator and denominator functions,
    which are integrated to calculate the total free energy contribution.
    """
    pmf = np.array([x,pmf])
    
    width = pmf[0][1] - pmf[0][0]

    restraint_center = theta_0

    x = pmf[0]
    y = pmf[1]

    numerator = width*np.exp(-beta*y)
    denominator = width*np.exp((-beta)*(y+0.5*k_restraint*(np.square(x-restraint_center))))

    # Normalise for plotting purposes
    return numerator/np.max(numerator), denominator/np.max(denominator)

def analyse_SepContribution(x, pmf, r_star):
    """
    Integrate the separation PMF to obtain the free energy contribution of separating the restrained proteins
    """

    pmf = np.array([x,pmf])
    
    w_r_star = pmf[1][0]
    for x, y in zip(pmf[0], pmf[1]):
        if x >= r_star:
            w_r_star = y
            break
        
    width = pmf[0][1] - pmf[0][0]

    integrand = width*np.exp(-beta*(pmf[1]-w_r_star))

    return integrand/np.max(integrand)

def visualise_integrands(system, stage, dof, run_number):
    """
    Plot the numerator and denominator integrands for a specific PMF
    """
    k_boresch = 100 # kcal/mol rad**2
    k_sep = 1000 # kcal/mol nm**-2
    k_rmsd = 500 # kcal/mol nm**-2

    if stage == 'Boresch':
        equilibration = None # CV samples already truncated for Boresch US
        bins = 50 # WHAM number of bins
        k=k_boresch
    else:
        equilibration = 'RED' # Use RED for separation and RMSD
        bins = 100
        k=k_sep

    # Find upper and lower limit for WHAM
    resultspath = obtain_dirpath(system, stage, dof, equilibration, run_number=run_number)
    CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(resultspath) if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
    upper_limit = np.max(CV_vals)
    lower_limit = np.min(CV_vals)

    # Convert to units of nm for RMSD US
    if stage == 'RMSD':
        upper_limit = np.round(upper_limit*0.1, 5) + 0.01
        lower_limit = np.round(lower_limit*0.1, 5)
        k=k_rmsd # Overwrite to RMSD force constant
    
    elif stage == 'separation':
        r_star = CV_vals[-2]

    wham_params = [lower_limit, upper_limit, bins, 1e-6, 300, 0]

    # Obtain the PMF
    generate_metafile(system, stage, dof, equilibration, run_number, k_CV=k, plot=False)
    perform_WHAM(wham_params, system, stage, dof, equilibration, run_number)
    x, pmf = obtain_PMF(system, stage, dof, equilibration, run_number, plot=False)
    
    # Filter out infinite values
    mask = np.isfinite(pmf)
    x = x[mask]
    pmf = pmf[mask]
    pmf_norm = pmf / np.nanmax(pmf[np.isfinite(pmf)]) # Normalise PMF

    # Calculate normalised denominator and numerator (equal to 1 for separation integral I*)
    if stage == 'Boresch':

        # Find the Boresch eq values
        with open(f"{system}/US/US_config.yaml", "r") as file:
            data = yaml.safe_load(file)

        boresch_eq = {
            'thetaA' : data['Boresch equilibrium values']['theta_A_0'],
            'thetaB' : data['Boresch equilibrium values']['theta_B_0'],
            'phiA' : data['Boresch equilibrium values']['phi_A_0'],
            'phiB' : data['Boresch equilibrium values']['phi_B_0'],
            'phiC' : data['Boresch equilibrium values']['phi_C_0']
        }

        num, den = analyse_BoreschContribution(x, pmf, boresch_eq[dof], k_boresch)

    elif stage == 'RMSD':
        num, den = analyse_RMSDContribution(x, pmf, k_rmsd)

    elif stage == 'separation':
        num = analyse_SepContribution(x, pmf, r_star)
        den = np.ones(len(num))
        
    plt.plot(x, pmf_norm, label='PMF (norm.)', linestyle='dotted', c='k')

    if stage == 'separation':
        plt.plot(x, num, label='$e^{-β W(r)}$ (norm.)')
    else:
        plt.plot(x, num, label='$n(\eta)$ (norm.)')
        plt.plot(x, den, label='$d(\eta)$ (norm.)')

    # Set axes
    if stage == 'separation':
        plt.xlabel('Separation (nm)')
        plt.title(f"Separation replica {run_number}")
    elif stage == 'RMSD':
        plt.xlabel('RMSD (nm)')
        plt.title(f"RMSD {dof} US replica {run_number}")
    else:
        plt.title(f"{dof} simulation {run_number}")
        plt.xlabel('Radians')

    plt.legend(fontsize='small')
    plt.show()

def test_array_decay(array, tolerance=0.01):
    """Test that the first and last values in a normalised numpy array have decayed to 0.001, i.e. 0.1%"""

    decayed = True

    if array[0]>tolerance:
        decayed = False
    if array[-1]>tolerance:
        decayed = False

    return decayed

def calc_total_DeltaG(system, tolerance=0.01, plot_pmfs=False, runs=[1,2,3]):
    """
    Calculate the total binding free energy for a given system

    Returns the total delta G, error, df of individual components
    """

    # Specify the various bound state RMSD DOFs
    bound_rmsd_systems = {
        'CRBN_CK1a' : ['CRBN_only', 'CK1awithCRBN'],
        'CRBN_len_CK1a' : ['CRBN_len_only', 'CK1awithCRBN_len'],
        'CRBN_len_CK1aI35G' : ['CRBN_len_only', 'CK1aI35GwithCRBN_len'],
        'CRBN_len_CK1aI37E' : ['CRBN_len_only', 'CK1aI37EwithCRBN_len'],
        'CRBN_len_CK1aN39G' : ['CRBN_len_only', 'CK1aN39GwithCRBN_len'],
        'CRBN_len_CK1aG40N' : ['CRBN_len_only', 'CK1aG40NwithCRBN_len'],
        'BD1_DCAF16' : ['DCAF16_only', 'BD1withDCAF16'],
        'BD2_DCAF16' : ['DCAF16_only', 'BD2withDCAF16'],
        'BRD3_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2withDCAF16andBD1'],
        'BRD4_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2withDCAF16andBD1'],
        'BRD4_IBG1_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG1withDCAF16andBD1'],
        'BRD4_IBG3_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG3withDCAF16andBD1'],
        'BRD3_IBG1_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG1withDCAF16andBD1'],
        'BRD4G386E_IBG1_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG1withDCAF16andBD1'],
        'BRD3E344G_IBG1_DCAF16' : ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG1withDCAF16andBD1']
    }

    # Specify the various bulk state RMSD DOFs
    bulk_rmsd_systems = {
        'CRBN_CK1a' : ['CRBN', 'CK1a'],
        'CRBN_len_CK1a' : ['CRBN_len', 'CK1a'],
        'CRBN_len_CK1aI35G' : ['CRBN_len', 'CK1aI35G'],
        'CRBN_len_CK1aI37E' : ['CRBN_len', 'CK1aI37E'],
        'CRBN_len_CK1aN39G' : ['CRBN_len', 'CK1aN39G'],
        'CRBN_len_CK1aG40N' : ['CRBN_len', 'CK1aG40N'],
        'BD1_DCAF16' : ['BD1', 'DCAF16'],
        'BD2_DCAF16' : ['BD2', 'DCAF16'],
        'BRD3_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2withBD1_bulk'],
        'BRD4_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2withBD1_bulk'],
        'BRD4_IBG1_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2_IBG1withBD1_bulk'],
        'BRD4_IBG3_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2_IBG3withBD1_bulk'],
        'BRD3_IBG1_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2_IBG1withBD1_bulk'],
        'BRD4G386E_IBG1_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2_IBG1withBD1_bulk'],
        'BRD3E344G_IBG1_DCAF16' : ['DCAF16', 'BD1_only_bulk', 'BD2_IBG1withBD1_bulk']
    }

    stages = [] # Separation, RMSD, Boresch (bound), Standard State Correction
    all_deltaGs = []
    errs = []

    # Separation contribution: 
    print('Calculating separation contribution...')
    resultspath = obtain_dirpath(system, free_energy_step='separation', dof=None, equilibration=None, run_number=1)
    CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(resultspath) if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
    upper_limit = np.max(CV_vals)
    lower_limit = np.min(CV_vals)
    wham_params = [lower_limit, upper_limit, 200, 1e-6, 300, 0]
    r_bulk = upper_limit-0.1 # Choose point in bulk state

    deltaGs = [] # List to store individual replicas

    for run_number in runs:

        generate_metafile(system, 'separation', dof=None, equilibration='RED', run_number=run_number, k_CV=1000, ignore_values=[], plot=False)
        perform_WHAM(wham_params, system, 'separation', dof=None, equilibration='RED', run_number=run_number)
        x, pmf = obtain_PMF(system, 'separation', dof=None, equilibration='RED', run_number=run_number, plot=False)
        deltaGs.append(SepContribution(x, pmf, r_bulk))

        # Test for sufficient sampling
        denominator = analyse_SepContribution(x, pmf, r_bulk)
        if test_array_decay(denominator, tolerance) == False:
            print(f"\nWARNING : Separation PMF for run {run_number} has insufficient sampling...\n")

    # Average PMF
    avs = obtain_av_PMF(runs, system, 'separation', None, 'RED', plot_indiv=plot_pmfs)

    deltaGs = np.array(deltaGs)
    stages.append('Separation')

    all_deltaGs.append(np.average(deltaGs))
    errs.append(np.std(deltaGs, ddof=1)/np.sqrt(len(runs)))

    # RMSD contribution
    dofs_bound = bound_rmsd_systems[system]
    dofs_bulk = bulk_rmsd_systems[system]

    # Bound state
    for dof in dofs_bound:
        print(f"Calculating {dof} RMSD contribution")

        resultspath = obtain_dirpath(system, free_energy_step='RMSD', dof=dof, equilibration=None, run_number=1)
        CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(resultspath) if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
        upper_limit = np.round(np.max(CV_vals) * 0.1 + 0.02, 4)
        lower_limit = np.round(np.min(CV_vals) * 0.1 + 0.02, 4)
        wham_params = [lower_limit, upper_limit, 100, 1e-6, 300, 0]

        deltaGs = []

        for run_number in runs:

            generate_metafile(system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number, k_CV=500, ignore_values=[], plot=False)
            perform_WHAM(wham_params, system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number)
            x, pmf = obtain_PMF(system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number, plot=False)
            deltaGs.append(RMSDContribution(x, pmf, 500, unbound=False))

            # Test for sufficient sampling
            num, denom = analyse_RMSDContribution(x, pmf, k_rmsd=500, unbound=False)
            if test_array_decay(num, tolerance) == False or test_array_decay(denom, tolerance) == False:
                print(f"\nWARNING : {dof} RMSD PMF for run {run_number} has insufficient sampling...")

        # Average PMF
        avs = obtain_av_PMF(runs, system, 'RMSD', dof, 'RED', plot_indiv=plot_pmfs)

        deltaGs = np.array(deltaGs)
        stages.append(f'RMSD {dof}')
        all_deltaGs.append(np.average(deltaGs))
        errs.append(np.std(deltaGs, ddof=1)/np.sqrt(len(runs)))

    # Bulk state
    for dof in dofs_bulk:
        print(f"Calculating {dof} RMSD contribution")

        resultspath = obtain_dirpath(system, free_energy_step='RMSD', dof=dof, equilibration=None, run_number=1)
        CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(resultspath) if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
        upper_limit = np.round(np.max(CV_vals) * 0.1 + 0.02, 4)
        lower_limit = np.round(np.min(CV_vals) * 0.1 + 0.02, 4)
        wham_params = [lower_limit, upper_limit, 100, 1e-6, 300, 0]

        deltaGs = []

        for run_number in runs:

            generate_metafile(system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number, k_CV=500, ignore_values=[], plot=False)
            perform_WHAM(wham_params, system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number)
            x, pmf = obtain_PMF(system, 'RMSD', dof=dof, equilibration='RED', run_number=run_number, plot=False)
            deltaGs.append(RMSDContribution(x, pmf, 500, unbound=True))

            # Test for sufficient sampling
            num, denom = analyse_RMSDContribution(x, pmf, k_rmsd=500, unbound=True)
            if test_array_decay(num, tolerance) == False or test_array_decay(denom, tolerance) == False:
                print(f"\nWARNING : {dof} RMSD PMF for run {run_number} has insufficient sampling...")

        # Average PMF
        avs = obtain_av_PMF(runs, system, 'RMSD', dof, 'RED', plot_indiv=plot_pmfs)

        deltaGs = np.array(deltaGs)
        stages.append(f'RMSD {dof}')
        all_deltaGs.append(np.average(deltaGs))
        errs.append(np.std(deltaGs, ddof=1)/np.sqrt(len(runs)))

    # Extract Boresch equilibrium values
    with open(f'{system}/US/US_config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    boresch_eq = {
        'thetaA' : data['Boresch equilibrium values']['theta_A_0'],
        'thetaB' : data['Boresch equilibrium values']['theta_B_0'],
        'phiA' : data['Boresch equilibrium values']['phi_A_0'],
        'phiB' : data['Boresch equilibrium values']['phi_B_0'],
        'phiC' : data['Boresch equilibrium values']['phi_C_0']
    }

    # Boresch bound state
    for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:

        print(f'Calculating {dof} contribution...')
        resultspath = obtain_dirpath(system, free_energy_step='Boresch', dof=dof, equilibration=None, run_number=1)
        CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(resultspath) if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
        upper_limit = np.max(CV_vals)
        lower_limit = np.min(CV_vals)
        wham_params = [lower_limit, upper_limit, 50, 1e-6, 300, 0]

        deltaGs = []

        for run_number in runs:

            generate_metafile(system, 'Boresch', dof=dof, equilibration=None, run_number=run_number, k_CV=500, ignore_values=[], plot=False)
            perform_WHAM(wham_params, system, 'Boresch', dof=dof, equilibration=None, run_number=run_number)
            x, pmf = obtain_PMF(system, 'Boresch', dof=dof, equilibration=None, run_number=run_number, plot=False)
            deltaGs.append(BoreschContribution(x, pmf, boresch_eq[dof], 100))

            # Test for sufficient sampling
            num, denom = analyse_BoreschContribution(x, pmf, boresch_eq[dof], 100)
            if test_array_decay(num, tolerance) == False or test_array_decay(denom, tolerance) == False:
                print(f"\nWARNING : {dof} Boresch PMF for run {run_number} has insufficient sampling...")

        # Average PMF
        avs = obtain_av_PMF(runs, system, 'Boresch', dof, None, plot_indiv=plot_pmfs)

        deltaGs = np.array(deltaGs)
        stages.append(dof)
        all_deltaGs.append(np.average(deltaGs))
        errs.append(np.std(deltaGs, ddof=1)/np.sqrt(len(runs)))
    
    # Standard state correction
    stages.append('Standard state corr')
    all_deltaGs.append(standard_state_correction(r_bulk, boresch_eq['thetaA'], boresch_eq['thetaB'], 100))
    errs.append(0.0)

    all_deltaGs = np.array(all_deltaGs)
    errs = np.array(errs)

    if len(all_deltaGs) != len(errs) and len(all_deltaGs) != len(stages):
        raise ValueError('Mismatch in number of stages :(')

    # Save all contributions to df
    df = pd.DataFrame()
    df['Stage'] = stages
    df['dG (kcal/mol)'] = all_deltaGs
    df['err (kcal/mol)'] = errs

    total_deltaG = np.sum(all_deltaGs)
    total_err = np.sqrt(np.sum(np.square(errs)))

    return total_deltaG, total_err, df




        

