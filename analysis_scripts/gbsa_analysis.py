"""This file contains scripts for analysing the binding free energies
calculated from the MM-GBSA calculations in Ambertools [1]

[1] Miller III, B. R., McGee Jr., T. D., Swails, J. M. Homeyer, N. Gohlke, H. and Roitberg, A. E.
   J. Chem. Theory Comput., 2012, 8 (9) pp 3314--3321
"""

import re
import numpy as np

def obtain_mmgbsa_deltaG(system, run_number):
    """Return the delta G obtained from MM-PBSA calculation"""

    resultsfile = f'{system}/mm_gbsa/run{run_number}/results'

    with open(resultsfile, 'r') as file:
        text_data = file.read()

    match = re.search(r'DELTA TOTAL\s+([\d\.\-]+)\s+([\d\.]+)\s+([\d\.]+)', text_data)

    if match:
        average = float(match.group(1))
        std_dev = float(match.group(2))
        std_err = float(match.group(3))
        
    return average

def obtain_av_mmgbsa_deltaG(system, replicas=[1,2,3]):
    """Calculate the average delta G with the standard error"""

    replicate_avs = []

    for run in replicas:

        replicate_avs.append(obtain_mmgbsa_deltaG(system, run))

    av = np.average(np.array(replicate_avs))
    std_err = np.std(np.array(replicate_avs), ddof=1)/np.sqrt(len(replicas))

    return av, std_err