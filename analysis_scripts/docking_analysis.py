"""This file contains scripts for analysing the trajectory-averaged docking scores
for the various ternary andb binary complexes using haddock3 [1]

[1] Giulini, M.; Reys, V.; Teixeira, J. M. C.; Jiménez-García, B.; V. Honorato, R.; Kravchenko,
A.; Xu, X.; Versini, R.; Engel, A.; Verhoeven, S.; Bonvin, A. M. J. J. Journal of Chemical
Information and Modeling 2025, 65, 7315-7324.

"""

import pandas as pd
import numpy as np

def obtain_scores(system, run_number):

    return pd.read_csv(f'{system}/docking/run{run_number}/HADDOCK_scores.csv')['HADDOCK score'].to_numpy()

def obtain_av_score(system, replicas=[1,2,3]):
    """
    Return the average and standard error
    Assume triplicate run
    """

    av_scores = []

    for n_run in replicas:
        
        av_scores.append(np.average(obtain_scores(system, n_run)))

    av_scores = np.array(av_scores)

    return np.average(av_scores), np.std(av_scores, ddof=1)/np.sqrt(len(replicas))