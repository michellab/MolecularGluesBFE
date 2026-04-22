import numpy as np
import os

def adjust_samples(CV_array):
    """Subtract 2*pi from all elements in the provided array"""
    for n in range(len(CV_array)):
        if CV_array[n, 1] < 0:
            CV_array[n, 1] =  CV_array[n, 1] + 2*np.pi

    return CV_array

for n_run in [1,2,3]:

    CV_vals = [float(filename.split('.txt')[0]) for filename in os.listdir(f"run{n_run}") if filename.endswith('.txt') and filename not in ('pmf.txt', 'metafile.txt')]
            
    for CV in CV_vals:

        CV = np.round(CV, 5)
        CV_array = np.loadtxt(f"run{n_run}/{CV}.txt")
        CV_array_modified = adjust_samples(CV_array)

        np.savetxt(f"run{n_run}/{CV}.txt", CV_array_modified)
