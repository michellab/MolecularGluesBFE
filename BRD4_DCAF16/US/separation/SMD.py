"""
Perform steered MD simulation and save the starting configurations for the separation US windows.
Adapted from https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/umbrella_sampling.html
"""

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import os
import pandas as pd
import yaml

"""r0 values to save"""

windows0 = np.arange(0.90, 2.11, 0.05)
windows1 = np.arange(2.2, 4.01, 0.1)
windows = np.round(np.append(windows0, windows1), 4)

"""Read in US configuration from yaml"""

with open('../US_config.yaml', 'r') as file:
    US_data = yaml.safe_load(file)

"""System setup"""

dt = 4*unit.femtoseconds

# Load param and coord files
prmtop = app.AmberPrmtopFile('../inputs/complex_eq.prmtop')
inpcrd = app.AmberInpcrdFile('../inputs/complex_eq.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(1.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
os.makedirs('results/separation', exist_ok=True)
simulation.reporters.append(app.DCDReporter('results/separation/sep_traj.dcd', 1000))
simulation.reporters.append(app.StateDataReporter('results/separation/separation.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))

# Minimise energy 
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(1.0000*unit.kelvin)

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

"""RMSD Restraints"""

reference_positions = inpcrd.positions

receptor_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(0, 172) and atom.name in ('CA', 'C', 'N')
]

ligand_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(173, 392) and atom.name in ('CA', 'C', 'N')
]

# Add restraining forces for receptor and ligand rmsd
receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
receptor_rmsd_force.addGlobalParameter('k_rec', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
system.addForce(receptor_rmsd_force)

ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*rmsd^2')
ligand_rmsd_force.addGlobalParameter('k_lig', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
system.addForce(ligand_rmsd_force)

simulation.context.reinitialize(preserveState=True)

for atom in simulation.topology.atoms():
    if atom.index==receptor_atoms[0] and atom.residue.name!='ASN':
        raise ValueError(f'Incorrect residue selection for DCAF16 - residue N1 is missing')
    if atom.index==receptor_atoms[-1] and atom.residue.name!='LEU':
        raise ValueError(f'Incorrect residue selection for DCAF16 - residue L172 is missing')

for atom in simulation.topology.atoms():
    if atom.index==ligand_atoms[0] and atom.residue.name!='THR':
        raise ValueError(f'Incorrect residue selection for BRD4 - residue THR174 is missing')
    if atom.index==ligand_atoms[-1] and atom.residue.name!='ASP':
        raise ValueError(f'Incorrect residue selection for BRD4 - residue ASP392 is missing')

"""Radial separation CV"""

# 1-indexing
rec_interface_res = US_data['Receptor interface residues']
lig_interface_res =  US_data['Ligand interface residues']

# Account for OpenMM residue 0-indexing
rec_interface_res = -1 + np.array(rec_interface_res) 
lig_interface_res = -1 + np.array(lig_interface_res)

rec_group = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in rec_interface_res and atom.name=='CA'
]

lig_group = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in lig_interface_res and atom.name=='CA'
]

# Define radial distance as collective variable which we will vary
cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
cv.addGroup(np.array(rec_group))
cv.addGroup(np.array(lig_group))

# Specify bond groups
bondGroups = [0, 1]
cv.addBond(bondGroups)

r_0 = 1.1 * unit.nanometers #Set initial separation of 11 Angstrom

# Define biasing potential
bias_pot = mm.CustomCVForce('0.5 * k_r * (cv-r_0)^2')
bias_pot.addGlobalParameter('k_r', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
bias_pot.addGlobalParameter('r_0', r_0)

bias_pot.addCollectiveVariable('cv', cv)
system.addForce(bias_pot)

simulation.context.reinitialize(preserveState=True)

"""Boresch restraints"""

def obtain_CA_idx(res_idx):

    """Function to obtain the index of the alpha carbon for a given residue index"""
    
    atom_idx = None

    for atom in simulation.topology.atoms():
        if atom.residue.index == res_idx and atom.name=='CA':
            atom_idx = atom.index
    
    return atom_idx
    
# Boresch_residues = [13, 8, 142, 370, 332, 244]

# Define anchor points (1-indexing)
res_b = US_data['Boresch anchor points']['res_b']
res_c = US_data['Boresch anchor points']['res_c']
res_B = US_data['Boresch anchor points']['res_B']
res_C = US_data['Boresch anchor points']['res_C']

# Account for OpenMM 0-indexing 
res_b -=1 
res_c -=1
res_B -=1
res_C -=1

# Find atomic indices
idx_b = obtain_CA_idx(res_b)
idx_c = obtain_CA_idx(res_c)
idx_B = obtain_CA_idx(res_B)
idx_C = obtain_CA_idx(res_C)

print('Anchor points:')
for atom in simulation.topology.atoms():
    if atom.index in [idx_b, idx_c, idx_B, idx_C]:
        print(atom)

# Check that we have only selected CA anchor points
all_atoms = rec_group + [idx_b] + [idx_c] + lig_group + [idx_B] + [idx_C]
for atom in simulation.topology.atoms():
    if atom.index in all_atoms and atom.name != 'CA':
        raise ValueError('Select only CA atoms as anchorpoints')
    
print("\nAtomic indices:")
print(f"rec_group = {rec_group}")
print(f"idx_b = {idx_b}")
print(f"idx_c = {idx_c}")
print(f"lig_group = {lig_group}")
print(f"idx_B = {idx_B}")
print(f"idx_C = {idx_C}\n")
    
# Equilibrium values of Boresch dof
theta_A_0 = US_data['Boresch equilibrium values']['theta_A_0']
theta_B_0 = US_data['Boresch equilibrium values']['theta_B_0']
phi_A_0 = US_data['Boresch equilibrium values']['phi_A_0']
phi_B_0 = US_data['Boresch equilibrium values']['phi_B_0']
phi_C_0 = US_data['Boresch equilibrium values']['phi_C_0'] 

k_Boresch = 200 * unit.kilocalories_per_mole / unit.radians**2 #Set global force constant

theta_A_pot = mm.CustomCentroidBondForce(3, '0.5 * k_Boresch * (angle(g1,g2,g3)-theta_A_0)^2')
theta_A_pot.addGlobalParameter('theta_A_0', theta_A_0)
theta_A_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
theta_A_pot.addGroup([idx_b])
theta_A_pot.addGroup(np.array(rec_group))
theta_A_pot.addGroup(np.array(lig_group))

# Add the centroid angle bond
theta_A_pot.addBond([0, 1, 2])

system.addForce(theta_A_pot)

theta_B_pot = mm.CustomCentroidBondForce(3, '0.5 * k_Boresch * (angle(g1,g2,g3)-theta_B_0)^2')
theta_B_pot.addGlobalParameter('theta_B_0', theta_B_0)
theta_B_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
theta_B_pot.addGroup(np.array(rec_group))
theta_B_pot.addGroup(np.array(lig_group))
theta_B_pot.addGroup([idx_B])

# Add the centroid angle bond
theta_B_pot.addBond([0, 1, 2])

system.addForce(theta_B_pot)

phi_A_pot = mm.CustomCentroidBondForce(4, "0.5*k_Boresch*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(dihedral(g1,g2,g3,g4)-phi_A_0); pi = 3.1415926535")
phi_A_pot.addGlobalParameter('phi_A_0', phi_A_0)
phi_A_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_A_pot.addGroup([idx_c])
phi_A_pot.addGroup([idx_b])
phi_A_pot.addGroup(np.array(rec_group))
phi_A_pot.addGroup(np.array(lig_group))

# Add the centroid angle bond
phi_A_pot.addBond([0, 1, 2, 3])

system.addForce(phi_A_pot)

phi_B_pot = mm.CustomCentroidBondForce(4, "0.5*k_Boresch*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(dihedral(g1,g2,g3,g4)-phi_B_0); pi = 3.1415926535")
phi_B_pot.addGlobalParameter('phi_B_0', phi_B_0)
phi_B_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_B_pot.addGroup([idx_b])
phi_B_pot.addGroup(np.array(rec_group))
phi_B_pot.addGroup(np.array(lig_group))
phi_B_pot.addGroup([idx_B])

# Add the centroid angle bond
phi_B_pot.addBond([0, 1, 2, 3])

system.addForce(phi_B_pot)

phi_C_pot = mm.CustomCentroidBondForce(4, "0.5*k_Boresch*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(dihedral(g1,g2,g3,g4)-phi_C_0); pi = 3.1415926535")
phi_C_pot.addGlobalParameter('phi_C_0', phi_C_0)
phi_C_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_C_pot.addGroup(np.array(rec_group))
phi_C_pot.addGroup(np.array(lig_group))
phi_C_pot.addGroup([idx_B])
phi_C_pot.addGroup([idx_C])

# Add the centroid angle bond
phi_C_pot.addBond([0, 1, 2, 3])

system.addForce(phi_C_pot)

simulation.context.reinitialize(preserveState=True)

"""Reducing RMSD equilibrium value"""

n_red = 500 # Total number of steps to reduce r_0 over
dr = (0.5/n_red) * unit.nanometers # Bring separation down

for i in range(1, n_red+1):

    print(f'Iteration {i}')
    
    simulation.step(100) # Run short equilibration
    current_cv_value = bias_pot.getCollectiveVariableValues(simulation.context)

    r_0 = r_0 - dr
    simulation.context.setParameter('r_0', r_0)

    if i % 10 == 0:
        print(f"r_0 is {r_0}")
        print(f"The radial separation is {current_cv_value}")

"""Protein-protein unbinding"""

# Total number of steps
total_steps = 500000

# Number of steps to run between incrementing r_0
increment_steps = 100

r_increment = ((np.max(windows)-np.min(windows)+0.2) / (total_steps // increment_steps)) * unit.nanometers

# During the pulling loop we will save specific configurations corresponding to the windows
window_coords = []
window_index = 0

# SMD pulling loop
for i in range(total_steps//increment_steps):

    if len(window_coords)==len(windows):
        break
    
    simulation.step(increment_steps)
    current_cv_value = bias_pot.getCollectiveVariableValues(simulation.context)

    if (i * increment_steps) % 1000 == 0:
        print("\nIteration " + str(i))
        print("r_0 = ", r_0, ", distance  = ", current_cv_value)
    
    # Increment the location of the CV
    r_0 = r_0 + r_increment
    simulation.context.setParameter('r_0', r_0)

    # Check if we should save this config as a window starting structure
    if (window_index < len(windows) and current_cv_value >= windows[window_index]):
        window_coords.append(simulation.context.getState(getPositions=True).getPositions())

        print(f"Configuration saved for window {window_index}")
        window_index += 1

    # Break condition
    if len(window_coords) == len(windows):
        break

"""Save input configurations for windows"""

for i in range(len(windows)):
    try:
        r0 = np.round(windows[i], 3)

        # Make directory if it doesn't exist
        dirname = f"windows/{r0}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        outfile = open(f'windows/{r0}/{r0}.pdb', 'w')
        app.PDBFile.writeFile(simulation.topology, window_coords[i], outfile)
        outfile.close()
    except:
        print(f'\nError encountered when saving configuration for window {i}')

