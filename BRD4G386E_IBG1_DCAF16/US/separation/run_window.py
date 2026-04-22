import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import os
from sys import stdout
import numpy as np
import yaml 
import pandas as pd

"""r0 value for window bias"""

r0 = np.round(float(sys.argv[1]), 4) # Take r0 value as command line argument
run_number = int(sys.argv[2]) # specify the replica

"""Read in US configuration from yaml"""

with open('../US_config.yaml', 'r') as file:
    US_data = yaml.safe_load(file)

# MD parameters
timestep = 4 # fs
sampling_steps = 7500000 # 30 ns
record_steps = 125 # sample every 0.5 ps

# Force constants
k_Boresch = 100 # kcal/mol/rad^2
k_RMSD = 5 # kcal/mol/AA^2
k_sep = 10 # kcal/mol/AA^2

# Directory to save all results
restraint_type = 'CA' # choose 'CA' or 'backbone'
inputdir = f"windows/{r0}" # pdb snapshot for window starting configuration
savedir = f"results/run{run_number}"

if not os.path.exists(inputdir): # Check if directory exists
    raise FileNotFoundError(f"Input directory does not exist for r0 = {r0}")

if os.path.exists(f'{savedir}/{r0}.txt'): # Check if a CV sample file already exists
    raise FileExistsError(f"A file of CV samples already exists for r0 = {r0}")

if not os.path.exists(savedir): # Make save directory if it doesn't yet exist
    os.makedirs(savedir)

"""Selection tuple for restraint type"""

if restraint_type == 'backbone':
    restraint_selection = ['C', 'N', 'CA']

elif restraint_type == 'CA':
    restraint_selection = ['CA']

else:
    raise ValueError('Select one of the following restraint type options: backbone, CA')

"""System setup"""

dt = timestep*unit.femtoseconds 

inpcrd = app.AmberInpcrdFile('../inputs/complex_eq.inpcrd')
prmtop = app.AmberPrmtopFile('../inputs/complex_eq.prmtop')

system = prmtop.createSystem(nonbondedMethod=app.PME, hydrogenMass=1.5*unit.amu, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(1.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors) # NPT equil box vectors

pdb = app.PDBFile(f"windows/{r0}/{r0}.pdb") # Set positions to suitable window
simulation.context.setPositions(pdb.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter(f'{savedir}/{r0}.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.StateDataReporter(stdout, 2000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(app.DCDReporter(f'{savedir}/{r0}.dcd', 2500))

# Minimise energy 
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(1.0000*unit.kelvin)

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

"""Find indices of all IBG1 heavy atoms"""

IBG1_indices = []
for atom in simulation.topology.atoms():
    if atom.residue.name == 'MOL':
        if not atom.name.startswith('H'):
            IBG1_indices.append(atom.index)

"""RMSD Restraints"""

reference_positions = simulation.context.getState(getPositions=True).getPositions()

DCAF16_interface_residx = np.append(np.arange(0,55), np.arange(123, 172))
DCAF16_DDB1_binding_residx = np.arange(71,114)

receptor_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in DCAF16_interface_residx and atom.name in restraint_selection
]
DDB1_binding_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in restraint_selection
]
BD1_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(173, 280) and atom.name in restraint_selection
]
BD2_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(280, 390) and atom.name in restraint_selection
]

# Add IBG1 heavy atom indices to restrained ligand atoms
BD2_IBG1_atoms = BD2_atoms + IBG1_indices

# Add restraining forces for receptor and ligand rmsd
receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
system.addForce(receptor_rmsd_force)

DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
system.addForce(DDB1_rmsd_force)

BD1_rmsd_force = mm.CustomCVForce('0.5*k_BD1*rmsd^2')
BD1_rmsd_force.addGlobalParameter('k_BD1', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
BD1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD1_atoms))
system.addForce(BD1_rmsd_force)

BD2_IBG1_rmsd_force = mm.CustomCVForce('0.5*k_BD2_IBG1*rmsd^2')
BD2_IBG1_rmsd_force.addGlobalParameter('k_BD2_IBG1', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
BD2_IBG1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD2_IBG1_atoms))
system.addForce(BD2_IBG1_rmsd_force)

simulation.context.reinitialize(preserveState=True)

for atom in simulation.topology.atoms():
    if atom.index==receptor_atoms[0] and atom.residue.name!='ASN':
        raise ValueError(f'Incorrect residue selection for DCAF16 - residue N1 is missing')
    if atom.index==receptor_atoms[-1] and atom.residue.name!='LEU':
        raise ValueError(f'Incorrect residue selection for DCAF16 - residue L172 is missing')
    if atom.index==BD1_atoms[0] and atom.residue.name!='THR':
        raise ValueError(f'Incorrect residue selection for BD1 - residue T174 is missing')
    if atom.index==BD1_atoms[-1] and atom.residue.name!='THR':
        raise ValueError(f'Incorrect residue selection for BD1 - residue T280 is missing')
    if atom.index==BD2_IBG1_atoms[0] and atom.residue.name!='LYS':
        raise ValueError(f'Incorrect residue selection for BD2 - residue K281 is missing')
    if atom.index==BD2_IBG1_atoms[-1] and atom.residue.name!='MOL':
        raise ValueError(f'Incorrect residue selection for IBG1!')
    
"""Radial separation CV"""

# 1-indexing from MDAnalysis
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

# Add IBG1 heavy atom indices to restrained ligand atoms
lig_group = lig_group + IBG1_indices

# Define radial distance as collective variable which we will vary
cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
cv.addGroup(np.array(rec_group))
cv.addGroup(np.array(lig_group))

# Specify bond groups
bondGroups = [0, 1]
cv.addBond(bondGroups)

# Define biasing potential
bias_pot = mm.CustomCVForce('0.5 * k_r * (cv-r0)^2')
bias_pot.addGlobalParameter('k_r', k_sep * unit.kilocalories_per_mole / unit.angstrom**2)
bias_pot.addGlobalParameter('r0', r0* unit.nanometers)

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
all_atoms = [idx_b] + [idx_c] + rec_group + [idx_B] + [idx_C]
for atom in simulation.topology.atoms():
    if atom.index in all_atoms and atom.name != 'CA':
        raise ValueError('Select only CA atoms as anchorpoints')
    
# Equilibrium values of Boresch dof
theta_A_0 = US_data['Boresch equilibrium values']['theta_A_0']
theta_B_0 = US_data['Boresch equilibrium values']['theta_B_0']
phi_A_0 = US_data['Boresch equilibrium values']['phi_A_0']
phi_B_0 = US_data['Boresch equilibrium values']['phi_B_0']
phi_C_0 = US_data['Boresch equilibrium values']['phi_C_0']

k_Boresch = k_Boresch * unit.kilocalories_per_mole / unit.radians**2 #Set global force constant

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

"""Collecting CV samples"""

def obtain_RMSDs(simulation):
    """
    Return the instantaneous values of the RMSDs restrained
    by the previously applied bias potentials
    """
    # Convert RMSD units to Angstrom
    RMSD_rec = 10*receptor_rmsd_force.getCollectiveVariableValues(simulation.context)[0]
    RMSD_BD1 = 10*BD1_rmsd_force.getCollectiveVariableValues(simulation.context)[0]
    RMSD_BD2_IBG1 = 10*BD2_IBG1_rmsd_force.getCollectiveVariableValues(simulation.context)[0]
    RMSD_DDB1 = 10*DDB1_rmsd_force.getCollectiveVariableValues(simulation.context)[0]

    return [RMSD_rec, RMSD_BD1, RMSD_BD2_IBG1, RMSD_DDB1]

print('running window for r0 = ', r0)

# Prepare sampling
n_samples = int(sampling_steps//record_steps) # Total number of samples
cv_values = np.zeros((n_samples, 2)) # Empty array to store samples
dof_data = np.zeros((n_samples, 6))

# Run the simulation and record the value of the CV.
for i in range(n_samples):

    simulation.step(record_steps)

    # get the current value of the cv
    current_cv_value = bias_pot.getCollectiveVariableValues(simulation.context)
    sample = current_cv_value[0]
    cv_values[i] = [i, sample]

    # Save the other dofs
    dofs = obtain_RMSDs(simulation)
    dof_data[i] = [i, i*timestep*1e-6] + dofs

# Final save
np.savetxt(f'{savedir}/{r0}.txt', cv_values)

# Save the RMSDs to csv
df = pd.DataFrame(data=dof_data, columns=[
    'Steps',
    'Time (ns)',
    'RMSD_rec',
    'RMSD_BD1',
    'RMSD_BD2-IBG1',
    'RMSD_DDB1'
])

df.to_csv(f'{savedir}/{r0}_RMSD.csv', index=False)

print('Completed window for r0 = ', r0)



