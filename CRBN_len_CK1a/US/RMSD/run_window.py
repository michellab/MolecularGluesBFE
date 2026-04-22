import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import os
from sys import stdout
import numpy as np

"""Command line arguments"""

RMSD_0 = np.round(float(sys.argv[1]), 3) # units of Angstrom
dof = str(sys.argv[2]) # Select CRBN_len, CK1a, CRBN_only, CK1awithCRBN
run_number = int(sys.argv[3]) # specify the replica

# MD parameters
timestep = 4 # fs
sampling_steps = 5000000 # 20 ns
record_steps = 125 # sample every 0.5 ps

# Force constants
k_RMSD = 5 # kcal/mol/AA^2

# Directory to save all results
restraint_type = 'CA' # choose 'CA' or 'backbone'
species = dof
savedir = f"results/{species}/run{run_number}"

if species == 'CK1a': # Unbound
    prmtop_filename= 'CK1a.prmtop'
    inpcrd_filename = 'CK1a.inpcrd'

elif species == 'CRBN_len': # Unbound
    prmtop_filename = 'CRBN_len.prmtop'
    inpcrd_filename = 'CRBN_len.inpcrd'    

elif species in ['CRBN_len_only', 'CK1a_only', 'CRBN_lenwithCK1a', 'CK1awithCRBN_len']: 
    prmtop_filename = 'complex_eq.prmtop' # Bound
    inpcrd_filename = 'complex_eq.inpcrd'

else:
    raise FileNotFoundError(f"Select one of the following species: CRBN_len, CK1a, CRBN_len_only, CK1a_only, CRBN_lenwithCK1a, CK1awithCRBN_len")    

# Check to see if there is an existing file
if os.path.exists(f'{savedir}/{RMSD_0}.txt'): # Check if a CV sample file already exists
    raise FileExistsError(f"A file of CV samples already exists for RMSD_0 = {RMSD_0}")

if not os.path.exists(savedir): # Make save directory if it doesn't yet exist
    os.makedirs(savedir)

"""System setup"""

dt = timestep*unit.femtoseconds 

# Load param and coord files
prmtop = app.AmberPrmtopFile(f'../inputs/{prmtop_filename}')
inpcrd = app.AmberInpcrdFile(f'../inputs/{inpcrd_filename}')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(1.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter(f'{savedir}/{RMSD_0}.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.StateDataReporter(stdout, 2000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(app.DCDReporter(f'{savedir}/{RMSD_0}.dcd', 2500))

# Minimise energy 
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(1.0000*unit.kelvin)

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)


"""Selection tuple for restraint type"""

if restraint_type == 'backbone':
    restraint_selection = ['C', 'N', 'CA']

elif restraint_type == 'CA':
    restraint_selection = ['CA']

else:
    raise ValueError('Select one of the following restraint type options: backbone, CA!')

"""Find indices of all lenalidomide heavy atoms"""

len_indices = []
for atom in simulation.topology.atoms():
    if atom.residue.name == 'MOL':
        if not atom.name.startswith('H'):
            len_indices.append(atom.index)

"""RMSD Restraints"""

reference_positions = inpcrd.positions

if species == 'CK1a':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 293) and atom.name in restraint_selection
    ]

elif species == 'CRBN_len' or species == 'CRBN_len_only':
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 312) and atom.name in restraint_selection
    ]

elif species == 'CK1a_only':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(313, 606) and atom.name in restraint_selection
    ]

else: # CK1awithCRBN_len or CRBN_lenwithCK1a
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 312) and atom.name in restraint_selection
    ]
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(313, 606) and atom.name in restraint_selection
    ]

# Add lenalidomide heavy atom indices to restrained receptor atoms
if 'CRBN_len' in species:
    receptor_atoms = receptor_atoms + len_indices

    print('\nReceptor atom selection')
    for atom in simulation.topology.atoms():
        if atom.index in receptor_atoms:
            print(atom)

"""Tests to ensure we have the right indices"""

if species in ['CRBN_len', 'CRBN_len_only', 'CRBN_lenwithCK1a']:
    if atom.index==receptor_atoms[0] and atom.residue.name!='ASP':
        raise ValueError(f'Incorrect residue selection for CRBN_len - residue D1 is missing')
    if atom.index==receptor_atoms[-1] and atom.residue.name!='MOL':
        raise ValueError(f'Incorrect residue selection for CRBN_len - lenalidomide is missing')

if species in ['CK1a', 'CK1awithCRBN_len', 'CK1a_only']:
    for atom in simulation.topology.atoms():
        if atom.index==ligand_atoms[0] and atom.residue.name!='PHE':
            raise ValueError(f'Incorrect residue selection for CK1a - residue F314 is missing')
        if atom.index==ligand_atoms[-1] and atom.residue.name!='GLN':
            raise ValueError(f'Incorrect residue selection for CK1a - residue Q606 is missing')

"""Applying RMSD forces"""

if species in ['CK1a', 'CK1a_only']:
    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

elif species == 'CRBN_len' or species == 'CRBN_len_only':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)

elif species == 'CRBN_lenwithCK1a':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*rmsd^2')
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

else: # CK1awithCRBN_len
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

# Reinitialise with added potentials
simulation.context.reinitialize(preserveState=True)

"""Collecting CV samples"""

print('running window', RMSD_0)

# Run the simulation and record the value of the CV.
cv_values=[]

for i in range(sampling_steps//record_steps):

    simulation.step(record_steps)

    if species in ['CRBN_len', 'CRBN_len_only', 'CRBN_lenwithCK1a']:
        current_cv_value = receptor_rmsd_force.getCollectiveVariableValues(simulation.context)

    else:
        current_cv_value = ligand_rmsd_force.getCollectiveVariableValues(simulation.context)    
    
    cv_values.append([i, current_cv_value[0]])

# Final save
np.savetxt(f'{savedir}/{RMSD_0}.txt', np.array(cv_values))

print('Completed window', RMSD_0)