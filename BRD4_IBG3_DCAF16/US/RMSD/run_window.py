import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
from sys import stdout
import numpy as np
import os

"""Command line arguments"""

RMSD_0 = np.round(float(sys.argv[1]), 3) # units of Angstrom
dof = str(sys.argv[2]) # Select DCAF16, BD1, DCAF16_only, BD1withDCAF16
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

if species in ['BD1_only_bulk', 'BD2_IBG3withBD1_bulk', 'BD2_IBG3_only_bulk', 'BD1withBD2_IBG3_bulk']:
    prmtop_filename= 'BRD4_IBG3.prmtop'
    inpcrd_filename = 'BRD4_IBG3.inpcrd'

elif species == 'DCAF16': 
    prmtop_filename = 'DCAF16.prmtop'
    inpcrd_filename = 'DCAF16.inpcrd'    

elif species in ['DCAF16_only', 'BD1withDCAF16', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3withDCAF16', 'BD1withDCAF16andBD2_IBG3', 'BD1_only', 'BD2_IBG3withBD1', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD1', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3_only', 'BD1withBD2_IBG3', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD2_IBG3', 'BD1withDCAF16andBD2_IBG3']:
    prmtop_filename = 'complex_eq.prmtop'
    inpcrd_filename = 'complex_eq.inpcrd'

else:
    raise FileNotFoundError(f"Select one of the following options for species of interest: DCAF16, BRD4, DCAF16_only, BRD4_only, DCAF16withBRD4, BRD4withDCAF16")    

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

# Set NPT-scaled box vectors
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

"""Find indices of all IBG3 heavy atoms"""

IBG3_indices = []
for atom in simulation.topology.atoms():
    if atom.residue.name == 'MOL':
        if not atom.name.startswith('H'):
            IBG3_indices.append(atom.index)

"""Selection tuple for restraint type"""

if restraint_type == 'backbone':
    restraint_selection = ['C', 'N', 'CA']

elif restraint_type == 'CA':
    restraint_selection = ['CA']

else:
    raise ValueError('Select one of the following restraint type options: backbone, CA!')

"""RMSD atom selection"""

# Make lists of residue indices distinguising between interface and DDB1-binding regions of DCAF16
DCAF16_interface_residx = np.append(np.arange(0,55), np.arange(123, 172))
DCAF16_DDB1_binding_residx = np.arange(71,114)

reference_positions = inpcrd.positions

# Bulk state BD1 and BD2_IBG3 selections
if species in ['BD1_only_bulk', 'BD2_IBG3_only_bulk', 'BD1withBD2_IBG3_bulk', 'BD2_IBG3withBD1_bulk']:
    BD1_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 107) and atom.name in restraint_selection
    ]
    BD2_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(107, 218) and atom.name in restraint_selection
    ]
    # Add IBG3 heavy atom indices to restrained ligand atoms
    BD2_IBG3_atoms = BD2_atoms + IBG3_indices

# Bulk state DCAF16 + complex selections
if species in ['DCAF16', 'DCAF16_only', 'BD1withDCAF16', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3withDCAF16', 'BD1withDCAF16andBD2_IBG3', 
    'BD1_only', 'BD2_IBG3withBD1', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD1', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3_only', 
    'BD1withBD2_IBG3', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD2_IBG3', 'BD1withDCAF16andBD2_IBG3']:
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
        if atom.residue.index in range(280, 391) and atom.name in restraint_selection
    ]
    # Add IBG3 heavy atom indices to restrained ligand atoms
    BD2_IBG3_atoms = BD2_atoms + IBG3_indices
 
       
"""Tests to ensure we have the right indices"""

if 'DCAF16' in species:
    for atom in simulation.topology.atoms():
        if atom.index==receptor_atoms[0] and atom.residue.name!='ASN':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue N1 is missing')
        if atom.index==receptor_atoms[-1] and atom.residue.name!='LEU':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue L172 is missing')

if 'BD1' in species:
    for atom in simulation.topology.atoms():
        if atom.index==BD1_atoms[0] and atom.residue.name!='THR':
            raise ValueError(f'Incorrect residue selection for BRD4 - residue THR174 is missing')
        if atom.index==BD1_atoms[-1] and atom.residue.name!='THR':
            raise ValueError(f'Incorrect residue selection for BRD4 - residue THR280 is missing')
        
if 'BD2_IBG3' in species:
    for atom in simulation.topology.atoms():
        if atom.index==BD2_IBG3_atoms[0] and atom.residue.name!='LYS':
            raise ValueError(f'Incorrect residue selection for BRD4 - residue LYS281 is missing')

"""Applying RMSD forces"""

# BD1 CV
if species in ['BD1_only_bulk', 'BD1withBD2_IBG3_bulk', 'BD1withDCAF16', 'BD1withDCAF16andBD2_IBG3', 'BD1_only', 
        'BD1withBD2_IBG3', 'BD1withDCAF16andBD2_IBG3']:
    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD1_atoms))
    system.addForce(ligand_rmsd_force)

# BD2_IBG3 CV
elif species in ['BD2_IBG3withBD1_bulk', 'BD2_IBG3_only_bulk', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3withDCAF16', 'BD2_IBG3withBD1',
        'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3_only']:
    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD2_IBG3_atoms))
    system.addForce(ligand_rmsd_force)

# DCAF16 CV
elif species in ['DCAF16', 'DCAF16_only', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD1', 'DCAF16withBD1andBD2_IBG3', 
        'DCAF16withBD2_IBG3']:
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)

# BD1 restrained
if species in ['BD2_IBG3withBD1_bulk', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3withBD1', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD1', 
        'BD2_IBG3withDCAF16andBD1', 'DCAF16withBD1andBD2_IBG3']:
    BD1_rmsd_force = mm.CustomCVForce('0.5*k_BD1*rmsd^2')
    BD1_rmsd_force.addGlobalParameter('k_BD1', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    BD1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD1_atoms))
    system.addForce(BD1_rmsd_force)

# BD2_IBG3 restrained
if species in ['BD1withBD2_IBG3_bulk', 'BD1withDCAF16andBD2_IBG3', 'DCAF16withBD1andBD2_IBG3', 'BD1withBD2_IBG3', 'DCAF16withBD1andBD2_IBG3', 
        'DCAF16withBD2_IBG3', 'BD1withDCAF16andBD2_IBG3' ]:
    BD2_IBG3_rmsd_force = mm.CustomCVForce('0.5*k_BD2_IBG3*rmsd^2')
    BD2_IBG3_rmsd_force.addGlobalParameter('k_BD2_IBG3', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    BD2_IBG3_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, BD2_IBG3_atoms))
    system.addForce(BD2_IBG3_rmsd_force)

# DCAF16 restrained
if species in ['BD1withDCAF16', 'BD2_IBG3withDCAF16andBD1', 'BD2_IBG3withDCAF16', 'BD1withDCAF16andBD2_IBG3', 'BD2_IBG3withDCAF16andBD1',
        'BD1withDCAF16andBD2_IBG3']:
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)

simulation.context.reinitialize(preserveState=True)

"""Collecting CV samples"""

print('running window', RMSD_0)

# Run the simulation and record the value of the CV.
cv_values=[]

for i in range(sampling_steps//record_steps):

    simulation.step(record_steps)

    if species in ['DCAF16', 'DCAF16_only', 'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD1', 
        'DCAF16withBD1andBD2_IBG3', 'DCAF16withBD2_IBG3']:
        current_cv_value = receptor_rmsd_force.getCollectiveVariableValues(simulation.context)

    else: # BD1 or BD2_IBG3 CV
        current_cv_value = ligand_rmsd_force.getCollectiveVariableValues(simulation.context)    
    
    cv_values.append([i, current_cv_value[0]])

# Final save
np.savetxt(f'{savedir}/{RMSD_0}.txt', np.array(cv_values))

print('Completed window', RMSD_0)
