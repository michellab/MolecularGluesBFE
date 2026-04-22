import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
from sys import stdout
import numpy as np
import os

"""Command line arguments"""

RMSD_0 = np.round(float(sys.argv[1]), 3) # units of Angstrom
dof = str(sys.argv[2]) # Select DCAF16, BD2, DCAF16_only, BD2withDCAF16
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

if species == 'BD2':
    prmtop_filename= 'BD2.prmtop'
    inpcrd_filename = 'BD2.inpcrd'

elif species == 'DCAF16': 
    prmtop_filename = 'DCAF16.prmtop'
    inpcrd_filename = 'DCAF16.inpcrd'    

elif species in ['DCAF16_only', 'BD2_only', 'DCAF16withBD2', 'BD2withDCAF16']:
    prmtop_filename = 'complex_eq.prmtop'
    inpcrd_filename = 'complex_eq.inpcrd'

else:
    raise FileNotFoundError(f"Select one of the following options for species of interest: DCAF16, BD2, DCAF16_only, BD2_only, DCAF16withBD2, BD2withDCAF16")    

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

"""Selection tuple for restraint type"""

if restraint_type == 'backbone':
    restraint_selection = ['C', 'N', 'CA']

elif restraint_type == 'CA':
    restraint_selection = ['CA']

else:
    raise ValueError('Select one of the following restraint type options: backbone, CA!')

"""RMSD atom selection"""
# Ligand = BD2, receptor = DCAF16

# Make lists of residue indices distinguising between interface and DDB1-binding regions of DCAF16
DCAF16_interface_residx = np.append(np.arange(0,55), np.arange(123, 172))
DCAF16_DDB1_binding_residx = np.arange(71,114)

reference_positions = inpcrd.positions

if species == 'BD2':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 109) and atom.name in restraint_selection
    ]
 
# Add extra restraint for DDB1-binding residues in DCAF16
elif species == 'DCAF16' or species == 'DCAF16_only':
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_interface_residx and atom.name in restraint_selection
    ]
    DDB1_binding_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in restraint_selection
    ]

elif species == 'BD2_only':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(173, 282) and atom.name in restraint_selection
    ]

else: # BD2withDCAF16 or DCAF16withBD2
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_interface_residx and atom.name in restraint_selection
    ]
    DDB1_binding_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in restraint_selection
    ]
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(173, 282) and atom.name in restraint_selection
    ]

"""Tests to ensure we have the right indices"""

if species in ['DCAF16', 'DCAF16_only', 'DCAF16withBD2']:
    for atom in simulation.topology.atoms():
        if atom.index==receptor_atoms[0] and atom.residue.name!='ASN':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue N1 is missing')
        if atom.index==receptor_atoms[-1] and atom.residue.name!='LEU':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue L172 is missing')

if species in ['BD2', 'BD2withDCAF16', 'BD2_only']:
    for atom in simulation.topology.atoms():
        if atom.index==ligand_atoms[0] and atom.residue.name!='SER':
            raise ValueError(f'Incorrect residue selection for BD2 - residue SER174 is missing')
        if atom.index==ligand_atoms[-1] and atom.residue.name!='ASP':
            raise ValueError(f'Incorrect residue selection for BD2 - residue ASP109 is missing')

"""Applying RMSD forces"""

if species in ['BD2', 'BD2_only']:
    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

elif species == 'DCAF16' or species == 'DCAF16_only':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)

elif species == 'DCAF16withBD2':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*rmsd^2')
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

else: #BD2withDCAF16
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

simulation.context.reinitialize(preserveState=True)

"""Collecting CV samples"""

print('running window', RMSD_0)

# Run the simulation and record the value of the CV.
cv_values=[]

for i in range(sampling_steps//record_steps):

    simulation.step(record_steps)

    if species in ('DCAF16', 'DCAF16_only', 'DCAF16withBD2'):
        current_cv_value = receptor_rmsd_force.getCollectiveVariableValues(simulation.context)

    else: # BD2, BD2, BD2_only, BD2withDCAF16
        current_cv_value = ligand_rmsd_force.getCollectiveVariableValues(simulation.context)    
    
    cv_values.append([i, current_cv_value[0]])

# Final save
np.savetxt(f'{savedir}/{RMSD_0}.txt', np.array(cv_values))

print('Completed window', RMSD_0)
