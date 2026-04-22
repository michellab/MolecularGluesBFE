import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import numpy as np
import pandas as pd
import os

run_number = int(sys.argv[1])
savedir = f"results/run{run_number}"

if not os.path.exists(savedir): # Make save directory if it doesn't yet exist
    os.makedirs(savedir)

"""System setup"""

dt = 4*unit.femtoseconds 

# Load param and coord files
prmtop = app.AmberPrmtopFile(f'structures/complex.prmtop')
inpcrd = app.AmberInpcrdFile(f'structures/complex.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(1.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter(f'{savedir}/system.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.DCDReporter(f'{savedir}/traj.dcd', 2500))

# Minimise energy 
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(1.0000*unit.kelvin)

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

"""100 ns simulation"""

print('-------------------------')
print('Starting NVT simulation!')
simulation.step(25000000)