import mdtraj

import parmed as pmd
import numpy as np

from parmed.tools import change
from qmhub import *
from get_charges import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


def setup_simulation(top, positions, j):
    '''Setup the openMM system with the current topology and
    the input coordinates or the current positions depending on
    the value of j.
    Standard conditions are assumed (298K, 1bar)
    Input:
    top : Topology object from OpenMM o ParmEd (Gromacs or Amber)
    positions: current positions of atoms
    j: integer of charge update cycle
    Returns:
    Simulation (OpenMM class)
    '''

    system = top.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    system.addForce(MonteCarloBarostat(1 * bar, 298 * kelvin))
    integrator = LangevinIntegrator(
        298 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(top.topology, system, integrator)
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 5000, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(DCDReporter(f'traj_{j}.dcd', 50000))
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    return simulation, system


def calculate_charges(simulation, system, parmed_selection, qm_charge, radius=10, method='B3LYP', basis='def2-TZVP'):

    positions = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))
    pdb = pmd.load_file('output.pdb')
    traj = mdtraj.load('output.pdb')
    traj.image_molecules()
    frame = pmd.openmm.load_topology(
        pdb.topology, system, traj.openmm_positions(0))
    qm_region = frame[parmed_selection]
    environment = frame[parmed_selection + '<@12.0 & !' + parmed_selection]
    qmmm = QMMM(qm_region, environment, qmSoftware='orca', mmSoftware='openmm', qmCharge=qm_charge, qmMult=1,
                qmEmbedNear='eed', qmEmbedFar=None, qmSwitchingType='Switch', qmCutoff=radius)
    qmmm.run_qm(method=method, basis=basis, calc_forces=False)
    qmmm.parse_output()
    epol = qmmm.system.qm_atoms.qm_pol_energy
    charges = get_charges('orca.molden.input', 'mbis')
    return positions, epol, charges


def make_new_top(top_file, box_vectors, charge_list, epol_list, parmed_selection):

    epol = numpy.array(epol_list)
    epol_mean, epol_std = epol.mean(), epol.std()
    charges = numpy.array(charge_list)
    charges_mean, charges_std = charges.mean(
        axis=0, dtype=np.float64), charges.std(axis=0, dtype=np.float64)
    _top = pmd.load_file(top_file)
    _top.box_vectors = box_vectors
    for i, atom in enumerate(_top[parmed_selection].atoms):
        mask = f'{parmed_selection}@{atom.name}'
        action = change(_top, mask, "charge", round(charges_mean[i], 5))
        action.execute()
    _top.save(top_file, overwrite=True)
    return _top, charges_mean, charges_std, epol_mean, epol_std
