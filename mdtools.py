import warnings
import mdtraj
import sys

import parmed as pmd
import numpy as np

from parmed.tools import change
from openmmtools import forces
from qmhub import *
from dynqmprop.ffparams import ForceFieldParams
from simtk.openmm import openmm
from simtk.openmm import app
from simtk import unit

# avoid warnings
warnings.filterwarnings('ignore')

def set_restrained_atoms(top_file, coords_file, ligand_selection, receptor_selection):

    _top = pmd.load_file(top_file, xyz=coords_file)
    atom_list = _top.atoms
    lig = _top[ligand_selection].atoms
    rec = _top[receptor_selection].atoms
    ligand_atom_list = [idx for at in lig for idx, atom in enumerate(atom_list) if (
        at.residue.number, at.name) == (atom.residue.number, atom.name)]
    receptor_atom_list = [idx for at in rec for idx, atom in enumerate(
        atom_list) if (at.residue.number, at.name) == (atom.residue.number, atom.name)]

    return ligand_atom_list, receptor_atom_list


def setup_simulation(top, positions, update, box_vectors=None, restraint=False, ligand_atom_list=None, receptor_atom_list=None):
    '''Setup the openMM system with the current topology and
    the input coordinates or the current positions depending on
    the value of update.
    Standard conditions are assumed (298K, 1bar)
    Input:
    top : Topology object from OpenMM o ParmEd (Gromacs or Amber)
    positions: current positions of atoms
    update: integer of charge update cycle
    Returns:
    Simulation (OpenMM class)
    '''

    system = top.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=1 * unit.nanometer, constraints=app.HBonds)
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin))
    if restraint:
        if ligand_atom_list is not None and receptor_atom_list is not None:
            restraint = forces.HarmonicRestraintForce(spring_constant=0.2 * unit.kilocalories_per_mole / unit.angstrom**2,
                                                      restrained_atom_indices1=ligand_atom_list,
                                                      restrained_atom_indices2=receptor_atom_list)
            system.addForce(restraint)
        else:
            raise Exception("Missing atom list to apply restraints")

    integrator = openmm.LangevinIntegrator(
        298 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(top.topology, system, integrator)
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, 5000, step=True, potentialEnergy=True, temperature=True, density=True))
    simulation.reporters.append(app.DCDReporter(f'traj_{update}.dcd', 50000))
    simulation.context.setPositions(positions)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.minimizeEnergy()
    return simulation, system


def calculate_charges(simulation, system, ligand_selection, qm_charge, radius=10, method='B3LYP', basis='def2-TZVP'):

    positions = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    _box_vectors = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(_box_vectors)
    app.PDBFile.writeFile(simulation.topology, positions,
                          open('output.pdb', 'w'))
    pdb = pmd.load_file('output.pdb')
    traj = mdtraj.load('output.pdb')
    traj.image_molecules()
    frame = pmd.openmm.load_topology(
        pdb.topology, system, traj.openmm_positions(0))
    qm_region = frame[ligand_selection]
    environment = frame[f'{ligand_selection}<@{float(radius)+2.0} & !{ligand_selection}']
    qmmm = QMMM(qm_region, environment, qmSoftware='orca', mmSoftware='openmm', qmCharge=qm_charge, qmMult=1,
                qmEmbedNear='eed', qmEmbedFar=None, qmSwitchingType='Switch', qmCutoff=radius)
    qmmm.run_qm(method=method, basis=basis, calc_forces=False)
    qmmm.parse_output()
    epol = qmmm.system.qm_atoms.qm_pol_energy
    params = ForceFieldParams('orca.molden.input')
    params.set_molgrid()
    params.do_partitioning(method='mbis')
    charges = params.get_charges()
    return positions, epol, charges


def charge_stats(charge_list):

    charges = np.array(charge_list)
    charges_mean, charges_std = charges.mean(
        axis=0, dtype=np.float64), charges.std(axis=0, dtype=np.float64)
    return charges_mean, charges_std


def epol_stats(epol_list):

    epol = np.array(epol_list)
    epol_mean, epol_std = epol.mean(), epol.std()
    return epol_mean, epol_std


def make_new_top(top_file, box_vectors, charges_mean, ligand_selection):

    _top = pmd.load_file(top_file)
    _top.box_vectors = box_vectors
    for i, atom in enumerate(_top[ligand_selection].atoms):
        mask = f'{ligand_selection}&@{atom.name}'
        action = change(_top, mask, 'charge', round(charges_mean[i], 6))
        action.execute()
    _top.save(top_file, overwrite=True)
    return _top
