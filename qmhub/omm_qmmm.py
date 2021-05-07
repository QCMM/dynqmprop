import os
import sys
import numpy as np

from itertools import combinations
from constforceplugin import ConstForce

from simtk import openmm
import simtk.unit as u
from .qmmm import QMMM as QMMMBase
from . import units


class QMMM(QMMMBase):
    def __init__(self, fin=None, baseDir=None, mmSoftware=None, **kwargs):

        if fin is None:
            raise ValueError("A QMMMForce object is needed.")

        if baseDir is None:
            baseDir = os.path.join(os.getcwd(), 'scratch')

        if not os.path.isdir(baseDir):
            os.mkdir(baseDir)

        mmSoftware='openmm'

        super(QMMM, self).__init__(fin=fin, baseDir=baseDir, mmSoftware=mmSoftware, **kwargs)

    def set_qm(self, **kwargs):
        """Set QM calculation parameters."""
        self.qm.get_qm_params(**kwargs)

    def run_qm(self):
        """Run QM calculation."""
        self.qm.gen_input()
        self.qm.run()
        if self.qm.exitcode != 0:
            sys.exit(self.qm.exitcode)

    def update_force(self):
        self.system.update_positions()
        self.embed.update()
        self.qm.update()
        self.run_qm()
        self.parse_output()

    @property
    def qm_energy(self):
        return self.system.qm_energy

    @property
    def qm_force(self):
        return self.system.qm_force

    @property
    def mm_force(self):
        return self.system.mm_force


class QMMMForce(ConstForce):
    def __init__(self, qm_index, mm_index, qm_charge, qm_mult,
                 qm_position, mm_position,
                 qm_atom_charge, mm_atom_charge,
                 qm_element, mm_element, cell_basis):
        super(QMMMForce, self).__init__()

        self.qm_index = qm_index
        self.mm_index = mm_index
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult
        self.qm_position = qm_position
        self.mm_position = mm_position
        self.qm_atom_charge = qm_atom_charge
        self.mm_atom_charge = mm_atom_charge
        self.qm_element = qm_element
        self.mm_element = mm_element
        self.cell_basis = cell_basis

        # Add QM and MM atoms to QMMMForce
        for i in self.qm_index:
            self.addParticle(i)
        for i in self.mm_index:
            self.addParticle(i)
        self.force_index = self.qm_index + self.mm_index

        # Make sure numbers of atoms are consistent
        self.n_qm_atoms = len(self.qm_index)
        self.n_mm_atoms = len(self.mm_index)

        assert len(self.qm_position) == self.n_qm_atoms, "Number of QM atom coordinates is wrong."
        assert len(self.mm_position) == self.n_mm_atoms, "Number of MM atom coordinates is wrong."
        assert len(self.qm_atom_charge) == self.n_qm_atoms, "Number of QM atom charges is wrong."
        assert len(self.mm_atom_charge) == self.n_mm_atoms, "Number of MM atom charges is wrong."
        assert len(self.qm_element) == self.n_qm_atoms, "Number of QM elements is wrong."

    def update_positions(self, positions):
        self.qm_position = positions[self.qm_index]
        self.mm_position = positions[self.mm_index]

    def setForce(self, force):
        for i, j in enumerate(self.force_index):
            self.setParticleForce(i, j, force[i])


def QMMMStruct(struct, qm_index, qm_charge, qm_mult, qm_cutoff=None, copy=True):

    class QMMMStruct(struct.__class__):
        QMMM_FORCE_GROUP = 13

        def createSystem(self, *args, **kwargs):
            system = super(QMMMStruct, self).createSystem(*args, **kwargs)

            mm_index = []
            for i in range(len(self.atoms)):
                if i not in qm_index:
                    mm_index.append(i)

            n_qm_atoms = len(qm_index)
            n_mm_atoms = len(mm_index)

            # Set dummy initial coordinates
            qm_position = np.zeros((n_qm_atoms, 3))
            mm_position = np.zeros((n_mm_atoms, 3))

            # Set QM and MM atom charges
            qm_atom_charge = np.zeros(n_qm_atoms)
            mm_atom_charge = np.zeros(n_mm_atoms)

            for i, j in enumerate(qm_index):
                qm_atom_charge[i] = self.atoms[j].charge
            for i, j in enumerate(mm_index):
                mm_atom_charge[i] = self.atoms[j].charge

            # Set QM atom elements
            qm_element = []
            mm_element = []
            for i, atom in enumerate(self.topology.atoms()):
                if i in qm_index:
                    qm_element.append(atom.element.symbol)
                elif i in mm_index:
                    mm_element.append(atom.element.symbol)

            # Set cell basis
            cell_basis = self.box_vectors.value_in_unit(u.angstrom)

            self.qmmm_force = self.omm_qmmm_force(qm_index, mm_index, qm_charge, qm_mult,
                                                  qm_position, mm_position,
                                                  qm_atom_charge, mm_atom_charge,
                                                  qm_element, mm_element, cell_basis)
            self._add_force_to_system(system, self.qmmm_force)
            self.remove_qmmm_elec_force(system, qm_index, mm_index, qm_cutoff)

            return system

        def omm_qmmm_force(self, *args, **kwargs):

            force = QMMMForce(*args, **kwargs)
            force.setForceGroup(self.QMMM_FORCE_GROUP)

            return force

        def remove_qmmm_elec_force(self, system, qm_index, mm_index, qm_cutoff):
            forces = { force.__class__.__name__ : force for force in system.getForces() }
            reference_force = forces['NonbondedForce']
            ONE_4PI_EPS0 = 138.935456

            if qm_cutoff is None:
                qm_cutoff = reference_force.getCutoffDistance()

            expression = '-1*ONE_4PI_EPS0*chargeprod/r;'
            expression += 'chargeprod = charge1*charge2;'
            expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)

            force = openmm.CustomNonbondedForce(expression)
            force.addPerParticleParameter("charge")
            force.setUseSwitchingFunction(False)
            force.setCutoffDistance(qm_cutoff)
            force.setUseLongRangeCorrection(False)
            for index in range(reference_force.getNumParticles()):
                [charge, sigma, epsilon] = reference_force.getParticleParameters(index)
                force.addParticle([charge])
            for index in range(reference_force.getNumExceptions()):
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(index)
                force.addExclusion(iatom, jatom)
            force.setForceGroup(reference_force.getForceGroup())
            force.addInteractionGroup(qm_index, mm_index)
            system.addForce(force)

            return force

        def update_qmmm_force(self, context, qmmm):
            positions = context.getState(getPositions=True).getPositions(asNumpy=True)
            self.qmmm_force.update_positions(positions.value_in_unit(u.angstrom))
            qmmm.update_force()

            qmmm_force = np.concatenate((qmmm.system.qm_force, qmmm.system.mm_force))
            self.qmmm_force.setForce(qmmm_force * units.KCAL_TO_JOULE * 10)
            self.qmmm_force.setEnergy(qmmm.qm_energy * units.KCAL_TO_JOULE)

            self.qmmm_force.updateForceInContext(context)

        def remove_qm_forcefield(self):
            bonds_to_delete = set()
            del_bonds = set()
            del_angles = set()
            del_dihedrals = set()
            del_rbtorsions = set()
            del_urey_bradleys = set()
            del_impropers = set()
            del_cmaps = set()
            del_trigonal_angles = set()
            del_oopbends = set()
            del_pi_torsions = set()
            del_strbnds = set()
            del_tortors = set()
            for i in qm_index:
                ai = self.atoms[i]
                for j in qm_index:
                    aj = self.atoms[j]
                    # Skip if these two atoms are identical
                    if ai is aj: continue
                    for bond in ai.bonds:
                        if aj not in bond: continue
                        bonds_to_delete.add(bond)
            # Find other valence terms we need to delete
            for i, bond in enumerate(self.bonds):
                if bond in bonds_to_delete:
                    del_bonds.add(i)
            for bond in bonds_to_delete:
                for i, angle in enumerate(self.angles):
                    if bond in angle:
                        del_angles.add(i)
                for i, dihed in enumerate(self.dihedrals):
                    if bond in dihed:
                        del_dihedrals.add(i)
                for i, dihed in enumerate(self.rb_torsions):
                    if bond in dihed:
                        del_rbtorsions.add(i)
                for i, urey in enumerate(self.urey_bradleys):
                    if bond in urey:
                        del_urey_bradleys.add(i)
                for i, improper in enumerate(self.impropers):
                    if bond in improper:
                        del_impropers.add(i)
                for i, cmap in enumerate(self.cmaps):
                    if bond in cmap:
                        del_cmaps.add(i)
                for i, trigonal_angle in enumerate(self.trigonal_angles):
                    if bond in trigonal_angle:
                        del_trigonal_angles.add(i)
                for i, oopbend in enumerate(self.out_of_plane_bends):
                    if bond in oopbend:
                        del_oopbends.add(i)
                for i, pitor in enumerate(self.pi_torsions):
                    if bond in pitor:
                        del_pi_torsions.add(i)
                for i, strbnd in enumerate(self.stretch_bends):
                    if bond in strbnd:
                        del_strbnds.add(i)
                for i, tortor in enumerate(self.torsion_torsions):
                    if bond in tortor:
                        del_tortors.add(i)

            if not del_bonds: return
            for i in sorted(del_bonds, reverse=True):
                self.bonds[i].delete()
                del self.bonds[i]
            for i in sorted(del_angles, reverse=True):
                self.angles[i].delete()
                del self.angles[i]
            for i in sorted(del_dihedrals, reverse=True):
                self.dihedrals[i].delete()
                del self.dihedrals[i]
            for i in sorted(del_rbtorsions, reverse=True):
                self.rb_torsions[i].delete()
                del self.rb_torsions[i]
            for i in sorted(del_urey_bradleys, reverse=True):
                self.urey_bradleys[i].delete()
                del self.urey_bradleys[i]
            for i in sorted(del_impropers, reverse=True):
                self.impropers[i].delete()
                del self.impropers[i]
            for i in sorted(del_cmaps, reverse=True):
                self.cmaps[i].delete()
                del self.cmaps[i]
            for i in sorted(del_trigonal_angles, reverse=True):
                del self.trigonal_angles[i]
            for i in sorted(del_oopbends, reverse=True):
                del self.out_of_plane_bends[i]
            for i in sorted(del_tortors, reverse=True):
                self.torsion_torsions[i].delete()
                del self.torsion_torsions[i]
            for i in sorted(del_strbnds, reverse=True):
                del self.stretch_bends[i]
            try:
                self.remake_parm()
            except AttributeError:
                self.prune_empty_terms()

            # Exclude non-bonded interactions between QM atoms in real space
            for i, j in combinations(qm_index, 2):
                self.atoms[i].exclude(self.atoms[j])

    struct = struct.from_structure(struct, copy=copy)
    struct.__class__ = QMMMStruct
    struct.remove_qm_forcefield()

    return struct
