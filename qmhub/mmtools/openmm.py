import numpy as np

from .mmbase import MMBase
from ..atomtools import QMAtoms, MMAtoms
import os as os
import parmed as pmd


class OpenMM(MMBase):
    """Class to communicate with NAMD.

    Attributes
    ----------
    n_qm_atoms : int
        Number of QM atoms including linking atoms
    n_mm_atoms : int
        Number of MM atoms including virtual particles
    n_atoms: int
        Number of total atoms in the whole system
    qm_charge : int
        Total charge of QM subsystem
    qm_mult : int
        Multiplicity of QM subsystem
    step : int
        Current step number

    """

    MMTOOL = 'OpenMM'

    def __init__(self, qmatoms, mmatoms):



        self.n_qm_atoms = len(qmatoms.atoms)
        self.n_mm_atoms = len(mmatoms.atoms)
        self.n_atoms = self.n_qm_atoms + self.n_mm_atoms

        self.qm_charge = int(sum(qmatoms.atoms[i].charge for i in range(len(qmatoms.atoms))))
        self.qm_mult = 1

        self.step = 0

        # Process QM atoms
        qm_pos_x = []
        qm_pos_y = []
        qm_pos_z = []
        qm_element = []
        qm_atom_charge = []
        qm_index = []
        for i in range(len(qmatoms.atoms)):
            qm_index.append(i)
            qm_atom_charge.append(qmatoms.atoms[i].charge)
            qm_element.append(pmd.periodic_table.element_by_mass(qmatoms.atoms[i].mass))
            qm_pos_x.append(qmatoms.atoms[i].xx)
            qm_pos_y.append(qmatoms.atoms[i].xy)
            qm_pos_z.append(qmatoms.atoms[i].xz)

        # qm_pos_x = self.fin.qm_position[:, 0]
        # qm_pos_y = self.fin.qm_position[:, 1]
        # qm_pos_z = self.fin.qm_position[:, 2]
        # qm_element = self.fin.qm_element
        # qm_atom_charge = self.fin.qm_atom_charge
        # qm_index = self.fin.qm_index
        box = qmatoms.get_box()[0]
        if box[3] == box[4] and box[4] == box[5]:
            cell = np.zeros((3,3))
            cell[0][0] = box[0]
            cell[1][1] = box[1]
            cell[2][2] = box[2]
        else:
            print('Box is not rectangular')
            exit(1)
        self.cell_basis = cell

        self.qm_atoms = QMAtoms(qm_pos_x, qm_pos_y, qm_pos_z, qm_element,
                                qm_atom_charge, qm_index, self.cell_basis)

        # Process MM atoms
        if self.n_mm_atoms > 0:
            mm_pos_x = []
            mm_pos_y = []
            mm_pos_z = []
            mm_atom_charge = []
            mm_element = []
            mm_index = []
            for i in range(len(mmatoms.atoms)):
                mm_index.append(i + self.n_qm_atoms)
                mm_atom_charge.append(mmatoms.atoms[i].charge)
                mm_pos_x.append(mmatoms.atoms[i].xx)
                mm_pos_y.append(mmatoms.atoms[i].xy)
                mm_pos_z.append(mmatoms.atoms[i].xz)
                mm_element.append(pmd.periodic_table.element_by_mass(mmatoms.atoms[i].mass))

                # mm_pos_x = self.fin.mm_position[:, 0]
                # mm_pos_y = self.fin.mm_position[:, 1]
                # mm_pos_z = self.fin.mm_position[:, 2]
                # mm_atom_charge = self.fin.mm_atom_charge
                # mm_index = self.fin.mm_index

            self.mm_atoms = MMAtoms(mm_pos_x, mm_pos_y, mm_pos_z,
                                        mm_atom_charge, mm_index, mm_atom_charge, self.qm_atoms)
            self.mm_atoms.element = mm_element
        else:
            self.mm_atoms = None

    def update_positions(self):
        self.qm_atoms.position = self.fin.qm_position
        self.mm_atoms.position = self.fin.mm_position
        self.step += 1

    def save_results(self):
        """Save the results of QM calculation to file."""
        fout = "qmmm.result"
        if os.path.isfile(fout):
            os.remove(fout)

        with open(fout, 'w') as f:
            #print(str(self.qm_energy))
            energy = self.qm_energy
            f.write(str(energy) + '\n')
            #if self.qm_force is not None:
                #np.savetxt(f, np.column_stack((self.qm_force, self.qm_atoms.charge)), fmt='%22.14e')
                #np.savetxt(f, self.mm_force, fmt='%22.14e')
