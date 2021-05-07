import numpy as np

from .atombase import AtomBase
from ..electools import ElecQMQM


class QMAtoms(AtomBase):
    """Class to hold QM atoms."""

    def __init__(self, x, y, z, element, charge, index, cell_basis):

        super(QMAtoms, self).__init__(x, y, z, charge, index)

        self._atoms.element = element

        self._elec = ElecQMQM(self, cell_basis)

        # Set initial QM energy and charges
        self._qm_energy = 0.0
        self._qm_pol_energy = 0.0
        self._qm_charge = np.zeros(self.n_atoms)

        # Set the box for PBC conditions
        self._box = cell_basis

    @property
    def qm_energy(self):
        return self._qm_energy

    @qm_energy.setter
    def qm_energy(self, energy):
        self._qm_energy = energy

    @property
    def qm_charge(self):
        return self._get_property(self._qm_charge)

    @qm_charge.setter
    def qm_charge(self, charge):
        self._set_property(self._qm_charge, charge)

    @property
    def qm_pol_energy(self):
        return self._qm_pol_energy

    @qm_pol_energy.setter
    def qm_pol_energy(self, pol_energy):
        self._qm_pol_energy = pol_energy

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, cell_basis):
        self._box = cell_basis
