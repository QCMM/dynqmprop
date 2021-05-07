import numpy as np

from .atombase import AtomBase
from ..electools import ElecQMMM


class MMAtoms(AtomBase):
    """Class to hold MM atoms."""

    def __init__(self, x, y, z, charge, index, orig_charge, qm_atoms):

        super(MMAtoms, self).__init__(x, y, z, charge, index)

        self._elec = ElecQMMM(self, qm_atoms)

        # Initialize original MM charges
        self._orig_charge = np.zeros(self.n_atoms, dtype=float)
        self._orig_charge[self._real_indices] = orig_charge

        self._esp_eed = np.zeros(self.n_atoms, dtype=float)

    @property
    def orig_charge(self):
        return self._get_property(self._orig_charge)

    @property
    def coulomb_mask(self):
        return self._get_property(self._elec.coulomb_mask)

    @coulomb_mask.setter
    def coulomb_mask(self, mask):
        self._set_property(self._elec.coulomb_mask, mask)

    @property
    def esp_eed(self):
        return self._get_property(self._esp_eed)

    @esp_eed.setter
    def esp_eed(self, esp):
        self._set_property(self._esp_eed, esp)
