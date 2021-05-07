import numpy as np

from .embed_base import EmbedBase


class EmbedEEqEEq(EmbedBase):

    EMBEDNEAR = 'EEq'
    EMBEDFAR = 'EEq'

    @staticmethod
    def check_unitcell(system):
        if system.n_atoms != system.n_real_qm_atoms + system.n_real_mm_atoms:
            raise ValueError("Unit cell is not complete.")

    def get_qm_charge_eeq(self):
        if self.qmRefCharge:
            self.qm_charge_eeq = self.qm_atoms.charge
        else:
            raise NotImplementedError()

    def scale_mm_charges(self):
        """Scale external point charges."""
        if self.qmSwitchingType is not None:
            raise ValueError("Switching MM charges is not necessary here.")
        else:
            self.charge_scale = np.ones(self.mm_atoms_near.n_atoms, dtype=float)
            self.scale_deriv = np.zeros((self.mm_atoms_near.n_atoms, 3), dtype=float)
            self.mm_atoms_near.charge_near = self.mm_atoms_near.charge
            self.mm_atoms_near.charge_comp = None

    def get_mm_charge(self):

        super(EmbedEEqEEq, self).get_mm_charge()

        self.mm_atoms_near.charge_eeq = self.mm_atoms_near.charge
        self.mm_atoms_far.charge_eeq = self.mm_atoms_far.orig_charge

    def get_mm_esp(self):

        return self.get_mm_esp_eeq().sum(axis=1)
