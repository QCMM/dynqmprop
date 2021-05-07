import numpy as np

from .embed_base import EmbedBase


class EmbedEEdEEq(EmbedBase):

    EMBEDNEAR = 'EEd'
    EMBEDFAR = 'EEq'

    @staticmethod
    def check_unitcell(system):
        if system.n_atoms != system.n_real_qm_atoms + system.n_real_mm_atoms:
            raise ValueError("Unit cell is not complete.")

    def get_near_mask(self):
        return np.array((self.mm_atoms.dij_min <= self.qmCutoff), dtype=bool)

    def get_qm_charge_eeq(self):
        if self.qmRefCharge:
            self.qm_charge_eeq = self.qm_atoms.charge
        else:
            raise NotImplementedError()

    def get_mm_charge(self):

        super(EmbedEEdEEq, self).get_mm_charge()

        self.mm_atoms_near.charge_eeq = self.mm_atoms_near.charge_comp
        self.mm_atoms_near.charge_eed = self.mm_atoms_near.charge_near
        self.mm_atoms_far.charge_eeq = self.mm_atoms_far.orig_charge

    def get_mm_esp(self):

        mm_esp_near = self.get_mm_esp_eed()

        if self.mm_atoms_near.charge_comp is None:
            return mm_esp_near
        else:
            mm_esp_comp = self.get_mm_esp_eeq().sum(axis=1)
            return mm_esp_near - mm_esp_comp
