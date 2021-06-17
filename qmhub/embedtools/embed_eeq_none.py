from .embed_base import EmbedBase


class EmbedEEqNone(EmbedBase):

    EMBEDNEAR = 'EEq'
    EMBEDFAR = 'None'

    def get_mm_charge(self):

        super(EmbedEEqNone, self).get_mm_charge()

        self.mm_atoms_near.charge_eeq = self.mm_atoms_near.charge_near

    def get_mm_esp(self):

        return self.get_mm_esp_eeq().sum(axis=1)
