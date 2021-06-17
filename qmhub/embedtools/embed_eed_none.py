from .embed_base import EmbedBase


class EmbedEEdNone(EmbedBase):

    EMBEDNEAR = 'EEd'
    EMBEDFAR = 'None'

    def get_mm_charge(self):

        super(EmbedEEdNone, self).get_mm_charge()

        self.mm_atoms_near.charge_eed = self.mm_atoms_near.charge_near

    def get_mm_esp(self):

        return self.get_mm_esp_eed()
