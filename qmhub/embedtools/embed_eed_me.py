from .embed_base import EmbedBase


class EmbedEEdME(EmbedBase):

    EMBEDNEAR = 'EEd'
    EMBEDFAR = 'ME'

    def get_mm_charge(self):

        super(EmbedEEdME, self).get_mm_charge()

        self.mm_atoms_near.charge_me = self.mm_atoms_near.charge_comp
        self.mm_atoms_near.charge_eed = self.mm_atoms_near.charge_near

    def get_mm_esp(self):

        mm_esp_near = self.get_mm_esp_eed()

        if self.mm_atoms_near.charge_comp is None:
            return mm_esp_near
        else:
            mm_esp_comp = self.get_mm_esp_me().sum(axis=1)
            return mm_esp_near - mm_esp_comp
