import math
import numpy as np

from .elec_core import elec_core_qmmm as elec_core

from .. import units


class ElecQMMM(object):

    def __init__(self, mm_atoms, qm_atoms):

        self._n_mm_atoms = mm_atoms.n_atoms
        self._n_qm_atoms = qm_atoms.n_atoms
        self._mm_position = mm_atoms.position
        self._qm_position = qm_atoms.position
        self._real_mm_indices = mm_atoms._real_indices
        self._real_qm_indices = qm_atoms._real_indices
        self._box = qm_atoms.box

        self.ewald = qm_atoms._elec.ewald

        # Get array mask to cancel 1-2 and 1-3 interactions for coulomb
        self.coulomb_mask = np.ones((self._n_mm_atoms, self._n_qm_atoms), dtype=bool)

        self._init_property()

    def _init_property(self):
        self._rij = None
        self._dij2 = None
        self._dij = None

        self._dij_min2 = None
        self._dij_min_j = None
        self._dij_min = None

        self._coulomb_esp = None
        self._coulomb_efield = None
        self._ewald_esp = None
        self._ewald_efield = None
        self._ewald_real_esp = None
        self._ewald_real_efield = None
        self._ewald_recip_esp = None
        self._ewald_recip_efield = None

    @property
    def rij(self):
        if self._rij is None:
            self._rij = (self._qm_position[np.newaxis, :, :]
                         - self._mm_position[:, np.newaxis, :])
            # correct for PBC condition
            for mmatom in range(self._n_mm_atoms):
                for qmatom in range(self._n_qm_atoms):
                    for dim in range(3):
                        dist = self._rij[mmatom][qmatom][dim]
                        box_length = self._box[dim][dim]
                        correction = box_length * np.rint(dist / box_length)
                        self._rij[mmatom][qmatom][dim] = dist - correction
        return self._rij

    @property
    def dij2(self):
        if self._dij2 is None:
            self._dij2 = np.sum(self.rij**2, axis=2)
        return self._dij2

    @property
    def dij(self):
        if self._dij is None:
            self._dij = np.sqrt(self.dij2)
        return self._dij

    @property
    def dij_min2(self):
        if self._dij_min2 is None:
            self._dij_min2 = np.zeros(self._n_mm_atoms, dtype=float)
            self._dij_min2[self._real_mm_indices] = self.dij2[self._real_mm_indices, self._real_qm_indices].min(axis=1)
        return self._dij_min2

    @property
    def dij_min_j(self):
        if self._dij_min_j is None:
            self._dij_min_j = -1 * np.ones(self._n_mm_atoms, dtype=int)
            self._dij_min_j[self._real_mm_indices] = self.dij2[self._real_mm_indices, self._real_qm_indices].argmin(axis=1)
        return self._dij_min_j

    @property
    def dij_min(self):
        if self._dij_min is None:
            self._dij_min = np.sqrt(self.dij_min2)
        return self._dij_min

    @property
    def coulomb_esp(self):
        if self._coulomb_esp is None:
            self._coulomb_esp, self._coulomb_efield = elec_core.get_coulomb(self.rij, self.dij, units.KE)
        return self._coulomb_esp

    @property
    def coulomb_efield(self):
        if self._coulomb_efield is None:
            self._coulomb_esp, self._coulomb_efield = elec_core.get_coulomb(self.rij, self.dij, units.KE)
        return self._coulomb_efield

    @property
    def ewald_esp(self):
        if self._ewald_esp is None:
            if self.ewald is not None:
                self._ewald_esp = self.ewald_real_esp + self.ewald_recip_esp
        return self._ewald_esp

    @property
    def ewald_efield(self):
        if self._ewald_efield is None:
            if self.ewald is not None:
                self._ewald_efield = self.ewald_real_efield + self.ewald_recip_efield
        return self._ewald_efield

    @property
    def ewald_real_esp(self):
        if self._ewald_real_esp is None:
            if self.ewald is not None:
                self._ewald_real_esp, self._ewald_real_efield = elec_core.get_ewald_real(self.rij, self.ewald.real_lattice, self.ewald.alpha, units.KE)
        return self._ewald_real_esp

    @property
    def ewald_real_efield(self):
        if self._ewald_real_efield is None:
            if self.ewald is not None:
                self._ewald_real_esp, self._ewald_real_efield = elec_core.get_ewald_real(self.rij, self.ewald.real_lattice, self.ewald.alpha, units.KE)
        return self._ewald_real_efield

    @property
    def ewald_recip_esp(self):
        if self._ewald_recip_esp is None:
            if self.ewald is not None:
                self._ewald_recip_esp, self._ewald_recip_efield = elec_core.get_ewald_recip(self.rij, self.ewald.recip_lattice, self.ewald.recip_prefactor, units.KE)
        return self._ewald_recip_esp

    @property
    def ewald_recip_efield(self):
        if self._ewald_recip_efield is None:
            if self.ewald is not None:
                self._ewald_recip_esp, self._ewald_recip_efield = elec_core.get_ewald_recip(self.rij, self.ewald.recip_lattice, self.ewald.recip_prefactor, units.KE)
        return self._ewald_recip_efield
