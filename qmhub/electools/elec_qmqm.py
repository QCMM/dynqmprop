import math
import numpy as np

from .ewaldsum import EwaldSum
from .elec_core import elec_core_qmqm as elec_core

from .. import units

class ElecQMQM(object):

    def __init__(self, qm_atoms, cell_basis):

        self._qm_position = qm_atoms.position
        self._cell_basis = cell_basis
        self._init_property()

    def _init_property(self):
        self._ewald = None

        self._rij = None
        self._dij2 = None
        self._dij = None

        self._coulomb_esp = None
        self._coulomb_efield = None
        self._ewald_esp = None
        self._ewald_efield = None
        self._ewald_real_esp = None
        self._ewald_real_efield = None
        self._ewald_recip_esp = None
        self._ewald_recip_efield = None
        self._ewald_self_esp = None

    @property
    def ewald(self):
        if self._ewald is None:
            if self._cell_basis is not None:
                self._ewald = EwaldSum(self._cell_basis)
            else:
                self._ewald = None
        return self._ewald

    @property
    def rij(self):
        if self._rij is None:
            self._rij = (self._qm_position[np.newaxis, :, :]
                        - self._qm_position[:, np.newaxis, :])
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
                self._ewald_esp = self.ewald_real_esp + self.ewald_recip_esp - self.ewald_self_esp
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

    @property
    def ewald_self_esp(self):
        if self._ewald_self_esp is None:
            if self.ewald is not None:
                self._ewald_self_esp = elec_core.get_ewald_self_esp(self.dij, self.ewald.alpha, units.KE)
        return self._ewald_self_esp
