import math
import numpy as np


class EwaldSum(object):

    def __init__(self, cell_basis, tolerance=1e-4, order='spherical'):

        self.cell_basis = cell_basis
        self.tolerance = tolerance
        self.order = order

        self._threshold = None
        self._volume = None
        self._recip_basis = None

        self._alpha = None
        self._nmax = None
        self._kmax = None

        self._real_vector_rectangular = None
        self._real_lattice_rectangular = None

        self._real_vector_spherical = None
        self._real_lattice_spherical = None

        self._recip_vector_rectangular = None
        self._recip_lattice_rectangular = None
        self._recip_prefactor_rectangular = None

        self._recip_vector_spherical = None
        self._recip_lattice_spherical = None
        self._recip_prefactor_spherical = None

    @property
    def threshold(self):
        if self._threshold is None:
            self._threshold = math.sqrt(-1 * math.log(self.tolerance))
        return self._threshold

    @property
    def volume(self):
        if self._volume is None:
            self._volume = np.linalg.det(self.cell_basis)
        return self._volume

    @property
    def recip_basis(self):
        if self._recip_basis is None:
            self._recip_basis = 2 * np.pi * np.linalg.inv(self.cell_basis).T
        return self._recip_basis

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = math.sqrt(math.pi) / np.diag(self.cell_basis).max()
        return self._alpha

    @property
    def nmax(self):
        if self._nmax is None:
            self._nmax = np.ceil(self.threshold / self.alpha / np.diag(self.cell_basis)).astype(int)
        return self._nmax

    @property
    def kmax(self):
        if self._kmax is None:
            self._kmax = np.ceil(2 * self.threshold * self.alpha / np.diag(self.recip_basis)).astype(int)
        return self._kmax

    @property
    def real_vector_rectangular(self):
        if self._real_vector_rectangular is None:
            self._real_vector_rectangular = self._get_vector(self.nmax)
        return self._real_vector_rectangular

    @property
    def real_vector_spherical(self):
        if self._real_vector_spherical is None:
            self._real_vector_spherical = self.real_vector_rectangular[np.sum(self.real_lattice_rectangular**2, axis=1) <= np.max(self.nmax * np.diag(self.cell_basis))**2]
        return self._real_vector_spherical

    @property
    def real_lattice_rectangular(self):
        if self._real_lattice_rectangular is None:
            self._real_lattice_rectangular = np.dot(self.real_vector_rectangular, self.cell_basis)
        return self._real_lattice_rectangular

    @property
    def real_lattice_spherical(self):
        if self._real_lattice_spherical is None:
            self._real_lattice_spherical = self.real_lattice_rectangular[np.sum(self.real_lattice_rectangular**2, axis=1) <= np.max(self.nmax * np.diag(self.cell_basis))**2]
        return self._real_lattice_spherical

    @property
    def recip_vector_rectangular(self):
        if self._recip_vector_rectangular is None:
            self._recip_vector_rectangular = self._get_vector(self.kmax)

            # Delete the central unit
            self._recip_vector_rectangular = self._recip_vector_rectangular[~np.all(self._recip_vector_rectangular == 0, axis=1)]

        return self._recip_vector_rectangular

    @property
    def recip_vector_spherical(self):
        if self._recip_vector_spherical is None:
            self._recip_vector_spherical = self.recip_vector_rectangular[np.sum(self.recip_lattice_rectangular**2, axis=1) <= np.max(self.kmax * np.diag(self.recip_basis))**2]
        return self._recip_vector_spherical

    @property
    def recip_lattice_rectangular(self):
        if self._recip_lattice_rectangular is None:
            self._recip_lattice_rectangular = np.dot(self.recip_vector_rectangular, self.recip_basis)
        return self._recip_lattice_rectangular

    @property
    def recip_lattice_spherical(self):
        if self._recip_lattice_spherical is None:
            self._recip_lattice_spherical = self.recip_lattice_rectangular[np.sum(self.recip_lattice_rectangular**2, axis=1) <= np.max(self.kmax * np.diag(self.recip_basis))**2]
        return self._recip_lattice_spherical

    @property
    def recip_prefactor_rectangular(self):
        if self._recip_prefactor_rectangular is None:
            k = self.recip_lattice_rectangular
            k2 = np.sum(k * k, axis=1)
            self._recip_prefactor_rectangular = (4 * np.pi / self.volume) * np.exp(-1 * k2 / (4 * self.alpha**2)) / k2
        return self._recip_prefactor_rectangular

    @property
    def recip_prefactor_spherical(self):
        if self._recip_prefactor_spherical is None:
            k = self.recip_lattice_spherical
            k2 = np.sum(k * k, axis=1)
            self._recip_prefactor_spherical = (4 * np.pi / self.volume) * np.exp(-1 * k2 / (4 * self.alpha**2)) / k2
        return self._recip_prefactor_spherical

    @property
    def real_vector(self):
        if self.order.lower() == 'spherical':
            return self.real_vector_spherical
        elif self.order.lower() == 'rectangular':
            return self.real_vector_rectangular

    @property
    def real_lattice(self):
        if self.order.lower() == 'spherical':
            return self.real_lattice_spherical
        elif self.order.lower() == 'rectangular':
            return self.real_lattice_rectangular

    @property
    def recip_vector(self):
        if self.order.lower() == 'spherical':
            return self.recip_vector_spherical
        elif self.order.lower() == 'rectangular':
            return self.recip_vector_rectangular

    @property
    def recip_lattice(self):
        if self.order.lower() == 'spherical':
            return self.recip_lattice_spherical
        elif self.order.lower() == 'rectangular':
            return self.recip_lattice_rectangular

    @property
    def recip_prefactor(self):
        if self.order.lower() == 'spherical':
            return self.recip_prefactor_spherical
        elif self.order.lower() == 'rectangular':
            return self.recip_prefactor_rectangular

    @staticmethod
    def _get_vector(_max):
        vector = np.mgrid[-_max[0]:_max[0]+1, -_max[1]:_max[1]+1, -_max[2]:_max[2]+1]
        vector = np.rollaxis(vector, 0, 4)
        vector = vector.reshape((_max[0]*2 + 1) * (_max[1]*2 + 1) * (_max[2]*2 + 1), 3)

        return vector
