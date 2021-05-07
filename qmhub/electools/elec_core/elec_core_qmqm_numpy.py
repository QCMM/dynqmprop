import math
import numpy as np
from scipy import special


def _get_coulomb(rij, dij, ke):
    if dij != 0.0:
        return ke / dij, ke * rij / dij**3
    else:
        return 0.0, np.zeros(3, dtype=float)


get_coulomb = np.vectorize(_get_coulomb, signature='(m),(),()->(),(m)')


def _get_ewald_real(rij, n, alpha, ke):
    r = rij + n
    d2 = np.sum(r**2, axis=1)
    r = r[d2 != 0.0]
    d2 = d2[d2 != 0.0]
    d = np.sqrt(d2)
    prod = special.erfc(alpha * d) / d
    prod2 = prod / d2 + 2 * alpha * np.exp(-1 * alpha**2 * d2) / math.sqrt(math.pi) / d2
    return ke * prod.sum(), ke * (prod2[:, np.newaxis] * r).sum(axis=0)


get_ewald_real = np.vectorize(_get_ewald_real, signature='(m),(n,m),(),()->(),(m)')


def _get_ewald_recip(rij, k, fac, ke):
    kr = np.dot(k, rij)
    prod = fac * np.sin(kr)
    return ke * np.sum(fac * np.cos(kr)), ke * (prod[:, np.newaxis] * k).sum(axis=0)


get_ewald_recip = np.vectorize(_get_ewald_recip, signature='(m),(n,m),(n),()->(),(m)')


def _get_ewald_self_esp(dij, alpha, ke):
    if dij == 0.0:
        return 2 * ke * alpha / math.sqrt(math.pi)
    else:
        return 0.0


get_ewald_self_esp = np.vectorize(_get_ewald_self_esp, signature='(),(),()->()')
