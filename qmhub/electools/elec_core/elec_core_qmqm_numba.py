import math
import numba as nb


SQRTPI = math.sqrt(math.pi)


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                '(m),(),()->(),(m)', nopython=True, target='parallel', cache=True)
def get_coulomb(rij, dij, ke, coulomb_esp, coulomb_efield):
    if dij[0] != 0.0:
        coulomb_esp[0] = ke[0] / dij[0]
        coulomb_efield[0] = ke[0] * rij[0] / dij[0]**3
        coulomb_efield[1] = ke[0] * rij[1] / dij[0]**3
        coulomb_efield[2] = ke[0] * rij[2] / dij[0]**3
    else:
        coulomb_esp[0] = 0.0
        coulomb_efield[0] = 0.0
        coulomb_efield[1] = 0.0
        coulomb_efield[2] = 0.0


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                '(m),(n,m),(),()->(),(m)', nopython=True, target='parallel', cache=True)
def get_ewald_real(rij, n, alpha, ke, ewald_real_esp, ewald_real_efield):
    ewald_real_esp[0] = 0.0
    ewald_real_efield[0] = 0.0
    ewald_real_efield[1] = 0.0
    ewald_real_efield[2] = 0.0

    alpha2 = alpha[0]**2

    for i in range(len(n)):
        r_0 = rij[0] + n[i, 0]
        r_1 = rij[1] + n[i, 1]
        r_2 = rij[2] + n[i, 2]
        d = math.sqrt(r_0**2 + r_1**2 + r_2**2)
        if d != 0:
            d2 = d**2
            prod = math.erfc(alpha[0] * d) / d
            prod2 = prod / d2 + 2 * alpha[0] * math.exp(-1 * alpha2 * d2) / SQRTPI / d2
            ewald_real_esp[0] += prod
            ewald_real_efield[0] += prod2 * r_0
            ewald_real_efield[1] += prod2 * r_1
            ewald_real_efield[2] += prod2 * r_2

    ewald_real_esp[0] *= ke[0]
    ewald_real_efield[0] *= ke[0]
    ewald_real_efield[1] *= ke[0]
    ewald_real_efield[2] *= ke[0]


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                '(m),(n,m),(n),()->(),(m)', nopython=True, target='parallel', cache=True)
def get_ewald_recip(rij, k, fac, ke, ewald_recip_esp, ewald_recip_efield):
    ewald_recip_esp[0] = 0.0
    ewald_recip_efield[0] = 0.0
    ewald_recip_efield[1] = 0.0
    ewald_recip_efield[2] = 0.0

    for n in range(len(k)):
        kr = k[n, 0] * rij[0] + k[n, 1] * rij[1] + k[n, 2] * rij[2]
        prod = fac[n] * math.sin(kr)
        ewald_recip_esp[0] += fac[n] * math.cos(kr)
        ewald_recip_efield[0] += prod * k[n, 0]
        ewald_recip_efield[1] += prod * k[n, 1]
        ewald_recip_efield[2] += prod * k[n, 2]

    ewald_recip_esp[0] *= ke[0]
    ewald_recip_efield[0] *= ke[0]
    ewald_recip_efield[1] *= ke[0]
    ewald_recip_efield[2] *= ke[0]


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                '(),(),()->()', nopython=True, target='cpu', cache=True)
def get_ewald_self_esp(dij, alpha, ke, ewald_self_esp):
    if dij[0] == 0.0:
        ewald_self_esp[0] = 2 * ke[0] * alpha[0] / math.sqrt(math.pi)
    else:
        ewald_self_esp[0] = 0.0
