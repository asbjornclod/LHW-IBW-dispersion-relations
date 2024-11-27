### This python file defines varous dispersion relations

import numpy as np
import scipy.constants as c
from scipy.special import roots_legendre

def Z_multipole_expansion(z): # multipole expansion of the plasma dispersion function
    Zeta = np.zeros_like(z, dtype=complex)
    bj = np.array([0.00383968430671409 - 0.0119854387180615j,
    -0.321597857664957 - 0.218883985607935j,
    2.55515264319988 + 0.613958600684469j,
    -2.73739446984183 + 5.69007914897806j])
    cj = np.array([2.51506776338386 - 1.60713668042405j,
    -1.68985621846204 - 1.66471695485661j,
    0.981465428659098 - 1.70017951305004j,
    -0.322078795578047 - 1.71891780447016j])
    bj = np.concatenate((bj, np.conj(bj[::-1])))
    cj = np.concatenate((cj, -np.conj(cj[::-1])))
    idx = np.imag(z) >= 0
    Zeta[~idx] = 2j * np.sqrt(np.pi) * np.exp(-(z[~idx])**2)
    for j in range(len(bj)):
        Zeta[idx] += bj[j] / (z[idx] - cj[j])
        Zeta[~idx] += np.conj(bj[j] / (np.conj(z[~idx]) - cj[j]))
    return Zeta

def integrand(ψ, x0, y): # integrant used to wvaluate eq. (3)
    return np.sin(ψ*y) * np.sin(ψ) * np.exp(-x0*(np.cos(ψ) + 1.0))

# Roots of Gauss-Legendre quadratures
Nquad = 100
t, w = roots_legendre(Nquad)
ψ = (t + 1.0)*np.pi/2.0

# eq. (1) with K_sigma form eq. (3)
def D_IBW(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*B*c.e/mi
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    vTi2 = 2.0*Ti/mi
    lami = (k/omgci)**2*vTi2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    yi = omg/omgci
    _ψ, _yi = np.meshgrid(ψ, yi)
    _ψ, _lami = np.meshgrid(ψ, lami)
    integrali = np.pi/2.0 * np.matmul(integrand(_ψ, _lami, _yi), w)
    K1_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    K1_i = omgpi2/omgci**2*integrali/np.sin(np.pi*_yi[:,0])
    return (1.0 + K1_e + K1_i)*k**2 

# eq. (19) with K_sigma form eq. (3)
def D_IBW_modified(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*B*c.e/mi
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    vTi2 = 2.0*Ti/mi
    lami = (k/omgci)**2*vTi2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    yi = omg/omgci
    _ψ, _yi = np.meshgrid(ψ, yi)
    _ψ, _lami = np.meshgrid(ψ, lami)
    integrali = np.pi/2.0 * np.matmul(integrand(_ψ, _lami, _yi), w)
    S = 1 - omgpe2/(omg**2 - omgce**2) - omgpi2/(omg**2 - omgci**2)
    D = -(omgce/omg)*omgpe2/(omg**2 - omgce**2) + (omgci/omg)*omgpi2/(omg**2 - omgci**2)
    K1_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    K1_i = omgpi2/omgci**2*integrali/np.sin(np.pi*_yi[:,0])
    return (1.0 + K1_e + K1_i)*k**2 - (omg/c.c)**2*(S**2-D**2)

# eq. (22)
def D_LH(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*B*c.e/mi
    vTe2 = 2.0*Te/c.m_e
    rLe = np.sqrt(vTe2)/(omgce)
    vTi2 = 2.0*Ti/mi
    omgLH2 = omgci*omgce*(omgpi2+omgpe2+omgci*omgce)/(omgci**2+omgce**2+omgpi2+omgpe2)
    one_plus_K1_E = omgpi2/omgLH2 - 3/8*k**2*rLe**2
    zeta = omg/(k*np.sqrt(vTi2))
    K1_i = 2*omgpi2/(vTi2*k**2)*(1+zeta*np.real(Z_multipole_expansion(zeta)))
    S = 1 - omgpe2/(omg**2 - omgce**2) - omgpi2/(omg**2 - omgci**2)
    D = -(omgce/omg)*omgpe2/(omg**2 - omgce**2) + (omgci/omg)*omgpi2/(omg**2 - omgci**2)
    return (one_plus_K1_E + K1_i)*k**2 - (omg/c.c)**2*(S**2-D**2)

# eq. (17)
def D_LH_electrostatic(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*c.e*B/mi
    vTe2 = 2.0*Te/c.m_e
    rLe = np.sqrt(vTe2)/(omgce)
    vTi2 = 2.0*Ti/mi
    omgLH2 = omgci*omgce*(omgpi2+omgpe2+omgci*omgce)/(omgci**2+omgce**2+omgpi2+omgpe2)
    one_plus_K1_E = omgpi2/omgLH2 - 3/8*k**2*rLe**2
    zeta = omg/(k*np.sqrt(vTi2))
    K1_i = 2*omgpi2/(vTi2*k**2)*(1+zeta*np.real(Z_multipole_expansion(zeta)))
    return (one_plus_K1_E + K1_i)*k**2

# eq. (23)
def D_LH_approx(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgUH2 = omgpe2 + omgce**2
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*c.e*B/mi
    vTi2 = 2.0*Ti/mi
    rLi = np.sqrt(vTi2)/(omgci)
    omgLH2 = omgci*omgce*(omgpi2+omgpe2+omgci*omgce)/(omgci**2+omgce**2+omgpi2+omgpe2)
    S = 1 - omgpe2/(omg**2 - omgce**2) - omgpi2/(omg**2 - omgci**2)
    D = -(omgce/omg)*omgpe2/(omg**2 - omgce**2) + (omgci/omg)*omgpi2/(omg**2 - omgci**2)
    return omgpi2/omgLH2*(1-omgLH2/omg**2-3/2*(omgci**2/omgLH2+Zi*Te/(4*Ti)*omgpi2/(omgUH2))*k**2*rLi**2)*k**2 - (omg/c.c)**2*(S**2-D**2)