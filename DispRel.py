### This python file defines varous dispersion relations

import numpy as np
import scipy.constants as c
from scipy.special import roots_legendre

# multipole expansion of the plasma dispersion function
def Z_multipole_expansion(z): 
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

# Cold electromagnetic LH wave
def D_LH_cold(omg, k, ne, ni, B, mi, Zi):
    S = 1
    D = 0
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    S = S - omgpe2/(omg**2 - omgce**2)
    D = D - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        mi = [mi]
        Zi = [Zi]
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        S = S - omgpi2/(omg**2 - omgci**2)
        D = D + np.sign(Zi[n])*(omgci/omg)*omgpi2/(omg**2 - omgci**2)
    return S*k**2 - (omg/c.c)**2*(S**2-D**2)

# Lower hybrid freuency
def omg_LH(B, ne, ni, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        mi = [mi]
        Zi = [Zi]
    LH_numerator = 0
    LH_denominator = omgce**2 + omgpe2
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        LH_numerator = LH_numerator + omgpe2*omgci**2 + omgce**2*(omgci**2+omgpi2)
        LH_denominator = LH_denominator + omgci**2 + omgpi2
    omgLH2 = LH_numerator/LH_denominator
    return np.sqrt(omgLH2)

# Explicit equation for elecrtostatic LH wave (eq. (18)) is used as initial guess electrostatic 
def omg_LH_electrostatic(k, B, ne, ni, Te, Ti, mi, Zi):
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgUH2 = omgpe2+omgce**2
    vTe2 = 2.0*Te/c.m_e
    omgLH2 = omg_LH(B, ne, ni, mi, Zi)**2
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    omg_LHi2_vTi2_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2  = 2.0*Ti[n]/mi[n]
        omg_LHi2 = (omgpe2*omgci**2 + omgce**2*(omgci**2+omgpi2)) / (omgce**2 + omgpe2 + omgci**2 + omgpi2)
        omg_LHi2_vTi2_sum = omg_LHi2_vTi2_sum + omg_LHi2*vTi2
    return np.sqrt(omgLH2 / (1 - 3/2 *(omgpe2*vTe2/(4*omgce**2*omgUH2) + omg_LHi2_vTi2_sum/omgLH2**2)*k**2))

# Integrant used for calculating IBWs
def integrand(ψ, x0, y): # integrant used to wvaluate eq. (3)
    return np.sin(ψ*y) * np.sin(ψ) * np.exp(-x0*(np.cos(ψ) + 1.0))

# Roots of Gauss-Legendre quadratures
Nquad = 100 # Number of quadrature points. Must be increased if the LH branch is at a very high harmonic
t, w = roots_legendre(Nquad)
ψ = (t + 1.0)*np.pi/2.0

# eq. (1) with K_sigma form eq. (3)
def D_IBW(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    K_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    K_i_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        lami = (k/omgci)**2*vTi2/2.0
        yi = omg/omgci
        _ψ, _yi = np.meshgrid(ψ, yi)
        _ψ, _lami = np.meshgrid(ψ, lami)
        integrali = np.pi/2.0 * np.matmul(integrand(_ψ, _lami, _yi), w)
        K_i = omgpi2/omgci**2*integrali/np.sin(np.pi*_yi[:,0])
        K_i_sum = K_i_sum + K_i
    return (1.0 + K_e + K_i_sum)*k**2 

# eq. (25) with K_sigma form eq. (3)
def D_IBW_modified(omg, k, ne, ni, Te, Ti, B, mi, Zi):
    S = 1
    D = 0
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    K_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    S = S - omgpe2/(omg**2 - omgce**2)
    D = D - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    K_i_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        lami = (k/omgci)**2*vTi2/2.0
        yi = omg/omgci
        _ψ, _yi = np.meshgrid(ψ, yi)
        _ψ, _lami = np.meshgrid(ψ, lami)
        integrali = np.pi/2.0 * np.matmul(integrand(_ψ, _lami, _yi), w)
        K_i = omgpi2/omgci**2*integrali/np.sin(np.pi*_yi[:,0])
        K_i_sum = K_i_sum + K_i
        S = S - omgpi2/(omg**2 - omgci**2)
        D = D + np.sign(Zi[n])*(omgci/omg)*omgpi2/(omg**2 - omgci**2)
    return (1.0 + K_e + K_i_sum)*k**2 - (omg/c.c)**2*(S**2-D**2)

# eq. 16
def D_LH_noncorrected(omg, k, ne, ni, Te, Ti, B, mi, Zi):
    S = 1
    D = 0
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgUH2 = omgpe2 + omgce**2
    vTe2 = 2.0*Te/c.m_e
    one_plus_K_e = omgUH2/omgce**2 - 3/8*omgpe2*vTe2*k**2/omgce**4
    S = S - omgpe2/(omg**2 - omgce**2)
    D = D - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    K_i_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        zeta = omg/(k*np.sqrt(vTi2))
        K_i = 2*omgpi2/(vTi2*k**2)*(1+zeta*np.real(Z_multipole_expansion(zeta)))
        K_i_sum = K_i_sum + K_i
        S = S - omgpi2/(omg**2 - omgci**2)
        D = D + np.sign(Zi[n])*(omgci/omg)*omgpi2/(omg**2 - omgci**2)
    return (one_plus_K_e + K_i_sum)*k**2 - (omg/c.c)**2*(S**2-D**2)

# eq. (26)
def D_LH(omg, k, ne, ni, Te, Ti, B, mi, Zi):
    S = 1
    D = 0
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    K_e_part_2 = - 3/8*omgpe2*vTe2*k**2/omgce**4
    S = S - omgpe2/(omg**2 - omgce**2)
    D = D - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    K_i_part_2_sum = 0
    LH_numerator = 0
    LH_denominator = omgce**2 + omgpe2
    omgpi2_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        zeta = omg/(k*np.sqrt(vTi2))
        omgpi2_sum = omgpi2_sum + omgpi2
        LH_numerator = LH_numerator + omgpe2*omgci**2 + omgce**2*(omgci**2+omgpi2)
        LH_denominator = LH_denominator + omgci**2 + omgpi2
        K_i_part_2 = 2*omgpi2/(vTi2*k**2)*(1+zeta*np.real(Z_multipole_expansion(zeta)))
        K_i_part_2_sum = K_i_part_2_sum + K_i_part_2
        S = S - omgpi2/(omg**2 - omgci**2)
        D = D + np.sign(Zi[n])*(omgci/omg)*omgpi2/(omg**2 - omgci**2)
    omgLH2 = LH_numerator/LH_denominator
    hybrid_term = omgpi2_sum/omgLH2
    return (hybrid_term + K_e_part_2 + K_i_part_2_sum)*k**2 - (omg/c.c)**2*(S**2-D**2)


# eq. (21)
def D_LH_electrostatic(omg, k, ne, ni, Te, Ti, B, mi, Zi):
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    K_e_part_2 = - 3/8*omgpe2*vTe2*k**2/omgce**4
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    K_i_part_2_sum = 0
    LH_numerator = 0
    LH_denominator = omgce**2 + omgpe2
    omgpi2_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        zeta = omg/(k*np.sqrt(vTi2))
        omgpi2_sum = omgpi2_sum + omgpi2
        LH_numerator = LH_numerator + omgpe2*omgci**2 + omgce**2*(omgci**2+omgpi2)
        LH_denominator = LH_denominator + omgci**2 + omgpi2
        K_i_part_2 = 2*omgpi2/(vTi2*k**2)*(1+zeta*np.real(Z_multipole_expansion(zeta)))
        K_i_part_2_sum = K_i_part_2_sum + K_i_part_2
    omgLH2 = LH_numerator/LH_denominator
    hybrid_term = omgpi2_sum/omgLH2
    return (hybrid_term + K_e_part_2 + K_i_part_2_sum)*k**2

# eq. (27)
def D_LH_approx(omg, k, ne, ni, Te, Ti, B, mi, Zi): 
    S = 1
    D = 0
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    S = S - omgpe2/(omg**2 - omgce**2)
    D = D - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    LH_numerator = 0
    LH_denominator = omgce**2 + omgpe2
    omgpi2_sum = 0
    omgpi2_vTi2_sum = 0
    for n in range(len(ni)):
        omgpi2 = Zi[n]**2*c.e**2*ni[n]/(c.epsilon_0*mi[n])
        omgci = np.abs(Zi[n])*B*c.e/mi[n]
        vTi2 = 2.0*Ti[n]/mi[n]
        omgpi2_sum = omgpi2_sum + omgpi2
        omgpi2_vTi2_sum = omgpi2_vTi2_sum + omgpi2*vTi2
        LH_numerator = LH_numerator + omgpe2*omgci**2 + omgce**2*(omgci**2+omgpi2)
        LH_denominator = LH_denominator + omgci**2 + omgpi2
        S = S - omgpi2/(omg**2 - omgci**2)
        D = D + np.sign(Zi[n])*(omgci/omg)*omgpi2/(omg**2 - omgci**2)
    omgLH2 = LH_numerator/LH_denominator
    
    return ((1/omgLH2 - 1/omg**2)*omgpi2_sum -3/2*k**2*( omgpe2*vTe2/(4*omgce**4) + omgpi2_vTi2_sum/omgLH2**2))*k**2 - (omg/c.c)**2*(S**2-D**2)


# EBW dispersion relation
def D_EBW(omg, k, ne, Te, B): 
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    K_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    #print(K_e)
    return (1.0 + K_e)*k**2

# EBW dispersion relation modified with cold electromagnetic response
def D_EBW_modified(omg, k, ne, Te, B):
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    vTe2 = 2.0*Te/c.m_e
    lame = (k/omgce)**2*vTe2/2.0
    ye = omg/omgce
    _ψ, _ye = np.meshgrid(ψ, ye)
    _ψ, _lame = np.meshgrid(ψ, lame)
    integrale = np.pi/2.0 * np.matmul(integrand(_ψ, _lame, _ye), w)
    K_e = omgpe2/omgce**2*integrale/np.sin(np.pi*_ye[:,0])
    S = 1 - omgpe2/(omg**2 - omgce**2)
    D = - (omgce/omg)*omgpe2/(omg**2 - omgce**2)
    return ((1.0 + K_e)*k**2 - (omg/c.c)**2*(S**2-D**2))


from scipy.special import ive

def ivpe(v,z):
    if v == 0:
        return (ive(v,z) + ive(v+1.0,z))/1.0
    else:
        return (ive(v-1.0,z) + ive(v+1.0,z))/2.0

# kinetic dispersion relation with electron response for EBWs
def D_full(omg, k, ne, Te, B):
    N_harm = 30
    vTe = np.sqrt(2.0*Te/c.m_e)
    omgce = -c.e*B/c.m_e
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    lame = k**2*vTe**2/(omgce**2*2.0)
    nn = np.arange(-N_harm,N_harm+1)
    ive_e_list = np.zeros(len(nn))
    ivpe_e_list = np.zeros(len(nn))
    for i in range(len(nn)):
        if nn[i]<0:
            ive_e_list[i] = ive(-nn[i],lame)
            ivpe_e_list[i] = ivpe(-nn[i],lame)
        else:
            ive_e_list[i] = ive(nn[i],lame)
            ivpe_e_list[i] = ivpe(nn[i],lame)

    M_0 = (-2.0*omgpe2/omg*lame*np.sum((ive_e_list-ivpe_e_list)/(omg+nn*omgce)))
    M_1 = (1.0 - omgpe2/(omg*lame)*np.sum(nn**2*ive_e_list/(omg+nn*omgce)))
    M_2 = (1.0j*omgpe2/(omg)*np.sum(nn*(ive_e_list-ivpe_e_list)/(omg+nn*omgce)))
    disp_rel = M_1*k**2 - omg**2/c.c**2*(M_0*M_1 + M_1**2 + M_2**2)
    return np.real(disp_rel)
