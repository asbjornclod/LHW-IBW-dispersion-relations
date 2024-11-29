import numpy as np
import scipy.constants as c
from scipy.optimize import fsolve
import DispRel

def omg_LH(B, ne, ni, mi, Zi):
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*B*c.e/mi
    omgLH = np.sqrt(omgci*omgce*(omgpi2+omgpe2+omgci*omgce)/(omgci**2+omgce**2+omgpi2+omgpe2))
    return omgLH

# Explicit equation for elecrtostatic LH wave (eq. (18)) is used as initial guess electrostatic 
def omg_LH_electrostatic(k, B, ne, ni, Te, Ti, mi, Zi):
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgce = c.e*B/c.m_e
    omgpi2 = Zi*c.e**2*ni/(c.epsilon_0*mi)
    omgci = Zi*B*c.e/mi
    omgUH2 = omgpe2+omgce**2
    omgLH2 = omgci*omgce*(omgpi2+omgpe2+omgci*omgce)/(omgci**2+omgce**2+omgpi2+omgpe2)
    vTi2 = 2.0*Ti/mi
    rLi = np.sqrt(vTi2)/(omgci)
    Zi = 1
    return np.sqrt(omgLH2/(1 - 3/2*(omgci**2/omgLH2 + Zi*Te/(4*Ti)*omgpi2/(omgUH2))*k**2*rLi**2))

# Function to evaluate the lower hybrid wave
def evaluate_LHW(ne, ni, Te, Ti, B, mi, Zi, k_min, k_max, k_res, dispersion_relation):
    print('Evaluating {}'.format(dispersion_relation))
    k_list = np.linspace(k_max, k_min, k_res)
    omg_list = []
    omg_init = omg_LH_electrostatic(k_max, B, ne, ni, Te, Ti, mi, Zi)
    print(omg_init/(1e6*2*np.pi))
    for index, k in enumerate(k_list):
        if index == 0:
            omg_guess = omg_init
        else:
            omg_guess = omg_list[-1]
        if dispersion_relation == 'D_LH':
            omg = fsolve(DispRel.D_LH, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
        elif dispersion_relation == 'D_LH_electrostatic':
            omg = fsolve(DispRel.D_LH_electrostatic, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
        elif dispersion_relation == 'D_LH_approx':
            omg = fsolve(DispRel.D_LH_approx, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
        else:
            print('invalid dispersion relation')
            return
        omg_list.append(omg[0])
    omg_list = np.array(omg_list)
    return k_list, omg_list
        
# Function to evaluate the ion Bernstein waves
def evaluate_IBWs(ne, ni, Te, Ti, B, mi, Zi, k_min, k_max, k_res, N_min, N_max, dispersion_relation):
    print('Evaluating {}'.format(dispersion_relation))
    N_harmonics = N_max - N_min + 1
    harmonics = np.flip(np.arange(N_min, N_max+1))
    k_list = np.linspace(k_max, k_min, k_res)
    omg_2d_array = np.ones((N_harmonics,k_res))
    omgci = Zi*c.e*B/mi
    omgLH = omg_LH(B, ne, ni, mi, Zi)
    LH_harmonic = np.floor(omgLH/omgci)
    print('Lower hybrid branch is at the {}th ion cyclotron harmonic'.format(LH_harmonic))
    for id, harmonic in enumerate(harmonics):
        omg_list_n = np.array([])
        for index, k in enumerate(k_list):
            if index > 3:
                if harmonic != LH_harmonic:
                    if np.abs((omg_list_n[-1]-omg_list_n[-2])/(harmonic*omgci)) > 0.01:
                        if harmonic > LH_harmonic:
                            omg_list_n= np.append(omg_list_n[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgci)
                        else:

                            omg_list_n= np.append(omg_list_n[:-4],np.ones(len(k_list)-index+4)*(harmonic+1)*omgci)
                        break
                    if harmonic < LH_harmonic:
                        if np.abs(omg_list_n[-1] - (harmonic+1)*omgci) < 0.001*omgci:
                            omg_list_n= np.append(omg_list_n[:-4],np.ones(len(k_list)-index+4)*(harmonic+1)*omgci)
                            break
                if id != 0:
                    if omg_list_n[-1] > omg_2d_array[id-1, index]:
                        omg_list_n= np.append(omg_list_n[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgci)
                        break
            if index == 0:
                omg_guess = harmonic*omgci*1.01
            else:
                omg_guess = omg_list_n[index - 1]
            if dispersion_relation == 'D_IBW':
                omg = fsolve(DispRel.D_IBW, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
            elif dispersion_relation == 'D_IBW_modified':
                omg = fsolve(DispRel.D_IBW_modified, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
            omg_list_n = np.append(omg_list_n, omg)
        omg_2d_array[id, :] = omg_list_n
    return k_list, omg_2d_array