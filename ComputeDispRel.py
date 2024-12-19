import numpy as np
import scipy.constants as c
from scipy.optimize import fsolve
import DispRel

# Function to evaluate the lower hybrid wave
def evaluate_LHW(ne, ni, Te, Ti, B, mi, Zi, k_min, k_max, k_res, dispersion_relation):
    print('Evaluating {}'.format(dispersion_relation))
    k_list = np.linspace(k_max, k_min, k_res)
    omg_list = []
    omg_init = DispRel.omg_LH_electrostatic(k_max, B, ne, ni, Te, Ti, mi, Zi)
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
        elif dispersion_relation == 'D_LH_cold':
            omg = fsolve(DispRel.D_LH_cold, omg_guess, args=(k, ne, ni, B, mi, Zi))
        else:
            print('invalid dispersion relation')
            return
        omg_list.append(omg[0])
    omg_list = np.array(omg_list)
    return k_list, omg_list
        
# Function to evaluate ion Bernstein waves
def evaluate_IBWs(ne, ni, Te, Ti, B, mi, Zi, k_min, k_max, k_res, N_min, N_max, dispersion_relation):
    print('Evaluating {}'.format(dispersion_relation))
    k_list = np.linspace(k_max, k_min, k_res)
    omg_array = np.array([])
    try:
        len(ni)
    except TypeError:
        ni = [ni]
        Ti = [Ti]
        mi = [mi]
        Zi = [Zi]
    
    omgLH = DispRel.omg_LH(B, ne, ni, mi, Zi)
    all_omgcis = np.array([])
    delta_omg_LH = np.array([])
    for n in range(len(ni)):
        N_max_n = N_max*mi[n]/mi[0]
        N_min_n = N_min*mi[n]/mi[0]
        N_harmonics = np.ceil(N_max_n - N_min_n + 1)
        harmonics = np.arange(N_min_n, N_max_n+1)
        omgci = Zi[n]*c.e*B/mi[n]
        omgcis = omgci*harmonics
        all_omgcis = np.append(all_omgcis, omgcis)
        try:
            delta = omgLH - np.max(omgcis[omgcis<omgLH])
            delta_omg_LH = np.append(delta_omg_LH, delta)
        except ValueError:
            pass

    all_omgcis = np.sort(all_omgcis)
    unique_omgcis = [all_omgcis[0]]  # Start with the first number
    for num in all_omgcis[1:]:
        if abs(num - unique_omgcis[-1]) > 1e3:
            unique_omgcis.append(num)
    unique_omgcis = np.array(unique_omgcis)
    try:
        n_LH = np.argmin(delta_omg_LH)
    except ValueError:
        n_LH = 0

    for n in range(len(ni)):
        N_max_n = N_max*mi[n]/mi[0]
        N_min_n = N_min*mi[n]/mi[0]
        N_harmonics = int(np.ceil(N_max_n - N_min_n + 1))
        harmonics = np.flip(np.arange(N_min_n, N_max_n+1))
        omg_array_n = np.ones((N_harmonics,k_res))
        omgci = Zi[n]*c.e*B/mi[n]
        
        LH_harmonic = np.floor(omgLH/omgci)
        if n == n_LH:
            if len(ni) == 1:
                print('Lower hybrid branch is at the {}th ion cyclotron harmonic'.format(LH_harmonic))
            else:
                print('Lower hybrid branch is at the {}th ion cyclotron harmonic of ion species number {}'.format(LH_harmonic, n+1))
        for id, harmonic in enumerate(harmonics):
            omgci_index = np.argmin(np.abs(unique_omgcis - omgci*harmonic))
            omg_list_id = np.array([])
            for index, k in enumerate(k_list):
                if index > 3:
                    if harmonic != LH_harmonic or n != n_LH:
                        if np.abs((omg_list_id[-1]-omg_list_id[-2])/(harmonic*omgci)) > 0.01:
                            if harmonic > LH_harmonic:
                                omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgci)
                            else:
                                omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*unique_omgcis[omgci_index + 1])
                            break
                        if harmonic < LH_harmonic:
                            if np.abs(omg_list_id[-1] - unique_omgcis[omgci_index + 1]) < 0.001*omgci:
                                omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*unique_omgcis[omgci_index + 1])
                                break
                        if harmonic >= LH_harmonic:
                            if omg_list_id[-1] < (harmonic)*omgci:
                                omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgci)
                                break
                    if id != 0:
                        if omg_list_id[-1] > omg_array_n[id-1, index]:
                            if harmonic != LH_harmonic or n != n_LH:
                                omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgci)
                                break
                if index == 0:
                    omg_guess = harmonic*omgci + 0.1*omgci
                else:
                    if harmonic == LH_harmonic and n == n_LH:
                        omg_guess = omg_list_id[index - 1] - 0.000001*omgci
                    else:
                        omg_guess = omg_list_id[index - 1]

                if dispersion_relation == 'D_IBW':
                    omg = fsolve(DispRel.D_IBW, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
                elif dispersion_relation == 'D_IBW_modified':
                    omg = fsolve(DispRel.D_IBW_modified, omg_guess, args=(k, ne, ni, Te, Ti, B, mi, Zi))
                omg_list_id = np.append(omg_list_id, omg)
            omg_array_n[id, :] = omg_list_id
        if n == 0:
            omg_array = omg_array_n
        else:
            omg_array = np.concatenate((omg_array, omg_array_n), axis = 0)
    return k_list, omg_array

# Function to evaluate electron Bernstein waves
def evaluate_EBWs(ne, Te, B, k_min, k_max, k_res, N_min, N_max, dispersion_relation):
    print('Evaluating {}'.format(dispersion_relation))
    k_list = np.linspace(k_max, k_min, k_res)
    N_harmonics = int(np.ceil(N_max - N_min + 1))
    harmonics = np.flip(np.arange(N_min, N_max+1))
    omg_array = np.ones((N_harmonics,k_res))
    omgce = c.e*B/c.m_e
    omgpe2 = c.e**2*ne/(c.epsilon_0*c.m_e)
    omgUH = np.sqrt(omgce**2 + omgpe2)
    UH_harmonic = np.floor(omgUH/omgce)
    if UH_harmonic == 1:
        print('Upper hybrid branch is at the 1st electron cyclotron harmonic')
    elif UH_harmonic == 2:
        print('Upper hybrid branch is at the 2nd electron cyclotron harmonic')
    elif UH_harmonic == 3:
        print('Upper hybrid branch is at the 3rd electron cyclotron harmonic')
    else:
        print('Upper hybrid branch is at the {}th electron cyclotron harmonic'.format(UH_harmonic))
        
    for id, harmonic in enumerate(harmonics):
        if harmonic > UH_harmonic:
            sgn = 1
        else:
            sgn = 1
        print('harmonic {}'.format(harmonic))
        omg_list_id = np.array([])
        for index, k in enumerate(k_list):
            if index > 3:
                if np.abs((omg_list_id[-1]-omg_list_id[-2])/(harmonic*omgce)) > 0.01 and dispersion_relation == 'D_EBW':
                                if harmonic > UH_harmonic:
                                    omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*(harmonic)*omgce)
                                else:
                                    omg_list_id= np.append(omg_list_id[:-4],np.ones(len(k_list)-index+4)*(harmonic+1)*omgce)
                                break
            if index == 0:
                omg_guess = harmonic*omgce + 0.1*omgce
            else:
                omg_guess = omg_list_id[index - 1] + sgn*0.0001*omgce
            if dispersion_relation == 'D_EBW':
                omg = fsolve(DispRel.D_EBW, omg_guess, args=(k, ne, Te, B))
            elif dispersion_relation == 'D_EBW_electromagnetic':
                omg = fsolve(DispRel.D_full, omg_guess, args=(k, ne, Te, B))
                if index > 0:
                    if np.abs((omg-omg_list_id[index - 1])/omg_list_id[index - 1]) > 0.001:
                        sgn = - sgn
                        omg_guess = omg_list_id[index -1] + sgn*0.0005*omgce
                        omg = fsolve(DispRel.D_full, omg_guess, args=(k, ne, Te, B))
            omg_list_id = np.append(omg_list_id, omg)
        omg_array[id, :] = omg_list_id
    return k_list, omg_array