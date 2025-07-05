#!/usr/bin/env python
# coding: utf-8

# In[125]:


# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:50 2024

@author: otavi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
from numpy import exp
from scipy.special import factorial
import scipy.optimize as spo
from scipy.optimize import minimize
import os
import math
from itertools import zip_longest
from itertools import combinations_with_replacement,product
from datetime import datetime  
import time
import imageio.v3 as iio
from matplotlib.ticker import FormatStrFormatter
from joblib import Parallel, delayed, parallel_config
from bisect import bisect_left
from scipy.interpolate import CubicSpline
from collections import defaultdict, Counter

hi_values = pd.read_csv('./hi.csv', sep = ';').to_numpy() #importa os valores de hi do arquivo csv e transforma em uma matriz 
# bonding_alpha = pd.read_csv('./bonding_alpha.csv', sep = ';').to_numpy() #importa os valores da energia de ligação da estrutura alpha
# bonding_delta = pd.read_csv('./bonding_delta.csv', sep = ';').to_numpy()#importa os valores da energia de ligação da estrutura delta
# atomic_mass = pd.read_csv('./atomic_mass.csv', sep = ';').to_numpy()#importa os valores da energia de ligação da estrutura delta
composition_vetor = np.zeros(len(hi_values)) #cria um vetor de zeros com a mesma quantidade de coordenadas que entradas de elementos na matriz hi_values
NbC14 = pd.read_csv('./NbC14_Nb21.csv', sep = ';').to_numpy() #importa os valores de hi do arquivo csv e transforma em uma matriz 
R = 0.00831446261815324 #kJ/mol


# # Code

# In[128]:


class Thermodynamics():
    def __init__(self):
        self.G = {}
        self.H = {}
        self.S = {}
        self.S_c = {}
        self.occupation_k_site = {}
        self.mu_H = {}
        self.mu_M = {}
        self.cH_plateau_start = {}
        self.cH_plateau_end = {}
        self.pressure = {}
        self.h_m = None #{}
        self.theta_b = {}
        self.G_eq = {}
        self.dS = {}
        self.dH = {}
        self.dS_c = {}   
        self.pressure_total = {}
        self.cHvals = {}

    def s0(self,T):
            t=T/1000
            A = 33.066178
            B = -11.363417
            C = 11.432816
            D = -2.772874
            E = -0.158558
            G = 172.707974
            S0 = A*ln(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G # Hydrogen standard entropy J/mol of H2- NIST
            S0 = S0/1000 #kJ/mol of H2
            return S0



    

    
    def Enthalpy_DSEM(self,occupation_k_type_sites):
        flat_array_theta_k= self.theta_k_tensor.flatten()
        flat_array_h_k=self.h_k_tensor.flatten()
        h_m = (sum((occupation_k_type_sites * flat_array_theta_k * flat_array_h_k)))
        h = h_m +self.H_M
        return h

    

    def Entropy_DSEM(self,occupation_k_type_sites,T):
        t=T/1000
        A = 33.066178
        B = -11.363417
        C = 11.432816
        D = -2.772874
        E = -0.158558
        G = 172.707974
        S_0 = A*ln(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G # Hydrogen standard entropy J/mol of H2- NIST
        S_0 = S_0/1000 #kJ/mol of H2
        flat_array_theta_k= self.theta_k_tensor.flatten()
        flat_array_h_k=self.h_k_tensor.flatten()
            
        #fh = sum(occupation_k_type_sites*self.k_type_sites_probability)
    
        if not isinstance(self.Nb,np.ndarray):
            Pb=0
        else:

    
            fh_ki = occupation_k_type_sites*self.p_k_tensor.flatten()
            fh_wk = np.sum(fh_ki.reshape(self.j_size,self.i_size),axis=1)
            fb_wk = np.einsum('ijk,ki->j', self.Nb_tensor, fh_ki.reshape(self.j_size,self.i_size))
            #result_array = np.where(alloy_DSEM.C14.theta_k_tensor != 0, 1, 0)
            p_matrix = (self.p_k_tensor.flatten()).reshape(self.j_size,self.i_size)
            fb= np.einsum('ij,i->ij',p_matrix,fb_wk)     
    
    
            
            Pb = 1 - np.exp(-fb).flatten()
        if T>0:
            # Evitar divisão por zero em Ph_k e Pv_k
            if np.any(1 - Pb <= 0):
                raise ValueError("Divisão por zero detectada: 1 - Pb é menor ou igual a zero.")
            
            # Calcular Ph_k e Pv_k
            Ph_k = occupation_k_type_sites / (1 - Pb)
            Pv_k = (1 - occupation_k_type_sites - Pb) / (1 - Pb)
            
            # Substituir valores inválidos antes de calcular o log
            Ph_k = np.clip(Ph_k, 1e-10, None)  # Garante que Ph_k >= 1e-10
            Pv_k = np.clip(Pv_k, 1e-10, None)  # Garante que Pv_k >= 1e-10
            # Ph_k = occupation_k_type_sites/(1-Pb) #correto?
            # Pv_k = (1 - occupation_k_type_sites - Pb)/(1-Pb)
            # Garantir que os valores de entrada para np.log() sejam válidos
            log_Ph_k = np.log(np.clip(Ph_k, 1e-10, None))  # Evita log(0)
            log_Pv_k = np.log(np.clip(Pv_k, 1e-10, None))  # Evita log(0)
            
            # Calcular sc com os logs corrigidos
            sc = -R * sum(flat_array_theta_k * (1 - Pb) * (Ph_k * log_Ph_k + Pv_k * log_Pv_k))
        
            # sc = -R * sum(flat_array_theta_k*(1-Pb)*(Ph_k*ln(Ph_k)+Pv_k*ln(Pv_k)))
            s = sc - ((sum(flat_array_theta_k * occupation_k_type_sites) * S_0)/2)
    
        if T==0:
            s=S_0/2
        return s
        

    def Gibbs_DSEM(self,fk,T):
        if T!= 0:
            DG = self.Enthalpy_DSEM(fk) - T* self.Entropy_DSEM(fk,T)
        if T== 0:
            DG = self.Enthalpy_DSEM(fk)
        return DG
        



    def Gibbs_minimization_DSEM(self,c_h,T): 

        def objective(x,T):
            return self.Gibbs_DSEM(x, T)  # Replace with your actual implementation
        

        
        # Set bounds for variables
        zero_indices = np.where(self.theta_k_tensor.flatten() == 0)[0]
        xl_d = np.full(self.i_size * self.j_size, 1e-6)  # Lower bounds
        xu_d = np.full(self.i_size * self.j_size, 0.999)  # Upper bounds
        xl_d[zero_indices] = 0.0
        xu_d[zero_indices] = 0.0
        # bounds = [(xl_d[i], max_fk[i]) for i in range(len(xl_d))]
        bounds = [(xl_d[i], self.fhk_limit[i]) for i in range(len(xl_d))]
        
        # Combine constraints into a list
        constraints = [
            #{'type': 'ineq', 'fun': inequality_constraints},  # Inequality constraints (g(x) <= 0)
            {'type': 'eq', 'fun':lambda x: c_h - np.sum(x * self.theta_k_tensor.flatten())}       # Equality constraints (h(x) == 0)
        ]
        
        # Initial guess for decision variables (can be random or heuristic-based)
        x0 = np.random.uniform(low=xl_d, high=xu_d)
        # x0[zero_indices]=0
        x0 = np.clip(x0, xl_d, xu_d)
        # x0[zero_indices]=0
        
        # Optimize using scipy.optimize.minimize
        Gibbs_min = spo.minimize(
            fun=self.Gibbs_DSEM,
            x0=x0,
            args=(T,),
            method='SLSQP',  # Sequential Least Squares Programming for constrained optimization
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 2000}  # Display convergence messages
        )
        return Gibbs_min


        
    def ThermodynamicsCalculation_DSEM(self,T,n_threads=-1,step_interpolation=0.00001,fit_exp = False, cH_custom = None):

        def s0(T):
            t=T/1000
            A = 33.066178
            B = -11.363417
            C = 11.432816
            D = -2.772874
            E = -0.158558
            G = 172.707974
            S0 = A*ln(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G # Hydrogen standard entropy J/mol of H2- NIST
            S0 = S0/1000 #kJ/mol of H2
            return S0
        flat_array_theta_k=self.theta_k_tensor.flatten()
        flat_array_h_k=self.h_k_tensor.flatten()
                
        
        begin = time.time()
        print(f"The calculation of {self.name} phase at {T} K has started.")
        self.occupation_k_site[T] = {}

        if fit_exp!= False:
            self.cH = cH_custom
            for i in self.cH:
                if i-0.01 not in self.cH:
                    self.cH = np.append(self.cH, i-0.01)
                if i+0.01 not in self.cH:
                    self.cH = np.append(self.cH, i+0.01)
            self.cH = sorted(self.cH)
                    


        
        Enthalpy_temp = np.zeros(len(self.cH))
        h_m_temp = np.zeros(len(self.cH))
        Entropy_temp = np.zeros(len(self.cH))
        Entropy_config_temp = np.zeros(len(self.cH))
        Gibbs_temp = np.zeros(len(self.cH))
        mu_H_temp = np.zeros(len(self.cH))
        mu_M_temp = np.zeros(len(self.cH))
        theta_b_temp = np.zeros(len(self.cH))
        
            
        
        results_par = Parallel(n_jobs=n_threads)(delayed(self.Gibbs_minimization_DSEM)(c_h,T) for c_h in self.cH)
        for Gibbs_minimized,c_h in zip(results_par,self.cH):
            occupation_k_type_sites = Gibbs_minimized.x 
            ch = sum(Gibbs_minimized.x*flat_array_theta_k)
            Gibbs_temp[np.where(self.cH==c_h)] = Gibbs_minimized.fun
            Enthalpy_temp[np.where(self.cH==c_h)] = self.Enthalpy_DSEM(occupation_k_type_sites)
            h_m_temp[np.where(self.cH==c_h)] = self.Enthalpy_DSEM(occupation_k_type_sites) - self.H_M
            if T != 0:
                Entropy_temp[np.where(self.cH==c_h)] = self.Entropy_DSEM(occupation_k_type_sites,T)
                Entropy_config_temp[np.where(self.cH == c_h)] = self.Entropy_DSEM(occupation_k_type_sites,T) + ((sum(flat_array_theta_k* occupation_k_type_sites) * s0(T))/2)
            occupation_k_type_sites= occupation_k_type_sites.reshape(self.j_size,self.i_size)
            self.occupation_k_site[T][c_h] = {}
            for site_type in self.k_type_sites:
                site_idx = list(self.k_type_sites_probability.keys()).index(site_type)
                self.occupation_k_site[T][c_h][site_type] = occupation_k_type_sites[site_idx]
            #print(c_h, ch)
            
            #theta_b_temp[np.where(self.cH==c_h)] = (self.theta - self.theta/(self.Nb+1))*(1-exp(-self.Nb*sum(occupation_k_type_sites*self.theta_k)))
            

        self.H[T] = Enthalpy_temp
        self.S[T] = Entropy_temp
        self.G[T] = Gibbs_temp
        self.S_c[T] = Entropy_config_temp
        self.h_m[T] = h_m_temp
        #self.theta_b[T] = theta_b_temp

        ############### Calculating chemical potentials
        G_interpolation = CubicSpline(self.cH, self.G[T])
        self.mu_H[T] = G_interpolation(self.cH, nu=1)
        self.mu_M[T] = self.G[T] - self.cH*self.mu_H[T]
        
        #self.mu_H[T] = (self.G[T][2:] - self.G[T][0:-2])/(2*(self.cH_step))
        #self.mu_M[T]= self.G[T][1:-1] - self.cH[1:-1] * self.mu_H[T][0:]  #remember to plot with .cH[1:-1]
        #self.dS[T] = (self.S[T][2:] - self.S[T][0:-2])/(2*(self.cH_step))
        #self.dH[T] = (self.H[T][2:] - self.H[T][0:-2])/(2*(self.cH_step))


        self.cHvals[T] = self.cH
        if step_interpolation != False:
        
            self.cHvals[T] = np.arange(0, self.cH_limit+step_interpolation, step_interpolation)
            self.H[T] = np.interp(self.cHvals[T], self.cH, self.H[T])
            self.S[T] = np.interp(self.cHvals[T], self.cH, self.S[T])
            self.S_c[T] = np.interp(self.cHvals[T], self.cH, self.S_c[T])
            self.G[T] = np.interp(self.cHvals[T], self.cH, self.G[T])
            self.mu_H[T] = np.interp(self.cHvals[T], self.cH[1:-1], self.mu_H[T])
            self.mu_M[T] = np.interp(self.cHvals[T], self.cH[1:-1], self.mu_M[T])

        #if fit_exp != False
             
            

        
        end = time.time()
        tempo = end - begin
        print(f"The calculation takes {round(tempo/60,2)} minutes")

    def Enthalpy_RSOM(self,T):
        self.H[T] = self.h_m * self.cH + self.H_M
        return None

    def Entropy_RSOM(self,T):

        if T!=0:
            t=T/1000
            A = 33.066178
            B = -11.363417
            C = 11.432816
            D = -2.772874
            E = -0.158558
            G = 172.707974
            S_0 = A*ln(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G # Hydrogen standard entropy J/mol of H2- NIST
            S_0 = S_0/1000 #kJ/mol of H2


            fh = self.cH/self.theta
    
            if self.Nb==0:
                Pb=0
            else:
                Pb =  1-exp(-self.Nb*fh)
    
            
            Ph = fh/(1-Pb) #correto?
            Pv = (1 - fh - Pb)/(1-Pb)
            self.S_c[T] = -R *self.theta*(1-Pb)*(Ph*ln(Ph)+Pv*ln(Pv))
            self.S[T] = self.S_c[T] - ((self.cH * S_0)/2)
        else:
            self.S_c[T] = np.zeros(len(self.cH))
            self.S[T] = self.S_c[T]

        return None


    def Gibbs_RSOM(self,T):
            if T!= 0:
                self.G[T] = self.H[T] - T* self.S[T]
            if T == 0:
                self.G[T] = self.H[T]

    def mu_H_RSOM(self,T):
        if T>0:
            fh = self.cH/self.theta
            self.mu_H[T] = self.h_m  - T * (-R*ln((fh*(exp(-(self.Nb)*fh))**((self.Nb)*exp(-(self.Nb)*fh))/(exp(-(self.Nb)*fh)-fh)**(1+(self.Nb)*exp(-(self.Nb)*fh))))- self.s0(T)/2)
        if T==0:
            self.mu_H[T] = self.h_m * np.ones(len(self.cH))
    def mu_M_RSOM(self,T):
        self.mu_M[T] = self.G[T] - self.cH * self.mu_H[T]

    def ThermodynamicsCalculation_RSOM(self,T,n_threads=-1,step_interpolation=0.00001):


        
        begin = time.time()
        print(f"The calculation of {self.name} phase at {T} K has started.")
        self.Enthalpy_RSOM(T)
        self.Entropy_RSOM(T)
        self.Gibbs_RSOM(T)
        self.mu_H_RSOM(T)
        self.mu_M_RSOM(T)
        
        self.cHvals[T] = self.cH
        if step_interpolation != False:
            if T>0:
                self.cHvals[T] = np.arange(0, self.cH_limit+step_interpolation, step_interpolation)
                self.H[T] = np.interp(self.cHvals[T], self.cH, self.H[T])
                self.S[T] = np.interp(self.cHvals[T], self.cH, self.S[T])
                self.S_c[T] = np.interp(self.cHvals[T], self.cH, self.S_c[T])
                self.G[T] = np.interp(self.cHvals[T], self.cH, self.G[T])
                self.mu_H[T] = np.interp(self.cHvals[T], self.cH, self.mu_H[T])
                self.mu_M[T] = np.interp(self.cHvals[T], self.cH, self.mu_M[T])

        
        end = time.time()
        tempo = end - begin
        print(f"The calculation takes {round(tempo/60,2)} minutes")



    
    

    def CalcPressure(self,T,P_0=1):
        #P_0 = 1#[atm]
        #lnpH = 2* self.mu_H[T][0:len(self.cHvals)]/(R*T)
        #peq = P_0*exp(2* self.mu_H[T][0:len(self.cHvals)]/(R*T))
        cte=2/(R*T)
        #self.pressure_total[T] = P_0*exp(2* self.mu_H[T][0:len(self.cHvals)]/(R*T))
        self.pressure_total[T] = P_0*exp(cte* self.mu_H[T][0:len(self.cHvals[T])])


# In[130]:


class Phase(Thermodynamics):
    def __init__(self,name,structure,theta,Nb,n_site=4,model ='DSEM'):
        Thermodynamics.__init__(self)
        self.name = name
        self.H_M = []
        self.Nb = Nb
        self.theta = theta
        self.model = model
        self.cH_step = []
        self.cH = []
        self.cH_equilibrium = {}
        #self.temperatures = []
        self.n_site_type = n_site # 4 for tetrahedral sites, 6 for octahedral sites 
        self.composition = None
        self.structure = None
        self.fhk_limit = None
        self.cH_limit = None
        if any(structure):
            self.structure = structure
        self.elements_A = None
        self.elements_B = None
        
        
        self.cH_limit = {}
        if self.model == 'DSEM': #Create dicts and lists necessary for the model calculation
            self.h_m = {}
            self.k_type_sites = {}
            self.k_type_sites_probability = {}
            self.theta_k = {}
            self.h_k_type_sites = {}
            self.fhk_limit ={}
            self.set_cH_limit = self.set_cH_limit_DSEM
            self.ThermodynamicsCalculation = self.ThermodynamicsCalculation_DSEM

        if self.model == 'RSOM': #Create dicts and lists necessary for the model calculation
            self.composition = None
            self.h_m = None
            self.set_cH_limit = self.set_cH_limit_RSOM
            self.ThermodynamicsCalculation = self.ThermodynamicsCalculation_RSOM

        

        
        if self.name == "alpha":
            self.structure = "BCC"
        if self.name == "beta":
            self.structure = "BCC_ord"
        if self.name == "delta":
            self.structure = "FCC"
        
        
    @property
    def h_m_calculation(self):
            
        if self.structure == "BCC":
            self.h_m = sum(self.composition * hi_values[0:,1])
            
        if self.structure == "BCC_ord":
            self.h_m = sum(self.composition * hi_values[0:,2])     
            
        if self.structure == "FCC":
            self.h_m = sum(self.composition * hi_values[0:,3])

        if self.structure == 'C14':
                self.h_m = sum(self.composition * hi_values[0:,1])

    
    @property
    def H_M_calculation(self):  #determines the Enthalpy due the phase transition from the reference -- H_M diff from 0 only for delta phase
        
        if self.structure == 'FCC':
            alpha_vetor = np.zeros([len(bonding_alpha),len(bonding_alpha)])
            for i in self.elements:
                for j in self.elements:
                    alpha_vetor[np.where(bonding_alpha==i)[0],np.where(bonding_alpha==j)[0]]=self.alloy_composition[i]*self.alloy_composition[j]

            average_bond_alpha = alpha_vetor * bonding_alpha[0:,1:]
            average_bond_alpha = sum(sum(average_bond_alpha))

            E_total_alpha = average_bond_alpha * 4


            delta_vetor = np.zeros([len(bonding_delta),len(bonding_delta)])

            for i in self.elements:
                for j in self.elements:
                    delta_vetor[np.where(bonding_delta==i)[0],np.where(bonding_delta==j)[0]]=self.alloy_composition[i]*self.alloy_composition[j]

            average_bond_delta = delta_vetor * bonding_delta[0:,1:]
            average_bond_delta = sum(sum(average_bond_delta))
            E_total_delta = average_bond_delta * 6

            H_delta = E_total_delta - E_total_alpha  #Talvez seja necessário algum efeito de temperatura?
            self.H_M = H_delta
        else:
            self.H_M = 0
        
#################################################  begin DSEM equations ############################################################################        
    @property    
    def k_type_sites_calculation(self):  #combinatorial analysis for determine all possible different k-sites


        def calculate_multinomial_probability(combination, composition_elements):
            """
            Calculates the probability for a single combination (e.g., (e1, e2, e3))
            applying the factorial permutation factor.
            """
            
            r = len(combination)
            
            if r == 0:
                return 1.0 # Probability of an empty combination is 1 for B4
                
            element_counts = Counter(combination)
    
            base_probability = 1.0
            for element, count in element_counts.items():
                base_probability *= (composition_elements[element] ** count)
    
            # Calculate the permutation factor: r! / (k1! * k2! * ...)
            permutation_factor_denominator = 1.0
            for count in element_counts.values():
                permutation_factor_denominator *= math.factorial(count)
    
            permutation_factor = math.factorial(r) / permutation_factor_denominator
    
            return base_probability * permutation_factor
    
        def calculate_probabilities(n_A, n_B):
            """
            Calculates probabilities for n_A n_B compounds.
    
            Args:
                n_A (int): The number of elements from set A in the combination.
                n_B (int): The number of elements from set B in the combination.
    
            Returns:
                dict: A dictionary where keys are (combination_A, combination_B) tuples
                      and values are their calculated probabilities.
            """
            # Generate combinations for A and B using combinations_with_replacement
            combinations_A = list(combinations_with_replacement(self.elements_A, r=n_A))
            combinations_B = list(combinations_with_replacement(self.elements_B, r=n_B))
    
            # Calculate probabilities for individual combinations_A and combinations_B
            prob_combinations_A = {}
            for combo in combinations_A:
                prob_combinations_A[combo] = calculate_multinomial_probability(combo, self.composition_elements_A)
    
            prob_combinations_B = {}
            for combo in combinations_B:
                prob_combinations_B[combo] = calculate_multinomial_probability(combo, self.composition_elements_B)
    
            # Generate all n_A n_B combinations
            n_A_n_B_list = list(product(combinations_A, combinations_B))
    
            # Calculate the total probability for each A_rA B_rB combination
            summed_probabilities = defaultdict(float)
    
            for combo_A, combo_B in n_A_n_B_list:
                current_probability = prob_combinations_A[combo_A] * prob_combinations_B[combo_B]
                normalized_key = (combo_A, combo_B) # combinations_with_replacement gives canonical forms
                summed_probabilities[normalized_key] += current_probability
    
            return summed_probabilities

        if self.structure in ['BCC','BCC_ord','FCC']:
            if self.n_site_type ==4:
                self.site_type = 'Tetrahedral'
            if self.n_site_type ==6: 
                self.site_type = 'Octahedral'
            self.k_type_sites[self.site_type] = list(combinations_with_replacement(self.elements, self.n_site_type))     



        
        if self.structure == 'C14':
            for site_type in A2B2:
                results_A2B2 = calculate_probabilities(n_A=2, n_B=2)
                self.k_type_sites[site_type] = [combination[0]+combination[1] for combination,probability in results_A2B2.items()]
                self.k_type_sites_probability[site_type] = np.array([probability for combination,probability in results_A2B2.items()])
            for site_type in AB3:
                results_AB3 = calculate_probabilities(n_A=1, n_B=3)
                self.k_type_sites[site_type] = [combination[0]+combination[1] for combination,probability in results_AB3.items()]
                self.k_type_sites_probability[site_type] = np.array([probability for combination,probability in results_AB3.items()])
            for site_type in B4:
                results_B4 = calculate_probabilities(n_A=0, n_B=4)
                self.k_type_sites[site_type] = [combination[1] for combination,probability in results_B4.items()]
                self.k_type_sites_probability[site_type] = np.array([probability for combination,probability in results_B4.items()])
   
        
        self.i_size =  max([len(site_compositions) for site_compositions in self.k_type_sites_probability.values()]) 
        self.j_size = len(self.k_type_sites_probability.keys())
        self.k_size = len(self.k_type_sites_probability.keys())
        self.Nb_tensor = np.stack([NbC14] * self.i_size, axis=0)
        for key in self.k_type_sites_probability:
            size = len(self.k_type_sites_probability[key])
            if size<self.i_size:
                self.k_type_sites_probability[key] = np.pad(self.k_type_sites_probability[key],(0,self.i_size-size))
        self.p_k_tensor = np.stack([self.k_type_sites_probability[key] for key in self.k_type_sites_probability],axis=0)

   
        
    @property
    def k_type_sites_probability_calculation(self): #combinatorial calculation for determine all probability of occurence for all different k-sites
        if self.structure in ['BCC','BCC_ord','FCC']:
            
            for site_type in self.k_type_sites: 
                self.k_type_sites_probability[site_type] = []
                prodx = []
                for interstitial_site in range(len(self.k_type_sites[site_type])):
                    temp_thetak = []
                    (unique,counts_k_site)= np.unique(self.k_type_sites[site_type][interstitial_site], return_counts = True)
                    temp_factorial = []
                    for element_atom in counts_k_site:
                        temp_factorial.append(factorial(element_atom))
                    delta = factorial(self.n_site_type)/np.prod(temp_factorial)
                    for m in self.k_type_sites[site_type][interstitial_site]:
                        temp_thetak.append(self.alloy_composition[m])
                    prodx.append(np.prod(temp_thetak))
                    self.k_type_sites_probability[site_type].append(delta * np.prod(temp_thetak)) 
                self.k_type_sites_probability[site_type] = np.array(self.k_type_sites_probability[site_type])
                
       
            
    @property
    def theta_k_calculation(self):
        for site_type in self.k_type_sites:
            self.theta_k[site_type] = []
            for p_k in self.k_type_sites_probability[site_type]:
                self.theta_k[site_type].append((self.theta[site_type])*p_k)
                #self.theta_k.append((self.theta/self.r)*p_k) Trying without /r, the limitation is on f_k
            self.theta_k[site_type] = np.array(self.theta_k[site_type])
        self.theta_k_tensor = np.stack([self.theta_k[key] for key in self.theta_k],axis=0)



    @property
    def h_k_type_sites_calculation(self):
        
        if self.name == "alpha":
            structure = 0
        if self.name == "beta":
            structure = 1
        if self.name == "delta":
            structure = 2
        if self.name == "C14": 
            structure = 3
        for site_type in self.k_type_sites:
            self.h_k_type_sites[site_type] = []
            for k_site in self.k_type_sites[site_type]:
                temp_h = []
                for element_atom in k_site:
                    temp_h.append(hi_values[np.where(hi_values==element_atom)[0],structure+1][0]) #mudar structure para name e mudar na planilha
                self.h_k_type_sites[site_type].append(sum(temp_h)/float(self.n_site_type))   
            self.h_k_type_sites[site_type] = np.array(self.h_k_type_sites[site_type])

        for key in self.h_k_type_sites:
            size = len(self.h_k_type_sites[key])
            if size<self.i_size:
                self.h_k_type_sites[key] = np.pad(self.h_k_type_sites[key],(0,self.i_size-size))
        self.h_k_tensor = np.stack([self.h_k_type_sites[key] for key in self.h_k_type_sites],axis=0)
        #self.h_k_tensor = np.concatenate([self.h_k_type_sites[key] for key in self.h_k_type_sites],axis=0)


    
    def set_cH_limit_DSEM(self):   
            # def Pb(x):
            #     return np.linalg.norm(x-exp(-(self.Nb)*sum(x*self.k_type_sites_probability))) #Determine fh  real
                 
          
        if len(self.Nb)>0:
            def objective(x):
                # fh_ki = x * self.p_k_tensor.flatten()
                # fh_wk = np.sum(fh_ki.reshape(self.j_size, self.i_size), axis=1)
                # fb_wk = np.einsum('ijk,ki->j', self.Nb_tensor,fh_ki.reshape(self.j_size, self.i_size))
                # p_matrix = (self.p_k_tensor.flatten()).reshape(self.j_size, self.i_size)
                # fb = np.einsum('ij,i->ij', p_matrix, fb_wk)
                # teste = -x.flatten() + np.exp(-fb).flatten()
                # dif = np.sqrt(len(np.where(teste==1)[0]))
                return self.Gibbs_DSEM(x, T) 
                
            # Inequality constraints: Pb_constraints
            def inequality_constraints(x):
                fh_ki = x * self.p_k_tensor.flatten()
                fh_wk = np.sum(fh_ki.reshape(self.j_size, self.i_size), axis=1)
                fb_wk = np.einsum('ijk,ki->j', self.Nb_tensor,fh_ki.reshape(self.j_size, self.i_size))
                p_matrix = (self.p_k_tensor.flatten()).reshape(self.j_size, self.i_size)
                fb = np.einsum('ij,i->ij', p_matrix, fb_wk)
                f_i_wk = -x.flatten() + np.exp(-fb).flatten()
                dif = np.sqrt(len(np.where(f_i_wk==1)[0])) #the values = 1 which results from the 0 dummy values for make all lists have same size
                return  (np.linalg.norm(f_i_wk)-dif)
    
    
    
            zero_indices = np.where(self.theta_k_tensor.flatten() == 0)[0]
            xl_d = np.full(self.i_size * self.j_size, 1e-6)  # Lower bounds
            xu_d = np.full(self.i_size * self.j_size, 0.999)  # Upper bounds
            xl_d[zero_indices] = 0
            xu_d[zero_indices] = 0
            bounds = [(xl_d[i], xu_d[i]) for i in range(len(xl_d))]


            constraints = [{'type': 'eq', 'fun': inequality_constraints}]  # Inequality constraints (g(x) <= 0)
            x0 = np.random.uniform(low=xl_d, high=xu_d)
            x0 = np.clip(x0, xl_d, xu_d)
            # x0[zero_indices] = 0
            res = spo.minimize(
                fun=objective,
                x0=x0,
                method='SLSQP',  # Sequential Least Squares Programming for constrained optimization
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 2000}  # Display convergence messages
            )
            
        
            self.fhk_limit = res.x
            self.cH_limit =  np.sum(res.x*self.theta_k_tensor.flatten())
        if isinstance(self.Nb,int):
            self.fhk_limit = np.full(len(self.theta_k_tensor.flatten()),0.9999)
            self.cH_limit = np.sum(self.fhk_limit*self.theta_k_tensor.flatten())


################################################# end DSEM equations ############################################################################   

################################################# begin RSOM equations ############################################################################       
    
    def set_cH_limit_RSOM(self):
            def Pb(x):
                fb_extended = (self.Nb)*(x/(self.theta)) #JMAK
                return x-self.theta*exp(-fb_extended) 
    
            x0 = 0.005
            bnds = [(0,self.theta)]
    
            cons = {'type': 'eq', 'fun': lambda x: Pb(x),
                           }
            cH_limit = spo.minimize(Pb,x0, options = {"disp": True, "maxiter": 2000}, method = 'SLSQP',  constraints = cons, bounds = bnds)
            self.cH_limit = np.round(cH_limit.x[0],4)


    
################################################# end RSOM equations ############################################################################       
 

    
    @property
    def set_cH(self):
        c_H = []
        for i in range(1,int(self.cH_limit/self.cH_step +1)):
            if (self.cH_step*i)< self.cH_limit:
                c_H.append(np.round(self.cH_step*i,4))
        self.cH = np.array(c_H)
        


# In[151]:


class Alloy(Phase):
    def __init__(self):
        self.TEMPERATURES = [273.15+25,273.15+100]
        self.P = {}
        self.cH = {}
        self.idx_cH_equilibrium = {}
        self.cH_equilibrium = {}
        self.mu_H_equilibrium ={} 
        self.equilibriums = {}
        self.phases = {}
        self.H_eq = {}
        self.S_eq = {}
        self.P_plat = {}
        self.composition_vetor = None
        self.alloy_composition ={}

    
        
    def alloy_initialisation(self,step,fhk_lim=None,cH_custom=None):
        ##########Informa ao usuário a liga que foi inserida##############
        self.alloy_name = []
        for element in self.alloy_composition.keys():
            self.alloy_name.append(element)
            self.alloy_name.append(str(np.round(self.alloy_composition[element],3)))
            
        separator = ''
        self.alloy = separator.join(self.alloy_name)
 
        for i,j in self.__dict__.items():
            if isinstance(j,Phase):
                self.__dict__[i].elements = self.elements      
                self.__dict__[i].alloy_composition = self.alloy_composition    
                self.__dict__[i].H_M_calculation
                #self.__dict__[i].model = model
            
                #self.__dict__[i].n_site_type = 4                            #occupation of tetrahedral sites
                if self.__dict__[i].model == 'DSEM':
                    self.__dict__[i].elements_A = self.elements_A
                    self.__dict__[i].composition_elements_A = self.composition_A
                    self.__dict__[i].elements_B = self.elements_B
                    self.__dict__[i].composition_elements_B = self.composition_B
                    self.__dict__[i].k_type_sites_calculation                   #calculates the each different k-site
                    # self.__dict__[i].k_type_sites_probability_calculation       #calculates the probability of each different k-site occurs
                    self.__dict__[i].theta_k_calculation                        #calculates the quantity of interstitial sites per metal atom for each different k-site
                    self.__dict__[i].h_k_type_sites_calculation                 #calculates the energy of interstitial sites for each different k-site
                    #self.__dict__[i].set_cH_limit_DSEM()
                    self.__dict__[i].fhk_limit = fhk_lim
                if self.__dict__[i].model == 'RSOM':
                    self.composition_vetor = np.zeros(len(hi_values))
                    for element in self.elements:
                        self.composition_vetor[np.where(hi_values==element)[0]] = self.alloy_composition[element]
                        
                    self.__dict__[i].composition = self.composition_vetor
                    self.__dict__[i].h_m_calculation
                    #self.__dict__[i].set_cH_limit_RSOM()
                if fhk_lim is None:
                    self.__dict__[i].set_cH_limit()                               #set the Hydrogen composition limit for the phase  #change C14
                    self.__dict__[i].cH_step = step                      #set the step variation in the Hydrogen composition for the termodynamic calculation
                if cH_custom is None: 
                    self.__dict__[i].set_cH                                     #set all Hydrogen composition possible for the phase
                else:
                    self.__dict__[i].cH_limit = max(cH_custom)
                    self.__dict__[i].cH = cH_custom
        return print(f"The alloy inserted is: {self.alloy}")   

    def set_composition(self,step=0.01,fhk_lim = None,cH_custom = None):#stepBCC=0.001,stepBCC_ord=0.001,stepFCC=0.001):

        self.elements = []
        stop = False

        self.elements_A = []
        self.composition_A ={}
        self.elements_B = []
        self.composition_B ={}
        ##########Entrada de dados pelo usuário################
        while (stop!= True):
            y = input("Enter with an element to sublattice A: ")
            #adicionar um if elemento not in lista de elementos: Elemento não implementado ao modelo
            self.elements_A.append(y) #adiciona a entrada y no fim da lista de elementos
            x = float(input("Atomic fraction of element {} in sublattice A: " .format(y)))
            # self.composition_A[y] = x #adiciona a entrada w no fim da lista de composições
            if y in self.composition_A:
                self.composition_A[y]= self.composition_A[y] + x
            else:
                self.composition_A[y]=x
            z = input("Would you like to add another element to sublattice A? Y/n ")
            if z == "n":
                stop = True
                
        stop = False
        while (stop!= True):
            y = input("Enter with an element to sublattice B: ")
            #adicionar um if elemento not in lista de elementos: Elemento não implementado ao modelo
            self.elements_B.append(y) #adiciona a entrada y no fim da lista de elementos
            x = float(input("Atomic fraction of element {} in sublattice B: " .format(y)))
            # self.composition_B[y] = x #adiciona a entrada w no fim da lista de composições
            if y in self.composition_B:
                self.composition_B[y]= self.composition_B[y] + x
            else:
                self.composition_B[y]=x
            z = input("Would you like to add another element to sublattice B? Y/n ")
            if z == "n":
                stop = True



        for element in self.composition_A.keys():
            self.elements.append(element)
            self.alloy_composition[element] = np.round(self.composition_A[element],2)     
        for element in self.composition_B.keys():
            if element not in self.elements:
                self.alloy_composition[element] = 2*np.round(self.composition_B[element],2)
            if element in self.elements:
                self.alloy_composition[element] = self.alloy_composition[element]+ 2*np.round(self.composition_B[element],2)
            self.elements.append(element)
        norm = np.sum(list(self.alloy_composition.values()))
        for i in self.alloy_composition:
            self.alloy_composition[i] = self.alloy_composition[i]/norm  
        self.alloy_initialisation(step,fhk_lim,cH_custom)





    
    def compositions_list(self,composition,step,fhk_lim = None,cH_custom = None):  

        self.elements = []
        self.alloy_composition ={}
        self.elements_A = list(composition['A'].keys())
        self.composition_A = composition['A']
        for element in self.composition_A.keys():
            self.elements.append(element)
            self.alloy_composition[element] = self.composition_A[element]
            
        self.elements_B = list(composition['B'].keys())
        self.composition_B = composition['B']
        for element in self.composition_B.keys():
            if element not in self.elements:
                self.alloy_composition[element] = 2*self.composition_B[element]
            if element in self.elements:
                self.alloy_composition[element] = self.alloy_composition[element]+ 2*self.composition_B[element]
            self.elements.append(element)
        norm = np.sum(list(self.alloy_composition.values()))
        for i in self.alloy_composition:
            self.alloy_composition[i] = self.alloy_composition[i]/norm  
        self.alloy_initialisation(step,fhk_lim,cH_custom)


    def set_temperatures(self):
        self.TEMPERATURES = []
        stop = False
        while (stop!= True):
            y = float(input("Enter a temperature in degrees Celsius for calculations: "))
            y= y + 273.15
            self.TEMPERATURES.append(y) 
            z = input("Would you like to add another temperature ? Y/n ")
            if z == "n":
                stop = True
        self.TEMPERATURES.sort() #sort the temperatures in ascending order


    def temperatures_list(self,temperature):
        self.TEMPERATURES = []
        for T in temperature:
            self.TEMPERATURES.append(T+273.15)
        self.TEMPERATURES.sort() #sort the temperatures in ascending order


    def find_plateau(self,Phase1,Phase2,temperature, search="binary",tol=0.005):
        
        T = temperature
        if T not in self.H_eq:
                self.H_eq[T] = {}
                self.S_eq[T] = {}
                self.P_plat[T] = {}
            
        if T not in self.cH_equilibrium:               
            self.cH_equilibrium[T]= {}
            self.mu_H_equilibrium[T]= {}    
        #Também funciona!
        #melhor tempo
        # t1 = time.time()
        
        v1 = np.zeros([len(Phase1.cHvals[T]), 2])
        v1[:, 0] = np.round(Phase1.mu_H[T], 4)
        v1[:, 1] = np.round(Phase1.mu_M[T], 4)
        
        v2 = np.zeros([len(Phase2.cHvals[T]), 2])
        v2[:, 0] = np.round(Phase2.mu_H[T], 4)
        v2[:, 1] = np.round(Phase2.mu_M[T], 4)
        
        if search == "old":
            # Encontre os índices das linhas com valores iguais nas duas colunas
            idx_cH = np.where(np.all(abs(v1[:, None, :]- v2)<=tol, axis=2))
        
        #if np.array(idx_cH).size> 0:
            if idx_cH:
                cH_eq=np.round([Phase1.cHvals[T][idx_cH[0][0]],Phase2.cHvals[T][idx_cH[1][0]]],4)
                mu_h_eq = Phase1.mu_H[T][idx_cH[0][0]]
                self.cH_equilibrium[T][f"{Phase1.structure} - {Phase2.structure}"] = cH_eq
                self.mu_H_equilibrium[T][f"{Phase1.structure} - {Phase2.structure}"] = mu_h_eq
                print("cH_eq encontrado:", cH_eq)
                eq = f"{Phase1.structure} - {Phase2.structure}"
                self.H_eq[T][eq] = (Phase2.H[T][idx_cH[1][0]] -  Phase1.H[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[1][0]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.S_eq[T][eq] = (Phase2.S[T][idx_cH[1][0]] -  Phase1.S[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[1][0]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.P_plat[T][eq] = exp(2* ((self.H_eq[T][eq]/(R*T) - (self.S_eq[T][eq]/R))))  
                
            else:
                print(f"None equilibrium found between {Phase1.structure} and {Phase2.structure}")
            
        if search == "binary":   

                        # Função auxiliar para busca binária
            def busca_binaria(arr, valor, tol):
                idx = bisect_left(arr, valor)
                if idx < len(arr) and abs(arr[idx] - valor) <= tol:
                    return idx
                elif idx > 0 and abs(arr[idx - 1] - valor) <= tol:
                    return idx - 1
                return None
            
            # Encontra os índices usando a busca binária
            idx_cH = []
            for i, val1 in enumerate(v1[:, 0]):
                idx = busca_binaria(v2[:, 0], val1,tol)
                if idx is not None and abs(v1[i, 1] - v2[idx, 1]) <= tol:
                    # Adiciona apenas se Phase1.cHvals[i] < Phase2.cHvals[idx]
                    if Phase1.cHvals[T][i] < Phase2.cHvals[T][idx]:
                        idx_cH.append((i, idx))
            
            if idx_cH:
                cH_eq = np.round([Phase1.cHvals[T][idx_cH[0][0]], Phase2.cHvals[T][idx_cH[0][1]]], 4)
                mu_h_eq = Phase1.mu_H[T][idx_cH[0][0]]
                self.cH_equilibrium[T][f"{Phase1.structure} - {Phase2.structure}"] = cH_eq
                self.mu_H_equilibrium[T][f"{Phase1.structure} - {Phase2.structure}"] = mu_h_eq
                print("cH_eq encontrado:", cH_eq)
                eq = f"{Phase1.structure} - {Phase2.structure}"
                self.H_eq[T][eq] = (Phase2.H[T][idx_cH[0][1]] -  Phase1.H[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[0][1]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.S_eq[T][eq] = (Phase2.S[T][idx_cH[0][1]] -  Phase1.S[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[0][1]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.P_plat[T][eq] = exp(2* ((self.H_eq[T][eq]/(R*T) - (self.S_eq[T][eq]/R))))           


                
                return cH_eq
        
            else:
                print(f"None equilibrium found between {Phase1.structure} and {Phase2.structure}")
        
            



    def calculatePCI(self,T,t=0.05):
        self.equilibriums[T] = {}


        def find_real_equilibrium(mu_H_equilibrium, cH_equilibrium,tol, initial_equilibrium=None):
            equilibriums = {}
            
            def busca_binaria(arr, valor, tol):
                idx = bisect_left(arr, valor)
                if (idx < len(arr) and abs(arr[idx] - valor)) <= tol:
                    return idx
                elif (idx > 0 and abs(arr[idx - 1] - valor))  <= tol:
                    return idx - 1
                return None
        
            # Identificar o primeiro equilíbrio se não for fornecido
            if initial_equilibrium is None:
        
                best_equilibrium = None
                best_cHplat = float('inf')
                for key, values in self.cH_equilibrium[T].items():
                    phase1, phase2 = key.split(' - ')
                    for key2, values2 in self.cH_equilibrium[T].items():
                        phase3, phase4 = key2.split(' - ')
                        if key!=key2: #CORREÇÃO
                            cHplat = values[0]
                            cHG = values2[0]
                            #print(cHG,cHplat,phase1,phase3,key,key2)
                            findcH = busca_binaria(self.__dict__[phase1].cHvals[T], cHplat, tol)
                            findcH2 = busca_binaria(self.__dict__[phase3].cHvals[T], cHG, tol)
                            g1=self.__dict__[phase1].G[T][findcH]
                            g2=self.__dict__[phase3].G[T][findcH2]
                            if g1<g2 and cHplat < best_cHplat:
                                best_equilibrium = key
                                best_cHplat = cHplat
                                #print(best_equilibrium)
                            
                
                first_equilibrium = best_equilibrium
                equilibriums[first_equilibrium] = cH_equilibrium[first_equilibrium]
                last_equilibrium = (first_equilibrium, cH_equilibrium[first_equilibrium])
            else:
                equilibriums[initial_equilibrium[0]] = initial_equilibrium[1]
                last_equilibrium = initial_equilibrium
        
            # Buscar equilíbrios subsequentes
            while True:
                next_equilibrium = {}
                for equilibrium_phase, chemical_potential in mu_H_equilibrium.items():
                    # Verifica se a fase anterior termina com a mesma fase que inicia a nova
                    if last_equilibrium[0].split(" - ")[1] in equilibrium_phase.split(" - ")[0]:
                        # Verifica se cH da última fase é menor que o cH candidato
                        if last_equilibrium[1][1] < cH_equilibrium[equilibrium_phase][0]:
                            next_equilibrium[equilibrium_phase] = chemical_potential
        
                if next_equilibrium:
                    # Escolhe o próximo equilíbrio baseado no menor potencial químico
                    actual_equilibrium = min(next_equilibrium, key=lambda k: next_equilibrium[k])
                    equilibriums[actual_equilibrium] = cH_equilibrium[actual_equilibrium]
                    last_equilibrium = (actual_equilibrium, cH_equilibrium[actual_equilibrium])
                else:
                    break
        
            return equilibriums

        cH_equilibrium = self.cH_equilibrium[T]
        mu_H_equilibrium = self.mu_H_equilibrium[T]
        equilibriums = find_real_equilibrium(mu_H_equilibrium, cH_equilibrium,tol=t)
        
        self.equilibriums[T] = equilibriums
        
        # Novo dicionário para armazenar os valores reorganizados por fase
        phases_dict = {}
        
        # Itera sobre o dicionário original
        for phases, values in self.equilibriums[T].items():
            # Separa as palavras da chave
            phase1, phase2 = phases.split(" - ")
            
            # Adiciona o valor correspondente de phase1 ao novo dicionário
            if phase1 not in phases_dict:
                phases_dict[phase1] = []
            phases_dict[phase1].append(values[0])
            
            # Adiciona o valor correspondente de phase2 ao novo dicionário
            if phase2 not in phases_dict:
                phases_dict[phase2] = []
            phases_dict[phase2].append(values[1])

        self.phases[T] = phases_dict

        for phase, _ in self.__dict__.items():
            if isinstance(_,Phase):
                if phase in phases_dict.keys():
                    self.__dict__[phase].cH_equilibrium = phases_dict[phase]


        for phase in phases_dict.keys():
            object = self.__dict__[phase]
            if len(object.cH_equilibrium)==2:
                mask1 = (object.cHvals[T]>=object.cH_equilibrium[0])
                mask2 = (object.cHvals[T]<=object.cH_equilibrium[1])
                if T in self.cH:
                    self.cH[T] = np.concatenate((self.cH[T],object.cHvals[T][mask1&mask2]))
                    self.P[T] = np.concatenate((self.P[T],object.pressure_total[T][mask1&mask2]))         
            if len(object.cH_equilibrium) == 1:
                mask1 = (object.cHvals[T]<=object.cH_equilibrium)
                mask2 = (object.cHvals[T]>=object.cH_equilibrium)
                if T not in self.cH:
                    self.cH[T] = object.cHvals[T][mask1]
                    self.P[T] =  object.pressure_total[T][mask1]
                elif T in self.cH:
                    self.cH[T] = np.concatenate((self.cH[T],object.cHvals[T][mask2]))
                    self.P[T] = np.concatenate((self.P[T],object.pressure_total[T][mask2]))

                    
    

############################################################Saving Data #######################################################################3        

    


    def SaveData(self,custom_name = None, custom_suffix = None, custom_structure = None, custom_temperature = None, save_occ = False,save_figures=False,cmmax=0.25,cfsize=4,ctol=0.05):
        if custom_name == None: 
            alloy_name = self.alloy
            
        if custom_suffix != None:
            alloy_name = alloy_name + custom_suffix
   
        if custom_structure == None: 
            structures = [ self.__dict__[i] for i,j in self.__dict__.items() if isinstance(j,Phase)] #Return all instance which are Phase()
        
        if custom_temperature == None: 
            Temperatures = self.TEMPERATURES
        
        #######################################################################################
        #  Cria pasta para salvar os arquivos da liga e arquivo txt com os dados principais   #  
        #######################################################################################  
        
        if not os.path.isdir(f'./{alloy_name}'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
            os.mkdir(f'./{alloy_name}')

        if save_occ == True:
            self.SaveOccPlateau(name=alloy_name,mmax=cmmax,fsize=cfsize,tol=ctol)

        if save_figures == True:
            self.SaveFigures(name=alloy_name,phases=structures)
        
        dnow = datetime.now()        
        my_path = os.path.abspath(alloy_name)
        my_file = f'./{alloy_name}.txt'
        if not os.path.isfile(os.path.join(my_path, my_file)):
            file = open(os.path.join(my_path, my_file), 'w+')
            file.close()
        file = open(os.path.join(my_path, my_file),'a')
        file.write(f"\nAlloy: {alloy_name} | {dnow.strftime('%d/%m/%Y %H:%M')} | \n")
        #file.write(f"hM[alpha]= {hm_alpha} | hM[beta]= {hm_beta} | hM[delta]= {hm_delta} |\n")
        for i,j in self.__dict__.items():
            if isinstance(j,Phase):
                file.write(f"HM[{self.__dict__[i].name}] = {self.__dict__[i].H_M}| ") 
        #file.write= ("\n")
        file.write("\nValues for Theta and Nb is in the format: structure = (theta,Nb)\n")
        for i,j in self.__dict__.items():
             if isinstance(j,Phase):
                file.write(f"{self.__dict__[i].structure} = {self.__dict__[i].theta,self.__dict__[i].Nb}| ") 

        if alloy_DSEM.equilibriums:
            for T in self.TEMPERATURES:
                if T>0:
                    if self.equilibriums[T]:
                        eq_list = [self.P_plat[T][key] for key in self.equilibriums[T].keys()]
                        for i,j in zip(self.equilibriums[T].keys(),range(len(self.equilibriums[T].keys()))):
                            file.write( f"\nTemperature: {T} K - Equilibrium  {i}  - Plateau Pressure [atm]: {eq_list[j]} - Plateau cH: {self.equilibriums[T][i]} - Plateau H [kJ]: {self.H_eq[T][i]} - Plateau S [kJ]: {self.S_eq[T][i]}" )
                            
                            
        
        file.close()
        ########################################################################################



    
    
        for T in Temperatures:
            if not os.path.isdir(f'./{alloy_name}/{T}'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                os.mkdir(f'./{alloy_name}/{T}')
            #my_path = os.path.abspath(alloy.alloy+ f"/{T - 273.15}") # Figures out the absolute path for you in case your working directory moves around.
 
            if not os.path.isdir(f'./{alloy_name}/{T}/data'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                    os.mkdir(f'./{alloy_name}/{T}/data')
            my_path = os.path.abspath(alloy_name+ f"/{T}/data") # Figures out the absolute path for you in case your working directory moves around.
        
                    

            properties = ['H','S','S_c','G','mu_H','mu_M']
            #properties2 = ['mu_H','mu_M']
            for structure in structures:
                transfer = np.zeros((len(structure.cHvals[T]),2))
                transfer[:,0] = np.round(structure.cHvals[T],4)
                
                for i in properties:
                    transfer[:,1] = structure.__dict__[i][T]  #Saving each propertie for each phase
                    my_file = f'{alloy_name}_{i}_{structure.name}_T{T}K.txt' #structure.name[0]
                    dataframe = pd.DataFrame(transfer)
                    dataframe = dataframe.rename(columns = { 0 : "c_H", 1 : f"{i}_{structure.name}"})
                    dataframe.to_csv(os.path.join(my_path, my_file), sep = '\t', encoding = "utf-8", index = False, columns=['c_H', f"{i}_{structure.name}"])
        
                
                if structure.model == 'DSEM':
                    if not os.path.isdir(f'./{alloy_name}/{T}/data/sites information'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                            os.mkdir(f'./{alloy_name}/{T}/data/sites information')
                    my_path2 = os.path.abspath(alloy_name+ f"/{T}/data/sites information") # Figures out the absolute path for you in case your working directory moves around.
                
                    
                    if not os.path.isdir(f'./{alloy_name}/{T}/data/sites occupation'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                            os.mkdir(f'./{alloy_name}/{T}/data/sites occupation')
                    my_path3 = os.path.abspath(alloy_name+ f"/{T}/data/sites occupation") # Figures out the absolute path for you in case your working directory moves around.


                    if structure.name != 'C14':
                        ###### Sites informations
                        transfer= np.zeros((len(structure.k_type_sites),3))
                        sites = []
                        for i in structure.k_type_sites:
                            s = ''
                            for j in i:
                                s +=j
                            sites.append(s)
                        
                        
                        transfer[:,0] = structure.k_type_sites_probability
                        transfer[:,1] = structure.theta_k
                        transfer[:,2] = structure.h_k_type_sites
                        
                        my_file = f'{alloy_name}_{structure.name}_sites_information_T{T}K.txt'
                        dataframe = pd.DataFrame(sites)
                        dataframe = dataframe.rename(columns = { 0 : "site"})
                        dataframe['p_k'] = transfer[:,0]
                        dataframe['theta_k'] = transfer[:,1]
                        dataframe['h_k'] = transfer[:,2]
                        dataframe = dataframe.sort_values(by=['h_k'])
                        dataframe.to_csv(os.path.join(my_path2, my_file), sep = '\t', encoding = "utf-8", index = False, columns=['site','p_k' ,'theta_k' ,'h_k'])
                
                        
                        transfer= np.zeros((len(structure.k_type_sites),2))
                        transfer[:,0] = structure.h_k_type_sites
                        
                        for cH in structure.occupation_k_site[T]:
                            transfer[:,1] = structure.occupation_k_site[T][cH]
                            my_file3 = f'{alloy_name}_{structure.name}_phase_site_occupation_cH_{cH}_T{T}K.txt'
                            dataframe = pd.DataFrame(transfer)
                            dataframe = dataframe.rename(columns = { 0 : "h_k", 1 : f"{structure.name}_occupation_cH_{cH}"})
                            dataframe = dataframe.sort_values(by=['h_k'])
                            dataframe.to_csv(os.path.join(my_path3, my_file3), sep = '\t', encoding = "utf-8", index = False, columns=['h_k', f"{structure.name}_occupation_cH_{cH}"])


                        if T!=0:
                            my_file = f'{alloy_name}_pressure_T{T}K.txt' 
                            transfer = np.zeros((len(self.cH[T]),2))
                            transfer[:,0] = np.round(self.cH[T],4)
                            transfer[:,1] = self.P[T]
                            dataframe = pd.DataFrame(transfer)
                            dataframe = dataframe.rename(columns = { 0 : "c_H", 1 : f"Pressure"})
                            dataframe.to_csv(os.path.join(my_path, my_file), sep = '\t', encoding = "utf-8", index = False, columns=['c_H', f"Pressure"])
    
                
                        
                    if structure.name == 'C14':
                        # data
                        dict1 = structure.k_type_sites
                        dict2 = structure.h_k_type_sites
                        dict3 = structure.k_type_sites_probability
                        dict4 = structure.theta_k
                        
                        # columns
                        col1_name = "Site"
                        col2_name = "h_k"
                        col3_name = "p_i"
                        col4_name = "theta_wk_i"
                        
                        for key in dict1:
                            n = len(dict1[key])
                            
                            # Cria um dicionário para montar o DataFrame
                            data = {
                                col1_name: [],
                                col2_name: [],
                                col3_name: [],
                                col4_name: []
                            }
                            
                            for i in range(n):
                                # Converte numpy scalars para float, se necessário
                                val1 = float(dict1[key][i]) if isinstance(dict1[key][i], np.generic) else dict1[key][i]
                                val2 = float(dict2[key][i]) if isinstance(dict2[key][i], np.generic) else dict2[key][i]
                                val3 = float(dict3[key][i]) if isinstance(dict3[key][i], np.generic) else dict3[key][i]
                                val4 = float(dict4[key][i]) if isinstance(dict4[key][i], np.generic) else dict4[key][i]
                                
                                data[col1_name].append(val1)
                                data[col2_name].append(val2)
                                data[col3_name].append(val3)
                                data[col4_name].append(val4)
                            
                            # Cria DataFrame
                            df = pd.DataFrame(data)
                            df= df.sort_values(by=['h_k'])
                            my_file = f'{alloy_name}_{structure.name}_{key}_sites_information_T{T}K.txt'
                            df.to_csv(os.path.join(my_path2, my_file), sep = '\t', encoding = "utf-8", index = False)

                        
                        my_file3 = f'{alloy_name}_{structure.name}_phase_site_occupation_T{T}K.txt'
                        rows = []
                        for cH, site_dict in structure.occupation_k_site[T].items():
                            for site, arr_fi in site_dict.items():
                                arr_extra = structure.h_k_type_sites[site]
                                
                                for idx in range(len(arr_fi)):
                                    rows.append({
                                        "cH": cH,
                                        "site": site,
                                        "idx": idx,
                                        "fi": arr_fi[idx],
                                        "h_k": arr_extra[idx],
                                    })

                        df = pd.DataFrame(rows)
                        df.to_csv(os.path.join(my_path3, my_file3), sep = '\t', encoding = "utf-8", index = False)

                        
                        if T!=0:
                            my_file = f'{alloy_name}_pressure_T{T}K.txt' 
                            transfer = np.zeros((len(structure.cHvals[T]),2))
                            transfer[:,0] = np.round(structure.cHvals[T],4)
                            transfer[:,1] = structure.pressure_total[T]
                            dataframe = pd.DataFrame(transfer)
                            dataframe = dataframe.rename(columns = { 0 : "c_H", 1 : f"Pressure"})
                            dataframe.to_csv(os.path.join(my_path, my_file), sep = '\t', encoding = "utf-8", index = False, columns=['c_H', f"Pressure"])
    
       

    def SaveOccPlateau(self,name,mmax=0.25,fsize=4,tol = 0.05):
        alloy_name = name
        def busca_binaria(arr, valor, tol):
            idx = bisect_left(arr, valor)
            if (idx < len(arr) and abs(arr[idx] - valor)) <= tol:
                return idx
            elif (idx > 0 and abs(arr[idx - 1] - valor))  <= tol:
                return idx - 1
            return None

        
        for T in self.TEMPERATURES:
            if T>0:
                if not os.path.isdir(f'./{alloy_name}/{T}'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                    os.mkdir(f'./{alloy_name}/{T}')
                my_path = os.path.abspath(self.alloy+ f"/{T}") # Figures out the absolute path for you in case your working directory moves around.
     
    
                
                for phase in self.phases[T]:
                    for cH_plat in self.phases[T][phase]:
                        idx = busca_binaria(list(self.__dict__[phase].occupation_k_site[T].keys()),cH_plat,tol)
                        ch = list(self.__dict__[phase].occupation_k_site[T].keys())[idx]
            
                        my_file = f'{self.alloy}_{phase}_cH_plat_{ch}_T_{T}K.png'
                        fig = plt.figure()
                        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1) 
                        axes.bar(self.__dict__[phase].h_k_type_sites,self.__dict__[phase].occupation_k_site[T][ch],color='#2d73b6',label = f'{phase} - cH = {ch}')
                        axes.bar([],[], label = f'Temperature {T} K')
            
                        
                        # # Adicionar texto na orientação vertical dentro da barra
                        # axes.text(allowed_energies[key], new_prob[key] / 2, a, ha='center', va='center', 
                        #           rotation=90, color='white', fontweight='bold')
            
                        for x,y,z in zip(self.__dict__[phase].h_k_type_sites,self.__dict__[phase].occupation_k_site[T][ch],self.__dict__[phase].k_type_sites):
                            # Obter a representação do elemento e do número de átomos
                            elemento, atomos = np.unique(z, return_counts=True)
                            a = ''.join([f"{i}{j}" for i, j in zip(atomos, elemento)])
                            # Checar se o texto cabe dentro da barra
                            # if len(a) * 0.005 < y:  # Ajuste o multiplicador 0.05 conforme necessário para dimensionamento
                            #     # Texto dentro da barra
                            #     axes.text(x, y / 2, a, ha='center', va='center', rotation=90, color='white', fontweight='bold', fontsize=8)
                            # else:
                            #     # Texto acima da barra se não couber
                            if y<0.002: #just to rise the legend of the site name
                                y=0.002
                            axes.text(x, y + y*0.05, a, ha='center', va='bottom', rotation=90, color='black', fontweight='bold', fontsize=fsize)
                        ymax = np.max(self.__dict__[phase].occupation_k_site[T][ch])
                        axes.set_ylim([0,ymax+ymax*mmax])        
                        axes.legend(loc=0)         
                        axes.set_xlabel(r'Site energy[kJ]') # Notice the use of set_ to begin methods
                        axes.set_ylabel(r"Occupation")
                        axes.set_title(f'{self.alloy}')
                        fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                        plt.close()
    

    def SaveFigures(self,name,phases):
        alloy_name = name
        #Example of plotting different properties

        my_path = os.path.abspath(alloy_name)
        my_file = f'{alloy_name}_PCT.png'
        fig = plt.figure()

        # Add set of axes to figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
        # Plot on that set of 
        for T in self.TEMPERATURES:
            if T>0:
                axes.plot(self.cH[T], self.P[T], label = f"PCI - T = {T} K")
        axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
        axes.set_ylabel(r"$P$ [atm]")
        axes.set_title(f'{self.alloy}')
        plt.yscale('log')
        axes.legend(loc = 0)
        fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
        plt.close()

        


        for phase in phases:
            my_file = 'Alloy_quantity_sites.png'
            keys = [key for key in phase.theta_k]
            try:
                # print(np.sum(r.x*phase.theta_k_tensor.flatten()))
                # Create a figure with subplots arranged in 2 columns
                fig, axes = plt.subplots(4, 2, figsize=(10, 15))
                
                # Flatten the axes array for easier indexing
                axes = axes.flatten()
                
                # Plot data on each subplot
                
                axes[0].bar(phase.h_k_type_sites['24l'],phase.theta_k['24l'])
                axes[0].set_title(f"{keys[0]}")
                
                axes[1].bar(phase.h_k_type_sites['12k2'],phase.theta_k['12k2'])
                axes[1].set_title(f"{keys[1]}")
                axes[2].bar(phase.h_k_type_sites['6h1'],phase.theta_k['6h1'])
                axes[2].set_title(f"{keys[2]}")
                axes[3].bar(phase.h_k_type_sites['6h2'],phase.theta_k['6h2'])
                axes[3].set_title(f"{keys[3]}")
                axes[4].bar(phase.h_k_type_sites['12k1'],phase.theta_k['12k1'],color='tab:orange')
                axes[4].set_title(f"{keys[4]}")
                axes[5].bar(phase.h_k_type_sites['4f'],phase.theta_k['4f'],color='tab:orange')
                axes[5].set_title(f"{keys[5]}")
                axes[6].bar(phase.h_k_type_sites['4e'],phase.theta_k['4e'],color='tab:green')
                axes[6].set_title(f"{keys[6]}")
            
                for i in range(7):
                    axes[i].set_ylabel(r'$\theta_i^{wk}$')
                    axes[i].set_xlabel(r'$h_i$ [kJ/mol H]')
                    axes[i].set_xlim([min(phase.h_k_type_sites['24l'])-5,5+max(max(phase.h_k_type_sites['4e']),max(phase.h_k_type_sites['4f']))])
                # Hide the unused subplot (8th subplot)
                # axes[-1].axis('off')
            
                A2B2 = ['24l','12k2','6h1','6h2']
                AB3 = ['12k1','4f']
                B4 = ['4e']
                A2B2_result=0
                AB3_result=0
                B4_result=0
                for j in A2B2:
                    A2B2_result+= phase.theta_k[j]
                for j in AB3:
                    AB3_result+= phase.theta_k[j]
                for j in B4:
                    B4_result+= phase.theta_k[j]
                
                axes[7].bar(phase.h_k_type_sites['24l'],A2B2_result,label = 'A2B2')
                axes[7].bar(phase.h_k_type_sites['12k1'],AB3_result,label = 'AB3')
                axes[7].bar(phase.h_k_type_sites['4e'],B4_result,label = 'B4')
                axes[7].legend(loc=0)
                axes[7].set_ylabel(r'$\theta_i$')
                axes[7].set_xlabel(r'$h_i$ [kJ/mol H]')
                
                for i in range(7):
                     axes[i].set_xlim([min(phase.h_k_type_sites['24l'])-5,5+max(max(phase.h_k_type_sites['4e']),max(phase.h_k_type_sites['4f']))])
                
                # Adjust layout for better spacing
                plt.tight_layout()
                # plt.suptitle()
                fig.subplots_adjust(top=0.94) # Adjust this value as needed
                # plt.show()
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()
            except:
                pass



        



        

        properties = ['H','S','S_c','G']
        properties2 = ['mu_H','mu_M']
        
        for T in self.TEMPERATURES:
            
            if not os.path.isdir(f'./{alloy_name}/{T}'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                os.mkdir(f'./{alloy_name}/{T}')
            my_path = os.path.abspath(alloy_name+ f"/{T}") # Figures out the absolute path for you in case your working directory moves around.

            

            if T>0:
                my_file = f'{alloy_name}_PCI_T = {T} K.png'
                fig = plt.figure()
        
                # Add set of axes to figure
                axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
                # Plot on that set of 
                axes.plot(self.cH[T], self.P[T], label = f"PCI - T = {T} K")
                axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
                axes.set_ylabel(r"$P$ [atm]")
                axes.set_title(f'{self.alloy}')
                plt.yscale('log')
                axes.legend(loc = 0)
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()




            
            
            for prop in properties:
             
            
                my_file = f'{alloy_name}_{prop}_T_{T}K.png'
                fig = plt.figure()
        
                # Add set of axes to figure
                axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
                # Plot on that set of 
                for phase in phases:
                    axes.plot(phase.cHvals[T], phase.__dict__[prop][T], label = r"$\Delta$" f"${prop}^{{{phase.name}}}$")
                axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
                axes.set_ylabel(r"$\Delta$" f"${prop}$ [kJ/mol H]")
                axes.set_title(f'T = {T} K')
                axes.legend(loc = 0)
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()
              
            for prop in properties2:
            
                my_file = f'{alloy_name}_{prop}_T_{T}K.png'
                fig = plt.figure()
        
                # Add set of axes to figure
                axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
        
                # Plot on that set of axes
                for phase in phases:
                    axes.plot(phase.cHvals[T], phase.__dict__[prop][T], label = r"$\Delta$" f"$\\{prop}^{{{phase.name}}}$")
                axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
                axes.set_ylabel(r"$\Delta$" f"$\\{prop}$ [kJ/mol H]")
                axes.set_title(f'T = {T} K')
                axes.legend(loc = 0)
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()


# # Example of DSE calculation for multicomponent C14 alloys

# In[154]:


hi_values = pd.read_csv('./hi.csv', sep = ';').to_numpy() #importa os valores de hi do arquivo csv e transforma em uma matriz 
####################################3
NbC14 = pd.read_csv('./NbC14_Nb21.csv', sep = ';').to_numpy() #importa os valores de hi do arquivo csv e transforma em uma matriz 

# composition = {'Mn':2,'Cr':2,'Ti':1,'Zr':1}
# composition = {'A':{'Ti':0.38,'Zr':0.62},'B':{'Mn':0.45,'Cr':0.55}}
composition = {'A':{'Ti':1},'B':{'Mn':0.24,'Cr':0.76}}

#Define an Object Alloy
alloy_DSEM = Alloy()

## set Structures
A2B2=['24l','12k2','6h1','6h2']
AB3=['12k1','4f']
B4 = ['4e']

alloy_DSEM.C14 = Phase(name='C14',structure = 'C14', theta={'24l':2,'12k2':1,'6h1':0.5,'6h2':0.5,'12k1':1,'4f':0.333,'4e':0.333},Nb=NbC14,model='DSEM')


#Set composition and temperature
#alloy_DSEM.temperatures_list([30,60,90])
alloy_DSEM.set_temperatures()
# alloy_DSEM.compositions_list(composition,step=0.01)



    
    
begin = time.time()
alloy_DSEM.cH = {}  #It is necessary define these dicts if one want to recalculate the same composition and T with different thermodynamic parameters
alloy_DSEM.P = {}
for T in alloy_DSEM.TEMPERATURES:
    if len(alloy_DSEM.alloy_composition) ==0:
        alloy_DSEM.set_composition(step=0.01)
    if len(alloy_DSEM.alloy_composition) !=0:
        alloy_DSEM.alloy_initialisation(step=0.01)
    # alloy_DSEM.compositions_list(composition,step=0.05)
    alloy_DSEM.C14.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=False)#0.001)
    if T>0:
        alloy_DSEM.C14.CalcPressure(T)
        alloy_DSEM.cH[T] = alloy_DSEM.C14.cHvals[T]
        alloy_DSEM.P[T] = alloy_DSEM.C14.pressure_total[T]


alloy_DSEM.SaveData(save_figures=True)

