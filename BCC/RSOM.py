#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import os
import math
from itertools import zip_longest
from itertools import combinations_with_replacement
from datetime import datetime  
import time
import imageio.v3 as iio
from matplotlib.ticker import FormatStrFormatter
from joblib import Parallel, delayed, parallel_config
from bisect import bisect_left

hi_values = pd.read_csv('./hi.csv', sep = ';').to_numpy() #importa os valores de hi do arquivo csv e transforma em uma matriz 
bonding_alpha = pd.read_csv('./bonding_alpha.csv', sep = ';').to_numpy() #importa os valores da energia de ligação da estrutura alpha
bonding_delta = pd.read_csv('./bonding_delta.csv', sep = ';').to_numpy()#importa os valores da energia de ligação da estrutura delta
composition_vetor = np.zeros(len(hi_values)) #cria um vetor de zeros com a mesma quantidade de coordenadas que entradas de elementos na matriz hi_values
R = 0.00831446261815324 #kJ/mol


# In[2]:


class Thermodynamics():
    """
    A class to perform thermodynamic calculations for various phases and states.
    It includes methods to compute enthalpy, entropy, Gibbs free energy, and chemical potentials
    based on the DSEM (Discrete Site Energy Model) and RSOM (Random Site Occupation Model) approaches.
    """
    
    def __init__(self):
        """
        Initializes the Thermodynamics class with default attributes for thermodynamic properties.
        """
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
            """
            Calculates the standard entropy of hydrogen gas (H2) at a given temperature.
    
            :param T: Temperature in Kelvin (K)
            :return: Standard entropy (S0) in kJ/mol of H2
            """
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
        """
        Calculates the enthalpy using the DSEM model.

        :param occupation_k_type_sites: Array representing occupation of k-type sites.
        :return: Enthalpy value.
        """
        h_m = (sum((occupation_k_type_sites * self.theta_k * self.h_k_type_sites)))
        h = h_m + self.H_M
        return h

    def Entropy_DSEM(self,occupation_k_type_sites,T):
        """
        Calculates the entropy using the DSEM model.

        :param occupation_k_type_sites: Array representing occupation of k-type sites.
        :param T: Temperature in Kelvin (K).
        :return: Entropy value.
        """
        t=T/1000
        A = 33.066178
        B = -11.363417
        C = 11.432816
        D = -2.772874
        E = -0.158558
        G = 172.707974
        S_0 = A*ln(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G # Hydrogen standard entropy J/mol of H2- NIST
        S_0 = S_0/1000 #kJ/mol of H2

        
        fh = sum(occupation_k_type_sites*self.k_type_sites_probability)

        if self.Nb==0:
            Pb=0
        else:
            Pb = 1-exp(-(self.Nb)*sum(occupation_k_type_sites*self.k_type_sites_probability))

        
        Ph_k = occupation_k_type_sites/(1-Pb) #correto?
        Pv_k = (1 - occupation_k_type_sites - Pb)/(1-Pb)
        sc = -R * sum(self.theta_k*(1-Pb)*(Ph_k*ln(Ph_k)+Pv_k*ln(Pv_k)))
        s = sc - ((sum(self.theta_k * occupation_k_type_sites) * S_0)/2)
        return s


    def Gibbs_DSEM(self,fk,T):
        """
        Calculates Gibbs free energy using the DSEM model.

        :param fk: Array representing occupation probabilities for k-type sites.
        :param T: Temperature in Kelvin (K).
        :return: Gibbs free energy value.
        """
        
        #fk = x
        if T!= 0:
            DG = self.Enthalpy_DSEM(fk) - T* self.Entropy_DSEM(fk,T)
        if T== 0:
            DG = self.Enthalpy_DSEM(fk)
        return DG
        


    def Gibbs_minimization_DSEM(self,c_h,T):
        """
        Minimizes the Gibbs free energy using the DSEM model for a given hydrogen concentration.

        :param c_h: Hydrogen concentration.
        :param T: Temperature in Kelvin (K).
        :return: Result of the minimization process containing optimized occupation values and minimized Gibbs energy.
        """

        lower_limit = 0.0001                #set the lower limit of occupation of different k-site, it can't be 0        
        bnds = []
        for i in self.fhk_limit:
            bnds.append((lower_limit,i)) #this is fh limit, but fh = sum (fhk *pk)    
        
        cons = ({'type': 'eq', 'fun': lambda fk: c_h - self.theta*sum(fk*self.k_type_sites_probability)}) #fh.theta = cH)
        if self.Nb==0:
            x0 = np.ones(len(self.theta_k))*(c_h/self.theta)
        else:
            x0 = np.ones(len(self.theta_k))*(c_h/self.theta)/2
        Gibbs_min = spo.minimize(self.Gibbs_DSEM,x0, args=(T,), options = {"disp": False, "maxiter": 2000}, method = 'SLSQP', constraints = cons, bounds = bnds)
        return Gibbs_min



    def ThermodynamicsCalculation_DSEM(self,T,n_threads=-1,step_interpolation=0.00001):
        """
        Performs thermodynamic calculations using the DSEM model for a given temperature.

        :param T: Temperature in Kelvin (K).
        :param n_threads: Number of threads for parallel computation (-1 uses all available cores).
        :param step_interpolation: Step size for interpolation of results.
        """
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


                
        
        begin = time.time()
        print(f"The calculation of {self.name} phase at {T} K has started.")
        self.occupation_k_site[T] = {}
                   
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
            ch = sum(Gibbs_minimized.x*self.theta_k)
            Gibbs_temp[np.where(self.cH==c_h)] = Gibbs_minimized.fun
            Enthalpy_temp[np.where(self.cH==c_h)] = self.Enthalpy_DSEM(occupation_k_type_sites)
            h_m_temp[np.where(self.cH==c_h)] = self.Enthalpy_DSEM(occupation_k_type_sites) - self.H_M
            if T != 0:
                Entropy_temp[np.where(self.cH==c_h)] = self.Entropy_DSEM(occupation_k_type_sites,T)
                Entropy_config_temp[np.where(self.cH == c_h)] = self.Entropy_DSEM(occupation_k_type_sites,T) + ((sum(self.theta_k * occupation_k_type_sites) * s0(T))/2)
            self.occupation_k_site[T][c_h] = occupation_k_type_sites
            #print(c_h, ch)
            
            theta_b_temp[np.where(self.cH==c_h)] = (self.theta - self.theta/(self.Nb+1))*(1-exp(-self.Nb*sum(occupation_k_type_sites*self.theta_k)))
            

        self.H[T] = Enthalpy_temp
        self.S[T] = Entropy_temp
        self.G[T] = Gibbs_temp
        self.S_c[T] = Entropy_config_temp
        self.h_m[T] = h_m_temp
        self.theta_b[T] = theta_b_temp

        ############### Calculating chemical potentials
        self.mu_H[T] = (self.G[T][2:] - self.G[T][0:-2])/(2*(self.cH_step))
        self.mu_M[T]= self.G[T][1:-1] - self.cH[1:-1] * self.mu_H[T][0:] 


        self.cHvals[T] = self.cH
        if step_interpolation != False:
        
            self.cHvals[T] = np.arange(0, self.cH_limit+step_interpolation, step_interpolation)
            self.H[T] = np.interp(self.cHvals[T], self.cH, self.H[T])
            self.S[T] = np.interp(self.cHvals[T], self.cH, self.S[T])
            self.S_c[T] = np.interp(self.cHvals[T], self.cH, self.S_c[T])
            self.G[T] = np.interp(self.cHvals[T], self.cH, self.G[T])
            self.mu_H[T] = np.interp(self.cHvals[T], self.cH[1:-1], self.mu_H[T])
            self.mu_M[T] = np.interp(self.cHvals[T], self.cH[1:-1], self.mu_M[T])

        
        end = time.time()
        tempo = end - begin
        print(f"The calculation takes {round(tempo/60,2)} minutes")



    def Enthalpy_RSOM(self,T):
        """
        Calculates the enthalpy using the RSOM (Random Site Occupation Model) approach.

        :param T: Temperature in Kelvin (K).
        :return: None. Updates the enthalpy (H) attribute for the given temperature.
        """
        self.H[T] = self.h_m * self.cH + self.H_M
        return None

    def Entropy_RSOM(self,T):
        """
        Calculates the entropy using the RSOM (Random Site Occupation Model) approach.

        :param T: Temperature in Kelvin (K).
        :return: None. Updates the entropy (S) and configurational entropy (S_c) attributes for the given temperature.
        """
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
            """
            Calculates the Gibbs free energy using the RSOM approach.
    
            :param T: Temperature in Kelvin (K).
            :return: None. Updates the Gibbs free energy (G) attribute for the given temperature.
            """
            if T!= 0:
                self.G[T] = self.H[T] - T* self.S[T]
            if T == 0:
                self.G[T] = self.H[T]

    def mu_H_RSOM(self,T):
            """
            Calculates the chemical potential of hydrogen using the RSOM approach.
    
            :param T: Temperature in Kelvin (K).
            :return: None. Updates the hydrogen chemical potential (mu_H) attribute for the given temperature.
            """
            if T>0:
                fh = self.cH/self.theta
                self.mu_H[T] = self.h_m  - T * (-R*ln((fh*(exp(-(self.Nb)*fh))**((self.Nb)*exp(-(self.Nb)*fh))/(exp(-(self.Nb)*fh)-fh)**(1+(self.Nb)*exp(-(self.Nb)*fh))))- self.s0(T)/2)
            if T==0:
                self.mu_H[T] = self.h_m * np.ones(len(self.cH))
    def mu_M_RSOM(self,T):
        """
        Calculates the chemical potential of metal using the RSOM approach.

        :param T: Temperature in Kelvin (K).
        :return: None. Updates the metal chemical potential (mu_M) attribute for the given temperature.
        """
        self.mu_M[T] = self.G[T] - self.cH * self.mu_H[T]

    def ThermodynamicsCalculation_RSOM(self,T,n_threads=-1,step_interpolation=0.00001):

        """
        Performs thermodynamic calculations using the RSOM model for a given temperature.

        :param T: Temperature in Kelvin (K).
        :param n_threads: Number of threads for parallel computation (-1 uses all available cores).
        :param step_interpolation: Step size for interpolation of results.
        :return: None. Updates various thermodynamic properties for the given temperature.
        """

                
        
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
        """
        Calculates pressure based on hydrogen chemical potential.

        :param T: Temperature in Kelvin (K).
        :param P_0: Reference pressure in bar.
        :return: None. Updates the total pressure attribute for the given temperature.
        """
        c=2/(R*T)
        self.pressure_total[T] = P_0*exp(c* self.mu_H[T][0:len(self.cHvals[T])])


# In[3]:


class Phase(Thermodynamics):
    """
    A class that represents a specific phase of a material and extends the Thermodynamics class.
    It includes attributes and methods for thermodynamic calculations using either the DSEM or RSOM models.
    """
    def __init__(self,name,structure,theta,Nb,n_site=4,model ='DSEM'):
        """
        Initializes the Phase class with specific properties for a given phase.

        :param name: Name of the phase (e.g., 'alpha', 'beta', 'delta').
        :param structure: Crystal structure of the phase (e.g., 'BCC', 'FCC').
        :param theta: Fraction of available interstitial sites.
        :param Nb: Blocking factor (number of blocked sites per occupied site).
        :param n_site: Number of interstitial sites per unit cell (default is 4 for tetrahedral sites).
        :param model: Thermodynamic model to use ('DSEM' or 'RSOM').
        """
        Thermodynamics.__init__(self)
        self.name = name
        self.H_M = []
        self.Nb = Nb
        self.theta = theta
        self.model = model
        self.cH_step = []
        self.cH = []
        self.cH_equilibrium = {}
        self.n_site_type = n_site # 4 for tetrahedral sites, 6 for octahedral sites 
        self.composition = None

        self.cH_limit = []
        if self.model == 'DSEM': #Create dicts and lists necessary for the model calculation
            self.h_m = {}
            self.k_type_sites = []
            self.k_type_sites_probability = []
            self.theta_k = []
            self.h_k_type_sites = []
            self.fhk_limit =[]
            self.set_cH_limit = self.set_cH_limit_DSEM
            self.ThermodynamicsCalculation = self.ThermodynamicsCalculation_DSEM

        if self.model == 'RSOM': #Create dicts and lists necessary for the model calculation
            self.composition = None
            self.h_m = None
            self.set_cH_limit = self.set_cH_limit_RSOM
            self.ThermodynamicsCalculation = self.ThermodynamicsCalculation_RSOM



        
        if self.name == "alpha":
            self.STRUCTURE = "BCC"
        if self.name == "beta":
            self.STRUCTURE = "BCC_ord"
        if self.name == "delta":
            self.STRUCTURE = "FCC"
        else:
            self.STRUCTURE == structure

    @property
    def h_m_calculation(self):
        """
        Calculates the molar enthalpy per site based on the phase name and alloy composition.

        :return: None. Updates the h_m attribute with the calculated value.
        """            
        if self.name == 'alpha':
            self.h_m = sum(self.composition * hi_values[0:,1])
            
        if self.name == 'beta':
            self.h_m = sum(self.composition * hi_values[0:,2])     
            
        if self.name == 'delta':
            self.h_m = sum(self.composition * hi_values[0:,3])

    
    @property
    def H_M_calculation(self):  #determines the Enthalpy due the phase transition from the reference -- H_M diff from 0 only for delta phase
        """
        Calculates the enthalpy difference due to phase transition from a reference state (H_M).
        
        This is only non-zero for the delta phase (FCC structure).

        :return: None. Updates the H_M attribute with the calculated value.
        """        
        if self.STRUCTURE == 'FCC':
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

            H_delta = E_total_delta - E_total_alpha  
            self.H_M = H_delta
        else:
            self.H_M = 0
        
#################################################  begin DSEM equations ############################################################################        
    @property    
    def k_type_sites_calculation(self):  #combinatorial analysis for determine all possible different k-sites
        """
        Performs combinatorial analysis to determine all possible k-type sites.
        
        :return: None. Updates the k_type_sites attribute with a list of all possible k-sites.
        """        
        self.k_type_sites = list(combinations_with_replacement(self.elements, self.n_site_type))
        
    @property
    def k_type_sites_probability_calculation(self): #combinatorial calculation for determine all probability of occurence for all different k-sites
        """
        Calculates the probability of occurrence for all different k-type sites using combinatorial methods.

        :return: None. Updates the k_type_sites_probability attribute with probabilities for each k-site.
        """
        prodx = []
        for interstitial_site in range(len(self.k_type_sites)):
            temp_thetak = []
            (unique,counts_k_site)= np.unique(self.k_type_sites[interstitial_site], return_counts = True)
            temp_factorial = []
            for element_atom in counts_k_site:
                temp_factorial.append(factorial(element_atom))
            delta = factorial(self.n_site_type)/np.prod(temp_factorial)
            for m in self.k_type_sites[interstitial_site]:
                temp_thetak.append(self.alloy_composition[m])
            prodx.append(np.prod(temp_thetak))
            self.k_type_sites_probability.append(delta * np.prod(temp_thetak)) 
        self.k_type_sites_probability = np.array(self.k_type_sites_probability)
            
    @property
    def theta_k_calculation(self):
        """
        Calculates the fraction of k-type sites occupied by hydrogen.

        :return: None. Updates the theta_k attribute with calculated values.
        """
        for p_k in self.k_type_sites_probability:
            self.theta_k.append((self.theta)*p_k)
        self.theta_k = np.array(self.theta_k)
    


    @property
    def h_k_type_sites_calculation(self):
        """
        Calculates the enthalpy associated with each k-type site based on phase structure.

        :return: None. Updates the h_k_type_sites attribute with calculated enthalpy values.
        """
        if self.name == "alpha":
            structure = 0
        if self.name == "beta":
            structure = 1
        if self.name == "delta":
            structure = 2
        for k_site in self.k_type_sites:
            temp_h = []
            for element_atom in k_site:
                temp_h.append(hi_values[np.where(hi_values==element_atom)[0],structure+1][0]) #mudar structure para name e mudar na planilha
            self.h_k_type_sites.append(sum(temp_h)/float(self.n_site_type))   
        self.h_k_type_sites = np.array(self.h_k_type_sites)



    
    def set_cH_limit_DSEM(self):
            """
            Determines the maximum hydrogen concentration limit (cH_limit) using the DSEM model.
    
            :return: None. Updates cH_limit and fhk_limit attributes.
            """
            def Pb(x):
                return np.linalg.norm(x-exp(-(self.Nb)*sum(x*self.k_type_sites_probability))) #Determine fh  real
                 
              
            if self.Nb!=0:
                x0 = np.random.randint(1,100, len(self.k_type_sites_probability))
                x0 = x0/100000
                #bnds = [(0,1)]
                bnds = []
                for i in range(len(self.theta_k)):
                    bnds.append((0.0001 ,0.9899))
                cons = {'type': 'eq', 'fun': lambda x: Pb(x),
                               }
                cH_limit = spo.minimize(Pb,x0, options = {"disp": False, "maxiter": 20000}, tol=10**-7, method = 'SLSQP',  constraints = cons, bounds = bnds)
                self.cH_limit = np.round(np.sum(cH_limit.x*self.k_type_sites_probability)*self.theta,4)
                self.fhk_limit = cH_limit.x
            if self.Nb ==0:
                self.cH_limit = 1.9999
                self.fhk_limit = np.full(len(self.theta_k),0.9999)


################################################# end DSEM equations ############################################################################   

################################################# begin RSOM equations ############################################################################       
    
    def set_cH_limit_RSOM(self):
            """
            Determines the maximum hydrogen concentration limit (cH_limit) using the RSOM model.
        
            :return: None. Updates cH_limit attribute.
            """
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
        """
        Sets an array of hydrogen concentrations (c_H) up to the calculated limit.

        :return: None. Updates the cH attribute with an array of hydrogen concentrations.
        """
        c_H = []
        for i in range(1,int(2/self.cH_step +1)):
            if (self.cH_step*i)< self.cH_limit:
                c_H.append(np.round(self.cH_step*i,4))
        self.cH = np.array(c_H)
        


# In[4]:


class Alloy(Phase):
    """
    A class representing an alloy system, inheriting from the Phase class.
    This class handles alloy-specific properties, initialization, thermodynamic calculations, and data saving.
    """
    def __init__(self):
        """
        Initializes the Alloy class with default attributes for temperature, pressure, hydrogen concentration,
        equilibrium data, and alloy composition.
        """
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

    def set_composition(self,step=0.01):#stepBCC=0.001,stepBCC_ord=0.001,stepFCC=0.001):
        """
        Allows the user to input the alloy composition interactively and initializes the alloy.

        :param step: Step size for hydrogen concentration variation (default is 0.01).
        :return: None. Updates alloy composition and initializes the alloy.
        """
        self.elements = []
        self.composition = []
        self.alloy_composition ={}
        stop = False
        
        ##########Entrada de dados pelo usuário################
        while (stop!= True):
            y = input("Enter with an element: ")
            #adicionar um if elemento not in lista de elementos: Elemento não implementado ao modelo
            self.elements.append(y) #adiciona a entrada y no fim da lista de elementos
            x = float(input("Atomic fraction of element {}: " .format(y)))
            self.composition.append(x) #adiciona a entrada w no fim da lista de composições
            if y in self.alloy_composition:
                self.alloy_composition[y]= self.alloy_composition[y] + x
            else:
                self.alloy_composition[y]=x
            z = input("Would you like to add another element? Y/n ")
            if z == "n":
                stop = True
        for i in self.alloy_composition:
            self.alloy_composition[i] = self.alloy_composition[i]/sum(self.composition)  
        self.alloy_initialisation(step)
        
    def alloy_initialisation(self,step):
        """
        Initializes alloy properties and phases based on the composition and step size.

        :param step: Step size for hydrogen concentration variation (default is provided by set_composition).
        :return: None. Updates attributes related to phases and thermodynamic calculations.
        """
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
                    self.__dict__[i].k_type_sites_calculation                   #calculates the each different k-site
                    self.__dict__[i].k_type_sites_probability_calculation       #calculates the probability of each different k-site occurs
                    self.__dict__[i].theta_k_calculation                        #calculates the quantity of interstitial sites per metal atom for each different k-site
                    self.__dict__[i].h_k_type_sites_calculation                 #calculates the energy of interstitial sites for each different k-site

                if self.__dict__[i].model == 'RSOM':
                    self.composition_vetor = np.zeros(len(hi_values))
                    for element in self.elements:
                        self.composition_vetor[np.where(hi_values==element)[0]] = self.alloy_composition[element]
                        
                    self.__dict__[i].composition = self.composition_vetor
                    self.__dict__[i].h_m_calculation
                self.__dict__[i].set_cH_limit()                               #set the Hydrogen composition limit for the phase
                self.__dict__[i].cH_step = step                      #set the step variation in the Hydrogen composition for the termodynamic calculation
                self.__dict__[i].set_cH                                     #set all Hydrogen composition possible for the phase
        
        return print(f"The alloy inserted is: {self.alloy}")   

    def compositions_list(self,composition,step):   
        """
        Sets the alloy composition from a given dictionary and initializes properties.

        :param composition: Dictionary mapping elements to their atomic fractions.
        :param step: Step size for hydrogen concentration variation (default is provided by set_composition).
        :return: None. Updates attributes related to compositions and phases.
        """
        self.elements = []
        self.composition = []
        self.alloy_composition ={}
        self.alloy_composition = composition
        for element,x in self.alloy_composition.items():
            self.elements.append(element)
            self.composition.append(x)
        for i in self.alloy_composition:
            self.alloy_composition[i] = self.alloy_composition[i]/sum(self.composition)  
        self.alloy_initialisation(step)


    def set_temperatures(self):
        """
        Allows the user to input a list of temperatures interactively for thermodynamic calculations.

        :return: None. Updates the TEMPERATURES attribute with user-provided values in Kelvin.
        """
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
        """
        Sets a list of temperatures based on a given input array.

        :param temperature: List of temperatures in Celsius.
        :return: None. Updates TEMPERATURES attribute with values converted to Kelvin.
        """
        self.TEMPERATURES = []
        for T in temperature:
            self.TEMPERATURES.append(T+273.15)
        self.TEMPERATURES.sort() #sort the temperatures in ascending order


    def find_plateau(self,Phase1,Phase2,temperature, search="binary",tol=0.005):
        """
        Finds equilibrium plateaus between two phases at a given temperature.

        :param Phase1: First phase object (e.g., alpha phase).
        :param Phase2: Second phase object (e.g., beta phase).
        :param temperature: Temperature at which to find equilibrium (in Kelvin).
        :param search: Search method ('binary' or 'old').
                       - 'binary': Uses binary search for faster performance.
                       - 'old': Uses a direct comparison approach.
        :param tol: Tolerance for matching chemical potentials (default is 0.005).
        :return: Equilibrium hydrogen composition (cH_eq) if found; otherwise, None.
        """        
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
        

            if idx_cH:
                cH_eq=np.round([Phase1.cHvals[T][idx_cH[0][0]],Phase2.cHvals[T][idx_cH[1][0]]],4)
                mu_h_eq = Phase1.mu_H[T][idx_cH[0][0]]
                self.cH_equilibrium[T][f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"] = cH_eq
                self.mu_H_equilibrium[T][f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"] = mu_h_eq
                print("cH_eq encontrado:", cH_eq)
                eq = f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"
                self.H_eq[T][eq] = (Phase2.H[T][idx_cH[1][0]] -  Phase1.H[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[1][0]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.S_eq[T][eq] = (Phase2.S[T][idx_cH[1][0]] -  Phase1.S[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[1][0]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.P_plat[T][eq] = exp(2* ((self.H_eq[T][eq]/(R*T) - (self.S_eq[T][eq]/R))))  
                
            else:
                print(f"None equilibrium found between {Phase1.STRUCTURE} and {Phase2.STRUCTURE}")
            
        if search == "binary":   

                        # Função auxiliar para busca binária
            def busca_binaria(arr, valor, tol):
                """
                Helper function to perform binary search with tolerance.

                :param arr: Array to search within.
                :param value: Value to find within tolerance.
                :param tol: Tolerance for matching values.
                :return: Index of matching value or None if not found.
                """
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
                self.cH_equilibrium[T][f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"] = cH_eq
                self.mu_H_equilibrium[T][f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"] = mu_h_eq
                print("cH_eq encontrado:", cH_eq)
                eq = f"{Phase1.STRUCTURE} - {Phase2.STRUCTURE}"
                self.H_eq[T][eq] = (Phase2.H[T][idx_cH[0][1]] -  Phase1.H[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[0][1]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.S_eq[T][eq] = (Phase2.S[T][idx_cH[0][1]] -  Phase1.S[T][idx_cH[0][0]])/ (Phase2.cHvals[T][idx_cH[0][1]] - Phase1.cHvals[T][idx_cH[0][0]])
                self.P_plat[T][eq] = exp(2* ((self.H_eq[T][eq]/(R*T) - (self.S_eq[T][eq]/R))))           
                return cH_eq
        
            else:
                print(f"None equilibrium found between {Phase1.STRUCTURE} and {Phase2.STRUCTURE}")
        
            



    def calculatePCI(self,T,t=0.05):
        """
        Calculates Pressure-Composition-Isotherms (PCI) for a given temperature.

        :param T: Temperature in Kelvin.
        :param t: Tolerance for identifying equilibrium points (default is 0.05).
        :return: None. Updates the equilibriums and phases attributes with calculated values.
        """
        self.equilibriums[T] = {}


        #Atual funcionando
        def find_real_equilibrium(mu_H_equilibrium, cH_equilibrium,tol, initial_equilibrium=None):
            """
            Finds real equilibrium points by comparing chemical potentials and hydrogen concentrations.

            :param mu_H_equilibrium: Dictionary of equilibrium chemical potentials for each phase pair.
            :param cH_equilibrium: Dictionary of equilibrium hydrogen concentrations for each phase pair.
            :param tol: Tolerance for comparison.
            :param initial_equilibrium: Initial equilibrium point to start the search (optional).
            :return: Dictionary of identified equilibrium points.
            """
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
                        if phase1 !=phase3:
                            cHplat = values[0]
                            cHG = values2[0]
                            findcH = busca_binaria(self.__dict__[phase1].cHvals[T], cHplat, tol)
                            g1=self.__dict__[phase1].G[T][findcH]
                            g2=self.__dict__[phase3].G[T][findcH]
                            if g1<g2 and cHplat < best_cHplat:
                                best_equilibrium = key
                                best_cHplat = cHplat

                
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
        """
        Saves data related to the alloy's thermodynamic properties and equilibrium calculations.

        :param custom_name: Custom name for the alloy (default is None, uses the alloy name).
        :param custom_suffix: Custom suffix to append to the alloy name (default is None).
        :param custom_structure: List of phases to include in the saved data (default is None, includes all phases).
        :param custom_temperature: Specific temperatures to include in saved data (default is None, includes all temperatures).
        :param save_occ: Whether to save site occupation data (default is False).
        :param save_figures: Whether to save figures for thermodynamic properties (default is False).
        :param cmmax: Maximum margin for occupation plots (default is 0.25).
        :param cfsize: Font size for occupation plots (default is 4).
        :param ctol: Tolerance for matching hydrogen concentrations in site occupation plots (default is 0.05).
        """
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
                file.write(f"{self.__dict__[i].STRUCTURE} = {self.__dict__[i].theta,self.__dict__[i].Nb}| ") 

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
            
            if not os.path.isdir(f'./{alloy_name}/{T}/data'): #Verifica se a pasta já existe e se não existe cria ela com a proxima linha
                    os.mkdir(f'./{alloy_name}/{T}/data')
            my_path = os.path.abspath(alloy_name+ f"/{T}/data") # Figures out the absolute path for you in case your working directory moves around.
        
                    

            properties = ['H','S','S_c','G','mu_H','mu_M']
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
                dataframe = dataframe.rename(columns = { 0 : "c_H", 1 : "Pressure"})
                dataframe.to_csv(os.path.join(my_path, my_file), sep = '\t', encoding = "utf-8", index = False, columns=['c_H', "Pressure"])
        
                            
                        
       

    def SaveOccPlateau(self,name,mmax=0.25,fsize=4,tol = 0.05):
        """
        Saves bar plots of site occupation probabilities for each phase at equilibrium hydrogen concentrations.
    
        :param name: Name of the alloy.
        :param mmax: Maximum margin for the y-axis of the bar plots (default is 0.25).
        :param fsize: Font size for site labels in the bar plots (default is 4).
        :param tol: Tolerance for matching hydrogen concentrations (default is 0.05).
        """
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
        """
        Saves figures related to thermodynamic properties such as PCI curves and property trends.

        :param name: Name of the alloy.
        :param phases: List of phases to include in the figures.
        """
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
                    axes.plot(phase.cHvals[T], phase.__dict__[prop][T], label = r"$\Delta$" f"${prop}^{{\\{phase.name}}}$")
                axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
                axes.set_ylabel(r"$\Delta$" f"${prop}$ [kJ/mol]")
                axes.set_title(f'T = {T} K')
                axes.legend(loc = 0)
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()
              
            for prop in properties2:
            
                my_file = f'{alloy_name}_{prop}_T_{T}K.png'
                fig = plt.figure()
        
                axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
        
                for phase in phases:
                    axes.plot(phase.cHvals[T], phase.__dict__[prop][T], label = r"$\Delta$" f"${prop}^{{\\{phase.name}}}$")
                axes.set_xlabel('$c_H$') # Notice the use of set_ to begin methods
                axes.set_ylabel(r"$\Delta$" f"$\\{prop}$ [kJ/mol]")
                axes.set_title(f'T = {T} K')
                axes.legend(loc = 0)
                fig.savefig(os.path.join(my_path, my_file), dpi=200, bbox_inches='tight')
                plt.close()


# # Example DSE model calculation

# In[64]:


# =============================================================================
# #Define an Object Alloy
# alloy_DSEM = Alloy()
# 
# ## set Structures
# alloy_DSEM.BCC = Phase(name='alpha',structure = 'BCC', theta=6,Nb=14,model='DSEM')
# alloy_DSEM.BCC_ord = Phase(name='beta',structure = 'BCC_ord', theta=4,Nb=6,model = 'DSEM')
# alloy_DSEM.FCC = Phase(name='delta',structure = 'FCC', theta=2,Nb=0,model='DSEM')
# 
# 
# #Set composition and temperature
# alloy_DSEM.set_composition(step=0.005) #This function asks for each element/composition separately
# #composition = {'Ti':75/3, 'V':75/3, 'Nb':75/3, 'Cr':25}  #This dictionary of elements is necessary to use .compositions_list method below
# #alloy_DSEM.compositions_list(composition,step=0.005) #Set composition and initialises the composition
# 
# alloy_DSEM.set_temperatures() #This function asks for each temperature separately
# #alloy.temperatures_list([100,150,200]) #set all temperatures which should be calculated
# 
# 
# 
# # In[25]:
# 
# 
# #alloy_DSEM.temperatures_list([25]) #Temperatures in degrees Celsius
# begin = time.time()
# alloy_DSEM.cH = {}  #It is necessary define these dicts if one want to recalculate the same composition and T with different thermodynamic parameters
# alloy_DSEM.P = {}
# for T in alloy_DSEM.TEMPERATURES:
#     alloy_DSEM.BCC.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
#     alloy_DSEM.BCC_ord.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
#     alloy_DSEM.FCC.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
#     if T>0:
#         alloy_DSEM.BCC.CalcPressure(T)
#         alloy_DSEM.BCC_ord.CalcPressure(T)
#         alloy_DSEM.FCC.CalcPressure(T)
#         alloy_DSEM.find_plateau(alloy_DSEM.BCC,alloy_DSEM.FCC,T,search='binary',tol=0.01)
#         alloy_DSEM.find_plateau(alloy_DSEM.BCC,alloy_DSEM.BCC_ord,T,search='binary',tol=0.01)
#         alloy_DSEM.find_plateau(alloy_DSEM.BCC_ord,alloy_DSEM.FCC,T,search='binary',tol=0.01)
#         alloy_DSEM.calculatePCI(T,t=0.00001)
# alloy_DSEM.SaveData(save_occ=True,save_figures=True)
# end = time.time()
# print(f" O calculo levou {end-begin} segundos")
# =============================================================================


# # Example RSO model calculation

# In[9]:


#Define an Object Alloy
alloy = Alloy()

## set Structures
alloy.BCC = Phase(name='alpha',structure = 'BCC', theta=6,Nb=14,model='RSOM')
alloy.BCC_ord = Phase(name='beta',structure = 'BCC_ord', theta=4,Nb=6,model = 'RSOM')
alloy.FCC = Phase(name='delta',structure = 'FCC', theta=2,Nb=0,model='RSOM')

#com = {T}
#Set composition and temperature
#alloy.compositions_list(comp)
alloy.set_composition(step=0.005)
alloy.set_temperatures()
#alloy.temperatures_list([100,150,200])



# In[114]:


alloy.temperatures_list([100]) #Temperatures in degrees Celsius
begin = time.time()
alloy.cH = {}  #It is necessary define these dicts if one want to recalculate the same composition and T with different thermodynamic parameters
alloy.P = {}
for T in alloy.TEMPERATURES:
    alloy.BCC.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
    alloy.BCC_ord.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
    alloy.FCC.ThermodynamicsCalculation(T,n_threads=7,step_interpolation=0.0001)
    if T>0:
        alloy.BCC.CalcPressure(T)
        alloy.BCC_ord.CalcPressure(T)
        alloy.FCC.CalcPressure(T)
        alloy.find_plateau(alloy.BCC,alloy.FCC,T,search='binary',tol=0.1)
        alloy.find_plateau(alloy.BCC,alloy.BCC_ord,T,search='binary',tol=0.1)
        alloy.find_plateau(alloy.BCC_ord,alloy.FCC,T,search='binary',tol=0.1)
        alloy.calculatePCI(T)
alloy.SaveData(custom_suffix = '_RSOM',save_figures=True)
end = time.time()
print(f" O calculo levou {end-begin} segundos")

