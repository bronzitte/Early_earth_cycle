#! /usr/local/bin/python

#########################################################################
# import necessary modules and define global variables

import time  # time module used to generate random seeds
from n_out_functions_no_land import *
from model_functions_reformate import Forward_Model  # import forward model function
import numpy as np
import pandas as pd
from scipy import interpolate
global W, F_outgass, n, climp, tdep_weath, mod_sea, alt_frac, Mp_frac, CWF, carb_exp, sed_thick, F_carbw, fpel, deep_grad, f
# import pylab
# import scipy.stats
# import functions used for sampling in Archean no land cases

##########################################################################
# Options
# Number of forward model calls in Monte Carlo calculations
# 1000 provides approximation distrubitions, 10000 used in manuscript
it_num = 1000

# Parallelize on/off number
# 0 - no parallelization (slow)
# number of threads (e.g. 4, 6, 8, 10, 12)
Parallelize = 30

# Tdep carbon chemistry constants
# 0 - carbon chemsitry assumes fixed temperature (faster, but slightly less accurate)
# 1 - carbon chemistry equilibrium constants are temperature dependent (slower, but more accurate)
Carbon_chem = 1

# methane
# 0 - no atmospheric methane at all times
# 1 - 100 ppm methane in Proterozoic and 1% methane in Archean (see manuscript).
Methane_on = 0

# Save options. Will later be called by the forward model.
options_array = np.array([Parallelize, Carbon_chem, Methane_on])
np.save('options_array.npy', options_array)

#########################################################################

# Dissolution_change_array=np.zeros(shape=it_num)
# This array will contain the mass conservation imbalance for each forward model run
imbalance_array = []

# fit the land growth function from local excel
df = pd.read_excel('TM_age.xlsx')
f = interpolate.interp1d(df.iloc[:, 0], df.iloc[:, 1])

# Run forward model once with arbitrary parameters to define size of array that will contain outputs
[temporary_array, imb] = Forward_Model(6.0e4, 6e12, 1.7, 0.3, 13.7, 0.5e12, 0.7,
                                       0.01, 0.9, 0.3, 1.0, 11e12, 0.0, 0.5, 1.0, 
                                       0.2, 0.3, 0.5, 0.5, 2.5, 0.01, 90000, f)
# This array will contain forward model outputs for all iterations
all_output = np.zeros([len(temporary_array), len(temporary_array[4]), it_num])


def try_run_forward(ii):
    ij = 0
    while ij < 1:
        print(ii + 1, " of ", it_num)
        # Generate random seed for sampling parameter distributions
        # use remainder of this division as the random seed
        mtime = int(time.time()) % (ii+100000)
        np.random.seed(mtime)

        ################################################################
        # Sample uniform distribution for unknown parameters
        # For each parameter, the two numbers in brackets define
        # the range of their uniform distribution.
        ################################################################
        # Modern outgassing flux (mol C/yr)
        F_outgass = np.random.uniform(6e12, 10e12)
        # Exponent for carbonate precipitation (see equation S19)
        n = np.random.uniform(1.0, 2.5)
        # Modern ratio of seafloor dissolution to carbonate precipitation (Table S2)
        alt_frac = np.random.uniform(0.5, 1.5)
        # e-folding temperature for continental weathering (see equation 1)
        tdep_weath = np.random.uniform(10., 40.0)
        # exponent for CO2-dependence of continental silicate weathering (see equation 1)
        climp = np.random.uniform(0.1, 0.5)
        # Mixing time (yrs) for pore space water.
        W = np.random.uniform(2e4, 1e6)
        # Water fraction in pore space relative to ocean
        Mp_frac = 0.01  
        # Archean land fraction (set negative for Archean ocean world e.g. lfrac=-0.2)
        lfrac = np.random.uniform(0.1, 0.75)
        # Timing for growth of continents (Ga)
        growth_timing = np.random.uniform(2.0, 3.0)
        # Archean Ca abundance (range 0.0 to 500.0 for sensitvitiy tests Fig. S8 and S9)
        new_add_Ca = 0.0
        # Modern seafloor dissolution (mol/yr)
        mod_sea = .45e12  
        # exponent of CO2-dependence continetnal carbonate weathering (see equation S2)
        carb_exp = np.random.uniform(0.1, 0.5)
        # sediment thickness in Archean relative to modern (see equation S5)
        sed_thick = np.random.uniform(0.2, 1.0)
        # Fraction pelagic carbonate - no longer in use this version of code
        fpel = 0.0  
        # Modern continental carbonate weathering (mol C/yr)
        F_carbw = np.random.uniform(7e12, 14e12)
        # Biological enhancement of weathering (Archean value relative to modern)
        # CWF changed in Jan 2022
        CWF = 0.25
        # Gradient determining linear relationship between deep ocean temperatures and surface temperatures (see equation S20)
        deep_grad = np.random.uniform(0.8, 1.4)
        # Exponent determining pH-dependence of seafloor basalt dissolution and pore space pH (see equation S3)
        coef_for_diss = np.random.uniform(0.0, 0.5)
        # Exponent determining relationship between spreading rate and seafloor dissolution (see equation S10)
        beta = np.random.uniform(0.0, 2.0)
        # Exponent determing relationhip between crustal production and outgassing (see equation S9)
        mm = np.random.uniform(1.0, 2.0)
        # Exponent determing relationship between internal heat flow and outgassing (see equation S8)
        n_out = np.random.uniform(0.0, 0.73)
        # Effective activation energy for seafloor dissolution (see equation S3)
        Ebas = np.random.uniform(60000., 100000.)

        # n_out=no_land_no_methane() # For no-land, no-methane case (see ReadMe)
        # n_out=no_land_with_methane() # For no-land, with methane case (see ReadMe)

        #################################################################

        # Attempt to run forward model
        try:
            [all_output[:, :, ii], imb] = Forward_Model(W, F_outgass, n, climp, tdep_weath, mod_sea, alt_frac, Mp_frac, lfrac,
                                                        carb_exp, sed_thick, F_carbw, fpel, CWF, deep_grad, coef_for_diss, beta, n_out, mm, growth_timing, new_add_Ca, Ebas, f)
            # Dissolution_change_array[ii]=all_output[19,np.size(all_output[19,:,:])/it_num-1,ii]/all_output[19,0,ii]
            # If non-sensical outputs, report error and try iteration again
            if (np.isnan(all_output[7, 98, ii])) or (all_output[14, 98, ii] < 0.0):
                print("error, forward model produced non physical outputs - try again")
                print(" ")
            elif abs(imb) > 0.2:  # Check mass conservation, print warning and retry if large imbalance
                print("error, model not conserving mass - try again")
            else:
                # Return iteration number, carbon cycle outputs, mass imbalance, and various input parameters.
                ij += 1
                return ii, all_output[:, :, ii], imb, n_out, beta, mm, coef_for_diss, Ebas, deep_grad, carb_exp, tdep_weath, climp, W, n
        except:  # if forward model call unsuccessful, print error message and try again
            print("error, forward model failed - try again")
            print(" ")


# Non-parallelized version, run all forward model calls in same thread:
if Parallelize == 0:
    kk = 0
    while kk < it_num:
        try:
            [jj, all_output[:, :, kk], imbalan, n_out, beta, mm, coef_for_diss, Ebas, deep_grad, carb_exp, tdep_weath,
                climp, W, n] = try_run_forward(kk)  # fill in kk-th element of output array, and record mass imbalance
            imbalance_array.append(imbalan)
            kk = kk+1
        except:
            print("Try again")

# Parallelized version, distribute forward model calls among 'Parallelize' number of threads
else:
    items = range(it_num)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(Parallelize) as executor:
        for row, result, imbal, n_out, beta, mm, coef_for_diss, Ebas, deep_grad, carb_exp, tdep_weath, climp, W, n in executor.map(try_run_forward, items):
            all_output[:, :, row] = result
            imbalance_array.append(imbal)

##################################
# Saving results
##################################

ppCO2 = 10**-6  # For converting ppm to partial pressures in bar

# create confidence intervals from ensemble of outputs
low_c = 2.5  # lower bound for 95% confidence interval
mid_c = 50.0  # median
high_c = 97.5  # upper bound for 95% confidence interval

# Create confidence interval arrays for output variables of interest:


# def get_confidence(x): return scipy.stats.scoreatpercentile(all_output[x, :, :],
#                                                             [low_c, mid_c,
#                                                              high_c],
#                                                             interpolation_method='fraction',
#                                                             axis=1)


# confidence_DICo = get_confidence(0)  # Ocean dissolved inorganic carbon
# confidence_ALKo = get_confidence(1)  # Ocean alkalinity
# confidence_DICp = get_confidence(2)  # Pore space dissolved inorganic carbon
# confidence_ALKp = get_confidence(3)  # Pore space alkalinity
# # 4-Time/Age
# confidence_pH_o = get_confidence(5)  # ocean pH
# confidence_CO2o = get_confidence(6)  # atmospheric pCO2
# confidence_pH_p = get_confidence(7)  # pore space pH
# # 8-
# confidence_Ca_o = get_confidence(9)  # calicum molality in ocean
# confidence_Ca_p = get_confidence(10)  # calcium molality in pore space
# confidence_CO3_o = get_confidence(11)  # Carbonate molality in ocean
# confidence_CO3_p = get_confidence(12)  # carbonate molality in pore space
# confidence_HCO3_o = get_confidence(13)  # bicarbonate molality in ocean
# confidence_HCO3_p = get_confidence(14)  # bicarbonate molality in pore space

# confidence_omega_o = get_confidence(15)  # Saturation state ocean
# confidence_omega_p = get_confidence(16)  # Saturation state pore space
# confidence_Tsurf = get_confidence(17)  # Average surface temperature
# confidence_Tdeep = get_confidence(18)  # Average deep water temperature

# confidence_Fd = get_confidence(19)  # Seafloor dissolution flux
# confidence_Fs = get_confidence(20)  # Continental silicate weathering flux
# confidence_Prec_o = get_confidence(21)  # Ocean cabonate precipitation
# confidence_Prec_p = get_confidence(22)  # Pore space carbonate precipitation

# confidence_Volc = get_confidence(24)  # Volcanic outgassing flux
# confidence_T_pore = get_confidence(25)  # Average pore space temperature


# Convert time axis from years to Ga
all_output[4, :, 0] = all_output[4, :, 0] / 1e9  

# Save the results in to csv files
res_dict = {
    4: "Time",
    5: "Ocean_pH",
    6: "pCO2",
    7: "Pore_pH",
    9: "Ca_molality_ocean",
    10: "Ca_molality_pore",
    11: "CO3_molality_ocean",
    12: "CO3_molality_pore",
    17: "Average_Surface_SeaT",
    18: "Average_Deep_SeaT",
    19: "Seafloor_dis_flux",
    20: "Cont_Si_weathering_flux",
    24: "Volc_out_flux",
    25: "Average_poreT",
    26: "Weather_rate_ratio",
    27: "Temp_e",
    28: "carb_exp",
    29: "carb_weather",
    30: "bio_factor"
}

for key, val in res_dict.items():
    np.savetxt('{}.csv'.format(val), all_output[key, :, :], delimiter=',')

################################
## Plotting figure #############
#################################

# strt_lim=0.0 # lower limit on x-axis (0 Ga)
# fin_lim=4.0 # upper limit on x-axis (4 Ga)

# pylab.figure(figsize=(13,8)) ## create multipanel figure for pH, CO2, outgassing, temp, continental and silicate weathering through time

# # Subplot for ocean pH through time
# pylab.subplot(2, 3, 1)
# pylab.plot(all_output[4,:,0],confidence_pH_o[1],'k',linewidth=2.5) # plot median ocean pH
# pylab.fill_between(all_output[4,:,0], confidence_pH_o[0], confidence_pH_o[2], color='grey', alpha='0.4') #plot ocean pH confidence interval
# pylab.xlabel('Time (Ga)')
# pylab.ylabel('Ocean pH')
# pylab.xlim([strt_lim,fin_lim]) # x axis limits
# HB_low=np.loadtxt('Halevy_Bachan_low.txt',delimiter=',') # load Halevy and Bachan 95% confidence lower bound
# HB_high=np.loadtxt('Halevy_Bachan_high.txt',delimiter=',') # load Halevy and Bachan 95% confidence upper bound
# pylab.plot(HB_low[:,0],HB_low[:,1],'r--',label='Halevy17 95% confidence') # plot Halevy and Bachan for comparison
# pylab.plot(HB_high[:,0],HB_high[:,1],'r--')# plot Halevy and Bachan for comparison
# pylab.legend(numpoints=1,frameon=False) #display legend
# pylab.text(-0.7, 9.5, 'A', fontsize=16, fontweight='bold', va='top') # label subplot

# #subplot for atmospheric CO2 through time
# pylab.subplot(2, 3, 2)
# pylab.semilogy(all_output[4,:,0],confidence_CO2o[1],'k',linewidth=2.5)
# pylab.fill_between(all_output[4,:,0], confidence_CO2o[0], confidence_CO2o[2], color='grey', alpha='0.4')
# ### Literature proxies for comparison
# Sheldon_CO2_low=np.array([8.2,7.67,10.1,15.,0.6,2.2])*370*ppCO2 # lower limit Sheldon 2006
# Sheldon_CO2_best=np.array([26.,23,31,45.9,1.6,7.0])*370*ppCO2 # median estimate Sheldon 2006
# Sheldon_CO2_hi=np.array([72.7,69,93,138,4.9,20.3])*370*ppCO2 # upper limit Sheldon 2006
# Sheldon_dates=np.array([2.5,2.2,2.0,1.8,1.1,0.98]) # time in Ga
# pylab.errorbar(Sheldon_dates,Sheldon_CO2_best,yerr=[Sheldon_CO2_best-Sheldon_CO2_low,Sheldon_CO2_hi-Sheldon_CO2_best],color='g',marker='o',linestyle="None",label='Sheldon06') #plot errorbar
# Driese_CO2_low=np.array([10])*370*ppCO2 # lower limit Driese 2011
# Driese_CO2_best=np.array([41])*370*ppCO2 # median estimate Driese 2011
# Driese_CO2_hi=np.array([50])*370*ppCO2 # upper limit Driese 2011
# Driese_dates=np.array([2.69]) # time in Ga
# pylab.errorbar(Driese_dates,Driese_CO2_best,yerr=[Driese_CO2_best-Driese_CO2_low,Driese_CO2_hi-Driese_CO2_best],color='r',marker='o',linestyle="None",label='Driese11') #plot errorbar
# KahRiding_CO2=np.array([10])*360*ppCO2 # Kah and Riding 07 upper limit
# KahRiding_dates=np.array([1.2]) # time in Ga
# pylab.errorbar(KahRiding_dates, KahRiding_CO2-1500*ppCO2, yerr=1500*ppCO2, lolims=True,linestyle='none',color='b',label='Kah07') #plot errorbar
# KanzakiMurakami_CO2_low=np.array([85,78,160,30,20,23])*370*ppCO2 # Kanzaki and Murakami 2015 lower limit
# KanzakiMurakami_CO2_hi=np.array([510,2500,490,190,620,210])*370*ppCO2 # Kanzaki and Murakami 2015 median
# KanzakiMurakami_CO2_best= 0.5*(KanzakiMurakami_CO2_low+ KanzakiMurakami_CO2_hi)*ppCO2 # Kanzaki and Murakami 15 upper limit
# KanzakiMurakami_dates=np.array([2.77,2.75,2.46,2.15,2.08,1.85]) #time in Ga
# pylab.errorbar(KanzakiMurakami_dates,KanzakiMurakami_CO2_best,yerr=[KanzakiMurakami_CO2_best-KanzakiMurakami_CO2_low,KanzakiMurakami_CO2_hi-KanzakiMurakami_CO2_best],color='m',linestyle="None",label='Kanzaki15') #plot errorbar
# ### End of literature proxy comparison
# pylab.xlabel('Time (Ga)')
# pylab.ylabel('Atmospheric CO2 (bar)')
# pylab.legend(loc=2,numpoints=1,frameon=False,bbox_to_anchor=(-.07, 1.0, 1.0, 0)) # plot legend
# pylab.xlim([strt_lim,fin_lim]) # x axis limit
# pylab.ylim([1e-4,10.0]) # limit y-axis from 1e-4 to 10 bar
# pylab.text(-0.7, 10., 'B', fontsize=16, fontweight='bold', va='top') # label subplot

# ### OPTION 1 - plot volcanic outgassing flux
# pylab.subplot(2,3,3)
# pylab.plot(all_output[4,:,0],confidence_Volc[1]/1e12,'k',label='Total',linewidth=2.5) # plot median outgassing flux
# pylab.fill_between(all_output[4,:,0], confidence_Volc[0]/1e12, confidence_Volc[2]/1e12, color='grey', alpha='0.4') #outgassing flux confidence interval
# pylab.ylabel('Outgassing (Tmol/yr)')
# pylab.xlabel('Time (Ga)')
# pylab.xlim([strt_lim,fin_lim]) # x axis limits
# pylab.ylim([0,120.0]) # y axis limits
# pylab.text(-0.7, 120, 'C', fontsize=16, fontweight='bold', va='top') # label subplot

# ## OPTION 2 (for sensitivity test) - plot Calcium ocean molality
# #pylab.subplot(2,3,3)
# #pylab.plot(all_output[4,:,0],confidence_Ca_o[1],'k',label='Total',linewidth=2.5) # median Ca abundance
# #pylab.fill_between(all_output[4,:,0], confidence_Ca_o[0], confidence_Ca_o[2], color='grey', alpha='0.4') # Ca abundance confidence interval
# #pylab.ylabel('Ocean Ca abundance (mol/kg)')
# #pylab.xlabel('Time (Ga)')
# #pylab.xlim([strt_lim,fin_lim]) # x axis limits
# #pylab.ylim([0.0,0.6]) # y axis limits
# #pylab.text(-0.7, 0.6, 'C', fontsize=16, fontweight='bold', va='top') # label subplot

# # Subplot for surface temperature
# pylab.subplot(2, 3, 4)
# pylab.plot(all_output[4,:,0],confidence_Tsurf[1],'k',linewidth=2.5) # plot median surface temperature
# pylab.fill_between(all_output[4,:,0], confidence_Tsurf[0], confidence_Tsurf[2], color='grey', alpha='0.4') # plot confidence interval surface temperature
# pylab.ylabel('Temperature (K)')
# pylab.xlabel('Time (Ga)')
# Blake_dates=np.array([3.35]) # time in Ga for Blake 2010 temperature constraint
# Blake_temp_low=np.array([26])+273.15 # lower estimate Blake 2010
# Blake_temp_high=np.array([35])+273.15 # upper estimate Blake 2010
# Blake_best=0.5*(Blake_temp_low+Blake_temp_high) # midpoint Blake 2010
# pylab.errorbar(Blake_dates,Blake_best,xerr=0.15,yerr=Blake_temp_high-Blake_best,color='b',linestyle="None",label='Blake10') # plot Blake 2010 temperature proxy
# Temp_dates=np.array([2.9,2.7,1.8,1.9,1.2,3.45]) # Ages for glacial constraints (see Appendix D).
# Up_limit=np.array([25.,25.,25.,25.,25.,25.])+273.15-2.5 # Define maximum global mean temperature during glacial events to be 25 K
# pylab.errorbar(Temp_dates, Up_limit-2.5, yerr=5, lolims=True,linestyle='none',color='g',label='Glacial dep.') # plot glacial temperature constraints
# pylab.errorbar([3.42], 273.15+40-2.5, yerr=5, lolims=True,linestyle='none',color='m',label='Hren09') # plot Hren 2009 temperature proxy
# pylab.legend(loc=2,numpoints=1,frameon=False) #display legend
# pylab.xlim([strt_lim,fin_lim]) # x axis limits
# pylab.ylim([260,330]) # y axis limits
# pylab.text(-0.7, 333, 'D', fontsize=16, fontweight='bold', va='top') # label subplot

# #subplot for continental weathering flux through time
# pylab.subplot(2, 3, 5)
# pylab.plot(all_output[4,:,0],confidence_Fs[1]/1e12,'k',label='Cont. weathering',linewidth=2.5) # plot median continental weathering
# pylab.fill_between(all_output[4,:,0], confidence_Fs[0]/1e12, confidence_Fs[2]/1e12, color='grey', alpha='0.4') # plot confidence interval
# pylab.ylabel('Continental weathering flux (Tmol/yr)')
# pylab.xlabel('Time (Ga)')
# pylab.ylim([0,50]) # y axis limits
# pylab.xlim([strt_lim,fin_lim]) # x axis limits
# pylab.text(-0.7, 53, 'E', fontsize=16, fontweight='bold', va='top') # label subplot

# #subplot for seafloor weathering flux through time
# pylab.subplot(2, 3, 6)
# pylab.plot(all_output[4,:,0],confidence_Fd[1]/1e12,'k',linewidth=2.5) # plot median seafloor dissolution
# pylab.fill_between(all_output[4,:,0], confidence_Fd[0]/1e12, confidence_Fd[2]/1e12, color='grey', alpha='0.4') # plot seafloor dissolution confidence intervals
# Nakamura_Kato_date=np.array([3.46]) # Age of Nakamura and Kato seafloor carbonate mesaurements (Ga)
# Nakamura_Kato_low=np.array([7.6e12])/1e12 # See Appendix D for lower bound calculation
# Nakamura_Kato_high=np.array([6.5e13])/1e12 # See Appendix D for upper bound calculation
# NK_best=0.5*(Nakamura_Kato_low+Nakamura_Kato_high) # Find mid point of range
# pylab.errorbar(Nakamura_Kato_date,NK_best,yerr=[NK_best-Nakamura_Kato_low,Nakamura_Kato_high-NK_best],color='b',linestyle="None",label='Nakamura04') # plot errorbar
# Shibuya2013=np.array([2.6]) # Age of Shibuya 2013 seafloor carbonate mesaurements (Ga)
# Shibuya2013_hi=np.array([9.1e12])/1e12 # See Appendix D
# Shibuya2013_lo=np.array([2.6e12])/1e12 # See Appendix D
# Shib2013_best=0.5*(Shibuya2013_lo+Shibuya2013_hi) # find mid point
# pylab.errorbar(Shibuya2013,Shib2013_best,yerr=[Shib2013_best- Shibuya2013_lo,Shibuya2013_hi-Shib2013_best],color='m',linestyle="None",label='Shibuya13') # plot errorbar
# Shibuya2012=np.array([3.2]) # Age of Shibuya 2012 seafloor carbonate mesaurements (Ga)
# Shibuya2012_hi=np.array([2.6e14])/1e12 # See Appendix D
# Shibuya2012_lo=np.array([42e12])/1e12 # See Appendix D
# Shib2012_best=0.5*(Shibuya2012_lo+Shibuya2012_hi) # find mid point
# pylab.errorbar(Shibuya2012,Shib2012_best,yerr=[Shib2012_best- Shibuya2012_lo,Shibuya2012_hi-Shib2012_best],color='g',linestyle="None",label='Shibuya12')# plot errorbar
# NrthPole=np.array([3.5]) # Age of Kitajima seafloor carbonate measurements (Ga)
# NrthPole_hi=np.array([250e12])/1e12 # See Appendix D
# NrthPole_lo=np.array([2.8e13])/1e12 # See Appendix D
# NrthPole_best=0.5*(NrthPole_lo+NrthPole_hi) # find mid point
# pylab.errorbar(NrthPole,NrthPole_best,yerr=[NrthPole_best- NrthPole_lo,NrthPole_hi-NrthPole_best],color='r',linestyle="None",label='Kitajima01') # plot errorbar
# pylab.text(-0.7, 53, 'F', fontsize=16, fontweight='bold', va='top') # label subplot
# pylab.legend(loc=2)
# pylab.xlim([strt_lim,fin_lim])
# pylab.ylim([0,50])
# pylab.ylabel('Seafloor weathering flux (Tmol/yr)')
# pylab.xlabel('Time (Ga)')
# pylab.legend(loc=2,numpoints=1,frameon=False)
# pylab.tight_layout()
# ##################################################

# ### Plot distribution of mass imbalance
# pylab.figure()
# pylab.hist(imbalance_array,500)

# ## Plot distribution of continental weathering flux to seafloor weathering flux ratio
# pylab.figure()
# ratio_cont_sea=all_output[20,:,:]/all_output[19,:,:] # continental to seafloor weathering ratio
# radio_confidence=scipy.stats.scoreatpercentile(ratio_cont_sea,[low_c,mid_c,high_c], interpolation_method='fraction',axis=1) #create confidence interval for ratio
# pylab.plot(all_output[4,:,0],radio_confidence[1]) #plot median
# pylab.fill_between(all_output[4,:,0], radio_confidence[0], radio_confidence[2], color='grey', alpha='0.4') #plot confidence interval
# pylab.xlabel('Time (Ga)')
# pylab.ylabel('Continental/Seafloor weathering')
# pylab.show()

# #################################################
