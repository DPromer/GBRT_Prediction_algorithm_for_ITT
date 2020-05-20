"""
CASE:           MATERIAL #2 (2024-T4 Aluminum)
Training Size:  4 x 4 = 16
Testing Size:   75
Author:         darren.r.promer
"""

## ----------------------------------------------------------------------
## BASIC PROGRAM FUNCTIONS/SETTINGS:
## ----------------------------------------------------------------------

# Python Modules to Import:
import numpy as np
import matplotlib
import pandas as pd
import os
from matplotlib.offsetbox import AnchoredText
import string
import os.path
from colorama import init
from termcolor import colored
from grid_plotting_functions import grid_study_plotting
init()

# matplotlib formatting:
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Dejavu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['legend.fontsize'] = 'x-small'
matplotlib.rcParams['xtick.labelsize'] = 'small'        ## fontsize of the tick labels
matplotlib.rcParams['ytick.labelsize'] = 'small'        ## fontsize of the tick labels
matplotlib.rcParams['axes.labelsize'] = 'medium'        ## fontsize of the x any y labels
matplotlib.rcParams['axes.titlesize'] = 'Medium'        ## fontsize of the axes title
matplotlib.rcParams['xtick.major.pad'] = 1.75           ## distance to major tick label in points
matplotlib.rcParams['ytick.major.pad'] = 1.75           ## distance to major tick label in points
matplotlib.rcParams['figure.subplot.wspace'] = 0.165    ## the amount of width reserved for space between subplots,
                                                        ## expressed as a fraction of the average axis width
matplotlib.rcParams['grid.alpha'] = 0.7                 ## transparency, between 0.0 and 1.0
matplotlib.rcParams['axes.grid'] = True                 ## display grid or not']
matplotlib.rcParams['grid.linewidth'] = 0.4             ## in points
matplotlib.rcParams['legend.facecolor'] = 'white'       ## inherit from axes.facecolor; or color spec
matplotlib.rcParams['legend.borderpad'] = 0.3           ## border whitespace
matplotlib.rcParams['legend.framealpha'] = 1.0          ## legend patch transparency

## ----------------------------------------------------------------------
## INPUT VARIABLES:
## ----------------------------------------------------------------------

# EXECUTION Options:
"""
plotting_on: string {'T', 'F'}
    Do you desire to create .png images of the input data sets

analysis_call: list {   'MULTIMODEL_ANALYSIS', 'ONEGAMMA_ANALYSIS', 'HN_INCLUSION_ANALYSIS',
                        'PREDICTION_BOOSTING', 'XGBOOST', 'FEATURE_REDUCTION'}
    What portions of the machine-learning analysis should be completed?
"""
plotting_on = 'T'
analysis_call = ['3MATERIAL_RESULTS']

# Function Name:
funct_name = r'MAT2_016'
material_string = '2024-T4 Aluminum'
color = 'red'

# Array defining training grid (Typically generated using np.linspace()):
H_ss = np.array([480, 693.33, 906.67, 1120])
n_ss = np.array([0.120, 0.173, 0.227, 0.280])
problem_iterations = []

# A few important files:
basefile = r'R:\RefinedMesh_PcklBase.inp'
OTP_dir = r'R:\SIMDIR_2024T4'
master_code_dir = r'R:'

"""
KNOWN MECHANICAL properites. These are the properties used in
the Ramberg-Osgood model to calculate the elastic-plastic behavior
for <<ALL>> iterations.

Format/units:
    E   : youngs_modulus : integer [units : MPa           ]
    nu  : poissons_ratio : float   [units : Dimensionless ]
"""
E = 73000
nu = 0.345

# TRAINING PAIRINGS (Generated using np.random.normal(mu, sigma, n))
"""
training_gaussian_pairings = np.array([
        [6.68969296e+02,     1.64354225e-01,     'H6_68969296e02n1_64354225eneg01.inp'],
        [7.86115059e+02,     1.91330152e-01,     'H7_86115059e02n1_91330152eneg01.inp'],
        [6.62255123e+02,     1.73945064e-01,     'H6_62255123e02n1_73945064eneg01.inp'],
        [9.39092580e+02,     1.79660133e-01,     'H9_39092580e02n1_79660133eneg01.inp'],
        [7.84556551e+02,     1.86939253e-01,     'H7_84556551e02n1_86939253eneg01.inp'],
        [7.69880421e+02,     2.07302465e-01,     'H7_69880421e02n2_07302465eneg01.inp'],
        [7.63918669e+02,     2.05631357e-01,     'H7_63918669e02n2_05631357eneg01.inp'],
        [8.53342046e+02,     1.46223770e-01,     'H8_53342046e02n1_46223770eneg01.inp'],
        [7.73514809e+02,     2.28981843e-01,     'H7_73514809e02n2_28981843eneg01.inp'],
        [7.96344964e+02,     2.11287068e-01,     'H7_96344964e02n2_11287068eneg01.inp'],
        [6.35437112e+02,     2.47340354e-01,     'H6_35437112e02n2_47340354eneg01.inp'],
        [5.33309765e+02,     1.97468365e-01,     'H5_33309765e02n1_97468365eneg01.inp'],
        [8.91560057e+02,     2.13907922e-01,     'H8_91560057e02n2_13907922eneg01.inp'],
        [8.24874475e+02,     1.57399228e-01,     'H8_24874475e02n1_57399228eneg01.inp'],
        [9.24557354e+02,     2.28465773e-01,     'H9_24557354e02n2_28465773eneg01.inp'],
        [7.62441797e+02,     1.49415380e-01,     'H7_62441797e02n1_49415380eneg01.inp'],
        [7.95349608e+02,     2.09230559e-01,     'H7_95349608e02n2_09230559eneg01.inp'],
        [7.82757395e+02,     2.02050298e-01,     'H7_82757395e02n2_02050298eneg01.inp'],
        [6.36838303e+02,     2.15912140e-01,     'H6_36838303e02n2_15912140eneg01.inp'],
        [8.28425200e+02,     1.78981577e-01,     'H8_28425200e02n1_78981577eneg01.inp'],
        [8.27614130e+02,     1.74985573e-01,     'H8_27614130e02n1_74985573eneg01.inp'],
        [9.42962097e+02,     1.89699103e-01,     'H9_42962097e02n1_89699103eneg01.inp'],
        [8.10733426e+02,     1.37685420e-01,     'H8_10733426e02n1_37685420eneg01.inp'],
        [7.77461600e+02,     2.09120994e-01,     'H7_77461600e02n2_09120994eneg01.inp'],
        [9.10351072e+02,     1.93135777e-01,     'H9_10351072e02n1_93135777eneg01.inp'],
        [1.01157392e+03,     1.68483822e-01,     'H1_01157392e03n1_68483822eneg01.inp'],
        [8.04245624e+02,     2.23572078e-01,     'H8_04245624e02n2_23572078eneg01.inp'],
        [9.39236830e+02,     2.16037816e-01,     'H9_39236830e02n2_16037816eneg01.inp'],
        [7.73700854e+02,     2.05056102e-01,     'H7_73700854e02n2_05056102eneg01.inp'],
        [7.11098765e+02,     1.44299013e-01,     'H7_11098765e02n1_44299013eneg01.inp'],
        [8.95726142e+02,     1.31981656e-01,     'H8_95726142e02n1_31981656eneg01.inp'],
        [7.26280110e+02,     2.13657290e-01,     'H7_26280110e02n2_13657290eneg01.inp'],
        [9.04241975e+02,     1.70058550e-01,     'H9_04241975e02n1_70058550eneg01.inp'],
        [7.49785817e+02,     1.76657361e-01,     'H7_49785817e02n1_76657361eneg01.inp'],
        [1.04090649e+03,     1.75839395e-01,     'H1_04090649e03n1_75839395eneg01.inp'],
        [8.29951792e+02,     1.98928362e-01,     'H8_29951792e02n1_98928362eneg01.inp'],
        [6.38448953e+02,     1.52800215e-01,     'H6_38448953e02n1_52800215eneg01.inp'],
        [8.58043595e+02,     2.11482097e-01,     'H8_58043595e02n2_11482097eneg01.inp'],
        [8.13054777e+02,     2.13018281e-01,     'H8_13054777e02n2_13018281eneg01.inp'],
        [8.52670955e+02,     1.98506303e-01,     'H8_52670955e02n1_98506303eneg01.inp'],
        [7.68016200e+02,     1.52090877e-01,     'H7_68016200e02n1_52090877eneg01.inp'],
        [8.16517481e+02,     2.13308216e-01,     'H8_16517481e02n2_13308216eneg01.inp'],
        [7.99352733e+02,     2.02441836e-01,     'H7_99352733e02n2_02441836eneg01.inp'],
        [8.19313565e+02,     1.63853673e-01,     'H8_19313565e02n1_63853673eneg01.inp'],
        [9.10380761e+02,     2.19281011e-01,     'H9_10380761e02n2_19281011eneg01.inp'],
        [9.43275874e+02,     1.84342194e-01,     'H9_43275874e02n1_84342194eneg01.inp'],
        [7.70419992e+02,     1.49284751e-01,     'H7_70419992e02n1_49284751eneg01.inp'],
        [7.73555961e+02,     1.93153363e-01,     'H7_73555961e02n1_93153363eneg01.inp'],
        [9.41195621e+02,     1.69770238e-01,     'H9_41195621e02n1_69770238eneg01.inp'],
        [8.03814562e+02,     1.88302947e-01,     'H8_03814562e02n1_88302947eneg01.inp'],
        [8.12248733e+02,     1.90938654e-01,     'H8_12248733e02n1_90938654eneg01.inp'],
        [8.37713977e+02,     2.16429924e-01,     'H8_37713977e02n2_16429924eneg01.inp'],
        [7.56893715e+02,     1.52184465e-01,     'H7_56893715e02n1_52184465eneg01.inp'],
        [7.12548412e+02,     2.46042844e-01,     'H7_12548412e02n2_46042844eneg01.inp'],
        [7.20382560e+02,     2.20395816e-01,     'H7_20382560e02n2_20395816eneg01.inp'],
        [8.39417778e+02,     2.48331539e-01,     'H8_39417778e02n2_48331539eneg01.inp'],
        [7.91853173e+02,     1.68883612e-01,     'H7_91853173e02n1_68883612eneg01.inp'],
        [8.99248841e+02,     1.96408539e-01,     'H8_99248841e02n1_96408539eneg01.inp'],
        [7.66327952e+02,     1.92542216e-01,     'H7_66327952e02n1_92542216eneg01.inp'],
        [8.33943516e+02,     1.64390906e-01,     'H8_33943516e02n1_64390906eneg01.inp'],
        [9.00916845e+02,     1.89127950e-01,     'H9_00916845e02n1_89127950eneg01.inp'],
        [8.27773902e+02,     1.73744277e-01,     'H8_27773902e02n1_73744277eneg01.inp'],
        [6.81848193e+02,     1.73053032e-01,     'H6_81848193e02n1_73053032eneg01.inp'],
        [8.22177943e+02,     2.32228080e-01,     'H8_22177943e02n2_32228080eneg01.inp'],
        [7.80079261e+02,     1.46504530e-01,     'H7_80079261e02n1_46504530eneg01.inp'],
        [9.31987028e+02,     1.75913916e-01,     'H9_31987028e02n1_75913916eneg01.inp'],
        [8.14685764e+02,     2.03983396e-01,     'H8_14685764e02n2_03983396eneg01.inp'],
        [7.97360413e+02,     1.72939567e-01,     'H7_97360413e02n1_72939567eneg01.inp'],
        [6.42057110e+02,     2.09994943e-01,     'H6_42057110e02n2_09994943eneg01.inp'],
        [9.71401907e+02,     2.11559063e-01,     'H9_71401907e02n2_11559063eneg01.inp'],
        [9.07209433e+02,     2.31227742e-01,     'H9_07209433e02n2_31227742eneg01.inp'],
        [7.07524756e+02,     2.32665932e-01,     'H7_07524756e02n2_32665932eneg01.inp'],
        [8.89871854e+02,     2.40445408e-01,     'H8_89871854e02n2_40445408eneg01.inp'],
        [8.53599851e+02,     2.23271475e-01,     'H8_53599851e02n2_23271475eneg01.inp'],
        [7.01066605e+02,     1.82880790e-01,     'H7_01066605e02n1_82880790eneg01.inp']])
"""
training_gaussian_pairings = np.array([
        [6.68969296e+02,     1.64354225e-01,     'H6_68969296e02n1_64354225eneg01.inp'],
        [7.86115059e+02,     1.91330152e-01,     'H7_86115059e02n1_91330152eneg01.inp'],
        [6.62255123e+02,     1.73945064e-01,     'H6_62255123e02n1_73945064eneg01.inp'],
        [9.39092580e+02,     1.79660133e-01,     'H9_39092580e02n1_79660133eneg01.inp'],
        [7.84556551e+02,     1.86939253e-01,     'H7_84556551e02n1_86939253eneg01.inp'],
        [7.69880421e+02,     2.07302465e-01,     'H7_69880421e02n2_07302465eneg01.inp'],
        [7.63918669e+02,     2.05631357e-01,     'H7_63918669e02n2_05631357eneg01.inp'],
        [8.53342046e+02,     1.46223770e-01,     'H8_53342046e02n1_46223770eneg01.inp'],
        [7.73514809e+02,     2.28981843e-01,     'H7_73514809e02n2_28981843eneg01.inp'],
        [7.96344964e+02,     2.11287068e-01,     'H7_96344964e02n2_11287068eneg01.inp'],
        [6.35437112e+02,     2.47340354e-01,     'H6_35437112e02n2_47340354eneg01.inp'],
        [5.33309765e+02,     1.97468365e-01,     'H5_33309765e02n1_97468365eneg01.inp'],
        [8.91560057e+02,     2.13907922e-01,     'H8_91560057e02n2_13907922eneg01.inp'],
        [8.24874475e+02,     1.57399228e-01,     'H8_24874475e02n1_57399228eneg01.inp'],
        [9.24557354e+02,     2.28465773e-01,     'H9_24557354e02n2_28465773eneg01.inp'],
        [7.62441797e+02,     1.49415380e-01,     'H7_62441797e02n1_49415380eneg01.inp'],
        [7.95349608e+02,     2.09230559e-01,     'H7_95349608e02n2_09230559eneg01.inp'],
        [7.82757395e+02,     2.02050298e-01,     'H7_82757395e02n2_02050298eneg01.inp'],
        [6.36838303e+02,     2.15912140e-01,     'H6_36838303e02n2_15912140eneg01.inp'],
        [8.28425200e+02,     1.78981577e-01,     'H8_28425200e02n1_78981577eneg01.inp'],
        [8.27614130e+02,     1.74985573e-01,     'H8_27614130e02n1_74985573eneg01.inp'],
        [9.42962097e+02,     1.89699103e-01,     'H9_42962097e02n1_89699103eneg01.inp'],
        [8.10733426e+02,     1.37685420e-01,     'H8_10733426e02n1_37685420eneg01.inp'],
        [7.77461600e+02,     2.09120994e-01,     'H7_77461600e02n2_09120994eneg01.inp'],
        [9.10351072e+02,     1.93135777e-01,     'H9_10351072e02n1_93135777eneg01.inp'],
        [1.01157392e+03,     1.68483822e-01,     'H1_01157392e03n1_68483822eneg01.inp'],
        [8.04245624e+02,     2.23572078e-01,     'H8_04245624e02n2_23572078eneg01.inp'],
        [9.39236830e+02,     2.16037816e-01,     'H9_39236830e02n2_16037816eneg01.inp'],
        [7.73700854e+02,     2.05056102e-01,     'H7_73700854e02n2_05056102eneg01.inp'],
        [7.11098765e+02,     1.44299013e-01,     'H7_11098765e02n1_44299013eneg01.inp'],
        [8.95726142e+02,     1.31981656e-01,     'H8_95726142e02n1_31981656eneg01.inp'],
        [7.26280110e+02,     2.13657290e-01,     'H7_26280110e02n2_13657290eneg01.inp'],
        [9.04241975e+02,     1.70058550e-01,     'H9_04241975e02n1_70058550eneg01.inp'],
        [7.49785817e+02,     1.76657361e-01,     'H7_49785817e02n1_76657361eneg01.inp'],
        [1.04090649e+03,     1.75839395e-01,     'H1_04090649e03n1_75839395eneg01.inp'],
        [8.29951792e+02,     1.98928362e-01,     'H8_29951792e02n1_98928362eneg01.inp'],
        [6.38448953e+02,     1.52800215e-01,     'H6_38448953e02n1_52800215eneg01.inp'],
        [8.58043595e+02,     2.11482097e-01,     'H8_58043595e02n2_11482097eneg01.inp'],
        [8.13054777e+02,     2.13018281e-01,     'H8_13054777e02n2_13018281eneg01.inp'],
        [8.52670955e+02,     1.98506303e-01,     'H8_52670955e02n1_98506303eneg01.inp'],
        [7.68016200e+02,     1.52090877e-01,     'H7_68016200e02n1_52090877eneg01.inp'],
        [8.16517481e+02,     2.13308216e-01,     'H8_16517481e02n2_13308216eneg01.inp'],
        [7.99352733e+02,     2.02441836e-01,     'H7_99352733e02n2_02441836eneg01.inp'],
        [8.19313565e+02,     1.63853673e-01,     'H8_19313565e02n1_63853673eneg01.inp'],
        [9.10380761e+02,     2.19281011e-01,     'H9_10380761e02n2_19281011eneg01.inp'],
        [9.43275874e+02,     1.84342194e-01,     'H9_43275874e02n1_84342194eneg01.inp'],
        [7.70419992e+02,     1.49284751e-01,     'H7_70419992e02n1_49284751eneg01.inp'],
        [7.73555961e+02,     1.93153363e-01,     'H7_73555961e02n1_93153363eneg01.inp'],
        [9.41195621e+02,     1.69770238e-01,     'H9_41195621e02n1_69770238eneg01.inp'],
        [8.03814562e+02,     1.88302947e-01,     'H8_03814562e02n1_88302947eneg01.inp'],
        [8.37713977e+02,     2.16429924e-01,     'H8_37713977e02n2_16429924eneg01.inp'],
        [7.56893715e+02,     1.52184465e-01,     'H7_56893715e02n1_52184465eneg01.inp'],
        [7.12548412e+02,     2.46042844e-01,     'H7_12548412e02n2_46042844eneg01.inp'],
        [7.20382560e+02,     2.20395816e-01,     'H7_20382560e02n2_20395816eneg01.inp'],
        [8.39417778e+02,     2.48331539e-01,     'H8_39417778e02n2_48331539eneg01.inp'],
        [7.91853173e+02,     1.68883612e-01,     'H7_91853173e02n1_68883612eneg01.inp'],
        [8.99248841e+02,     1.96408539e-01,     'H8_99248841e02n1_96408539eneg01.inp'],
        [7.66327952e+02,     1.92542216e-01,     'H7_66327952e02n1_92542216eneg01.inp'],
        [8.33943516e+02,     1.64390906e-01,     'H8_33943516e02n1_64390906eneg01.inp'],
        [9.00916845e+02,     1.89127950e-01,     'H9_00916845e02n1_89127950eneg01.inp'],
        [8.27773902e+02,     1.73744277e-01,     'H8_27773902e02n1_73744277eneg01.inp'],
        [6.81848193e+02,     1.73053032e-01,     'H6_81848193e02n1_73053032eneg01.inp'],
        [8.22177943e+02,     2.32228080e-01,     'H8_22177943e02n2_32228080eneg01.inp'],
        [7.80079261e+02,     1.46504530e-01,     'H7_80079261e02n1_46504530eneg01.inp'],
        [9.31987028e+02,     1.75913916e-01,     'H9_31987028e02n1_75913916eneg01.inp'],
        [7.97360413e+02,     1.72939567e-01,     'H7_97360413e02n1_72939567eneg01.inp'],
        [6.42057110e+02,     2.09994943e-01,     'H6_42057110e02n2_09994943eneg01.inp'],
        [9.71401907e+02,     2.11559063e-01,     'H9_71401907e02n2_11559063eneg01.inp'],
        [9.07209433e+02,     2.31227742e-01,     'H9_07209433e02n2_31227742eneg01.inp'],
        [7.07524756e+02,     2.32665932e-01,     'H7_07524756e02n2_32665932eneg01.inp'],
        [8.89871854e+02,     2.40445408e-01,     'H8_89871854e02n2_40445408eneg01.inp'],
        [8.53599851e+02,     2.23271475e-01,     'H8_53599851e02n2_23271475eneg01.inp'],
        [7.01066605e+02,     1.82880790e-01,     'H7_01066605e02n1_82880790eneg01.inp']])

# ----------------------------------------------------------------------
# Study Execution
# ----------------------------------------------------------------------

sensitivity_study = np.zeros(((len(H_ss)*len(n_ss)), 2))

ii = 0
jj = 0
kk =  0

while ii < len(H_ss):
    while jj < len(n_ss):
        sensitivity_study[kk, 0] = H_ss[ii]
        sensitivity_study[kk, 1] = n_ss[jj]
        jj += 1
        kk += 1
    jj = 0
    ii+=1

# Plotting Call
os.chdir(master_code_dir)
if plotting_on == 'T':
    grid_study_plotting(    sensitivity_study,
                            training_gaussian_pairings,
                            E, nu, material_string,
                            funct_name, color, len(H_ss))

Hmax_ss = H_ss.max()
Hmin_ss = H_ss.min()
nmax_ss = n_ss.max()
nmin_ss = n_ss.min()

"""
THEORY:
H_mu = mean of the standard distribution of H values
n_mu = mean of the standard distribution of n values

H_sd = standard deviation of H distribution
n_sd = standard deviation of n distribution

    standard deviation calculated from z-score of 3...
        z = (x-mu)/sigma
        sigma = (x-mu)/3
"""

H_mu = (Hmax_ss+Hmin_ss)/2
H_sigma = (Hmax_ss-H_mu)/3

n_mu = (nmax_ss+nmin_ss)/2
n_sigma = (nmax_ss-n_mu)/3

H_dist = training_gaussian_pairings[:,0]
n_dist = training_gaussian_pairings[:,1]
inp_names =training_gaussian_pairings[:,2]

i_vect = np.linspace(0,(len(inp_names)-1),len(inp_names))

# Define array containing data 
data = np.column_stack((i_vect, H_dist, n_dist, inp_names))

# Convert the array into DataFrame
df = pd.DataFrame(data, columns=['i', 'H_gauss', 'n_gauss', 'inp_names'])

# Write dataframe to .csv
path = '/'+funct_name
if not os.path.exists(path):
    os.makedirs(path)
df.to_csv(path+'/training_gaussian_pairings.csv')

def reindent(s, numSpaces):
    s = string.split(s, '\n')
    s = [(numSpaces * ' ') + line for line in s]
    s = string.join(s, '\n')
    return s

def p_calc(H,n,E,nu):
    """
    Purpose:\n
    Calculates Ramberg-Osgood elasto-plastic material model [Yield Stress, Plastic Stress & Strain Arrays (up to 0.8 strain)] based on input material coefficients.\n
    Inputs:\n
    H: Strength Coefficient (MPa)\n
    n: Strain Hardening Coefficient\n
    E: Young's Modulus (MPa)\n
    nu: Poisson's Ratio\n
    Outputs:\n
    Output[0]: Yield Stress (MPa)\n
    Output[1]: Plastic Strain (ranges from 0-0.8)\n
    Output[2]: Plastic Stress (MPa)\n
    Output[3]: Doublet cotaining [E (MPa), nu]\n
    """
    Yield_Stress_002 = H*np.power(0.002, n)
    Plastic_Strain = np.arange(0, 0.801, 0.001)
    Plastic_Stress = H*np.power((Plastic_Strain+0.002), n)
    Elastic = [E, nu] 

    return Yield_Stress_002, Plastic_Strain, Plastic_Stress, Elastic

def p_inp(H, n, E, nu, PStrain, PStress, basefile, OTP_dir, iteration_count, inp_name):
    """
    Purpose:

    Generates an ABAQUS .inp file from a given base .inp file. The 'base' file must be complete except for the inputted material properties which must be presented in the form of:\n
    *Material, name=AISI_1020
    *Elastic
    *Plastic
    **
    
    Inputs:
    H:              Strength Coefficient (MPa)
    n:              Strain Hardening Coefficient
    elasticstring:  presented as: E,nu\n
    plasticstring:  presented as: PStress,PStrain\n
    basefile:       base .inp file without material properties\n 
    OTP_dir:        directory for generated input file to be saved within
    
    Outputs:
    .inp saved to file with material properties corresponding to given input properties (HXXXNXXX.inp)

    """

    newfile           = inp_name+".inp"
    filename_notype   = inp_name
    
    print
    print("H = "+str(H)+' [MPa]')
    print("n = "+str(n))

    print("Generating "+iteration_count+" input file from the above properties:")
    print
    print('From the following basefile:')
    print(basefile)
    print
    print('Saved in the following output directory:')
    print(OTP_dir)
    print
    print('With the following file name:')
    print(newfile)

    # Generating the elastic & plastic strings to be printed to the output from the p_calc module
    np.set_printoptions(threshold=np.inf, precision = 1)
    commalist = [',']*801
    plasticstring = str(np.column_stack((PStress, commalist, PStrain))).replace("[", "").replace("]","").replace("'","").replace(" ","")
    elasticstring = str([E, nu]).replace("[", "").replace("]","").replace("'","").replace(" ","")

    # Actual generation of the .inp file (bookeeping)
    import os
    os.chdir(OTP_dir)

    print("Mirror of text being inserted into to base input file:")

    with open(basefile, "r") as base, open(newfile, "w") as output:
      for line in base:
          l = line.strip()
          if l.endswith("*Elastic"):
            output.write('*Elastic'+'\n'+elasticstring)
            #print(reindent(('*Elastic'+'\n'+elasticstring), 12))
          if l.endswith("*Plastic"):
            output.write('*Plastic'+'\n'+plasticstring+'\n')
            #print(reindent(('*Plastic'+'\n'+plasticstring+'\n'), 12))
          else:
            output.write(l.replace('*Elastic',"")+'\n')
    print("Input file generation completed for : "+iteration_count)

    inp_path = OTP_dir + '\\' + newfile
    print inp_path

    return filename_notype

def odb_postproc(workdir, name, iteration_count, H, n, E, nu, iteration_number, P10_test):
    import os
    from abapy.misc import load
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from prettytable import PrettyTable

    strain_hardening_exponent = n

    # Extracting the raw data from the .pck1 file
    os.chdir(workdir)
    data = load(name + '.pckl')
    ref_node_label = data['ref_node_label']
    force_hist = -data['RF2']['Node I_INDENTER.{0}'.format(ref_node_label)]
    disp_hist = -data['U2']['Node I_INDENTER.{0}'.format(ref_node_label)]

    # Getting back force & displacemt during loading and scaling units:
    displacement_loading = [1000*x for x in (disp_hist[0,0.5].toArray()[1])]
    force_loading = [1000*x for x in (force_hist[0,0.5].toArray()[1])]

    # Getting back force & displacemt during unloading and scaling units:
    disp_long = [1000*x for x in (disp_hist[1].toArray()[1])]
    force_long = [1000*x for x in (force_hist[1].toArray()[1])]

    # Trimming the trailing zeroes from the unloading dataset
    force_unloading = [i for n, i in enumerate(force_long) if i not in force_long[:n]]
    displacement_unloading = disp_long[:len(force_unloading)]

    # Parabolic Curve fit (E. Buckingham. Physical review, 4, 1914.) shows that the loading curve must be parabolic of the form P=C*h^2
    Parray = force_loading
    h = displacement_loading
    hsquared = [x**2 for x in h]
    C_factor_array = np.divide(Parray[100:], hsquared[100:])
    C_factor = np.mean(C_factor_array)
    disaplacement_loading_fit = displacement_loading
    force_loading_fit = C_factor*np.power(h,2)

    # Modify the loading curve to start at the same point at the curve fit:
    unload_dispstart = disaplacement_loading_fit[-1]
    Pmax = force_loading_fit[-1]
    if Pmax >= force_unloading[0]:
      force_unloading_mod = [Pmax]+force_unloading
      displacement_unloading_mod = [unload_dispstart]+displacement_unloading
    else:
      force_unloading_mod = np.minimum(force_unloading, Pmax)
      displacement_unloading_mod = displacement_unloading

    # Extracting all properties of interest for table
    ITERATION_Pmax  = np.amax(force_loading)

    z = -0.001*np.concatenate([force_loading, force_unloading])
    z = z[::-1]
    z = z.astype(np.float)
    z = -1000*z
    mZ = np.amax(z)
    mZ_index = np.where(z == mZ)
    iteration_force_unloading = z[:mZ_index[0][0]]
 
    w = -0.001*np.concatenate([displacement_loading, displacement_unloading])
    w = w[::-1]
    w = w.astype(np.float)
    w = -1000*w
    iteration_displacment_unloading = w[:mZ_index[0][0]]

    if P10_test == 0:
        P10 = 0.1*ITERATION_Pmax
        point_type = "test"
    elif P10_test == 1:
        P10 = 0.1*ITERATION_Pmax
        #P10 = P10_test
        point_type  = "train"
    else:
        print("ERROR in calculation of P10 location")

    hf1  = np.interp(1*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf2  = np.interp(2*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf3  = np.interp(3*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf4  = np.interp(4*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf5  = np.interp(5*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf6  = np.interp(6*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf7  = np.interp(7*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf8  = np.interp(8*P10, iteration_force_unloading, iteration_displacment_unloading)
    hf9  = np.interp(9*P10, iteration_force_unloading, iteration_displacment_unloading)

    # Creation of dataframe row:
    df = pd.DataFrame([[
        point_type,
        iteration_number,
        round(float(H), 10),
        round(float(strain_hardening_exponent), 10),
        E,
        nu,
        round(ITERATION_Pmax, 10),
        round(hf1, 10),
        round(hf2, 10),
        round(hf3, 10),
        round(hf4, 10),
        round(hf5, 10),
        round(hf6, 10),
        round(hf7, 10),
        round(hf8, 10),
        round(hf9, 10),
        round(P10, 10),
        round(C_factor, 10)]],
        columns = [
            'point_type',
            'i',
            'H',
            'n',
            'E',
            'nu',
            'Pmax',
            'hf1',
            'hf2',
            'hf3',
            'hf4',
            'hf5',
            'hf6',
            'hf7',
            'hf8',
            'hf9',
            'P10',
            'C'])
    
    return df

def abq_pckl_execution(master_code_dir, OTP_dir, filename_notype):
    """
    Purpose:
    Executes the abaqus python abaqus_nanoindentation.pcklcreate(workdir, name)
    command from the windows command prompt.
    Inputs:
    master_code_dir : The location of the MASTER_CODE directory
    filename_notype : The name of the .inp file without .inp extension
    OTP_dir         : The directory where the .inp file is stored
    """
    import os
    from colorama import init
    from termcolor import colored
    init()

    os.chdir(master_code_dir)
    cmd = "abaqus python -c \"import abaqus_nanoindentation; print abaqus_nanoindentation.pcklcreate('"+OTP_dir+"', '"+filename_notype+"')\""

    print
    print("Accessing ABAQUS .odb file. Contents dumped to "+filename_notype+".pckl")
    print
    print("Sending to command line:")
    print(cmd)
    print("Executing .pckl creation code:")
    
    os.system(cmd)
    
    print
    print("decryption of "+filename_notype+".inp is COMPLETED")

# Defining a few vectors for easy reference...
inp_names = df[['inp_names']].values
H_val = df[['H_gauss']].values
n_val = df[['n_gauss']].values
results_df = pd.DataFrame(columns = ['i', 'H_test', 'H_pred', 'n_test', 'n_pred']) 

# CREATION OF TESTING DATASET [GAUSSIAN]
test_df = pd.DataFrame(columns = [
        'point_type', 'i', 'H', 'n',
        'E', 'nu', 'Pmax', 'hf1',
        'hf2', 'hf3', 'hf4', 'hf5',
        'hf6', 'hf7', 'hf8', 'hf9',
        'P10', 'C'])
i = 0
while i < len(inp_names):
    # From the input variables, contruct the name of the input file:
    filename_notype = str(inp_names[i]).lstrip("['").rstrip(".inp']")
    
    # Checking the status of the current iteration
    # (i.e.: Does this simulation need to be run? or can results be accesses)
    os.chdir(OTP_dir)
    
    print
    #print(colored("TEST Set Simulation :: Iteration "+str(i), 'blue'))

    if os.path.isfile(filename_notype+'.pckl'):
        print(colored(filename_notype+": Simulation already exists. Results Processed", 'green'))
    else:
        print(colored(filename_notype+": Simulation does not exist. Results Being Generated", 'red'))
        # Calculation of the elastic-plastic model for the training point
        print("Creation of .inp file: " + str(i))
        ROmod = p_calc(H_dist[i].astype(float), n_dist[i].astype(float), E, nu)

        # Creation of the .inp file for the training point
        if os.path.isfile(filename_notype+'.inp'):
            print("ABAQUS .inp file already exists. No .inp created, but proceeding to cmd execution...")
        else:
            p_inp(  H_dist[i].astype(float), n_dist[i].astype(float),
                    E, nu, ROmod[1], ROmod[2], basefile, OTP_dir,
                    str(i_vect[i]), filename_notype)
        
        # Execution of the .inp within the ABAQUS-cli using os command
        cmd = 'abaqus j='+filename_notype+' interactive\n'
        print
        print("Sending to command line: >>>"+cmd)
        os.system(cmd)
        print

        # Decryption of the .odb file to be read in ABAPY 
        import os
        print
        print("##  Executing decryption of .odb to .pckl... ")
        abq_pckl_execution(master_code_dir, OTP_dir, filename_notype)

    # using ABAPY to extract both train and test df
    """
    Exporting to a dataframe:
    train_df = [
        columns =
        'Point Type' (train or test)
        'H'    : Strength coefficient
        'n'    : Strain hardening exponent
        'E'    : Elastic Modulus
        'nu'   : Poisson's Ratio
        'Pmax' : Maximum Load
        'hf'   : Residual Displacement (recorded at P10)
        'P10'  : 0.1*P10 from test point
        'C'    : Kick's Coefficient for Fitting
    ]
    """
    
    # testing dataframe
    test_df_temp = odb_postproc(
        OTP_dir, filename_notype,
        str(i), H_val[i], n_val[i],
        E, nu, i,
        0)
    test_df.loc[i, 'point_type'] = test_df_temp.iloc[0]['point_type']
    test_df.loc[i, 'i']          = test_df_temp.iloc[0]['i']
    test_df.loc[i, 'H']          = test_df_temp.iloc[0]['H']
    test_df.loc[i, 'n']          = test_df_temp.iloc[0]['n']
    test_df.loc[i, 'E']          = test_df_temp.iloc[0]['E']
    test_df.loc[i, 'nu']         = test_df_temp.iloc[0]['nu']
    test_df.loc[i, 'Pmax']       = test_df_temp.iloc[0]['Pmax']
    test_df.loc[i, 'hf1']        = test_df_temp.iloc[0]['hf1']
    test_df.loc[i, 'hf2']        = test_df_temp.iloc[0]['hf2']
    test_df.loc[i, 'hf3']        = test_df_temp.iloc[0]['hf3']
    test_df.loc[i, 'hf4']        = test_df_temp.iloc[0]['hf4']
    test_df.loc[i, 'hf5']        = test_df_temp.iloc[0]['hf5']
    test_df.loc[i, 'hf6']        = test_df_temp.iloc[0]['hf6']
    test_df.loc[i, 'hf7']        = test_df_temp.iloc[0]['hf7']
    test_df.loc[i, 'hf8']        = test_df_temp.iloc[0]['hf8']
    test_df.loc[i, 'hf9']        = test_df_temp.iloc[0]['hf9']
    test_df.loc[i, 'P10']        = test_df_temp.iloc[0]['P10']
    test_df.loc[i, 'C']          = test_df_temp.iloc[0]['C']

    i += 1    

# CREATION OF TRAINING DATASET [GRID]
train_df = pd.DataFrame(columns = [
        'point_type', 'i', 'H', 'n',
        'E', 'nu', 'Pmax', 'hf1',
        'hf2', 'hf3', 'hf4', 'hf5',
        'hf6', 'hf7', 'hf8', 'hf9',      
        'P10', 'C'])           
ii = 0
i_vect2 = np.linspace(0,(len(sensitivity_study[:,0])-1),len(sensitivity_study[:,0]))
while ii < len(sensitivity_study[:,0]):
    # From the input variables, contruct the name of the input file:
    filename_notype = "H"+str(int(sensitivity_study[ii,0])).replace(".","_")+"N"+str(sensitivity_study[ii,1]).replace(".","_")

    # Checking the status of the current iteration
    # (i.e.: Does this simulation need to be run? or can results be accesses)
    os.chdir(OTP_dir)
    
    print
    #print(colored("TRAIN Set Simulation :: Iteration "+str(ii), 'cyan'))

    if ii in problem_iterations:
        print filename_notype+": PROBLEM ITERATION, SKIPPING FOR NOW"
    else:
        if os.path.isfile(filename_notype+'.pckl'):
            print(colored(filename_notype+": Simulation already exists. Results Processed", 'green'))
        else:
            print(colored(filename_notype+": Simulation does not exist. Results Being Generated", 'red'))
            # Calculation of the elastic-plastic model for the training point
            print("Creation of .inp file: " + str(ii))
            ROmod = p_calc(sensitivity_study[ii,0], sensitivity_study[ii,1], E, nu)

            # Creation of the .inp file for the training point
            if os.path.isfile(filename_notype+'.inp'):
                print("ABAQUS .inp file already exists. No .inp created, but proceeding to cmd execution...")
            else:
                p_inp(  sensitivity_study[ii, 0], sensitivity_study[ii, 1],
                        E, nu, ROmod[1], ROmod[2], basefile, OTP_dir,
                        str(i_vect2[ii]), filename_notype)

            # Execution of the .inp within the ABAQUS-cli using os command
            cmd = 'abaqus j='+filename_notype+' interactive\n'
            print
            print("Sending to command line: >>>"+cmd)
            os.system(cmd)
            print

            # Decryption of the .odb file to be read in ABAPY 
            import os
            print
            print("##  Executing decryption of .odb to .pckl: ")
            print("##  Executing decryption for file: i = ", str(ii))
            abq_pckl_execution(master_code_dir, OTP_dir, filename_notype)

        # using ABAPY to extract both train and test df
        """
        Exporting to a dataframe:
        train_df = [
            columns =
            'Point Type' (train or test)
            'H'    : Strength coefficient
            'n'    : Strain hardening exponent
            'E'    : Elastic Modulus
            'nu'   : Poisson's Ratio
            'Pmax' : Maximum Load
            'hf'   : Residual Displacement (recorded at P10)
            'P10'  : 0.1*P10 from test point
            'C'    : Kick's Coefficient for Fitting
        ]
        """

        name = "H"+str(int(sensitivity_study[ii,0])).replace(".","_")+"N"+str(sensitivity_study[ii,1]).replace(".","_")
        train_df_temp = odb_postproc(
            OTP_dir, name,
            str(ii), sensitivity_study[ii,0], sensitivity_study[ii,1],
            E, nu, ii, 1)
        train_df.loc[ii, 'point_type'] = train_df_temp.iloc[0]['point_type']
        train_df.loc[ii, 'i']    = int(ii)
        train_df.loc[ii, 'H']    = train_df_temp.iloc[0]['H']
        train_df.loc[ii, 'n']    = train_df_temp.iloc[0]['n']
        train_df.loc[ii, 'E']    = train_df_temp.iloc[0]['E']
        train_df.loc[ii, 'nu']   = train_df_temp.iloc[0]['nu']
        train_df.loc[ii, 'Pmax'] = train_df_temp.iloc[0]['Pmax']
        train_df.loc[ii, 'hf1']  = train_df_temp.iloc[0]['hf1']
        train_df.loc[ii, 'hf2']  = train_df_temp.iloc[0]['hf2']
        train_df.loc[ii, 'hf3']  = train_df_temp.iloc[0]['hf3']
        train_df.loc[ii, 'hf4']  = train_df_temp.iloc[0]['hf4']
        train_df.loc[ii, 'hf5']  = train_df_temp.iloc[0]['hf5']
        train_df.loc[ii, 'hf6']  = train_df_temp.iloc[0]['hf6']
        train_df.loc[ii, 'hf7']  = train_df_temp.iloc[0]['hf7']
        train_df.loc[ii, 'hf8']  = train_df_temp.iloc[0]['hf8']
        train_df.loc[ii, 'hf9']  = train_df_temp.iloc[0]['hf9']
        train_df.loc[ii, 'P10']  = train_df_temp.iloc[0]['P10']
        train_df.loc[ii, 'C']    = train_df_temp.iloc[0]['C']
        
    ii += 1

pd.set_option("display.max_rows", None)
    
print
print(colored("TRAINING Pandas DataFrame:", 'blue'))
print(colored(train_df.head(), 'yellow'))
train_df.to_csv(path+'/train_df.csv')

print
print(colored("TESTING Pandas DataFrame:", 'blue'))
print(colored(test_df.head(), 'yellow'))
test_df.to_csv(path+'/test_df.csv')

os.chdir(master_code_dir)

from GBRT_main import GBRT_fullanalysis

GBRT_fullanalysis(  train_df, test_df, funct_name,
                    analysis_call, material_string)