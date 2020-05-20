"""
CASE:           MATERIAL #3 (1020 SAE Hot Rolled Steel)
Training Size:  25 x 25 = 625
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
analysis_call = ['3MATERIAL_RESULTS', 'HN_INCLUSION_ANALYSIS']

# Function Name:
funct_name = 'MAT3_1020SAE_REFINED2'
material_string = '1020 SAE (Hot-Rolled)'
color = 'blue'

# Array defining training grid (Typically generated using np.linspace()):
H_ss = np.array([450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050])
n_ss = np.array([0.115,0.12125,0.1275,0.13375,0.140,0.14625,0.1525,0.15875,0.165,0.17125,0.1775,0.18375,0.190,0.19625,0.2025,0.20875,0.215,0.22125,0.2275,0.23375,0.240,0.24625,0.2525,0.25875,0.265])
problem_iterations = [0, 438, 439, 602]

# A few important files:
basefile = r'R:\RefinedMesh_PcklBase.inp'
OTP_dir = r'R:\SIMDIR_1020SAE'
master_code_dir = r'R:'

"""
KNOWN MECHANICAL properites. These are the properties used in
the Ramberg-Osgood model to calculate the elastic-plastic behavior
for <<ALL>> iterations.

Format/units:
    E   : youngs_modulus : integer [units : MPa           ]
    nu  : poissons_ratio : float   [units : Dimensionless ]
"""
E = 206000
nu = 0.293

# TRAINING PAIRINGS (Generated using np.random.normal(mu, sigma, n))
training_gaussian_pairings = np.array([
        [6.84138018e+02,     2.00347426e-01,     'H6_84138018e02n2_00347426eneg01.inp'],
        [8.66304988e+02,     1.51183157e-01,     'H8_66304988e02n1_51183157eneg01.inp'],
        [5.57096594e+02,     1.88329549e-01,     'H5_57096594e02n1_88329549eneg01.inp'],
        [8.63291889e+02,     2.18077398e-01,     'H8_63291889e02n2_18077398eneg01.inp'],
        [7.22419030e+02,     2.06046658e-01,     'H7_22419030e02n2_06046658eneg01.inp'],
        [7.45232382e+02,     2.33939835e-01,     'H7_45232382e02n2_33939835eneg01.inp'],
        [6.90673962e+02,     1.81247020e-01,     'H6_90673962e02n1_81247020eneg01.inp'],
        [7.11684633e+02,     1.81530832e-01,     'H7_11684633e02n1_81530832eneg01.inp'],
        [6.42550278e+02,     1.97165974e-01,     'H6_42550278e02n1_97165974eneg01.inp'],
        [7.19319401e+02,     1.84557918e-01,     'H7_19319401e02n1_84557918eneg01.inp'],
        [6.83903679e+02,     1.95489588e-01,     'H6_83903679e02n1_95489588eneg01.inp'],
        [7.86785582e+02,     1.39504304e-01,     'H7_86785582e02n1_39504304eneg01.inp'],
        [8.49410238e+02,     2.36454325e-01,     'H8_49410238e02n2_36454325eneg01.inp'],
        [5.15633683e+02,     2.14962696e-01,     'H5_15633683e02n2_14962696eneg01.inp'],
        [9.33309951e+02,     2.55520219e-01,     'H9_33309951e02n2_55520219eneg01.inp'],
        [6.87681163e+02,     2.11500031e-01,     'H6_87681163e02n2_11500031eneg01.inp'],
        [8.05712038e+02,     1.48615287e-01,     'H8_05712038e02n1_48615287eneg01.inp'],
        [8.23425678e+02,     1.92656480e-01,     'H8_23425678e02n1_92656480eneg01.inp'],
        [8.20514437e+02,     1.83096544e-01,     'H8_20514437e02n1_83096544eneg01.inp'],
        [6.83935061e+02,     1.62517511e-01,     'H6_83935061e02n1_62517511eneg01.inp'],
        [9.91174308e+02,     1.46072880e-01,     'H9_91174308e02n1_46072880eneg01.inp'],
        [7.81217355e+02,     1.74204914e-01,     'H7_81217355e02n1_74204914eneg01.inp'],
        [6.40012054e+02,     2.19113493e-01,     'H6_40012054e02n2_19113493eneg01.inp'],
        [7.34991731e+02,     1.88876624e-01,     'H7_34991731e02n1_88876624eneg01.inp'],
        [7.80793499e+02,     1.97477524e-01,     'H7_80793499e02n1_97477524eneg01.inp'],
        [6.51434193e+02,     1.76975307e-01,     'H6_51434193e02n1_76975307eneg01.inp'],
        [7.01469589e+02,     1.74362509e-01,     'H7_01469589e02n1_74362509eneg01.inp'],
        [8.70398919e+02,     1.85222503e-01,     'H8_70398919e02n1_85222503eneg01.inp'],
        [7.06821546e+02,     1.75650274e-01,     'H7_06821546e02n1_75650274eneg01.inp'],
        [8.64236852e+02,     1.85480560e-01,     'H8_64236852e02n1_85480560eneg01.inp'],
        [7.10817244e+02,     2.27430722e-01,     'H7_10817244e02n2_27430722eneg01.inp'],
        [6.30944870e+02,     2.21714487e-01,     'H6_30944870e02n2_21714487eneg01.inp'],
        [8.19822039e+02,     2.04288726e-01,     'H8_19822039e02n2_04288726eneg01.inp'],
        [7.10774894e+02,     1.96324779e-01,     'H7_10774894e02n1_96324779eneg01.inp'],
        [7.80538897e+02,     2.18978091e-01,     'H7_80538897e02n2_18978091eneg01.inp'],
        [7.98639058e+02,     2.03673933e-01,     'H7_98639058e02n2_03673933eneg01.inp'],
        [7.64733841e+02,     1.71693068e-01,     'H7_64733841e02n1_71693068eneg01.inp'],
        [8.87523608e+02,     2.15068968e-01,     'H8_87523608e02n2_15068968eneg01.inp'],
        [6.51750405e+02,     1.53083788e-01,     'H6_51750405e02n1_53083788eneg01.inp'],
        [8.38131635e+02,     1.70268478e-01,     'H8_38131635e02n1_70268478eneg01.inp'],
        [7.29909403e+02,     2.08801530e-01,     'H7_29909403e02n2_08801530eneg01.inp'],
        [7.63179298e+02,     1.86906040e-01,     'H7_63179298e02n1_86906040eneg01.inp'],
        [7.03194059e+02,     1.71825759e-01,     'H7_03194059e02n1_71825759eneg01.inp'],
        [5.88077161e+02,     2.04817114e-01,     'H5_88077161e02n2_04817114eneg01.inp'],
        [7.59672422e+02,     1.31773297e-01,     'H7_59672422e02n1_31773297eneg01.inp'],
        [7.03108746e+02,     1.68811934e-01,     'H7_03108746e02n1_68811934eneg01.inp'],
        [8.77309269e+02,     1.90954503e-01,     'H8_77309269e02n1_90954503eneg01.inp'],
        [6.99153119e+02,     1.92965737e-01,     'H6_99153119e02n1_92965737eneg01.inp'],
        [7.71123585e+02,     2.10198040e-01,     'H7_71123585e02n2_10198040eneg01.inp'],
        [6.08088215e+02,     2.16146197e-01,     'H6_08088215e02n2_16146197eneg01.inp'],
        [5.96746563e+02,     2.07084510e-01,     'H5_96746563e02n2_07084510eneg01.inp'],
        [7.15752469e+02,     2.20808413e-01,     'H7_15752469e02n2_20808413eneg01.inp'],
        [8.78547448e+02,     1.71152838e-01,     'H8_78547448e02n1_71152838eneg01.inp'],
        [7.66201377e+02,     2.24121826e-01,     'H7_66201377e02n2_24121826eneg01.inp'],
        [6.79636854e+02,     1.67813279e-01,     'H6_79636854e02n1_67813279eneg01.inp'],
        [7.72785594e+02,     2.17233289e-01,     'H7_72785594e02n2_17233289eneg01.inp'],
        [6.03051101e+02,     1.80993899e-01,     'H6_03051101e02n1_80993899eneg01.inp'],
        [8.57131217e+02,     1.79617941e-01,     'H8_57131217e02n1_79617941eneg01.inp'],
        [6.21924990e+02,     1.56692397e-01,     'H6_21924990e02n1_56692397eneg01.inp'],
        [6.87108350e+02,     1.72043419e-01,     'H6_87108350e02n1_72043419eneg01.inp'],
        [1.02901084e+03,     1.74484138e-01,     'H1_02901084e03n1_74484138eneg01.inp'],
        [7.57447104e+02,     1.46581973e-01,     'H7_57447104e02n1_46581973eneg01.inp'],
        [7.11690660e+02,     1.71276059e-01,     'H7_11690660e02n1_71276059eneg01.inp'],
        [5.81692243e+02,     1.94859431e-01,     'H5_81692243e02n1_94859431eneg01.inp'],
        [6.64526754e+02,     1.79076777e-01,     'H6_64526754e02n1_79076777eneg01.inp'],
        [7.48307913e+02,     1.87979977e-01,     'H7_48307913e02n1_87979977eneg01.inp'],
        [7.42285554e+02,     1.94865995e-01,     'H7_42285554e02n1_94865995eneg01.inp'],
        [7.83952503e+02,     1.56697625e-01,     'H7_83952503e02n1_56697625eneg01.inp'],
        [5.46608704e+02,     1.80282758e-01,     'H5_46608704e02n1_80282758eneg01.inp'],
        [8.80787910e+02,     2.11014215e-01,     'H8_80787910e02n2_11014215eneg01.inp'],
        [7.28235351e+02,     1.99937663e-01,     'H7_28235351e02n1_99937663eneg01.inp'],
        [8.29952428e+02,     1.83337359e-01,     'H8_29952428e02n1_83337359eneg01.inp'],
        [6.39190015e+02,     1.85421089e-01,     'H6_39190015e02n1_85421089eneg01.inp'],
        [8.65028910e+02,     1.45731489e-01,     'H8_65028910e02n1_45731489eneg01.inp'],
        [6.49907077e+02,     1.57696659e-01,     'H6_49907077e02n1_57696659eneg01.inp']])

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