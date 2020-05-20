"""
CASE:           MATERIAL #1 (7075-T6 Aluminum)
Training Size:  7 x 7 = 49
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
                        'PREDICTION_BOOSTING', 'XGBOOST', 'FEATURE_RECUTION', '3MATERIAL_RESULTS'}
    What portions of the machine-learning analysis should be completed?
"""
plotting_on = 'T'
analysis_call = ['3MATERIAL_RESULTS']

# Function Name:
funct_name = r'MAT1_7075T6'
material_string = '7075-T6 Aluminum'
color = 'black'

# Array defining training grid (Typically generated using np.linspace()):
H_ss = np.array([500, 608.33, 716.67, 825, 933.33, 1041.67, 1150])
n_ss = np.array([0.068, 0.083, 0.098, 0.113, 0.128, 0.143, 0.158])
problem_iterations = []

# A few important files:
basefile = r'R:\RefinedMesh_PcklBase.inp'
OTP_dir = r'R:\SIMDIR_7075T6'
master_code_dir = r'R:'

"""
KNOWN MECHANICAL properites. These are the properties used in
the Ramberg-Osgood model to calculate the elastic-plastic behavior
for <<ALL>> iterations.

Format/units:
    E   : youngs_modulus : integer [units : MPa           ]
    nu  : poissons_ratio : float   [units : Dimensionless ]
"""
E = 70300
nu = 0.345

# TRAINING PAIRINGS (Generated using np.random.normal(mu, sigma, n))
training_gaussian_pairings = np.array([
        [8.42345053e+02,     1.05770989e-01,     'H8_42345053e02n1_05770989eneg01.inp'],
        [8.61230220e+02,     1.04825859e-01,     'H8_61230220e02n1_04825859eneg01.inp'],
        [9.69135834e+02,     9.52395402e-02,     'H6_79664691e02n9_52395402eneg02.inp'],
        [6.79664691e+02,     1.37655518e-01,     'H9_69135834e02n1_37655518eneg01.inp'],
        [8.21176241e+02,     1.10737772e-01,     'H8_21176241e02n1_10737772eneg01.inp'],
        [9.78976777e+02,     1.25597635e-01,     'H9_78976777e02n1_25597635eneg01.inp'],
        [7.28717355e+02,     1.19531117e-01,     'H7_28717355e02n1_19531117eneg01.inp'],
        [8.69795074e+02,     1.22969229e-01,     'H8_69795074e02n1_22969229eneg01.inp'],
        [7.01822309e+02,     1.31999357e-01,     'H7_01822309e02n1_31999357eneg01.inp'],
        [8.98045239e+02,     1.05083085e-01,     'H8_98045239e02n1_05083085eneg01.inp'],
        [7.69564473e+02,     1.27676255e-01,     'H7_69564473e02n1_27676255eneg01.inp'],
        [7.48770337e+02,     1.24154148e-01,     'H7_48770337e02n1_24154148eneg01.inp'],
        [8.23118029e+02,     1.15679708e-01,     'H8_23118029e02n1_15679708eneg01.inp'],
        [9.41432527e+02,     1.24979794e-01,     'H9_41432527e02n1_24979794eneg01.inp'],
        [7.04019909e+02,     1.04896645e-01,     'H7_04019909e02n1_04896645eneg01.inp'],
        [9.15591161e+02,     1.22168066e-01,     'H9_15591161e02n1_22168066eneg01.inp'],
        [7.54830405e+02,     1.23326596e-01,     'H7_54830405e02n1_23326596eneg01.inp'],
        [8.21589912e+02,     1.13352310e-01,     'H8_21589912e02n1_13352310eneg01.inp'],
        [8.84737989e+02,     1.27832125e-01,     'H8_84737989e02n1_27832125eneg01.inp'],
        [8.26149617e+02,     1.05617575e-01,     'H8_26149617e02n1_05617575eneg01.inp'],
        [7.46788254e+02,     1.20869703e-01,     'H7_46788254e02n1_20869703eneg01.inp'],
        [8.79708116e+02,     1.39794075e-01,     'H8_79708116e02n1_39794075eneg01.inp'],
        [5.08990268e+02,     1.11220726e-01,     'H5_08990268e02n1_11220726eneg01.inp'],
        [8.80310829e+02,     1.36466487e-01,     'H8_80310829e02n1_36466487eneg01.inp'],
        [9.74585863e+02,     1.35766890e-01,     'H9_74585863e02n1_35766890eneg01.inp'],
        [7.89738483e+02,     1.27117216e-01,     'H7_89738483e02n1_27117216eneg01.inp'],
        [6.99463187e+02,     1.11523380e-01,     'H6_99463187e02n1_11523380eneg01.inp'],
        [6.47672753e+02,     1.15103179e-01,     'H6_47672753e02n1_15103179eneg01.inp'],
        [1.02954232e+03,     1.17842596e-01,     'H1_02954232e03n1_17842596eneg01.inp'],
        [7.69603396e+02,     9.20550448e-02,     'H7_69603396e02n9_20550448eneg02.inp'],
        [1.01680241e+03,     1.27412425e-01,     'H1_01680241e03n1_27412425eneg01.inp'],
        [9.64582004e+02,     1.15315551e-01,     'H9_64582004e02n1_15315551eneg01.inp'],
        [7.74684039e+02,     1.11713183e-01,     'H7_74684039e02n1_11713183eneg01.inp'],
        [7.89973971e+02,     1.23509827e-01,     'H7_89973971e02n1_23509827eneg01.inp'],
        [8.61715525e+02,     1.12805255e-01,     'H8_61715525e02n1_12805255eneg01.inp'],
        [9.39156715e+02,     1.35258306e-01,     'H9_39156715e02n1_35258306eneg01.inp'],
        [7.32778579e+02,     9.38322629e-02,     'H7_32778579e02n9_38322629eneg02.inp'],
        [7.11999449e+02,     1.24224808e-01,     'H7_11999449e02n1_24224808eneg01.inp'],
        [7.63074259e+02,     1.25840346e-01,     'H7_63074259e02n1_25840346eneg01.inp'],
        [8.98172751e+02,     1.25247103e-01,     'H8_98172751e02n1_25247103eneg01.inp'],
        [7.74439202e+02,     1.23068303e-01,     'H7_74439202e02n1_23068303eneg01.inp'],
        [8.82262350e+02,     1.47273212e-01,     'H8_82262350e02n1_47273212eneg01.inp'],
        [7.65382442e+02,     8.66651771e-02,     'H7_65382442e02n8_66651771eneg02.inp'],
        [8.50743683e+02,     9.74483642e-02,     'H8_50743683e02n9_74483642eneg02.inp'],
        [7.49181217e+02,     1.17917800e-01,     'H7_49181217e02n1_17917800eneg01.inp'],
        [6.20668346e+02,     1.43009673e-01,     'H6_20668346e02n1_43009673eneg01.inp'],
        [7.05813700e+02,     1.16575057e-01,     'H7_05813700e02n1_16575057eneg01.inp'],
        [7.44528542e+02,     1.26765398e-01,     'H7_44528542e02n1_26765398eneg01.inp'],
        [7.29278080e+02,     1.07093271e-01,     'H7_29278080e02n1_07093271eneg01.inp'],
        [8.49923178e+02,     1.32168460e-01,     'H8_49923178e02n1_32168460eneg01.inp'],
        [8.98981123e+02,     1.15358530e-01,     'H8_98981123e02n1_15358530eneg01.inp'],
        [7.55888247e+02,     1.07423922e-01,     'H7_55888247e02n1_07423922eneg01.inp'],
        [6.95308570e+02,     1.18472450e-01,     'H6_95308570e02n1_18472450eneg01.inp'],
        [9.69705708e+02,     1.07451832e-01,     'H9_69705708e02n1_07451832eneg01.inp'],
        [1.00777401e+03,     1.06169073e-01,     'H1_00777401e03n1_06169073eneg01.inp'],
        [7.06115388e+02,     1.27136523e-01,     'H7_06115388e02n1_27136523eneg01.inp'],
        [8.82424992e+02,     1.23513493e-01,     'H8_82424992e02n1_23513493eneg01.inp'],
        [7.92839104e+02,     1.05557068e-01,     'H7_92839104e02n1_05557068eneg01.inp'],
        [7.13190034e+02,     1.07837416e-01,     'H7_13190034e02n1_07837416eneg01.inp'],
        [6.81429607e+02,     1.19305255e-01,     'H6_81429607e02n1_19305255eneg01.inp'],
        [8.22154241e+02,     1.13094847e-01,     'H8_22154241e02n1_13094847eneg01.inp'],
        [7.84994755e+02,     1.08783554e-01,     'H7_84994755e02n1_08783554eneg01.inp'],
        [8.58628956e+02,     1.05160561e-01,     'H8_58628956e02n1_05160561eneg01.inp'],
        [6.58060062e+02,     9.78925249e-02,     'H6_58060062e02n9_78925249eneg02.inp'],
        [8.72466071e+02,     1.26412031e-01,     'H8_72466071e02n1_26412031eneg01.inp'],
        [6.06881224e+02,     1.07303995e-01,     'H6_06881224e02n1_07303995eneg01.inp'],
        [7.27331087e+02,     1.10238273e-01,     'H7_27331087e02n1_10238273eneg01.inp'],
        [7.64155817e+02,     1.01973322e-01,     'H7_64155817e02n1_01973322eneg01.inp'],
        [8.26808807e+02,     1.10791404e-01,     'H8_26808807e02n1_10791404eneg01.inp'],
        [9.48258222e+02,     1.08038827e-01,     'H9_48258222e02n1_08038827eneg01.inp'],
        [8.03159706e+02,     1.19438654e-01,     'H8_03159706e02n1_19438654eneg01.inp'],
        [8.40658778e+02,     1.28529645e-01,     'H8_40658778e02n1_28529645eneg01.inp'],
        [8.13597838e+02,     1.18993790e-01,     'H8_13597838e02n1_18993790eneg01.inp'],
        [8.01753104e+02,     1.13803888e-01,     'H8_01753104e02n1_13803888eneg01.inp'],
        [6.85519216e+02,     9.36858958e-02,     'H6_85519216e02n9_36858958eneg02.inp']])

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