"""
GRADIENT BOOSTING REGRESSION METHOD FUNCTION
(Per the Reccomendation of Dr. Lee)

Callable Program written to use Gradient Boosting Regression Method to prdict both
the H and n values from a feature matrix containing both Pmax and hf (10%)

"""
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def multi_model(    test_df, train_df, case,
                    loss_vector, n_estimators, gamma, mx_dp,
                    X_inp, X_known_inp, y_inp, y_known_inp,
                    crit, col, funct_name):

    if case == 0:
        path = '/'+funct_name+'/MULTIMODEL_ANALYSIS/GBRT_1a'
    elif case == 1:
        path = '/'+funct_name+'/MULTIMODEL_ANALYSIS/GBRT_1b'
    
    if not os.path.exists(path):
        os.makedirs(path)

    if int(case) == 0 or int(case) == 1:
        y  = train_df[y_inp]
        y_known  = test_df[y_known_inp]
        X  = train_df[X_inp]
        X_known  = test_df[X_known_inp]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    train_min = y.min()
    train_max = y.max()

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known']+loss_vector)
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    errors = pd.DataFrame(columns = ['n_estimators', 'gamma', 'mse'])

    zz  = 0
    jj = 0
    kk = 0
    ll = 0

    ovr_min = pd.DataFrame(columns = ['loss', 'n_estimators', 'gamma', 'mse'])

    for ii in loss_vector:
        errors = pd.DataFrame(columns = ['loss', 'n_estimators', 'gamma', 'mae', 'mse', 'mape', 'mpe', 'rmspe'])
        errors.n_estimators = n_estimators
        while jj < len(n_estimators):
            while kk < len(gamma):
                model1 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators[jj]),
                         learning_rate = float(gamma[kk]),
                         loss = str(ii),
                         random_state = 1,
                         criterion = crit)

                #Fitting of model to training data
                model1.fit(X, np.ravel(y, order='C'))

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred = model1.predict(X_known)

                # ---------------------------------------------- #
                # ERROR ANALYSIS:
                # ---------------------------------------------- #
                def mean_absolute_percentage_error(y_true, y_pred): 
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                def mean_percentage_error(y_true, y_pred): 
                    return np.mean((y_true - y_pred) / y_true) * 100
                def root_mean_squared_percentage_error(y_true, y_pred):
                    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))*100
                y_known_np = y_known.values

                mae = mean_absolute_error(y_known, y_pred)
                mse =   mean_squared_error(y_known, y_pred)
                mape =  mean_absolute_percentage_error(y_known_np, y_pred)
                mpe =   mean_percentage_error(y_known_np, y_pred)
                rmspe = root_mean_squared_percentage_error(y_known_np, y_pred)

                errors.loc[ll, 'loss']  = str(ii)
                errors.loc[ll, 'n_estimators']  = int(n_estimators[jj])
                errors.loc[ll, 'gamma'] = float(gamma[kk])
                errors.loc[ll, 'mae'] = float(mae)
                errors.loc[ll, 'mse'] = float(mse)
                errors.loc[ll, 'mape'] = float(mape)
                errors.loc[ll, 'mpe'] = float(mpe)
                errors.loc[ll, 'rmspe'] = float(rmspe)
                kk += 1
                ll += 1
            jj += 1
            kk =  0

        err_list = ['mae', 'mse', 'mape', 'mpe', 'rmspe']
        
        for err in err_list:
            # Return the index at which the minimum MSE value is present:
            errors[err] = pd.to_numeric(errors[err])
            minid = errors[err].abs().idxmin()
            
            print(" ")
            print("OPTIMUM VALUES FOR "+str(err)+":")
            print("Location:")
            print(minid)
            print("df row: ")
            print(errors.loc[minid, :])
            
            err_opt_model = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(errors.loc[minid, 'n_estimators']),
                         learning_rate = float(errors.loc[minid, 'gamma']),
                         loss = str(errors.loc[minid, 'loss']),
                         random_state = 1,
                         criterion = crit)

            err_opt_model.fit(X, np.ravel(y, order='C'))
            y_pred = err_opt_model.predict(X_known)

            f, (ax1) = plt.subplots(1, 1, figsize=(2.5, 3.5))
            ax1.plot(np.linspace(train_min,train_max,5), np.linspace(train_min,train_max,5), 'k--', linewidth=0.5)
            ax1.scatter(y_known, y_pred, marker='o', color=col, s=3.5)
            if case == 0:
                ax1.set_xlabel("H : Test Set")
                ax1.set_ylabel("H : Prediction")
                ax1.set_title("H [MPa] Test v. Pred\nloss function: "+str(errors.loc[minid, 'loss'])+'\nlearning rate: '+str(round(float(errors.loc[minid, 'gamma']), 2))+'\nEstimators: '+str(int(errors.loc[minid, 'n_estimators']))+'\nError Metric: '+err, loc='left', fontsize=10)
                plt.xticks(np.linspace(train_min, train_max, 2), (str(int(train_min)), str(int(train_max))))
                plt.yticks(np.linspace(train_min, train_max, 2), (str(int(train_min)), str(int(train_max))))
                fig_name = '/H__crit_'+crit+'__loss_'+str(errors.loc[minid, 'loss'])+'__err_'+err+'.png'
            elif case  == 1:
                ax1.set_xlabel("n : Test Set")
                ax1.set_ylabel("n : Prediction")
                ax1.set_title("n Test v. Pred\nloss function: "+str(errors.loc[minid, 'loss'])+'\nlearning rate: '+str(round(float(errors.loc[minid, 'gamma']), 2))+'\nEstimators: '+str(int(errors.loc[minid, 'n_estimators']))+'\nError Metric: '+err, loc='left', fontsize=10)
                plt.xticks(np.linspace(train_min, train_max, 2), (str(round(float(train_min), 3)), str(round(float(train_max), 3))))
                plt.yticks(np.linspace(train_min, train_max, 2), (str(round(float(train_min), 3)), str(round(float(train_max), 3))))
                fig_name = '/n__crit_'+crit+'__loss_'+str(errors.loc[minid, 'loss'])+'__err_'+err+'.png'
            lt1 = err+': '+str(round(errors.iloc[minid][err], 5))
            at1 = AnchoredText(lt1, frameon=False, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            plt.savefig(path+fig_name, dpi=800)
            plt.close()

        #heatmap1_data = pd.pivot_table(errors.round({'gamma':2}),
        #                        values='mse',
        #                        index=['gamma'],
        #                        columns=errors.loc[:,'n_estimators'],
        #                        aggfunc=np.sum)
        #
        #plt.figure(figsize=(6.5, 3))
        #ax = plt.axes()
        #if case == 0:
        #    sns.heatmap( heatmap1_data,
        #                 ax = ax,
        #                 cmap='coolwarm',
        #                 vmin=20,
        #                 vmax=260,
        #                 cbar_kws={'label': 'H: MSE for "'+str(ii)+ '" loss'})
        #    # OPTIONAL: Top of Plot Title...
        #    #ax.set_title('H: MSE for "'+str(ii)+ '" loss')
        #elif case == 1:
        #    sns.heatmap( heatmap1_data,
        #                 ax = ax,
        #                 cmap='coolwarm',
        #                 vmin=4,
        #                 vmax=15,
        #                 cbar_kws={'label': 'n: MSE'+r'$\bullet10^{6}$'+" for '"+str(ii)+ "' loss"})
        #    # OPTIONAL: Top of Plot Title...
        #    #ax.set_title('n: MSE'+r'$\bullet10^{6}$'+" for '"+str(ii)+ "' loss")
        #labels = [item.get_text() for item in ax.get_yticklabels()]
        #ax.set_yticklabels([str(round(float(label), 2)) for label in labels])
        #ax.set_ylabel("Learning Rate, "+r'$\gamma$')
        #plt.show()
        #plt.close()

        jj = 0
        kk = 0
        ll = 0
        zz = zz + 1

    return results_df, errors

def singlegamma_model(  test_df, train_df, case,
                        loss_vector, n_estimators, gamma, mx_dp,
                        X_inp, X_known_inp, y_inp, y_known_inp,
                        crit):

    if int(case) == 0:
        y  = train_df[y_inp]
        y_known  = test_df[y_known_inp]
        X  = train_df[X_inp]
        X_known  = test_df[X_known_inp]
    elif int(case) == 1:
        y = train_df[y_inp]
        y_known = test_df[y_known_inp]
        X  = train_df[X_inp]
        X_known  = test_df[X_known_inp]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known']+loss_vector)
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    errors = pd.DataFrame(columns = ['n_estimators', 'gamma', 'mse'])

    zz  = 0
    jj = 0
    kk = 0
    ll = 0

    for ii in loss_vector:
        errors = pd.DataFrame(columns = [   'loss', 'n_estimators', 'gamma',
                                            'mae', 'mse', 'mape', 'mpe', 'rmspe'])
        errors.n_estimators = n_estimators
        while jj < len(n_estimators):
            while kk < len(gamma):
                model1 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators[jj]),
                         learning_rate = float(gamma[kk]),
                         loss = str(ii),
                         random_state = 1,
                         criterion = crit)

                #Fitting of model to training data
                model1.fit(X, np.ravel(y, order='C'))

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred = model1.predict(X_known)

                def mean_absolute_percentage_error(y_true, y_pred): 
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                def mean_percentage_error(y_true, y_pred): 
                    return np.mean((y_true - y_pred) / y_true) * 100
                def root_mean_squared_percentage_error(y_true, y_pred):
                    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))*100
                y_known_np = y_known.values

                mae     = mean_absolute_error(y_known, y_pred)
                mse     = mean_squared_error(y_known, y_pred)
                mape    = mean_absolute_percentage_error(y_known_np, y_pred)
                mpe     = mean_percentage_error(y_known_np, y_pred)
                rmspe   = root_mean_squared_percentage_error(y_known_np, y_pred)

                #results_df.loc[:, ii] = y_pred

                errors.loc[ll, 'loss']  = str(ii)
                errors.loc[ll, 'n_estimators']  = int(n_estimators[jj])
                errors.loc[ll, 'gamma'] = float(gamma[kk])
                errors.loc[ll, 'mae'] = float(mae)
                errors.loc[ll, 'mse'] = float(mse)
                errors.loc[ll, 'mape'] = float(mape)
                errors.loc[ll, 'mpe'] = float(mpe)
                errors.loc[ll, 'rmspe'] = float(rmspe)
                kk += 1
                ll += 1
            jj += 1
            kk =  0
        print(" ")
        print("# -------------------------------------------------------- #")
        print("Error Method used: "+ii)
        print(errors.head())

        # Return the index at which the minimum MSE value is present:
        errors['mse'] = pd.to_numeric(errors['mse'])
        minid = errors['mse'].idxmin()
        print(" ")
        print("Location of minimum MSE: ")
        print(minid)
        print(" ")
        print("df row with minimum MSE: ")
        print(errors.loc[minid, :])
        
        plt.figure(figsize=(6, 3.5))
        ax1 = plt.axes()
        ax2 = ax1.twinx()
        title = 'Estimator Optimization for Minimum Error (MSE & MAE)'
        if int(case) == 0:
            lns1 = ax1.plot(errors['n_estimators'], errors['mse'], 'b-', label='MSE', linewidth=1)
            lns2 = ax2.plot(errors['n_estimators'], errors['rmspe'], 'g-', label= 'RMSPE', linewidth=1)
            ax1.set_ylabel("MSE")
            ax2.set_ylabel("rmspe")
            #ax1.set_ylim(40,140)
            ax1.set_xlim(0, 400)
            #ax2.set_ylim(5,10)
        if int(case) == 1:
            l1 = "MSE"+r'$\bullet10^{6}$'
            l2 = "MAE"+r'$\bullet10^{3}$'
            lns1 = ax1.plot(errors['n_estimators'], 1000000*errors['mse'], 'b-', label=l1, linewidth=1)
            lns2 = ax2.plot(errors['n_estimators'], 1000*errors['mae'], 'g-', label=l2, linewidth=1)
            ax1.set_ylabel("MSE"+r'$\bullet10^{6}$')
            ax2.set_ylabel("MAE"+r'$\bullet10^{3}$')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='upper right', facecolor='white', framealpha=1)
        ax1.set_title(title)
        ax1.set_xlabel("Number of Estimators")
        plt.show()

        jj = 0
        kk = 0
        ll = 0
        zz = zz + 1

    return results_df, errors

def opt_pred_plot(  test_df, train_df, case,
                    loss_vector, n_estimators, gamma, mx_dp,
                    plt_opt,
                    X_inp, X_inp_known, y_inp, y_inp_known, crit):
    """
    test_df:
        Pandas df containing the test set (gaussian distribution) data. Should be of the form which was stored from the executable file's call of:
            >>> test_df.to_csv(path+'/test_df.csv')
    train_df:
        Pandas df containing the training set (gridded) data. Should be of the form which was stored from the executable file's call of:
            >>> train_df.to_csv(path+'/train_df.csv')
    case:
        A boolean indicator which indicates which variable is to be predicted.
            > 0 for Strength Coefficient estimation
            > 1 for Strain Hardening Exponent estimation
    loss_vector:
        A list containing the desired loss functions to be called as strings
        (i.e.: ['Huber', 'ls', 'lad'])
    n_estimators:
        A list containing the desired numbers of estimators to be analyzed
    gamma:
        A list contaning the desired learning rates to be analyzed
    mx_dp:
        The mamimum depth of the GRBT model
    plt_opt:
        T : 'True' meaning, yes show me the plots!
        F : 'False' meaning, no don't show the plots!
    X_inp:
        (type: list) A list containing the heading of variables to be included in the feature matrix for the testing dataset
    X_inp_known:
        (type: list) A list containing the heading of variables to be included in the feature matrix for the training dataset
    y_inp:
        (type: list) A list containing a single string to be used as the variable in the output vector fot the testing dataset
    y_inp_known:
        (type: list) A list containing a single string to be used as the variable in the output vector fot the training dataset
    crit:
        A string containing the criterion to be used in the GradientBoostedRegressor model. More details about the implications can be found on at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        (i.e.: 'mse', 'friedman_mse', 'mae')
    """

    if int(case) == 0 or int(case) == 1:
        y  = train_df[y_inp]
        X  = train_df[X_inp]
        y_known  = test_df[y_inp_known]
        X_known  = test_df[X_inp_known]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    train_min = y.min()
    train_max = y.max()

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known', 'y_pred'])
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    jj = 0
    kk = 0
    ll = 0

    for ii in loss_vector:
        while jj < len(n_estimators):
            while kk < len(gamma):
                model1 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators[jj]),
                         learning_rate = float(gamma[kk]),
                         loss = str(ii),
                         random_state = 1,
                         criterion = crit)

                #Fitting of model to training data
                model1.fit(X, np.ravel(y, order='C'))

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred = model1.predict(X_known)
                results_df.loc[:, 'y_pred'] = y_pred

                # ---------------------------------------------- #
                # ERROR ANALYSIS:
                # ---------------------------------------------- #
                y_known_np = y_known.values
                def MAPE(y_true, y_pred): 
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                def root_mean_squared_percentage_error(y_true, y_pred):
                    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))*100

                mae = str(round(float(mean_absolute_error(y_known, y_pred)), 4))
                mape = str(round(float(MAPE(y_known_np, y_pred)), 4))
                rmspe = str(round(float(root_mean_squared_percentage_error(y_known_np, y_pred)), 4))
                if case ==0:
                    mse = str(round(float(mean_squared_error(y_known, y_pred)), 4))
                if case ==1: 
                    mse = str(round(1000000*float(mean_squared_error(y_known, y_pred)), 4))

                kk += 1
                ll += 1
            jj += 1
            kk =  0
        jj = 0
        kk = 0
        ll = 0

    from matplotlib.offsetbox import AnchoredText

    if plt_opt == 'T':
        if case == 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 5))
            ax1.plot(np.linspace(train_min,train_max,5), np.linspace(train_min,train_max,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred)
            ax1.set_xlabel('H [MPa] : Testing Set')
            ax1.set_ylabel('H [MPa] : Prediction Set')
            title = 'Strength Coefficient\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma[0])+r'$,     estimators=$'+str(n_estimators[0])+r'$, loss=$'+str(loss_vector[0])+r'$)$'
            ax1.set_title(title)
            lt1 = 'MAPE: '+mape+' %\nRMSPE: '+rmspe+'%\nMSE'+': '+mse+"\nMAE: "+mae
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            #plt.show()

            
            # Plot feature importance
            feature_importance = model1.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            #f, (ax1) = plt.subplots(1, 1, figsize=(1.5, 3))
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            labels = np.asarray(X_inp_known)
            plt.yticks(pos, labels[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.show()
        if case == 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 5))
            ax1.plot(np.linspace(train_min,train_max,5), np.linspace(train_min,train_max,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred)
            ax1.set_xlabel('n : Testing Set')
            ax1.set_ylabel('n : Prediction Set')
            title = 'Strain Hardening Index\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma[0])+r'$,   estimators=$'+str(n_estimators[0])+r'$, loss=$'+str(loss_vector[0])+r'$)$'
            ax1.set_title(title)
            lt1 = 'MAPE: '+mape+' %\nRMSPE: '+rmspe+'%\nMSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            #plt.show()

            # Plot feature importance
            feature_importance = model1.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            #f, (ax1) = plt.subplots(1, 1, figsize=(1.5, 3))
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            labels = np.asarray(X_inp_known)
            plt.yticks(pos, labels[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.show()

    return results_df, mae, mse

def opt_pred_plot_dual(test_df, train_df, case, loss_vector1, n_estimators1, gamma1, mx_dp, loss_vector2, n_estimators2, gamma2, plt_opt):
        
    """
    test_df:
        pandas DataFrame with testing data
    train_df:
        pandas DataFrame with training data
    case:
        0 for Strength Coefficient estimation
        1 for Strain Hardening Exponent estimation
    loss:
        A vector containing the desired loss functions to be called as strings
        (i.e.: ['Huber', 'ls', 'lad'])
    n_estimators:
        A list containing the desired numbers of estimators to be analyzed
    gamma:
        A list contaning the desired learning rates to be analyzed
    mx_dp:
        The mamimum depth of the GRBT model
    plt_opt:
        T : 'True' meaning, yes plot!
        F : 'False' meaning, no don't plot!
    """

    if int(case) == 0:
        y  = train_df[['H']]
        y_known  = test_df[['H']]
        
        X1  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]
        X_known1  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]

        X2  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n']]
        X_known2  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n']]

    elif int(case) == 1:
        y = train_df[['n']]
        y_known = test_df[['n']]

        X1  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]
        X_known1  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]

        X2  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H']]
        X_known2  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H_prediction1']]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known', 'y_pred1', 'y_pred2'])
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    jj = 0
    kk = 0
    ll = 0

    for ii in loss_vector1:
        while jj < len(n_estimators1):
            while kk < len(gamma1):
                model1 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators1[jj]),
                         learning_rate = float(gamma1[kk]),
                         loss = str(ii),
                         random_state = 1,
                         criterion = "mse")
                
                model2 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators2[jj]),
                         learning_rate = float(gamma2[kk]),
                         loss = str(ii),
                         random_state = 1,
                         criterion = "mse")

                #Fitting of model to training data
                model1.fit(X1, np.ravel(y, order='C'))
                model2.fit(X2, np.ravel(y, order='C'))

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred1 = model1.predict(X_known1)
                y_pred2 = model2.predict(X_known2)

                results_df.loc[:, 'y_pred1'] = y_pred1
                results_df.loc[:, 'y_pred2'] = y_pred2

                if case ==0: 
                    mse1 = str(round(float(mean_squared_error(y_known, y_pred1)), 4))
                    mse2 = str(round(float(mean_squared_error(y_known, y_pred2)), 4))
                if case ==1: 
                    mse1 = str(round(1000000*float(mean_squared_error(y_known, y_pred1)), 4))
                    mse2 = str(round(1000000*float(mean_squared_error(y_known, y_pred2)), 4))

                kk += 1
                ll += 1
            jj += 1
            kk =  0
        jj = 0
        kk = 0
        ll = 0

    from matplotlib.offsetbox import AnchoredText

    if plt_opt == 'T':
        if case == 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))
            ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=0.75)
            ax2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=0.75)
            ax1.scatter(results_df.y_known, results_df.y_pred1, label="n NOT INCLUDED", marker='o', color='blue', s=3.5)
            ax2.scatter(results_df.y_known, results_df.y_pred2, label="n INCLUDED", marker='o', color='green', s=3.5)
            ax1.set_xlabel('H [MPa] : Testing Set\n [strain hardening index: NOT INCLUDED]')
            ax1.set_ylabel('H [MPa] : Prediction Set')
            ax2.set_xlabel('H [MPa] : Testing Set\n [strain hardening index: INCLUDED]')
            ax2.set_ylabel('H [MPa] : Prediction Set')
            title1 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma1[0])+'\n'+r'$estimators=$'+str(n_estimators1[0])+'\n'+r'$loss=$'    +str(loss_vector1[0])
            ax1.set_title(title1)
            title2 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma2[0])+'\n'+r'$estimators=$'+str(n_estimators2[0])+'\n'+r'$loss=$'    +str(loss_vector2[0])
            ax2.set_title(title2)

            t1 = "MSE: "+mse1
            t2 = "MSE: "+mse2
            at1 = AnchoredText(t1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)

            at2 = AnchoredText(t2, frameon=True, loc='lower right')
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax2.add_artist(at2)

            #title = 'Strength Coefficient\nTest vs. Prediction Set\n'+r'$(depth=4, \gamma=0.04, estimators=33, loss=lad)$'
            #ax1.set_title(title)
            plt.show()
        if case == 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))
            ax1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred1, label="H NOT INCLUDED", marker='o', color='blue', s=3.5)
            ax2.scatter(results_df.y_known, results_df.y_pred2, label="H INCLUDED", marker='o', color='green', s=3.5)
            ax1.set_xlabel('n : Testing Set\n [strength coefficient: NOT INCLUDED]')
            ax1.set_ylabel('n : Prediction Set')
            ax2.set_xlabel('n : Testing Set\n [strength coefficient: INCLUDED]')
            ax2.set_ylabel('n : Prediction Set')
            title1 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma1[0])+'\n'+r'$estimators=$'+str(n_estimators1[0])+'\n'+r'$loss=$'    +str(loss_vector1[0])
            ax1.set_title(title1)
            title2 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma2[0])+'\n'+r'$estimators=$'+str(n_estimators2[0])+'\n'+r'$loss=$'    +str(loss_vector2[0])
            ax2.set_title(title2)

            lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse1
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)

            lt2 = 'MSE'+r'$\bullet10^{6}$'+': '+mse2
            at2 = AnchoredText(lt2, frameon=True, loc='lower right')
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax2.add_artist(at2)

            #title = 'Strain Hardening Index\nTest vs. Prediction Set\n'+r'$(depth=3, \gamma=0.25, estimators=10, loss=lad)$'
            #ax1.set_title(title)
            plt.show()

    return results_df

def opt_pred_plot_tri(test_df, train_df, case, loss_vector1, n_estimators1, gamma1, mx_dp, loss_vector2, n_estimators2, gamma2, loss_vector3, n_estimators3, gamma3, plt_opt):
    """
    test_df:
        pandas DataFrame with testing data
    train_df:
        pandas DataFrame with training data
    case:
        0 for Strength Coefficient estimation
        1 for Strain Hardening Exponent estimation
    loss:
        A vector containing the desired loss functions to be called as strings
        (i.e.: ['Huber', 'ls', 'lad'])
    n_estimators:
        A list containing the desired numbers of estimators to be analyzed
    gamma:
        A list contaning the desired learning rates to be analyzed
    mx_dp:
        The mamimum depth of the GRBT model
    plt_opt:
        T : 'True' meaning, yes plot!
        F : 'False' meaning, no don't plot!
    """

    if int(case) == 0:
        y = train_df[['H']]
        y_known = test_df[['H']]

        X1  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]
        X_known1  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]

        X2  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n']]
        X_known2  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n']]

        X3  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n']]
        X_known3  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'n_prediction1']]

    elif int(case) == 1:
        y = train_df[['n']]
        y_known = test_df[['n']]

        X1  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]
        X_known1  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C']]

        X2  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H']]
        X_known2  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H']]

        X3  = train_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H']]
        X_known3  = test_df[['Pmax',
            'hf1', 'hf2', 'hf3', 'hf4',
            'hf5', 'hf6', 'hf7', 'hf8',
            'hf9', 'C', 'H_prediction1']]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known', 'y_pred1', 'y_pred2', 'y_pred3'])
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    jj = 0
    kk = 0
    ll = 0

    for ii in loss_vector1:
        while jj < len(n_estimators1):
            while kk < len(gamma1):
                model1 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators1[jj]),
                         learning_rate = float(gamma1[kk]),
                         loss = str(loss_vector1[jj]),
                         random_state = 1,
                         criterion = "mse")
                
                model2 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators2[jj]),
                         learning_rate = float(gamma2[kk]),
                         loss = str(loss_vector2[jj]),
                         random_state = 1,
                         criterion = "mse")
                
                model3 = GradientBoostingRegressor(
                         max_depth = mx_dp,
                         n_estimators = int(n_estimators3[jj]),
                         learning_rate = float(gamma3[kk]),
                         loss = str(loss_vector3[jj]),
                         random_state = 1,
                         criterion = "mse")

                #Fitting of model to training data
                model1.fit(X1, np.ravel(y, order='C'))
                model2.fit(X2, np.ravel(y, order='C'))
                model3.fit(X3, np.ravel(y, order='C'))

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred1 = model1.predict(X_known1)
                y_pred2 = model2.predict(X_known2)
                y_pred3 = model3.predict(X_known3)

                results_df.loc[:, 'y_pred1'] = y_pred1
                results_df.loc[:, 'y_pred2'] = y_pred2
                results_df.loc[:, 'y_pred3'] = y_pred3

                if case ==0: 
                    mse1 = str(round(float(mean_squared_error(y_known, y_pred1)), 4))
                    mse2 = str(round(float(mean_squared_error(y_known, y_pred2)), 4))
                    mse3 = str(round(float(mean_squared_error(y_known, y_pred3)), 4))
                if case ==1: 
                    mse1 = str(round(1000000*float(mean_squared_error(y_known, y_pred1)), 4))
                    mse2 = str(round(1000000*float(mean_squared_error(y_known, y_pred2)), 4))
                    mse3 = str(round(1000000*float(mean_squared_error(y_known, y_pred3)), 4))

                kk += 1
                ll += 1
            jj += 1
            kk =  0
        jj = 0
        kk = 0
        ll = 0

    from matplotlib.offsetbox import AnchoredText

    if plt_opt == 'T':
        if case == 0:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,  3.5))
            
            ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=0.75)
            ax1.scatter(results_df.y_known, results_df.y_pred1, label="n NOT INCLUDED", marker='o', color='blue', s=3.5)
            ax1.set_xlabel('H : Testing Set\n [Strain Hardening Index: NOT INCLUDED]')
            ax1.set_ylabel('H : Prediction Set')
            title1 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma1[0])+'\n'+r'$estimators=$'+str(n_estimators1[0])+'\n'+r'$loss=$'    +str(loss_vector1[0])
            ax1.set_title(title1)
            lt1 = 'MSE : '+mse1
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)

            ax2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=0.75)
            ax2.scatter(results_df.y_known, results_df.y_pred2, label="KNOWN n INCLUDED", marker='o', color='green', s=3.5)
            ax2.set_xlabel('H : Testing Set\n [Strain Hardening Index: KNOWN INCLUDED]')
            ax2.set_ylabel('H : Prediction Set')
            title2 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma2[0])+'\n'+r'$estimators=$'+str(n_estimators2[0])+'\n'+r'$loss=$'    +str(loss_vector2[0])
            ax2.set_title(title2)
            lt2 = 'MSE : '+mse2
            at2 = AnchoredText(lt2, frameon=True, loc='lower right')
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax2.add_artist(at2)
            
            ax3.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=0.75)
            ax3.scatter(results_df.y_known, results_df.y_pred3, label="PRED n INCLUDED", marker='o', color='red', s=3.5)
            ax3.set_xlabel('H : Testing Set\n [Strain Hardening Index: PRED INCLUDED]')
            ax3.set_ylabel('H : Prediction Set')
            title3 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma3[0])+'\n'+r'$estimators=$'+str(n_estimators3[0])+'\n'+r'$loss=$'    +str(loss_vector3[0])
            ax3.set_title(title3)
            lt3 = 'MSE : '+mse3
            at3 = AnchoredText(lt3, frameon=True, loc='lower right')
            at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax3.add_artist(at3)

            plt.show()

        if case == 1:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,  3.5))
            
            ax1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred1, label="H NOT INCLUDED", marker='o', color='blue', s=3.5)
            ax1.set_xlabel('n : Testing Set\n [strength coefficient: NOT INCLUDED]')
            ax1.set_ylabel('n : Prediction Set')
            title1 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma1[0])+'\n'+r'$estimators=$'+str(n_estimators1[0])+'\n'+r'$loss=$'    +str(loss_vector1[0])
            ax1.set_title(title1)
            lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse1
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            
            ax2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax2.scatter(results_df.y_known, results_df.y_pred2, label="KNOWN H INCLUDED", marker='o', color='green', s=3.5)
            ax2.set_xlabel('n : Testing Set\n [strength coefficient: KNOWN INCLUDED]')
            ax2.set_ylabel('n : Prediction Set')
            title2 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma2[0])+'\n'+r'$estimators=$'+str(n_estimators2[0])+'\n'+r'$loss=$'    +str(loss_vector2[0])
            ax2.set_title(title2)
            lt2 = 'MSE'+r'$\bullet10^{6}$'+': '+mse2
            at2 = AnchoredText(lt2, frameon=True, loc='lower right')
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax2.add_artist(at2)
            
            ax3.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax3.scatter(results_df.y_known, results_df.y_pred3, label="PRED H INCLUDED", marker='o', color='red', s=3.5)
            ax3.set_xlabel('n : Testing Set\n [strength coefficient: PRED INCLUDED]')
            ax3.set_ylabel('n : Prediction Set')
            title3 = r'$depth=$'+str(mx_dp)+'\n'+r'$\gamma=$'+str(gamma3[0])+'\n'+r'$estimators=$'+str(n_estimators3[0])+'\n'+r'$loss=$'    +str(loss_vector3[0])
            ax3.set_title(title3)
            lt3 = 'MSE'+r'$\bullet10^{6}$'+': '+mse3
            at3 = AnchoredText(lt3, frameon=True, loc='lower right')
            at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax3.add_artist(at3)

            plt.show()

    return results_df

def opt_pred_plot_XGB(  test_df, train_df, case,
                        loss_vector,
                        n_estimators,
                        gamma, mx_dp, plt_opt,
                        X_inp, X_inp_known, y_inp, y_inp_known,
                        crit):
    """
    test_df:
        pandas DataFrame with testing data
    train_df:
        pandas DataFrame with training data
    case:
        0 for Strength Coefficient estimation
        1 for Strain Hardening Exponent estimation
    loss:
        A vector containing the desired loss functions to be called as strings
        (i.e.: ['Huber', 'ls', 'lad'])
    n_estimators:
        A list containing the desired numbers of estimators to be analyzed
    gamma:
        A list contaning the desired learning rates to be analyzed
    mx_dp:
        The mamimum depth of the GRBT model
    plt_opt:
        T : 'True' meaning, yes plot!
        F : 'False' meaning, no don't plot!
    """

    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV

    if int(case) == 0 or int(case) == 1:
        y  = train_df[y_inp].astype(float)
        X  = train_df[X_inp].astype(float)
        
        y_known  = test_df[y_inp_known].astype(float)
        X_known  = test_df[X_inp_known].astype(float)
    
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known', 'y_pred'])
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    jj = 0
    kk = 0
    ll = 0

    for ii in loss_vector:
        while jj < len(n_estimators):
            while kk < len(gamma):
                model1 = xgb.XGBRegressor(
                            tree_method = 'exact',
                            random_state = 1)

                parameters = {
                    'max_depth': [2,3,4,5],
                    'n_estimators': [100,250,350,450,550,650,750,800],
                    'learning_rate': [0.01, 0.045,0.05,0.06]}
                
                clf = GridSearchCV(model1, parameters, cv=5, scoring='neg_mean_squared_error')

                #Fitting of model to training data
                clf.fit(X, y)
                print(clf.best_params_)

                # Predicting the result for the KNOWN VAULES!
                # Should be expecting:
                # y_pred = y_known
                y_pred = clf.predict(X_known)
                results_df.loc[:, 'y_pred'] = y_pred

                # ---------------------------------------------- #
                # ERROR ANALYSIS:
                # ---------------------------------------------- #
                mae = str(round(float(mean_absolute_error(y_known, y_pred)), 4))
                if case ==0: 
                    mse = str(round(float(mean_squared_error(y_known, y_pred)), 4))
                if case ==1: 
                    mse = str(round(1000000*float(mean_squared_error(y_known, y_pred)), 4))
                
                print "mse:"
                print mse
                print "mae:"
                print mae

                kk += 1
                ll += 1
            jj += 1
            kk =  0
        jj = 0
        kk = 0
        ll = 0

    from matplotlib.offsetbox import AnchoredText

    if plt_opt == 'T':
        if case == 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 5))
            ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred)
            ax1.set_xlabel('H [MPa] : Testing Set')
            ax1.set_ylabel('H [MPa] : Prediction Set')
            title = 'Strength Coefficient\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma[0])+r'$,     estimators=$'+str(n_estimators[0])+r'$, loss=$'+str(loss_vector[0])+r'$)$'
            ax1.set_title(title)
            t1 = "MSE: "+mse+"\nMAE: "+mae
            at1 = AnchoredText(t1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            #plt.show()
            
            # Plot feature importance
            feature_importance = model1.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            #f, (ax1) = plt.subplots(1, 1, figsize=(1.5, 3))
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            labels = np.asarray(X_inp_known)
            plt.yticks(pos, labels[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.show()
        if case == 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 5))
            ax1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
            ax1.scatter(results_df.y_known, results_df.y_pred)
            ax1.set_xlabel('n : Testing Set')
            ax1.set_ylabel('n : Prediction Set')
            title = 'Strain Hardening Index\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma[0])+r'$,   estimators=$'+str(n_estimators[0])+r'$, loss=$'+str(loss_vector[0])+r'$)$'
            ax1.set_title(title)
            lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
            at1 = AnchoredText(lt1, frameon=True, loc='lower right')
            at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.add_artist(at1)
            #plt.show()

            # Plot feature importance
            feature_importance = model1.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            #f, (ax1) = plt.subplots(1, 1, figsize=(1.5, 3))
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            labels = np.asarray(X_inp_known)
            plt.yticks(pos, labels[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.show()
            
    return results_df, mae, mse

def feature_reduction_analysis(  test_df, train_df, case,
                    loss_funct, n_estimators, gamma, mx_dp,
                    plt_opt,
                    X_inp, X_inp_known, y_inp, y_inp_known, crit, err_metric):

    if int(case) == 0 or int(case) == 1:
        y  = train_df[y_inp]
        X  = train_df[X_inp]
        y_known  = test_df[y_inp_known]
        X_known  = test_df[X_inp_known]
    else:
        print('ERROR: Cannot determine which parameter to estimate.')

    train_min = y.min()
    train_max = y.max()

    # Printing out a comparison of the results:
    results_df = pd.DataFrame(columns = ['y_known', 'y_pred'])
    if case == 0:
        results_df.y_known = test_df['H'].tolist()
    elif case == 1:
        results_df.y_known = test_df['n'].tolist()

    model1 = GradientBoostingRegressor(
                max_depth = mx_dp,
                n_estimators = n_estimators,
                learning_rate = gamma,
                loss = loss_funct,
                random_state = 1,
                criterion = crit)

    #Fitting of model to training data
    model1.fit(X, np.ravel(y, order='C'))

    # Predicting the result for the KNOWN VAULES!
    # Should be expecting:
    # y_pred = y_known
    y_pred = model1.predict(X_known)
    results_df.loc[:, 'y_pred'] = y_pred

    # ---------------------------------------------- #
    # ERROR ANALYSIS:
    # ---------------------------------------------- #
    y_known_np = y_known.values
    if err_metric == 'mape':
        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        err =  mean_absolute_percentage_error(y_known_np, y_pred)
    if err_metric == 'mpe':
        def mean_percentage_error(y_true, y_pred): 
            return np.mean((y_true - y_pred) / y_true) * 100
        err =   mean_percentage_error(y_known_np, y_pred)
    if err_metric == 'rmspe':
        def root_mean_squared_percentage_error(y_true, y_pred):
            return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))*100
        err = root_mean_squared_percentage_error(y_known_np, y_pred)
    if err_metric == 'mae':
        err = mean_absolute_error(y_known, y_pred)
    if err_metric == 'mse':
        err =   mean_squared_error(y_known, y_pred)
    
    from matplotlib.offsetbox import AnchoredText

    if case == 0:
        f, (ax1, ax2)= plt.subplots(1, 2, figsize=(6.5, 5))
        ax1.plot(np.linspace(train_min,train_max,5), np.linspace(train_min,train_max,5), 'k--', linewidth=1)
        ax1.scatter(results_df.y_known, results_df.y_pred)
        ax1.set_xlabel('H [MPa] : Testing Set')
        ax1.set_ylabel('H [MPa] : Prediction Set')
        title = 'Strength Coefficient\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma)+r'$,     estimators=$'+str(n_estimators)+r'$, loss=$'+loss_funct+r'$)$'
        ax1.set_title(title)
        #lt1 = 'MSE : '+mse+"\nMAE : "+mae+"\nRMSPE : "+str(round(rmspe, 4))
        #at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        #at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        #ax1.add_artist(at1)
        # Plot feature importance
        feature_importance = model1.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        labels = np.asarray(X_inp_known)
        plt.yticks(pos, labels[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        if plt_opt == 'T':
            plt.show() 
        plt.close()
    if case == 1:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 5))
        ax1.plot(np.linspace(train_min,train_max,5), np.linspace(train_min,train_max,5), 'k--', linewidth=1)
        ax1.scatter(results_df.y_known, results_df.y_pred)
        ax1.set_xlabel('n : Testing Set')
        ax1.set_ylabel('n : Prediction Set')
        title = 'Strain Hardening Index\nTest vs. Prediction Set\n'+r'$(depth=$'+str(mx_dp)+r'$, \gamma=$'+str(gamma)+r'$,   estimators=$'+str(n_estimators)+r'$, loss=$'+loss_funct+r'$)$'
        ax1.set_title(title)
        #lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae+"\nRMSPE : "+str(round(rmspe, 4))
        #at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        #at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        #ax1.add_artist(at1)
        # Plot feature importance
        feature_importance = model1.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        labels = np.asarray(X_inp_known)
        plt.yticks(pos, labels[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        if plt_opt == 'T':
            plt.show() 
        plt.close()

    feature_priority_list = labels[sorted_idx]
    rem_f = feature_priority_list[0]

    return results_df, err, rem_f