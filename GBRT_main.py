"""
PRIMARY RESULTS-GENERATING FILE FOR RESEARCH INCLUDED IN DARREN PROMER'S THESIS.

THE FINAL LINE OF ANY EXECUTABLE MATERIAL FILE IS A CALL FOR:
    GBRT_main.GBRT_fullanalysis(train_df, test_df, funct_name, analysis_call)

    >>> Note: In order to recieve ALL possible output plots/results,
    >>> all options must be included in analysis_call
"""

def GBRT_fullanalysis(  train_df, test_df, funct_name,
                        analysis_call, material_string):
    import numpy as np
    import matplotlib.pyplot as plt
    
    if 'MULTIMODEL_ANALYSIS' in analysis_call:
        # ------------------------ #
        #  GBRT ANALYSIS NUMBER 1  #
        # ------------------------ #
        from GBRT_Functions import multi_model
        # ---------- PART A ------------ #
        print
        print("#---------------------------------------#")
        print "GBRT ANALYSIS NUMBER 1a"
        print "Analysis for Strength Coefficient"
        loss         = ['ls', 'lad', 'huber']
        n_estimators = np.linspace(201, 201, 1)
        gamma        = np.linspace(0.01, 0.01, 1)
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df = multi_model( test_df, train_df, 0,
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "mse", "red", funct_name)
        H_df = multi_model( test_df, train_df, 0, 
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "friedman_mse", "green", funct_name)
        H_df = multi_model( test_df, train_df, 0, 
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "mae", "blue", funct_name)
        # ---------- PART B ------------ #
        print
        print("#---------------------------------------#")
        print "GBRT ANALYSIS NUMBER 1b"
        print "Analysis for Strain Hardening Index"
        loss         = ['ls', 'lad', 'huber']
        n_estimators = np.linspace(201, 201, 1)
        gamma        = np.linspace(0.01, 0.01, 1)
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['n']
        y_known = ['n']
        n_df = multi_model( test_df, train_df, 1,
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "mse", "red", funct_name)
        n_df = multi_model( test_df, train_df, 1,
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "friedman_mse", "green", funct_name)
        n_df = multi_model( test_df, train_df, 1,
                            loss, n_estimators, gamma, 4,
                            X, X_known, y, y_known,
                            "mae", "blue", funct_name)

    
    if 'ONEGAMMA_ANALYSIS' in analysis_call:
        # ------------------------ #
        #  GBRT ANALYSIS NUMBER 2  #
        # ------------------------ #
        from GBRT_Functions import singlegamma_model
        # ---------- PART A ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(1,    399,   200)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df = singlegamma_model(   test_df, train_df, 0,
                                    loss, n_estimators, gamma, 4,
                                    X, X_known, y, y_known,
                                    "mse")
        # ---------- PART B ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(1,    399,   200)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['n']
        y_known = ['n']
        n_df = singlegamma_model(   test_df, train_df, 1,
                                    loss, n_estimators, gamma, 4,
                                    X, X_known, y, y_known,
                                    "mse")


    if 'HN_INCLUSION_ANALYSIS' in analysis_call:
        from GBRT_Functions import opt_pred_plot
        # ---------- PART A ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(401,    401,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 9, "T", X, X_known, y, y_known, "friedman_mse")

        print H_df

        '''
        loss         = ['lad']
        n_estimators = np.linspace(201,    201,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['n']
        y_known = ['n']
        n_df, mae, mse = opt_pred_plot(test_df, train_df, 1, loss, n_estimators, gamma, 4, "T", X, X_known, y, y_known, "mse")

        # ---------- PART B ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(201,    201,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 4, "T", X, X_known, y, y_known, "mse")
        
        loss         = ['lad']
        n_estimators = np.linspace(201,    201,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y       = ['n']
        y_known = ['n']
        n_df, mae, mse = opt_pred_plot(test_df, train_df, 1, loss, n_estimators, gamma, 4, "T", X, X_known, y, y_known, "mse")
        '''
        # ---------- PART C ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(401,    401,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 9, "F", X, X_known, y, y_known, "friedman_mse")
        test_df['H_pred'] = H_df['y_pred']
        #H_df = GRBT_method.opt_pred_plot(train_df, test_df, 1, loss, n_estimators, gamma, 4, "F", X, X_known, y, y_known, "mse")
        #train_df['H_pred'] = H_df['y_pred']
        loss         = ['huber']
        n_estimators = np.linspace(77,    77,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred']
        y       = ['n']
        y_known = ['n']
        n_df, mae, mse = opt_pred_plot(test_df, train_df, 1, loss, n_estimators, gamma, 3, "T", X, X_known, y, y_known, "mae")
        
        print n_df
        
        '''
        loss         = ['lad']
        n_estimators = np.linspace(201,    201,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['n']
        y_known = ['n']
        n_df, mae, mse = opt_pred_plot(test_df, train_df, 1, loss, n_estimators, gamma, 4, "F", X, X_known, y, y_known, "mse")
        test_df['n_pred'] = n_df['y_pred']
        #n_df = GRBT_method.opt_pred_plot(train_df, test_df, 1, loss, n_estimators, gamma, 4, "F", X, X_known, y, y_known, "mse")
        #train_df['n_pred'] = n_df['y_pred']
        loss         = ['lad']
        n_estimators = np.linspace(201,    201,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 4, "T", X, X_known, y, y_known, "mse")
        '''


    if 'PREDICTION_BOOSTING' in analysis_call:
        # ---------- PART D ------------ #
        b_iters = 7
        loss         = ['lad']
        gamma        = np.linspace(0.01, 0.01, 1 )
        mx_dp        = 4
        H_est = 65
        n_est = 71
        crit = "mse"
        
        # ---------------------------------- Methodology 1 ---------------------------------- #
        
        ii = 0
        f, ((ax0, ax1, ax2, ax3, ax4),
            (bx0, bx1, bx2, bx3, bx4)) = plt.subplots(2, 5, figsize=(12, 6))
        
        # iteration 0:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred0'] = df['y_pred']
        ax0.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax0.scatter(test_df['H'], test_df['H_pred0'], color='blue')
        ax0.set_xlabel('H [MPa] : Testing Set')
        ax0.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 0\nTest vs. Prediction Set\nH_pred0'
        ax0.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax0.add_artist(at1)
        ii += 1
        
        # iteration 1:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred1'] = df['y_pred']
        bx0.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx0.scatter(test_df['n'], test_df['n_pred1'], color='blue')
        bx0.set_xlabel('n : Testing Set')
        bx0.set_ylabel('n : Prediction Set')
        title = 'Iteration 1\nTest vs. Prediction Set\nn_pred1'
        bx0.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx0.add_artist(at1)
        ii += 1
        
        # iteration 2:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred2'] = df['y_pred']
        ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax1.scatter(test_df['H'], test_df['H_pred2'], color='green')
        ax1.set_xlabel('H [MPa] : Testing Set')
        ax1.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 2\nTest vs. Prediction Set\nH_pred2'
        ax1.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at1)
        ii += 1
        
        # iteration 3:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred3'] = df['y_pred']
        bx1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx1.scatter(test_df['n'], test_df['n_pred3'], color='green')
        bx1.set_xlabel('n : Testing Set')
        bx1.set_ylabel('n : Prediction Set')
        title = 'Iteration 3\nTest vs. Prediction Set\nn_pred3'
        bx1.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx1.add_artist(at1)
        ii += 1
        
        # iteration 4:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred4'] = df['y_pred']
        ax2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax2.scatter(test_df['H'], test_df['H_pred4'], color='red')
        ax2.set_xlabel('H [MPa] : Testing Set')
        ax2.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 4\nTest vs. Prediction Set\nH_pred4'
        ax2.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at1)
        ii += 1
        
        # iteration 5:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred4']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred5'] = df['y_pred']
        bx2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx2.scatter(test_df['n'], test_df['n_pred5'], color='red')
        bx2.set_xlabel('n : Testing Set')
        bx2.set_ylabel('n : Prediction Set')
        title = 'Iteration 5\nTest vs. Prediction Set\nn_pred5'
        bx2.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx2.add_artist(at1)
        ii += 1
        
        # iteration 6:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred5']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred6'] = df['y_pred']
        ax3.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax3.scatter(test_df['H'], test_df['H_pred6'], color='purple')
        ax3.set_xlabel('H [MPa] : Testing Set')
        ax3.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 6\nTest vs. Prediction Set\nH_pred6'
        ax3.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax3.add_artist(at1)
        ii += 1
        
        # iteration 7:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred6']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred7'] = df['y_pred']
        bx3.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx3.scatter(test_df['n'], test_df['n_pred7'], color='purple')
        bx3.set_xlabel('n : Testing Set')
        bx3.set_ylabel('n : Prediction Set')
        title = 'Iteration 7\nTest vs. Prediction Set\nn_pred7'
        bx3.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx3.add_artist(at1)
        ii += 1
        
        # iteration 8:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred7']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred8'] = df['y_pred']
        ax4.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax4.scatter(test_df['H'], test_df['H_pred8'], color='orange')
        ax4.set_xlabel('H [MPa] : Testing Set')
        ax4.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 8\nTest vs. Prediction Set\nH_pred8'
        ax4.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at1)
        ii += 1
        
        # iteration 9:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred8']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred9'] = df['y_pred']
        bx4.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx4.scatter(test_df['n'], test_df['n_pred9'], color='orange')
        bx4.set_xlabel('n : Testing Set')
        bx4.set_ylabel('n : Prediction Set')
        title = 'Iteration 9\nTest vs. Prediction Set\nn_pred9'
        bx4.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx4.add_artist(at1)
        ii += 1
        
        plt.savefig(r"C:\Users\Darren Promer\OneDrive - Western Michigan University\Thesis_Report\ANN_rpt\methodology1.png", dpi=500)
        plt.show()
        
        # ---------------------------------- Methodology 2 ---------------------------------- #
        
        ii = 0
        f, ((ax0, ax1, ax2, ax3, ax4),
            (bx0, bx1, bx2, bx3, bx4)) = plt.subplots(2, 5, figsize=(12, 6))
        
        # iteration 0:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred0'] = df['y_pred']
        ax0.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        ax0.scatter(test_df['n'], test_df['n_pred0'], color='blue')
        ax0.set_xlabel('n : Testing Set')
        ax0.set_ylabel('n : Prediction Set')
        title = 'Iteration 0\nTest vs. Prediction Set\nn_pred0'
        ax0.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax0.add_artist(at1)
        ii += 1
        
        # iteration 1:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred1'] = df['y_pred']
        bx0.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        bx0.scatter(test_df['H'], test_df['H_pred1'], color='blue')
        bx0.set_xlabel('H [MPa] : Testing Set')
        bx0.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 1\nTest vs. Prediction Set\nH_pred1'
        bx0.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx0.add_artist(at1)
        ii += 1
        
        # iteration 2:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred2'] = df['y_pred']
        ax1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        ax1.scatter(test_df['n'], test_df['n_pred2'], color='green')
        ax1.set_xlabel('n : Testing Set')
        ax1.set_ylabel('n : Prediction Set')
        title = 'Iteration 2\nTest vs. Prediction Set\nn_pred2'
        ax1.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at1)
        ii += 1
        
        # iteration 3:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred3'] = df['y_pred']
        bx1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        bx1.scatter(test_df['H'], test_df['H_pred3'], color='green')
        bx1.set_xlabel('H [MPa] : Testing Set')
        bx1.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 3\nTest vs. Prediction Set\nH_pred3'
        bx1.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx1.add_artist(at1)
        ii += 1
        
        # iteration 4:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred4'] = df['y_pred']
        ax2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        ax2.scatter(test_df['n'], test_df['n_pred4'], color='red')
        ax2.set_xlabel('n : Testing Set')
        ax2.set_ylabel('n : Prediction Set')
        title = 'Iteration 4\nTest vs. Prediction Set\nn_pred4'
        ax2.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at1)
        ii += 1
        
        # iteration 5:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred4']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred5'] = df['y_pred']
        bx2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        bx2.scatter(test_df['H'], test_df['H_pred5'], color='red')
        bx2.set_xlabel('H [MPa] : Testing Set')
        bx2.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 5\nTest vs. Prediction Set\nH_pred5'
        bx2.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx2.add_artist(at1)
        ii += 1
        
        # iteration 6:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred5']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred6'] = df['y_pred']
        ax3.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        ax3.scatter(test_df['n'], test_df['n_pred6'], color='purple')
        ax3.set_xlabel('n : Testing Set')
        ax3.set_ylabel('n : Prediction Set')
        title = 'Iteration 6\nTest vs. Prediction Set\nn_pred6'
        ax3.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax3.add_artist(at1)
        ii += 1
        
        # iteration 7:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred6']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred7'] = df['y_pred']
        bx3.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        bx3.scatter(test_df['H'], test_df['H_pred7'], color='purple')
        bx3.set_xlabel('H [MPa] : Testing Set')
        bx3.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 7\nTest vs. Prediction Set\nH_pred7'
        bx3.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx3.add_artist(at1)
        ii += 1
        
        # iteration 8:
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred7']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred8'] = df['y_pred']
        ax4.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        ax4.scatter(test_df['n'], test_df['n_pred8'], color='orange')
        ax4.set_xlabel('n : Testing Set')
        ax4.set_ylabel('n : Prediction Set')
        title = 'Iteration 8\nTest vs. Prediction Set\nn_pred8'
        ax4.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at1)
        ii += 1
        
        # iteration 9:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred8']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred9'] = df['y_pred']
        bx4.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        bx4.scatter(test_df['H'], test_df['H_pred9'], color='orange')
        bx4.set_xlabel('H [MPa] : Testing Set')
        bx4.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 9\nTest vs. Prediction Set\nH_pred9'
        bx4.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx4.add_artist(at1)
        ii += 1
        
        plt.savefig(r"C:\Users\Darren Promer\OneDrive - Western Michigan University\Thesis_Report\ANN_rpt\methodology2.png", dpi=500)
        plt.show()
        
        # ---------------------------------- Methodology 3 ---------------------------------- #
        
        f, ((ax0, ax1, ax2, ax3, ax4),
            (bx0, bx1, bx2, bx3, bx4)) = plt.subplots(2, 5, figsize=(12, 6))
        
        # iteration 0:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred0'] = df['y_pred']
        ax0.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax0.scatter(test_df['H'], test_df['H_pred0'], color='blue')
        ax0.set_xlabel('H [MPa] : Testing Set')
        ax0.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 0\nTest vs. Prediction Set\nH_pred0'
        ax0.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax0.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred0'] = df['y_pred']
        bx0.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx0.scatter(test_df['n'], test_df['n_pred0'], color='blue')
        bx0.set_xlabel('n : Testing Set')
        bx0.set_ylabel('n : Prediction Set')
        title = 'n_pred0'
        bx0.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx0.add_artist(at1)
        ii += 1
        
        # iteration 1:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred1'] = df['y_pred']
        ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax1.scatter(test_df['H'], test_df['H_pred1'], color='green')
        ax1.set_xlabel('H [MPa] : Testing Set')
        ax1.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 1\nTest vs. Prediction Set\nH_pred1'
        ax1.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred1'] = df['y_pred']
        bx1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx1.scatter(test_df['n'], test_df['n_pred1'], color='green')
        bx1.set_xlabel('n : Testing Set')
        bx1.set_ylabel('n : Prediction Set')
        title = 'n_pred1'
        bx1.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx1.add_artist(at1)
        ii += 1
        
        # iteration 2:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred2'] = df['y_pred']
        ax2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax2.scatter(test_df['H'], test_df['H_pred2'], color='red')
        ax2.set_xlabel('H [MPa] : Testing Set')
        ax2.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 2\nTest vs. Prediction Set\nH_pred2'
        ax2.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred2'] = df['y_pred']
        bx2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx2.scatter(test_df['n'], test_df['n_pred2'], color='red')
        bx2.set_xlabel('n : Testing Set')
        bx2.set_ylabel('n : Prediction Set')
        title = 'n_pred2'
        bx2.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx2.add_artist(at1)
        ii += 1
        
        # iteration 3:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred3'] = df['y_pred']
        ax3.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax3.scatter(test_df['H'], test_df['H_pred3'], color='purple')
        ax3.set_xlabel('H [MPa] : Testing Set')
        ax3.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 3\nTest vs. Prediction Set\nH_pred3'
        ax3.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax3.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred3'] = df['y_pred']
        bx3.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx3.scatter(test_df['n'], test_df['n_pred3'], color='purple')
        bx3.set_xlabel('n : Testing Set')
        bx3.set_ylabel('n : Prediction Set')
        title = 'n_pred3'
        bx3.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx3.add_artist(at1)
        ii += 1
        
        # iteration 4:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred4'] = df['y_pred']
        ax4.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax4.scatter(test_df['H'], test_df['H_pred4'], color='orange')
        ax4.set_xlabel('H [MPa] : Testing Set')
        ax4.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 4\nTest vs. Prediction Set\nH_pred4'
        ax4.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred4'] = df['y_pred']
        bx4.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx4.scatter(test_df['n'], test_df['n_pred4'], color='orange')
        bx4.set_xlabel('n : Testing Set')
        bx4.set_ylabel('n : Prediction Set')
        title = 'n_pred4'
        bx4.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx4.add_artist(at1)
        ii += 1
        
        plt.savefig(r"C:\Users\Darren Promer\OneDrive - Western Michigan University\Thesis_Report\ANN_rpt\methodology3.png", dpi=500)
        plt.show()
        
        # ---------------------------------- Methodology 4 ---------------------------------- #
        
        f, ((ax0, ax1, ax2, ax3, ax4),
            (bx0, bx1, bx2, bx3, bx4)) = plt.subplots(2, 5, figsize=(12, 6))
        
        # iteration 0:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred0'] = df['y_pred']
        ax0.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax0.scatter(test_df['H'], test_df['H_pred0'], color='blue')
        ax0.set_xlabel('H [MPa] : Testing Set')
        ax0.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 0\nTest vs. Prediction Set\nH_pred0'
        ax0.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax0.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred0'] = df['y_pred']
        bx0.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx0.scatter(test_df['n'], test_df['n_pred0'], color='blue')
        bx0.set_xlabel('n : Testing Set')
        bx0.set_ylabel('n : Prediction Set')
        title = 'n_pred0'
        bx0.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx0.add_artist(at1)
        ii += 1
        
        # iteration 1:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n', 'H']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred0', 'H_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred1'] = df['y_pred']
        ax1.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax1.scatter(test_df['H'], test_df['H_pred1'], color='green')
        ax1.set_xlabel('H [MPa] : Testing Set')
        ax1.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 1\nTest vs. Prediction Set\nH_pred1'
        ax1.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H', 'n']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred0', 'n_pred0']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred1'] = df['y_pred']
        bx1.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx1.scatter(test_df['n'], test_df['n_pred1'], color='green')
        bx1.set_xlabel('n : Testing Set')
        bx1.set_ylabel('n : Prediction Set')
        title = 'n_pred1'
        bx1.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx1.add_artist(at1)
        ii += 1
        
        # iteration 2:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n', 'H']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred1', 'H_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred2'] = df['y_pred']
        ax2.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax2.scatter(test_df['H'], test_df['H_pred2'], color='red')
        ax2.set_xlabel('H [MPa] : Testing Set')
        ax2.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 2\nTest vs. Prediction Set\nH_pred2'
        ax2.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H', 'n']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred1', 'n_pred1']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred2'] = df['y_pred']
        bx2.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx2.scatter(test_df['n'], test_df['n_pred2'], color='red')
        bx2.set_xlabel('n : Testing Set')
        bx2.set_ylabel('n : Prediction Set')
        title = 'n_pred2'
        bx2.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx2.add_artist(at1)
        ii += 1
        
        # iteration 3:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n', 'H']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred2', 'H_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred3'] = df['y_pred']
        ax3.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax3.scatter(test_df['H'], test_df['H_pred3'], color='purple')
        ax3.set_xlabel('H [MPa] : Testing Set')
        ax3.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 3\nTest vs. Prediction Set\nH_pred3'
        ax3.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax3.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H', 'n']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred2', 'n_pred2']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred3'] = df['y_pred']
        bx3.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx3.scatter(test_df['n'], test_df['n_pred3'], color='purple')
        bx3.set_xlabel('n : Testing Set')
        bx3.set_ylabel('n : Prediction Set')
        title = 'n_pred3'
        bx3.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx3.add_artist(at1)
        ii += 1
        
        # iteration 4:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n', 'H']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'n_pred3', 'H_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 0,
                                        loss,
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp,
                                        crit)
        test_df['H_pred4'] = df['y_pred']
        ax4.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax4.scatter(test_df['H'], test_df['H_pred4'], color='orange')
        ax4.set_xlabel('H [MPa] : Testing Set')
        ax4.set_ylabel('H [MPa] : Prediction Set')
        title = 'Iteration 4\nTest vs. Prediction Set\nH_pred4'
        ax4.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H', 'n']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred3', 'n_pred3']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = GRBT_method.opt_pred_plot( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred4'] = df['y_pred']
        bx4.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx4.scatter(test_df['n'], test_df['n_pred4'], color='orange')
        bx4.set_xlabel('n : Testing Set')
        bx4.set_ylabel('n : Prediction Set')
        title = 'n_pred4'
        bx4.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx4.add_artist(at1)
        ii += 1
        
        plt.savefig(r"C:\Users\Darren Promer\OneDrive - Western Michigan University\Thesis_Report\ANN_rpt\methodology4.png", dpi=500)
        plt.show()
        

    if 'XGBOOST' in analysis_call:
        from GBRT_Functions import opt_pred_plot_XGB
        from matplotlib.offsetbox import AnchoredText
        loss         = ['lad']
        gamma        = np.linspace(0.01, 0.01, 1 )
        mx_dp        = 4
        H_est = 65
        n_est = 71
        crit = "mse"

        ii = 0
        f, ((ax0),
            (bx0)) = plt.subplots(2, 1, figsize=(4.5, 7))
        
        # iteration 0:
        n_estimators = np.linspace(H_est,    H_est,   1)
        y_inp       = ['H']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['H']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = opt_pred_plot_XGB(   test_df, train_df, 0,
                                            loss,
                                            n_estimators,
                                            gamma, mx_dp, "F",
                                            X_inp, X_known_inp, y_inp, y_known_inp,
                                            crit)
        test_df['H_pred0'] = df['y_pred']
        ax0.plot(np.linspace(680,760,5), np.linspace(680,760,5), 'k--', linewidth=1)
        ax0.scatter(test_df['H'], test_df['H_pred0'], color='blue')
        ax0.set_xlabel('H [MPa] : Testing Set')
        ax0.set_ylabel('H [MPa] : Prediction Set')
        title = 'XGBoost Implementation\n\nH_pred0'
        ax0.set_title(title)
        t1 = "MSE: "+mse+"\nMAE: "+mae
        at1 = AnchoredText(t1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax0.add_artist(at1)
        
        n_estimators = np.linspace(n_est,    n_est,   1)
        y_inp       = ['n']
        X_inp       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y_known_inp = ['n']
        X_known_inp = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        df_row = 'pred_'+str(int(ii))
        df, mae, mse = opt_pred_plot_XGB( test_df, train_df, 1,
                                        loss, 
                                        n_estimators,
                                        gamma, mx_dp, "F",
                                        X_inp, X_known_inp, y_inp, y_known_inp, 
                                        crit)
        test_df['n_pred0'] = df['y_pred']
        bx0.plot(np.linspace(0.175,0.195,5), np.linspace(0.175,0.196,5), 'k--', linewidth=1)
        bx0.scatter(test_df['n'], test_df['n_pred0'], color='blue')
        bx0.set_xlabel('n : Testing Set')
        bx0.set_ylabel('n : Prediction Set')
        title = 'n_pred0'
        bx0.set_title(title)
        lt1 = 'MSE'+r'$\bullet10^{6}$'+': '+mse+"\nMAE: "+mae
        at1 = AnchoredText(lt1, frameon=True, loc='lower right')
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        bx0.add_artist(at1)
        ii += 1
        
        plt.savefig(r"C:\Users\Darren Promer\OneDrive - Western Michigan University\Thesis_Report\ANN_rpt\xgboost_methodology.png", dpi=500)
        plt.show()


    if 'FEATURE_REDUCTION' in analysis_call:
        from GBRT_Functions import feature_reduction_analysis
        import os

        path = '/'+funct_name+'/FEATURE_REDUCTION'
        if not os.path.exists(path):
            os.makedirs(path)

        error_metric_vector = ['mse']
        for err_metric in error_metric_vector:
            loss         = 'lad'
            gamma        = 0.01

            # Feature Reduction (Strength Coefficient):
            n_estimators = 281
            X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
            X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
            y       = ['H']
            y_known = ['H']

            HFR = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            Herr = np.zeros(11)

            print
            print "FEATURE REDUCTION ANALYSIS: STRENGTH COEFFICIENT"
            print "Error Metric         : "+err_metric
            ii = 0
            max_iter = len(X_known)-1
            while ii <= max_iter:
                # Execution of current iteration training...
                H_df, err, rem_f = feature_reduction_analysis(  test_df, train_df, 0, 
                                                                loss, n_estimators, gamma, 4,
                                                                "F",
                                                                X, X_known, y, y_known, "mse", err_metric)
                Herr[ii] = err

                # Printout of results to the cmd prompt...
                print
                print "# Features Removed   : " +str(ii)
                print "Feature Matrix       : "+str(X_known)
                print "err_value            : "+str(err)
                print "Feature to Remove    : "+rem_f

                # Removal of the minimum contributing feature from the feature matricies
                X.remove(rem_f)
                X_known.remove(rem_f)

                ii += 1

            # Feature Reduction (Strain Hardening Index):
            n_estimators = 281
            HX       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
            HX_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
            Hy       = ['H']
            Hy_known = ['H']
            H_df, err, rem_f = feature_reduction_analysis( test_df, train_df, 0,
                                                                loss, n_estimators, gamma, 4,
                                                                "F",
                                                                HX, HX_known, Hy, Hy_known, "mse", err_metric)
            test_df['H_pred'] = H_df['y_pred']

            n_estimators = 69
            X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
            X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred']
            y       = ['n']
            y_known = ['n']

            nFR = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            nerr = np.zeros(12)

            print
            print "FEATURE REDUCTION ANALYSIS: STRAIN HARDENING INDEX"
            print "Error Metric         : "+err_metric
            ii = 0
            max_iter = len(X_known)-1
            while ii <= max_iter:
                # Execution of current iteration training...
                n_df, err, rem_f = feature_reduction_analysis(    test_df, train_df, 1, 
                                                                        loss, n_estimators, gamma, 4,
                                                                        "F",
                                                                        X, X_known, y, y_known, "mse", err_metric)
                nerr[ii] = err
                
                # Printout of results to the cmd prompt...
                print
                print "# Features Removed   : " +str(ii)
                print "Feature Matrix       : "+str(X_known)
                print "err_value            : "+str(err)
                print "Feature to Remove    : "+rem_f
                
                # Removal of the minimum contributing feature from the feature matricies
                if rem_f == 'H_pred':
                    X.remove('H')
                    X_known.remove('H_pred')
                else:
                    X.remove(rem_f)
                    X_known.remove(rem_f)
    
                ii += 1
            
            def feature_red_plot(Hx, Hy, nx, ny, err_metric, path):
                f, (ax1, bx1) = plt.subplots(2, 1, figsize=(4, 6))

                #strength coefficient:
                ax1.set_title(material_string+": Error Metric with Feature Reduction\nMetric: "+err_metric, loc='left', fontsize=11)
                ax1.plot(Hx, Hy, 'b-', linewidth=1)
                ax1.set_ylabel(err_metric)

                #strain hardening exponent
                bx1.plot(nx, ny, 'b-', linewidth=1)
                bx1.set_ylabel(err_metric)
                bx1.set_xlabel("Number of Features Removed")
                

                plt.savefig(path+'/FEATURE_REDUCTION__metric_'+err_metric+'.png', dpi=800)
                #plt.show()
                plt.close()

            feature_red_plot(HFR, Herr, nFR, nerr, err_metric, path)


    if '3MATERIAL_RESULTS' in analysis_call:

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

        print " "
        print "Commencing from 'analysis_call:"
        print "3MATERIAL_RESULTS"

        
        print " "
        print "Analysis of all comibations of: loss, criterion, maximum depth, estimators"
        loss         = ['ls', 'lad', 'huber']
        crit         = ['mse', 'friedman_mse', 'mae']
        max_depth    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        n_estimators = np.linspace(1, 401, 101)
        gamma        = 0.01
        
        path = '/'+funct_name
        if not os.path.exists(path):
            os.makedirs(path)
        
        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        def mean_percentage_error(y_true, y_pred): 
            return np.mean((y_true - y_pred) / y_true) * 100
        def root_mean_squared_percentage_error(y_true, y_pred):
            return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))*100

        print " "
        print "Strength Coefficient Prediction Results:"

        y  = train_df[['H']]
        y_known  = test_df[['H']]
        X  = train_df[['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']]
        X_known  = test_df[['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']]

        if os.path.isfile(path+'/'+funct_name+'_Herrors.csv'):
            error_results = pd.read_csv(path+'/'+funct_name+'_Herrors.csv')
        else:
            # Printing out a comparison of the results:
            error_results = pd.DataFrame(columns = ['loss','crit','max_depth','n_estimators','gamma','mae','mse','mape','mpe','rmspe'])

            idx = 0
            for ii in loss:
                for jj in crit:
                    for kk in max_depth:
                        for hh in n_estimators:
                            model1 = GradientBoostingRegressor(
                                max_depth = int(kk),
                                n_estimators = int(hh),
                                learning_rate = gamma,
                                loss = str(ii),
                                random_state = 1,
                                criterion = str(jj))

                            #Fitting of model to training data
                            model1.fit(X, np.ravel(y, order='C'))

                            # Predicting the result for the KNOWN VAULES!
                            # Should be expecting:
                            # y_pred = y_known
                            y_pred = model1.predict(X_known)

                            # ---------------------------------------------- #
                            # ERROR ANALYSIS:
                            # ---------------------------------------------- #
                            y_known_np = y_known.values
                            mae = mean_absolute_error(y_known, y_pred)
                            mse =   mean_squared_error(y_known, y_pred)
                            mape =  mean_absolute_percentage_error(y_known_np, y_pred)
                            mpe =   mean_percentage_error(y_known_np, y_pred)
                            rmspe = root_mean_squared_percentage_error(y_known_np, y_pred)

                            error_results.loc[idx, 'loss']  = str(ii)
                            error_results.loc[idx, 'crit']  = str(jj)
                            error_results.loc[idx, 'max_depth'] = int(kk)
                            error_results.loc[idx, 'n_estimators'] = int(hh)
                            error_results.loc[idx, 'gamma'] = gamma
                            error_results.loc[idx, 'mae'] = float(mae)
                            error_results.loc[idx, 'mse'] = float(mse)
                            error_results.loc[idx, 'mape'] = float(mape)
                            error_results.loc[idx, 'mpe'] = float(mpe)
                            error_results.loc[idx, 'rmspe'] = float(rmspe)

                            if int(hh) in [401, 301, 201, 101, 1]:
                                print error_results.loc[[idx]]

                            idx += 1

            error_results.to_csv(path+'/'+funct_name+'_Herrors.csv')

        err_list = ['mse']
        
        for err in err_list:
            error_results[err] = pd.to_numeric(error_results[err])
            minid = error_results[err].abs().idxmin()
            
            print("OPTIMUM VALUES FOR "+str(err)+":")
            print("Location:")
            print(minid)
            print("df row: ")
            print(error_results.loc[minid, :])
            print " "
        
        print " "
        print "Strain Hardening Index Prediction Results:"
        
        results_df = pd.DataFrame(columns = ['y_known', 'y_pred'])
        results_df.y_known = test_df['H'].tolist()

        model1 = GradientBoostingRegressor(
                         max_depth = int(error_results.loc[minid, 'max_depth']),
                         n_estimators = int(error_results.loc[minid, 'n_estimators']),
                         learning_rate = gamma,
                         loss = str(error_results.loc[minid, 'loss']),
                         random_state = 1,
                         criterion = str(error_results.loc[minid, 'crit']))
        model1.fit(X, np.ravel(y, order='C'))
        y_pred = model1.predict(X_known)
        results_df.loc[:, 'y_pred'] = y_pred
        test_df['H_pred'] = results_df['y_pred']

        y  = train_df[['n']]
        y_known  = test_df[['n']]
        X  = train_df[['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']]
        X_known  = test_df[['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred']]

        if os.path.isfile(path+'/'+funct_name+'_nerrors.csv'):
            error_results = pd.read_csv(path+'/'+funct_name+'_nerrors.csv')
        else:
            # Printing out a comparison of the results:
            error_results = pd.DataFrame(columns = ['loss','crit','max_depth','n_estimators','gamma','mae','mse','mape','mpe','rmspe'])

            idx = 0
            for ii in loss:
                for jj in crit:
                    for kk in max_depth:
                        for hh in n_estimators:
                            model1 = GradientBoostingRegressor(
                                max_depth = int(kk),
                                n_estimators = int(hh),
                                learning_rate = gamma,
                                loss = str(ii),
                                random_state = 1,
                                criterion = str(jj))

                            #Fitting of model to training data
                            model1.fit(X, np.ravel(y, order='C'))

                            # Predicting the result for the KNOWN VAULES!
                            # Should be expecting:
                            # y_pred = y_known
                            y_pred = model1.predict(X_known)

                            # ---------------------------------------------- #
                            # ERROR ANALYSIS:
                            # ---------------------------------------------- #
                            y_known_np = y_known.values
                            mae = mean_absolute_error(y_known, y_pred)
                            mse =   mean_squared_error(y_known, y_pred)
                            mape =  mean_absolute_percentage_error(y_known_np, y_pred)
                            mpe =   mean_percentage_error(y_known_np, y_pred)
                            rmspe = root_mean_squared_percentage_error(y_known_np, y_pred)

                            error_results.loc[idx, 'loss']  = str(ii)
                            error_results.loc[idx, 'crit']  = str(jj)
                            error_results.loc[idx, 'max_depth'] = int(kk)
                            error_results.loc[idx, 'n_estimators'] = int(hh)
                            error_results.loc[idx, 'gamma'] = gamma
                            error_results.loc[idx, 'mae'] = float(mae)
                            error_results.loc[idx, 'mse'] = float(mse)
                            error_results.loc[idx, 'mape'] = float(mape)
                            error_results.loc[idx, 'mpe'] = float(mpe)
                            error_results.loc[idx, 'rmspe'] = float(rmspe)

                            if int(hh) in [401, 301, 201, 101, 1]:
                                print error_results.loc[[idx]]

                            idx += 1

            error_results.to_csv(path+'/'+funct_name+'_nerrors.csv')

        err_list = ['mse']
        
        for err in err_list:
            error_results[err] = pd.to_numeric(error_results[err])
            minid = error_results[err].abs().idxmin()
            
            print(" ")
            print("OPTIMUM VALUES FOR "+str(err)+":")
            print("Location:")
            print(minid)
            print("df row: ")
            print(error_results.loc[minid, :])
            print " "
        
        """
        from GBRT_Functions import opt_pred_plot

        # ---------- PART A ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(177,    177,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 5, "T", X, X_known, y, y_known, "mae")

        # ---------- PART A ------------ #
        loss         = ['lad']
        n_estimators = np.linspace(177,    177,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C']
        y       = ['H']
        y_known = ['H']
        H_df, mae, mse = opt_pred_plot(test_df, train_df, 0, loss, n_estimators, gamma, 5, "F", X, X_known, y, y_known, "mae")
        test_df['H_pred'] = H_df['y_pred']
        #H_df = GRBT_method.opt_pred_plot(train_df, test_df, 1, loss, n_estimators, gamma, 4, "F", X, X_known, y, y_known, "mse")
        #train_df['H_pred'] = H_df['y_pred']
        loss         = ['lad']
        n_estimators = np.linspace(25,    25,   1)
        gamma        = np.linspace(0.01, 0.01, 1 )
        X       = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H']
        X_known = ['Pmax', 'hf1', 'hf2', 'hf3', 'hf4', 'hf5', 'hf6', 'hf7', 'hf8', 'hf9', 'C', 'H_pred']
        y       = ['n']
        y_known = ['n']
        n_df, mae, mse = opt_pred_plot(test_df, train_df, 1, loss, n_estimators, gamma, 10, "T", X, X_known, y, y_known, "mae")
        """