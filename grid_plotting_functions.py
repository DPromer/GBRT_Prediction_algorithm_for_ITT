"""
Plotting module for all of the 3 Materials.

Basic input plots:
1. Histograms of Test Distribution
2. Scatter of the Training Grid
3. Scatter plot and histogram plots combined
"""

def grid_study_plotting(    sensitivity_study,
                            training_gaussian_pairings,
                            E, nu, mat,
                            funct_name, color, grid_length):

    size = 10
    width = 0.5

    # Plotting Modules to Import:
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    from matplotlib.offsetbox import AnchoredText
    import os.path

    # matplotlib formatting:
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Dejavu Serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 8
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

    path = '/'+funct_name
    if not os.path.exists(path):
        os.makedirs(path)

    grid_ = sensitivity_study
    rand_ = training_gaussian_pairings[:,:2].astype(float)

    Hmax = grid_[:,0].max()
    Hmin = grid_[:,0].min()
    nmax = grid_[:,1].max()
    nmin = grid_[:,1].min()

    H_mu = (Hmax+Hmin)/2
    H_sigma = (Hmax-H_mu)/3
    n_mu = (nmax+nmin)/2
    n_sigma = (nmax-n_mu)/3

    H_bins = [ (H_sigma*-3)+H_mu,
            (H_sigma*-2)+H_mu,
            (H_sigma*-1)+H_mu,
            (H_sigma* 0)+H_mu,
            (H_sigma* 1)+H_mu,
            (H_sigma* 2)+H_mu,
            (H_sigma* 3)+H_mu ]
    n_bins = [ (n_sigma*-3)+n_mu,
            (n_sigma*-2)+n_mu,
            (n_sigma*-1)+n_mu,
            (n_sigma* 0)+n_mu,
            (n_sigma* 1)+n_mu,
            (n_sigma* 2)+n_mu,
            (n_sigma* 3)+n_mu]

    # Histograms
    fig, axs = plt.subplots(1, 2, figsize=(6.5,2.75))
    
    axs[0].hist(    rand_[:,0],
                    bins=H_bins,
                    histtype='bar',
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1,
                    alpha=1.0)
    axs[0].set_ylabel('frequency')
    axs[0].set_xlabel(r'$H  [MPa] \sim \mathbb{N}(\mu_{H}, \sigma_{H}^{2})$')
    axs[0].set_ylim(0, 40)
    axs[0].set_xlim(H_mu-(4*H_sigma),H_mu+(4*H_sigma)) 
    axs[0].set_title(mat+'\n'+'Strength Coefficient TEST Distribution', loc='left', fontsize=10)
    lt0 = r'$\mu_{H} = $'+str(float(round(H_mu, 3)))+' MPa\n'+r'$\sigma_{H} = $'+str(float(round(H_sigma, 3)))+' MPa'
    at0 = AnchoredText(lt0, frameon=False, loc='upper right')
    at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[0].add_artist(at0)

    axs[1].hist(    rand_[:,1],
                    bins=n_bins,
                    histtype='bar',
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1,
                    alpha=1.0)
    axs[1].set_yticklabels([])
    axs[1].set_xlabel('n')
    axs[1].set_xlim(n_mu-(4*n_sigma), n_mu+(4*n_sigma)) 
    axs[1].set_xlabel(r'$n \sim \mathbb{N}(\mu_{n}, \sigma_{n}^{2})$')
    axs[1].set_ylim(0, 40)
    axs[1].set_title('Strain Hardening Index TEST Distribution', loc='left', fontsize=10)
    lt1 = r'$\mu_{n} = $'+str(float(round(n_mu, 6)))+'\n'+r'$\sigma_{n} = $'+str(float(round(n_sigma, 6)))
    at1 = AnchoredText(lt1, frameon=False, loc='upper right')
    at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[1].add_artist(at1)
    plt.savefig(path+'/TEST_Hists.png', dpi=800)
    plt.close()

    x  = rand_[:,1]
    y  = rand_[:,0]
    x2 = grid_[:,1]
    y2 = grid_[:,0]

    # grid.png
    fig = plt.figure(figsize=(3.5, 3.5))
    ax_scatter = plt.axes()
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_scatter.scatter(x2, y2, s=size,facecolors='none', edgecolors='k', linewidth=width)
    ax_scatter.set_xlabel("Strain Hardening Index, n")
    ax_scatter.set_ylabel("Strength Coefficient, H, [MPa]")
    #ax_scatter.legend(loc='lower left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0), prop={'size':11})
    ax_scatter.set_xlim((nmin-0.35*n_mu, nmax+0.35*n_mu))
    ax_scatter.set_ylim((Hmin-0.5*H_mu, Hmax+0.2*H_mu))
    title = 'Material : '+mat+'\n'+str(grid_length)+'x'+str(grid_length)+' training domain'
    ax_scatter.set_title(title, loc='left', fontsize=10)
    
    lt1 = r'$H = \{r:r=$'+str(float(round(Hmin, 1)))+r'$+$'+str(float(round((Hmax-Hmin)/(grid_length-1), 1)))+r'$ \times i,  i \in \{0,1, \ldots, $'+str(grid_length-1)+r'$ \} \}$'+'\n'+r'$n = \{r:r=$'+str(float(round(nmin, 4)))+r'$+$'+str(float(round((nmax-nmin)/(grid_length-1), 4)))+r'$ \times i,  i \in \{0,1, \ldots, $'+str(grid_length-1)+r'$ \} \}$'
    at1 = AnchoredText(lt1, frameon=False, loc='lower center')
    at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax_scatter.add_artist(at1)
    plt.savefig(path+'/TRAIN_grid.png', dpi=800)
    plt.close()

    # grid_scat.png
    left, width = 0.1, 0.65
    bottom, height = 0.0825, 0.65
    spacing = 0.0025
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(figsize=(6.5,6.5))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_scatter.scatter(x2, y2, facecolors='none', s=size, linewidth=width, edgecolors='k', label='Train Set (i='+str(len(grid_[:,0]))+')')
    ax_scatter.scatter(x, y, label='Test Set (i='+str(len(rand_[:,0]))+')', marker='.', color=color)
    
    ax_scatter.set_xlabel("n")
    ax_scatter.set_ylabel("H, [MPa]")
    ax_scatter.legend(loc='lower left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0), prop={'size':11})
    ax_scatter.set_xlim((nmin-0.5*n_mu, nmax+0.5*n_mu))
    ax_scatter.set_ylim((Hmin-0.5*H_mu, Hmax+0.5*H_mu))
    ax_histx.hist(x, bins=n_bins, facecolor=color, edgecolor='black', linewidth=1, alpha=1.0)
    ax_histy.hist(y, bins=H_bins, orientation='horizontal', facecolor=color, edgecolor='black', linewidth=1, alpha=1.0)
    ax_histx.set_title(mat+'\nTEST and TRAIN Datasets', loc='left', fontsize=10)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histx.set_ylim((0,40))
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.set_xlim((0,40))
    ax_histx.set_xlabel(r'$H \sim \mathbb{N}(\mu_{H}, \sigma_{H}^{2})$')
    ax_histy.set_ylabel(r'$n \sim \mathbb{N}(\mu_{n}, \sigma_{n}^{2})$')
    plt.savefig(path+'/TESTandTRAIN.png', dpi=800)
    plt.close()