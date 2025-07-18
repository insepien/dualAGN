import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scp

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{mathptmx}",  # Times Roman
    "hatch.linewidth": 3.0,
})
sns.set_context("paper",font_scale=1.75)

if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="/home/insepien/research-data/pop-result/lit_rev_frac/", help="input directory")
    parser.add_argument("--inFile", type=str, default="dualfrac_rev2.csv", help="input dual fraction csv")
    parser.add_argument("--outFile", type=str, default="dualfrac.png")
    
    args = parser.parse_args()
    # load obs fracs and remove any nan fraction rows
    dfr = pd.read_csv(args.inDir+args.inFile)
    dfr_agn = dfr[~ dfr['lower dual frac'].isna()]
    dfobs = dfr_agn[~(dfr_agn['selection']=='cosmological simulation')]
    # define markers and colors
    all_markers = ["o","s","x","d",'p',"H"]
    #selmet = np.array(list(set(dfobs['selection'])))
    # set selmet manually for color purposes
    selmet = np.array(['optical','double-peak', 'LIRG', 'radio','X-ray'])
    all_colors = sns.color_palette("colorblind", len(selmet))
    ### observation fractions
    fig,ax = plt.subplots(1,2,figsize=(11,5),dpi=500,sharex=True,sharey=True)
    for i in dfobs.index.values: # plot each row, viz label by selection
        ind = np.where(selmet == dfobs.loc[i]['selection'])[0][0]
        sym = all_markers[ind]
        color = all_colors[ind]
        # fix label capitalization
        lbl = dfobs.loc[i]['selection'].capitalize() if i == dfobs[dfobs['selection']==selmet[ind]].index.values[0] else None
        lbl = "LIRG" if lbl=="Lirg" else lbl
        # plot error bars if exist
        if pd.isna(dfobs.loc[i]['lower dual frac error']):
            # only has lower bound
            if pd.isna(dfobs.loc[i]['upper dual frac']): 
                ax[0].scatter((dfobs.loc[i]['min z']+dfobs.loc[i]['max z'])/2,dfobs.loc[i]['lower dual frac'],
                        c=color,marker=sym,label=lbl)
            # has upper and lower bound
            else: 
                x = (dfobs.loc[i]['min z']+dfobs.loc[i]['max z'])/2
                ax[0].plot(x,dfobs.loc[i]['lower dual frac'],
                        c=color,marker="^")
                ax[0].plot(x,float(dfobs.loc[i]['upper dual frac']),
                        c=color,marker="v")
                ax[0].plot([x,x],[dfobs.loc[i]['lower dual frac'],float(dfobs.loc[i]['upper dual frac'])],
                        c=color,label=lbl,linestyle="--")
        else:
            ax[0].errorbar((dfobs.loc[i]['min z']+dfobs.loc[i]['max z'])/2,dfobs.loc[i]['lower dual frac'], yerr=[[np.abs(dfobs.loc[i]['lower dual frac'] - float(j))] for j in dfobs.loc[i]['lower dual frac error'].split(",")],
                    c=color,fmt=sym,label=lbl)
    # observation with large z range
    for f,sel_type in zip(['li2024.csv', 'silverman2020.csv'],["X-ray",'optical']):
        path = args.inDir + f
        df = pd.read_csv(path,header=None,names=['z',"f"])
        # set all redshift value for upper,lower,mean to 1 value
        for i in range(0,len(df),3):
            df.loc[i+1,"z"] = df.loc[i,"z"]
            df.loc[i+2,"z"] = df.loc[i,"z"]
        # now sort by fraction val
        df.sort_values(by=['z','f'],inplace=True)
        df.reset_index(inplace=True,drop=True)
        # each datapoint is every 3 rows
        x = [df.loc[i]['z'] for i in range(1,len(df),3)]
        f = np.array([df.loc[i]['f'] for i in range(1,len(df),3)])
        err_l = np.array([df.loc[i]['f'] for i in range(0,len(df),3)])
        err_u = np.array([df.loc[i]['f'] for i in range(2,len(df),3)])
        #set plot styles based on selection method
        ind = np.where(selmet == sel_type)[0][0]
        sym = all_markers[ind]
        color = all_colors[ind]
        ax[0].plot(x,f,c=color,linestyle="--")
        ax[0].fill_between(x, err_l,err_u, alpha=0.2,color=color)
    
    # magellan fraction
    ndual=2
    nsample=39
    myfrac = ndual/nsample
    err = np.array(scp.poisson.interval(0.95, ndual))/nsample
    ax[0].errorbar((0.14+0.22)/2,myfrac,yerr=[[myfrac-err[0]],[err[1]-myfrac]],
                fmt="*",markersize=14,c='k',label='Our fraction')

    ### theoretical fractions, had to split up to 2 groups because data is not uniform
    # fractions with no error bars
    for f,l,i in zip(['volonteri2022.csv','yu2011.csv'],['Volonteri+2022','Yu+2011'],range(2)):
        path = args.inDir + f
        d = pd.read_csv(path,header=None,names=['z',"f"])
        ax[1].plot(d['z'],d['f'],marker=all_markers[i],c=all_colors[i],label=l)
    # fractions with upper error saved
    for f,l,i in zip(['chen2023.csv','rosa2019.csv'],['Chen+2023','Rosas-Guevara+2019'],range(2,4)):
        path = args.inDir + f
        df = pd.read_csv(path,header=None,names=['z',"f"])
        # sort to find lw--up error later
        df.sort_values(by=['z','f'],inplace=True)
        df.reset_index(inplace=True)
        x = [df.loc[i]['z'] for i in range(1,len(df),2)]
        f = np.array([df.loc[i]['f'] for i in range(1,len(df),2)])
        err = np.array([np.abs(df.loc[i]['f'] - df.loc[i-1]['f']) for i in range(1,len(df),2)])
        ax[1].plot(x,f,c=all_colors[i],marker=all_markers[i],label=l)
        ax[1].fill_between(x, f-err,f+err, alpha=0.3,color=all_colors[i])
    # add 1 data point from steinborn
    sb = [2,0.005, 0.00214592, 0.00804721]
    ax[1].errorbar(sb[0],sb[1],yerr=[[sb[-2]],[sb[-1]]],fmt=all_markers[4],c=all_colors[4],label='Steinborn+2016')
    # formatting ticks
    ax[0].set_xscale("log")
    ax[0].set_yscale('log')
    ax[0].set_xticks([0.1,0.5,1,2,5],[0.1,0.5,1,2,5])
    ax[0].set_yticks([1e-4,1e-3,1e-2,1e-1,5e-1],np.array([1e-4,1e-3,1e-2,1e-1,5e-1]))
    # setting labels
    ax[0].set_ylabel("Dual fraction")
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='y', labelright=True, labelleft=False)
    [ax[i].set_xlabel("Redshift") for i in range(2)]
    [ax[i].legend(loc='lower left',fontsize=10) for i in range(2)]

    [ax[i].text(0.025, 0.975, ["Observed","Simulated"][i], transform=ax[i].transAxes, ha='left', va='top',fontsize=22) for i in range(2)]

    fig.tight_layout()
    fig.savefig(args.inDir+args.outFile)
    print("Done saving figure")