import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from astropy.wcs import WCS
from astropy.coordinates import angular_separation
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'monospace'

### CHI-SQUARED TABLE
def make_chi_table(df):
    df1 = df.loc[:,["Obj Name", "Type", "chi"]].copy()
    # fill in chi for 1 and 2 cores
    chi_1core = [df1.loc[n,"chi"][int(df1.loc[n,"Type"]!="1")] for n in range(len(df1))]
    chi_2core = [df1.loc[n,"chi"][int(df1.loc[n,"Type"]=="1")] for n in range(len(df1))]
    df1['chi 1 core'] = chi_1core
    df1['chi 2 core'] = chi_2core
    # calculate chi difference
    df1['del chi'] = np.abs(df1['chi 1 core'] - df1['chi 2 core'])
    return df1


def plot_chi(args,df1):
    """plot distribution of chi-square"""
    fig,ax = plt.subplots(1,3,sharey=True,figsize=(10,3))
    ax[0].hist([df1.loc[n,'chi'][0] for n in range(len(df))])
    ax[0].set_xlabel("$\chi$")
    ax[0].set_ylabel("number of objects")
    ax[0].set_title("distribution of best $\chi$")

    ax[1].hist(df1['del chi'].sort_values())
    ax[1].set_xlabel("$\Delta\chi$")
    ax[1].set_title("distribution of $\Delta\chi$")

    ax[2].hist(df1['del chi'].sort_values()[:-3])
    ax[2].set_xlabel("$\Delta\chi$")
    ax[2].set_title("distribution of $\Delta\chi$\nwithout 3 highest outliers")
    fig.tight_layout()
    savepath = os.path.expanduser(args.outDir+"chi.pdf")
    fig.savefig(savepath)

### SEPARATION
def ang_sep(pos1,pos2,on,z=0.2):
    """calculate angular separation given dictionary of parameter names and values
        returns angular sep in arcsec and physical separation in kpc"""
    imageFile = glob.glob(os.path.expanduser("/home/insepien/research-data/agn-result/box/final_cut/"+on+"*"))[0]
    w = WCS(imageFile)
    ra1,dec1 = w.pixel_to_world_values(pos1[0],pos1[1])
    ra2,dec2 = w.pixel_to_world_values(pos2[0],pos2[1])
    sep_rad = (angular_separation(ra1*u.deg,dec1*u.deg,ra2*u.deg,dec2*u.deg)).to(u.rad)
    sep_kpc = (cosmo.angular_diameter_distance(z)*sep_rad.value).to('kpc')
    return sep_rad.to(u.arcsec), sep_kpc


def pix_to_arcsec(imageFile):
    """convert pixel to arcsec"""
    w = WCS(imageFile)
    # convert pixel to degree
    framelim = fits.getdata(imageFile).shape[0]
    ra1,dec1 = w.pixel_to_world_values(0,0)
    ra2,dec2 = w.pixel_to_world_values(framelim,framelim)
    # find pixel-arsec scale
    arcsec_per_pix = 0.16*u.arcsec
    return arcsec_per_pix, [ra1,dec1,ra2,dec2]


def plot_sep(df2, args):
    sep_kpc = df2['sep_kpc']

    # use random galaxy to find scale of 1 pixel to draw 1 pix line
    on = "J0820+1801"
    imf = glob.glob(os.path.expanduser("~/research-data/agn-result/box/final_cut/"+on+"*"))[0]
    arcsec_per_pix, _ = pix_to_arcsec(imf)
    median_z = np.median(list(alpaka['Z']))
    pix_in_kpc = (arcsec_per_pix.to(u.rad).value*cosmo.angular_diameter_distance(median_z)).to(u.kpc)

    plt.hist(sep_kpc, 
             color='darkseagreen',edgecolor='k', bins=np.logspace(-1,1.5,10))
    plt.axvline(x=pix_in_kpc.value,
                label=f"1 pixel={pix_in_kpc.value:.2f}",c='cornflowerblue')
    plt.title("Figure 2: Separation of 2-core AGNs")
    plt.xlabel(" Separation (kpc)")
    plt.ylabel("Number of AGN")
    plt.xscale('log')
    plt.legend()
    savepath = os.path.expanduser(args.outDir+"sep.pdf")
    plt.savefig(savepath)



### OIII BOLO LUM
def oiii_bol_lum(df,alpaka):
    df["OIII_5007_LUM_DERRED"] = None
    df['sersic index'] = None
    df['I1'] = None
    df['I2'] = None
    # add sersic index, intensity, luminosity
    for j in range(len(df)):
        # save alpaka oii lum
        # for agns with 2 measurements, only take meas from type 2    
        name_mask = alpaka['Names'] == df.loc[j,"Obj Name"]
        type_mask  = alpaka['AGN_TYPE'] == 2
        # get OIII luminosity from Mullaney
        oiii_lum = alpaka[name_mask & type_mask]['OIII_5007_LUM_DERRED'].values 
        # there should be only 1 galaxy with dup entry in Mullaney
        if len(oiii_lum) > 1:
            print(df.loc[j,"Obj Name"])  
        df.at[j,"OIII_5007_LUM_DERRED"] = np.sum(oiii_lum)

        ## GET SERSIC INDEX
        param_names = list(df.loc[j,'param_vals_best'].keys())
        ns = [i for i in param_names if i[0]=="n"]
        sersic_index = [df.loc[j,'param_vals_best'][n] for n in ns]

        ## get all intensities for each positions
        ind2 = param_names.index("X0_2")
        core1_intens = [df.loc[j,'param_vals_best'][i] for i in param_names[:ind2] if i[0]=="I"]
        core2_intens = [df.loc[j,'param_vals_best'][i] for i in param_names[ind2:] if i[0]=="I"]
        df.at[j,"sersic index"] = sersic_index
        df.at[j, "I1"] = core1_intens
        df.at[j, "I2"] = core2_intens
        # get summed intensity ratio 
        df.at[j,'I1/I2'] = np.sum(df.loc[j,'I1'])/np.sum(df.loc[j,'I2'])

        ## CALCULATE OIII BOLOMETRIC LUMINOSITY 
        # Kauffman bolo correction
        oiii_bol = oiii_lum*800
            
        # save oiii bolo lums
        if oiii_lum.shape[0] == 2:
            # try to assign the right lum in Mullaney
            if df.loc[j,'I1/I2'] < 1:
                df.at[j,"OIII_BOL_2"] = np.max(oiii_bol)
                df.at[j,"OIII_BOL_1"] = np.min(oiii_bol)
            else:
                df.at[j,"OIII_BOL_1"] = np.max(oiii_bol)
                df.at[j,"OIII_BOL_2"] = np.min(oiii_bol)
        else:
            df.at[j,"OIII_BOL_2"] = oiii_bol/(df.loc[j,'I1/I2']+1)
            df.at[j,"OIII_BOL_1"] = oiii_bol - df.at[j,"OIII_BOL_2"]
    return df


def update_all_results():
    """load all sample df and get OIII lum """
    df_all = pd.read_pickle(os.path.expanduser("/home/insepien/research-data/pop-result/all_results.pkl"))
    alpaka = pd.read_pickle("/home/insepien/research-data/alpaka/alpaka_39fits.pkl")
    # add corrected bolometric luminosity
    df_all["OIII_5007_LUM_DERRED"] = None
    for j in range(len(df_all)):
        name_mask = alpaka['Desig'] == df_all.loc[j,"Obj Name"]
        type_mask  = alpaka['AGN_TYPE'] == 2
        oiiilum = alpaka[name_mask & type_mask]['OIII_5007_LUM_DERRED'].values
        # if 2 source add both lum
        df_all.at[j,"OIII_5007_LUM_DERRED"] = np.sum(oiiilum)
    df_all['OIII_BOL'] = df_all['OIII_5007_LUM_DERRED']*800

    df_all['sersic index'] = None
    # add sersic index
    for j in range(len(df_all)):
        param_names = list(df_all.loc[j,'param_vals_best'].keys())
        ns = [i for i in param_names if i[0]=="n"]
        sersic_index = [df_all.loc[j,'param_vals_best'][n] for n in ns]
        df_all.at[j,"sersic index"] = sersic_index
    df_all.to_pickle("/home/insepien/research-data/pop-result/all_results_updated.pkl")
    return df_all


def plot_dualFrac_lum(ax):
    # get df with bolo lum
    df_all = update_all_results()
    # make luminosity bin and get dual fraction
    log_lum_all = np.log10(list(df_all['OIII_BOL']))
    log_bin = np.linspace(np.min(log_lum_all), np.max(log_lum_all) ,9)
    log_lum_2 = np.log10(list(df_all[df_all['Type']=='2']['OIII_BOL']))
    # plotting
    ax.hist(log_lum_all, bins=log_bin, color="cornflowerblue", label='all', edgecolor='k')
    ax.set_xlabel("Log(L$_{\mathrm{OIII \ Bol}}$ [ergs s$^{-1}$])")
    ax.set_ylabel("Number of AGN")
    ax.set_title("Dual AGN fraction")
    ax.hist(log_lum_2, bins=log_bin, color="darkseagreen",stacked=True, label="2 core", edgecolor='k')
    ax.legend()


def plot_Lbol(df,args):
    Lbol_2core = list(np.concatenate([df['OIII_BOL_1'],df['OIII_BOL_2']]))
    logbol = np.log10(Lbol_2core)
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].hist(np.log10(Lbol_2core), bins=np.linspace(np.min(logbol), np.max(logbol),11),edgecolor="black",color="darkseagreen")
    ax[0].set_xlabel("Log (L$_{\mathrm{OIII \ Bol}}$ [ergs s$^{-1}$])")
    ax[0].set_ylabel("Number of AGN")
    ax[0].set_title("L$_{\mathrm{OIII \ Bol}}$ for each AGN in 2-core \n(flux summed at each core position)")

    ax[1].hist([x if x>1 else 1/x for x in df['I1/I2'] ],edgecolor="black",color="darkseagreen",bins=np.logspace(0,2,10))
    ax[1].set_title("Flux ratio in 2 core AGNs ")
    ax[1].set_xlabel("I1/I2")
    ax[1].set_ylabel("Number of AGN")
    ax[1].set_xscale('log')

    # plot dual fraction v. bolo lum
    plot_dualFrac_lum(ax[2])
    fig.suptitle("Figure 1: Population luminosity")
    fig.tight_layout()
    fig.savefig(args.outDir+"oiiiBol.pdf")



### SAVE FUNCTIONS
def save_pkl(args,df,filename):
    savepath = os.path.expanduser(args.outDir+filename)
    df.to_pickle(savepath)

def save_csv(args,df,filename):  
    savepath = os.path.expanduser(args.outDir+filename)
    df.to_csv(savepath, sep='\t', index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="/home/insepien/research-data/pop-result/", help="input directory")
    parser.add_argument("--inFile", type=str, default="all_results.pkl", help="input file")
    parser.add_argument("--outDir", type=str, default="/home/insepien/research-data/pop-result/", help="output directory")
    parser.add_argument("--save_chi", action="store_true", help='use this flag to save chi csv table')
    parser.add_argument("--save_sep", action="store_true", help='use this flag to save separation table+histogram')
    parser.add_argument("--save_bolo", action="store_true", help='use this flag to save bolometric luminosity')
    args = parser.parse_args()

    df = pd.read_pickle(args.inDir+args.inFile)
    # make chi table
    if args.save_chi:
        df_chi = make_chi_table(df)
        plot_chi(args,df_chi)
        df_chi.drop('chi',axis=1,inplace=True)
        save_csv(args,df_chi,"chi.csv")
        print("Done saving chi table and plot")

    # make separation histogram
    # make copy of df and select only 2 core rows
    if args.save_sep:
        df2 = df[df.Type == '2'].loc[:,["Obj Name", "best model", "param_vals_best"]].copy().reset_index(drop=True)
        # load alpaka to get redshift for angular diameter distance
        alpaka = pd.read_pickle("/home/insepien/research-data/alpaka/alpaka41.pkl")
        # get core separation of each object
        seps_arcsec = []
        seps_kpc = []
        for n in range(len(df2)):
            pos1 = [df2.loc[n,'param_vals_best']['X0_1'],df2.loc[n,'param_vals_best']['Y0_1']]
            pos2 = [df2.loc[n,'param_vals_best']['X0_2'],df2.loc[n,'param_vals_best']['Y0_2']]
            redshift = alpaka[alpaka['Names']==df2.loc[n,'Obj Name']]['Z'].values[0]
            seps = ang_sep(pos1,pos2, on = df2.loc[n,'Obj Name'], z=redshift)
            seps_arcsec.append(seps[0].value)
            seps_kpc.append(seps[1].value)
        # write to df
        df2['sep_arcsec'] = seps_arcsec
        df2['sep_kpc'] = seps_kpc
        save_pkl(args,df2,"separation.pkl")
        plot_sep(df2, args)
        print("Done saving sep table and plot")


    # calculate OIII bolometric luminosity and sersic indices
    if args.save_bolo:
        df = pd.read_pickle(os.path.expanduser(args.inDir+"separation.pkl"))
        alpaka = pd.read_pickle("/home/insepien/research-data/alpaka/alpaka41.pkl")
        df = oiii_bol_lum(df,alpaka)
        plot_Lbol(df,args)
        save_pkl(args,df,"oiiiBol.pkl")
        print("Done saving bolo lum")

