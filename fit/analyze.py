import os
import numpy as np
import pandas as pd
import pickle
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
    df1 = df.loc[:,["Obj Name", "Type", "chi_r"]].copy()
    # fill in chi for 1 and 2 cores
    chi_1core = [df1.loc[n,"chi_r"][int(df1.loc[n,"Type"]!="1")] for n in range(len(df1))]
    chi_2core = [df1.loc[n,"chi_r"][int(df1.loc[n,"Type"]=="1")] for n in range(len(df1))]
    df1['chi 1 core'] = chi_1core
    df1['chi 2 core'] = chi_2core
    # calculate chi difference
    df1['del chi'] = np.abs(df1['chi 1 core'] - df1['chi 2 core'])
    return df1


def plot_chi(args,df1):
    """plot distribution of chi-square"""
    fig,ax = plt.subplots(1,3,sharey=True,figsize=(10,3))
    ax[0].hist([df1.loc[n,'chi_r'][0] for n in range(len(df))])
    ax[0].set_xlabel("$\chi_r$")
    ax[0].set_ylabel("number of objects")
    ax[0].set_title("distribution of best $\chi_r$")

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

def indiv_sep_cal(df2,separation):
    """add separation for weird objects"""
    # find 1st object index in df
    on = 'J0932+1611'
    ind = np.where(df2['Obj Name']==on)[0][0]
    # get small bulge params
    with open(os.path.expanduser("/home/insepien/research-data/agn-result/fit/final_fit_nb/"+on+".pkl"),"rb") as f:
        d = pickle.load(f)
    small_bul_d = dict(zip(d['paramNames'][-6:],d['fitResults'][0].params[-6:]))
    # get positions
    pos1 = [df2.loc[ind,'param_vals_best']['X0_2'],df2.loc[ind,'param_vals_best']['Y0_2']]
    pos2 = [small_bul_d['X0_2'],small_bul_d['Y0_2']]
    # add to df
    df2.loc[ind,'param_vals_best']['core 2'] = small_bul_d
    df2.at[ind,'separation'] = ang_sep(pos1,pos2,on)
    separation.append(df2.loc[ind,'separation'])

    # find 2nd object index in df
    on = 'J1204+0335'
    ind = np.where(df2['Obj Name']==on)[0][0]
    # get small bulge params
    with open(os.path.expanduser("/home/insepien/research-data/agn-result/fit/final_fit_nb/"+on+"_bulges.pkl"),"rb") as f:
        d = pickle.load(f)
    bulges_d = dict(zip(d['paramNames'],d['fitResults'][0].params))
    # main bulge position
    pos0 = [df2.loc[ind,'param_vals_best']['X0_2'],df2.loc[ind,'param_vals_best']['Y0_2']]
    # small bulges position
    pos1 = [bulges_d["X0_1"],bulges_d['Y0_1']]
    pos2 = [bulges_d["X0_2"],bulges_d['Y0_2']]
    # add to df
    df2.loc[ind,'param_vals_best']['core 2'] = bulges_d
    df2.at[ind,'separation'] = [ang_sep(pos0,pos1,on,z=0.2),ang_sep(pos0,pos2,on,z=0.2)]
    separation.append(df2.loc[ind,'separation'][0])
    separation.append(df2.loc[ind,'separation'][1])
    
def pix_to_arcsec(imageFile):
    """convert pixel to arcsec"""
    w = WCS(imageFile)
    # convert pixel to degree
    framelim = fits.getdata(imageFile).shape[0]
    ra1,dec1 = w.pixel_to_world_values(0,0)
    ra2,dec2 = w.pixel_to_world_values(framelim,framelim)
    framelim_deg = angular_separation(ra1*u.degree,dec1*u.degree,ra2*u.degree,dec2*u.degree)
    # find pixel-arsec scale
    framelim_arcsec = framelim_deg.to('arcsec')
    arcsec_per_pix = framelim_arcsec/framelim
    return arcsec_per_pix, [ra1,dec1,ra2,dec2]


def plot_sep(df2, args):
    seps = df2['separation']
    sep_kpc = [seps[i][1].value for i in range(len(seps)) if i!=14]
    sep_kpc.append(seps[14][0][1].value)
    sep_kpc.append(seps[14][1][1].value)

    # find scale of 1 pixel
    on = "J0820+1801"
    imf = glob.glob(os.path.expanduser("~/research-data/agn-result/box/final_cut/"+on+"*"))[0]
    arcsec_per_pix, _ = pix_to_arcsec(imf)
    median_z = np.median(list(alpaka['Z']))
    pix_in_kpc = (arcsec_per_pix.to(u.rad).value*cosmo.angular_diameter_distance(median_z)).to(u.kpc)

    plt.hist(sep_kpc, color='darkseagreen',edgecolor='k', bins=np.logspace(-1,1.5,10))
    plt.axvline(x=pix_in_kpc.value,label="1 pixel scale",c='cornflowerblue')
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
        oiii_lum = alpaka[name_mask & type_mask]['OIII_5007_LUM_DERRED'].values   
        df.at[j,"OIII_5007_LUM_DERRED"] = np.sum(oiii_lum)
        if df.loc[j,"comp fit"] == "":
            param_names = list(df.loc[j,'param_vals_best'].keys())

            ## GET SERSIC INDEX
            ns = [i for i in param_names if i[0]=="n"]
            sersic_index = [df.loc[j,'param_vals_best'][n] for n in ns]

            ## GET INTENSITY RATIO
            ind2 = param_names.index("X0_2")
            core1_intens = [df.loc[j,'param_vals_best'][i] for i in param_names[:ind2] if i[0]=="I"]
            core2_intens = [df.loc[j,'param_vals_best'][i] for i in param_names[ind2:] if i[0]=="I"]
            df.at[j,"sersic index"] = sersic_index
            df.at[j, "I1"] = core1_intens
            df.at[j, "I2"] = core2_intens
            df.at[j,'I1/I2'] = np.sum(df.loc[j,'I1'])/np.sum(df.loc[j,'I2'])

            ## CALCULATE OIII BOLOMETRIC LUMINOSITY 
            # Kauffman bolo correction
            oiii_bol = oiii_lum*800
            
            # save oiii bolo lums
            if oiii_lum.shape[0] == 2:
                if df.loc[j,'I1/I2'] < 1:
                    df.at[j,"OIII_BOL_2"] = np.max(oiii_bol)
                    df.at[j,"OIII_BOL_1"] = np.min(oiii_bol)
                else:
                    df.at[j,"OIII_BOL_1"] = np.max(oiii_bol)
                    df.at[j,"OIII_BOL_2"] = np.min(oiii_bol)
            else:
                df.at[j,"OIII_BOL_2"] = oiii_bol/(df.loc[j,'I1/I2']+1)
                df.at[j,"OIII_BOL_1"] = oiii_bol - df.at[j,"OIII_BOL_2"]
    ## COMP FIT AGNS
    for j in[8,14]:
        # get sersic index
        ser_ind = [df.loc[j,'param_vals_best'][i] for i in list(df.loc[j,'param_vals_best'].keys()) if i[0]=="n"]
        ser_ind.append([df.loc[j,'param_vals_best']['core 2'][i] for i in list(df.loc[j,'param_vals_best']['core 2'].keys()) if i[0]=="n"])
        # get intensity
        i1 = [df.loc[j,'param_vals_best'][i] for i in list(df.loc[j,'param_vals_best'].keys()) if i[0]=="I"]
        i2 = [df.loc[j,'param_vals_best']['core 2'][i] for i in df.loc[j,'param_vals_best']['core 2'].keys() if i[0]=="I"]
        df.at[j,"sersic index"] = ser_ind
        df.at[j,"I1"] = i1
        df.at[j,"I2"] = i2
    # 2 core intensity and luminosity
    df.at[8,"I1/I2"] = np.sum(i1)/np.sum(i2)
    df.at[8,"OIII_BOL_2"] = df.loc[8,"OIII_5007_LUM_DERRED"]*800/(df.loc[8,'I1/I2']+1)
    df.at[8,"OIII_BOL_1"] = df.loc[8,"OIII_5007_LUM_DERRED"]*800 - df.at[8,"OIII_BOL_2"]
    # 3 core agn intensity and luminosity
    df.at[14,"I2"] = df.loc[14,"I2"][:-1]
    df.at[14,"I3"] = df.loc[14,"I2"][-1]
    i1_i2 = np.sum(df.loc[14,"I1"])/np.sum(df.loc[14,"I2"])
    i3_i2 = np.sum(df.loc[14,"I3"])/np.sum(df.loc[14,"I2"])
    df.at[14,"OIII_BOL_2"] = df.loc[14,"OIII_5007_LUM_DERRED"]*800/(1+i1_i2+i3_i2)
    df.at[14,"OIII_BOL_1"] = df.loc[14,"OIII_BOL_2"]*i1_i2
    df.at[14,"OIII_BOL_3"] = df.loc[14,"OIII_BOL_2"]*i3_i2
    return df


def plot_Lbol(df,args):
    Lbol_2core = list(np.concatenate([df['OIII_BOL_1'],df['OIII_BOL_2']]))
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(np.log10(Lbol_2core), bins=np.linspace(45,49,11),edgecolor="black",color="darkseagreen")
    ax[0].set_xlabel("Log L$_{\mathrm{OIII \ Bol}}$ (ergs s$^{-1}$)")
    ax[0].set_ylabel("Number of AGN")
    ax[0].set_title("L$_{\mathrm{OIII \ Bol}}$ for each AGN in 2-core \n(flux summed at each core position)")

    ax[1].hist(np.log10([x if x>1 else 1/x for x in df['I1/I2'] ]),edgecolor="black",color="darkseagreen")
    ax[1].set_title("Flux ratio in 2 core AGNs ")
    ax[1].set_xlabel("Log (I1/I2)")
    ax[1].set_ylabel("Number of AGN")
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
    parser.add_argument("--inDir", type=str, default="/home/insepien/research-data/", help="input directory")
    parser.add_argument("--inFile", type=str, default="all_results.pkl", help="input file")
    parser.add_argument("--outDir", type=str, default="/home/insepien/research-data/", help="output directory")
    parser.add_argument("--save_chi", action="store_true", help='use this flag to save chi csv table')
    parser.add_argument("--save_sep", action="store_true", help='use this flag to save separation table+histogram')
    parser.add_argument("--save_bolo", action="store_true", help='use this flag to save bolometric luminosity')
    args = parser.parse_args()

    df = pd.read_pickle(args.inDir+args.inFile)
    # make chi table
    if args.save_chi:
        df1 = make_chi_table(df)
        plot_chi(args,df1)
        df1.drop('chi_r',axis=1,inplace=True)
        save_csv(args,df1,"chi.csv")
        print("Done saving chi table")

    # make separation histogram
    # make copy of df and select only 2 core rows
    if args.save_sep:
        df2 = df[df.Type == '2'].loc[:,["Obj Name", "best model", "comp fit", "param_vals_best"]].copy().reset_index(drop=True)
        alpaka = pd.read_pickle("/home/insepien/research-data/alpaka41.pkl")
        # get core separation and write to df
        separations = []
        for n in range(len(df2)):
            if df2.loc[n,'comp fit'] == "":
                pos1 = [df2.loc[n,'param_vals_best']['X0_1'],df2.loc[n,'param_vals_best']['Y0_1']]
                pos2 = [df2.loc[n,'param_vals_best']['X0_2'],df2.loc[n,'param_vals_best']['Y0_2']]
                redshift = alpaka[alpaka['Names']==df2.loc[n,'Obj Name']]['Z'].values[0]
                separations.append(ang_sep(pos1,pos2, on = df2.loc[n,'Obj Name'], z=redshift))
            else:
                separations.append("")
        df2['separation'] = separations
        indiv_sep_cal(df2,separations)
        save_pkl(args,df2,"separation.pkl")
        plot_sep(df2, args)
        print("Done saving sep table")


    # calculate OIII bolometric luminosity
    if args.save_bolo:
        df = pd.read_pickle(os.path.expanduser(args.inDir+"separation.pkl"))
        alpaka = pd.read_pickle(args.inDir+"alpaka41.pkl")
        df = oiii_bol_lum(df,alpaka)
        plot_Lbol(df,args)
        save_pkl(args,df,"oiiiBol.pkl")
        print("Done saving bolo lum")

