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

def check(df):
    """some checks of df"""
    print("Chi best < chi compare: ", np.sum([df.loc[n,"chi_r"][0] < df.loc[n,"chi_r"][1] for n in range(len(df))]))
    [print(n, df.loc[n,"Obj Name"], df.loc [n, "chi_r"]) for n in np.random.randint(0,len(df),3) ];

def save_pkl(args,df,filename):
    savepath = os.path.expanduser(args.outDir+filename)
    df.to_pickle(savepath)


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

def save_csv(args,df,filename):  
    savepath = os.path.expanduser(args.outDir+filename)
    df.to_csv(savepath, sep='\t', index=False)


def plot_chi(args,df1):
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


def ang_sep(pos1,pos2,on,z=0.2):
    """calculate angular separation given dictionary of parameter names and values
        returns angular sep in arcsec and physical separation in kpc"""
    imageFile = glob.glob(os.path.expanduser("/home/insepien/research-data/agn-result/box/final_cut/"+on+"*"))[0]
    w = WCS(imageFile)
    ra1,dec1 = w.pixel_to_world_values(pos1[0],pos1[1])
    ra2,dec2 = w.pixel_to_world_values(pos2[0],pos2[1])
    sep_rad = (angular_separation(ra1,dec1,ra2,dec2))*u.rad
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
    


def plot_sep(args,separations):
    plt.hist([a[1].value for a in separations if a!=''],bins=np.arange(11)*100)
    plt.title("distribution of core separation")
    plt.xlabel("Separation(kpc)")
    plt.ylabel("Number of dual")
    savepath = os.path.expanduser(args.outDir+"sep.pdf")
    plt.savefig(savepath)

if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outDir", type=str, default="/home/insepien/research-data/", help="output directory")
    parser.add_argument("--save_main", action="store_true", help='use this flag to save all_result df')
    parser.add_argument("--save_chi", action="store_true", help='use this flag to save chi table')
    parser.add_argument("--save_sep", action="store_true", help='use this flag to save sep histogram')
    args = parser.parse_args()

    # read text file of model selection
    with open(os.path.expanduser(args.outDir+"blank.txt"),"r") as f:
        d = f.read().splitlines()
    # turn into df
    df = pd.DataFrame([d[1:][i].split('\t') for i in range(len(d[1:]))],columns=d[0].split('\t'))
    df['chi'] = None
    df['chi_r'] = None
    df['param_vals_best'] = None
    df['param_vals_compare'] = None

    # fill in df by row
    for n in range(len(df)):
        oname = df.loc[n]['Obj Name']
        # if being 1 of 2 extra weird object, use different file path
        if df.loc[n, 'comp fit'] != "" and df.loc[n, 'Use massfit'] =="1":
            best_path = os.path.expanduser("/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit/"+oname+".pkl")
            compare_path = os.path.expanduser("/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit/"+oname+"_all.pkl")
            fix = False
        # if normal object, best model from mass fit
        elif df.loc[n, 'comp fit'] == "" and df.loc[n, 'Use massfit'] =="1":
            best_path = os.path.expanduser("/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit/"+oname+".pkl")
            compare_path = best_path
            fix = False
        # for best model from nb
        else:
            best_path = os.path.expanduser("/home/insepien/research-data/agn-result/fit/final_fit_nb/"+oname+".pkl")
            compare_path = os.path.expanduser("/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit/"+oname+".pkl")
            fix = True

        # open fit result files
        with open(best_path,"rb") as f:
            d_best = pickle.load(f)
        with open(compare_path,"rb") as f:
            d_compare = pickle.load(f)

        # if using nb, fix format error in final_fit_nb pkl
        if fix:
            d_best['paramNames'] = [d_best['paramNames']]

        # find index of best model
        if len(d_best['modelNames'])==1 and list(d_best['modelNames'].keys())[0].replace('\n',"") .replace('/n',"")== df.loc[n, "best model"]:
            ind_best = 0
        else: 
            ind_best = list(d_best['modelNames'].keys()).index(df.loc[n]['best model'])
    
        # find index of compare model
        ind_compare = list(d_compare['modelNames'].keys()).index(df.loc[n]['compare model'])

        # add chi and chi_r values
        df.at[n,'chi'] = [d['fitResults'][i].fitStat for d,i in zip([d_best,d_compare],[ind_best,ind_compare])]
        df.at[n,'chi_r'] = [d['fitResults'][i].fitStatReduced for d,i in zip([d_best,d_compare],[ind_best,ind_compare])]
        
        # add dictionary of param names-param values
        df.at[n,'param_vals_best'] = dict(zip(d_best['paramNames'][ind_best],d_best['fitResults'][ind_best].params))
        df.at[n,'param_vals_compare'] = dict(zip(d_compare['paramNames'][ind_compare],d_compare['fitResults'][ind_compare].params))
    
    #check main df and save
    check(df)
    if args.save_main:
        save_pkl(args,df,"all_results.pkl")
        print("Done saving main df")

    # make chi table
    df1 = make_chi_table(df)
    if args.save_chi:
        plot_chi(args,df1)
        df1.drop('chi_r',axis=1,inplace=True)
        save_csv(args,df1,"chi.csv")
        print("Done saving chi table")

    # make separation histogram
    # make copy of df and select only 2 core rows
    if args.save_sep:
        df2 = df[df.Type == '2'].loc[:,["Obj Name", "best model", "comp fit", "param_vals_best"]].copy().reset_index(drop=True)
        # get core separation and write to df
        separations = []
        for n in range(len(df2)):
            if df2.loc[n,'comp fit'] == "":
                pos1 = [df2.loc[n,'param_vals_best']['X0_1'],df2.loc[n,'param_vals_best']['Y0_1']]
                pos2 = [df2.loc[n,'param_vals_best']['X0_2'],df2.loc[n,'param_vals_best']['Y0_2']]
                separations.append(ang_sep(pos1,pos2, on = df2.loc[n,'Obj Name']))
            else:
                separations.append("")
        df2['separation'] = separations
        indiv_sep_cal(df2,separations)
        save_pkl(args,df2,"separation.pkl")
        plot_sep(args,separations)
        print("Done saving sep table")


        
