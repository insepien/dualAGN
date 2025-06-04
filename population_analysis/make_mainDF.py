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
    chi_check = [df.loc[n,"chi_r"][0] < df.loc[n,"chi_r"][1] for n in range(len(df))]
    print(f"Chi best < chi compare: {np.sum(chi_check)}/{len(df)}")
    if np.sum(chi_check) != len(df):
        print("Error in objects:")
        print(df.loc[np.where(np.array(chi_check)==False)[0][0], 'Obj Name'])
    [print(n, df.loc[n,"Obj Name"], df.loc [n, "chi_r"]) for n in np.random.randint(0,len(df),3) ];

def save_pkl(args,df,filename):
    savepath = os.path.expanduser(args.outDir+filename)
    df.to_pickle(savepath)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--fitPath", type=str, default="/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit/", help="fit directory")
    parser.add_argument("--outDir", type=str, default="/home/insepien/research-data/pop-result/", help="output directory")
    parser.add_argument("--save_main", action="store_true", help='use this flag to save all_result df')
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
        fit_path = os.path.expanduser(args.fitPath+oname+".pkl")
        fix = False
        # open fit result files
        with open(fit_path,"rb") as f:
            d_fit = pickle.load(f)

        # find index of best model
        ind_best = list(d_fit['modelNames'].keys()).index(df.loc[n]['best model'])
        # find index of compare model
        ind_compare = list(d_fit['modelNames'].keys()).index(df.loc[n]['compare model'])

        # add chi and chi_r values
        df.at[n,'chi'] = [d['fitResults'][i].fitStat for d,i in zip([d_fit,d_fit],[ind_best,ind_compare])]
        df.at[n,'chi_r'] = [d['fitResults'][i].fitStatReduced for d,i in zip([d_fit,d_fit],[ind_best,ind_compare])]
        
        # add dictionary of param names-param values
        df.at[n,'param_vals_best'] = dict(zip(d_fit['paramNames'][ind_best],d_fit['fitResults'][ind_best].params))
        df.at[n,'param_vals_compare'] = dict(zip(d_fit['paramNames'][ind_compare],d_fit['fitResults'][ind_compare].params))
    
    #check main df and save
    check(df)
    if args.save_main:
        save_pkl(args,df,"all_results.pkl")
        print("Done saving main df")

    




        
