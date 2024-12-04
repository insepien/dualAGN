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


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outDir", type=str, default="/home/insepien/research-data/", help="output directory")
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

    




        
