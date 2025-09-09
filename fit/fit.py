import numpy as np
import pickle
import os
import pyimfit
from astropy.io import fits
from tqdm import tqdm
import glob
from modelComponents import makeModelDict
import pandas as pd


def find_highest_indices(arr):
    """returns a tuple of ys, xs - indices of pixels with highest intensity counts"""
    flattened_arr = np.array(arr).flatten()
    max_indices = np.unravel_index(np.argsort(flattened_arr)[-2:], arr.shape)
    return max_indices


def galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, 
                    Xlim, Ylim, Xsslim, Ysslim,
                    PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                    PA_lim, ell_lim,  Iss_lim, rss_lim, Itot_lim,
                    midf, h1,h2,h_lim,alpha,alpha_lim,sky):
    """Returns a function set dictionary with keys as model name, 
       values as model function set"""
    sersic_n1_dict, sersic2_dict, sersic1_dict, sersic_dict, psf_dict, flatbar_dict, exp_dict, flatsky_dict = makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                                                                    PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                                                                    h1,h2,h_lim,alpha,alpha_lim,sky)
    #========function dictionary
    # psf
    funcset_dict_psf0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict,flatsky_dict]}
    funcset_dict_psf1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict]}
    # same center psf+sersic
    funcset_dict_psfser0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,psf_dict,flatsky_dict]}
    funcset_dict_psfser1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,psf_dict]}  
    # separate sersic
    funcset_dict_sersic0 = {'X0': [Xss0,Xsslim[0],Xsslim[1]], 'Y0': [Yss0,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict,flatsky_dict]}
    funcset_dict_sersic1 = {'X0': [Xss1,Xsslim[0],Xsslim[1]], 'Y0': [Yss1,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic1_dict]}
    funcset_dict_serser0 = {'X0': [Xss0,Xsslim[0],Xsslim[1]], 'Y0': [Yss0,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict,sersic1_dict,flatsky_dict]}
    funcset_dict_serser1 = {'X0': [Xss1,Xsslim[0],Xsslim[1]], 'Y0': [Yss1,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict,sersic1_dict]}
    funcset_dict_serserser = {'X0': [midf,Xsslim[0],Xsslim[1]], 'Y0': [midf,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic2_dict,sersic_dict,sersic1_dict,flatsky_dict]}

    # exponential
    funcset_dict_serexp= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,exp_dict]}
    funcset_dict_serexppsf= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,exp_dict,psf_dict,flatsky_dict]}
    # try sub exp with sersic n=1
    funcset_dict_sersern1= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,sersic_n1_dict]}
    funcset_dict_sersern1psf= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,sersic_n1_dict,psf_dict,flatsky_dict]}
    
    #========model dict
    funcset = {
        "sersic": [funcset_dict_sersic0],
        "sersic+sersic":[funcset_dict_serser0],
        "sersic,sersic": [funcset_dict_sersic0, funcset_dict_sersic1],
        # 1 core
        "psf+sersic": [funcset_dict_psfser0],
        "psf,sersic": [funcset_dict_psf0,funcset_dict_sersic1],
        "psf,sersic+exp":[funcset_dict_psf0,funcset_dict_serexp],
        "psf,sersic+sersic(n1)":[funcset_dict_psf0,funcset_dict_sersern1],
        "exp+sersic+psf":[funcset_dict_serexppsf],
        "sersic+sersic(n1)+psf":[funcset_dict_sersern1psf],
        # 2 core
        "sersic+psf,psf": [funcset_dict_psfser0,funcset_dict_psf1],
        "sersic+psf,sersic+psf": [funcset_dict_psfser0,funcset_dict_psfser1],
        # intersting models
        "sersic+sersic+sersic":[funcset_dict_serserser],
        "sersic+psf,sersic": [funcset_dict_psfser0,funcset_dict_sersic1],
        "sersic+sersic,sersic":[funcset_dict_serser0, funcset_dict_sersic1],
        "sersic+sersic,sersic+sersic": [funcset_dict_serser0, funcset_dict_serser1],
    }
    return funcset


def galaxy_model(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim, Xsslim, Ysslim,
                PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                midf, h1,h2,h_lim,alpha,alpha_lim,sky):
    """return a dictionary of galaxy model with keys as model name"""
    funcset = galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, 
                                Xlim, Ylim, Xsslim, Ysslim,
                                PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                                PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                                midf, h1,h2,h_lim,alpha,alpha_lim, sky);
    models = {}
    for model in funcset:
        models[model]= pyimfit.ModelDescription.dict_to_ModelDescription({'function_sets':funcset[model]})
    return models


def get_dofit_val(objectName):
    """get image parameters for fitting"""
    mosfile = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+objectName+"*.mos.fits"))[0]
    with fits.open(mosfile) as hdul:
        hdu0 = hdul[0]
    sky_level = hdu0.header['BACKGND'] #[e-/s] native pixels, value should be in the same units as the data pixels
    gain = hdu0.header['EGAIN'] #[e-/DU] in header
    exptime= hdu0.header['EXPORG'] # actual exp time
    noise = 0.5 #[e-] in .exp header, checked that is the same for all targets
    numcom = hdu0.header['NCOADD'] #Number of Averaged Frames   
    return exptime, noise, sky_level,numcom,gain


def dofit_no_oversp(modelName, dataImage, psf, readnoise, expT, skylevel, ncom, gainlev, solver="NM"):
    """do fit with not oversampled psf
       """
    fitter = pyimfit.Imfit(models_n1[modelName],psf=psf)
    fitter.loadData(dataImage, gain=gainlev,exp_time=expT, 
                    read_noise=readnoise, original_sky=skylevel,n_combined=ncom)
    fitter.doFit(solver)
    fitConfig = fitter.getModelDescription()
    fitModelImage = fitter.getModelImage()
    fitResult = fitter.getFitResult()
    param_names = fitter.numberedParameterNames
    return fitConfig, fitModelImage, fitResult, param_names


def fit_multi(models, epsf, image,noise,exptime,skylev,numcom,gain):
    """fit all models in models
       return lists of model config, model images, fit results, and parameter names"""
    models = list(models.keys())
    configs = []
    modelIms = []
    fitResults = []
    pnames= []

    # fit all models
    for modelName in tqdm(models, desc="Fitting Models"):
        try:
            config, modelIm, fitRes, pname = dofit_no_oversp(modelName, dataImage=image, psf=epsf, 
                                                             readnoise=noise, expT=exptime, skylevel = skylev, 
                                                             ncom=numcom, gainlev=gain, solver="LM")
            configs.append(config)
            modelIms.append(modelIm)
            fitResults.append(fitRes)
            pnames.append(pname)
        except Exception as e:
            error_message = f"An error occurred for {modelName}: {e}"
            print(error_message) 
            continue  
    return configs, modelIms, fitResults, pnames


def save_data(models,configs,modelIms,fitResults,pnames,objectName):
    savedata = {}
    savedata['modelNames'] = models
    savedata['configs'] = configs
    savedata['modelImage'] = modelIms
    savedata['fitResults'] = fitResults
    savedata['paramNames'] = pnames
    filename = os.path.join(args.outDir, objectName+".pkl")
    try:
        pickle.dump(savedata,open(os.path.expanduser(filename),"wb"))
    except:
        pickle.dump(savedata,open("./"+objectName+".pkl"),"wb")
    

if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to fit AGN cutouts
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--fitsDir", type=str, default="~/research-data/agn-result/box", help="path to cut out directory, cut out must be .fits")
    parser.add_argument("--oname", type=str, help="object name")
    parser.add_argument("--outDir", type=str, default="~/research-data/agn-result/fit/final_fit", help="output directory") 
    parser.add_argument("--psfPath", type=str, default="~/research-data/psf-results/psf_pkls", help="path to psf directory")
    # parser.add_argument("--plotSkyHist", action="store_true")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    
    # load cutout image
    cutoutPath = glob.glob(os.path.expanduser(os.path.join(args.fitsDir, "*"+args.oname+"*.fits")))[0]
    imageAGN = fits.getdata(cutoutPath)
    # load psf file
    psf_fileName = "psf_"+args.oname+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (os.path.expanduser(psfPath), "rb") as f:
        d_psf = pickle.load(f)
    epsf = d_psf['psf'].data
    # get do fit params
    exptime, noise, sky_level,numcom,gain = get_dofit_val(args.oname)
    # find centers for initial guess
    ys,xs = find_highest_indices(imageAGN)
    Imax = imageAGN.max()
    framelim = imageAGN.shape[0]
    midF=framelim//2
    # load background dict
    skylev = pd.read_pickle("sky.pkl")
    background = skylev[args.oname]
    print(f"Estimated background: {background}")
    # read initial guesses
    guess = pd.read_csv("guessVal_kpcbox.csv")
    gmask = guess['Name'] == args.oname
    PA,ell,RE,X0,Y0,X1,Y1,_= guess[gmask].values[0][1:].astype(float)
    if np.isfinite(X1):
        if np.isfinite(X0):
            xs[0] = X0
            ys[0] = Y0
        else:
            xs[0] = midF
            ys[0] = midF
        xs[1] = X1
        ys[1] = Y1  
    print("Guess vals: ", PA,ell,RE,xs[0],ys[0],xs[1],ys[1])
    # make models
    models_n1 = galaxy_model(X0=xs[0], Y0=ys[0], 
                    X1=xs[1], Y1=ys[1], 
                    Xss0=xs[0], Yss0=ys[0], 
                    Xss1=xs[1], Yss1=ys[1],
                    Xlim=[0,framelim], Ylim=[0,framelim], Xsslim = [0,framelim], Ysslim=[0,framelim],
                    PA_ss=PA, ell_ss=ell, n_ss=1, I_ss=1, r_ss=RE, Itot=1500,
                    PA_lim=[0,360], ell_lim=[0.0,1.0],
                    Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
                    midf=midF, 
                    h1=10,h2=10,h_lim=[0.1,10000],alpha=0.1,alpha_lim=[0.1,framelim], sky=background)
    # fit and save results
    if args.fit:
        configs, modelIms, fitResults, pnames = fit_multi(models_n1, epsf, imageAGN,noise,exptime, sky_level,numcom,gain)
        save_data(models_n1,configs,modelIms,fitResults,pnames,args.oname)
    print('Done: ', args.oname)