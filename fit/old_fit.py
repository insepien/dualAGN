import numpy as np
import pickle
import os
import pyimfit
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging

medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def find_highest_indices(arr):
    """returns a tuple of ys, xs - indices of pixels with highest counts"""
    flattened_arr = np.array(arr).flatten()
    max_indices = np.unravel_index(np.argsort(flattened_arr)[-2:], arr.shape)
    return max_indices

def crop_image(image, size=40):
    """cropping agn cut out to and return new centers"""
    ysO,xsO = find_highest_indices(image)
    # find the center of AGNs
    ycO = int(np.sum(ysO)/2)
    xcO = int(np.sum(xsO)/2)
    # crop
    px=int(size/2)
    imagecrop = image[ycO-px:ycO+px,xcO-px:xcO+px]
    # find agn centers in cropped image
    ys,xs = find_highest_indices(imagecrop)
    return imagecrop, ys, xs

def find_sky(image, objectName, args):
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(image.flatten(),bins=np.arange(np.min(image),np.max(image)))
    bgr = ax[1].hist(image.flatten(),bins=np.arange(np.min(image),20))
    [ax[i].set_title(['Whole intensity range',"min to 20"][i]) for i in range(2)]
    [ax[i].set_xlabel('intensity') for i in range(2)]
    ax[0].set_ylabel('number of agns') 
    sky = (bgr[1][np.where(bgr[0]==np.max(bgr[0]))]+bgr[1][np.where(bgr[0]==np.max(bgr[0]))[0][0]+1])/2
    if args.plotSkyHist:
        fig.savefig("skyHist_"+objectName+".jpg")
    else:
        return sky


def makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                 sigma, sigma_lim, Isky, Isky_lim):
    """Return Sersic, PSF, and Gaussian model parameter dictionary"""
    # Sersic
    """sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 'fixed'],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}"""
    sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 0, 10],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}
    sersic_dict = {'name': "Sersic", 'label': "bulge", 'parameters': sersic}
    # PSF
    psf = {'I_tot' : [Itot, Itot_lim[0], Itot_lim[1]]}
    psf_dict = {'name': "PointSource", 'label': "psf", 'parameters': psf}
    """psf = {'I_tot' : [Itot, Itot_lim[0], Itot_lim[1]], 'PA':[PA_ss, PA_lim[0],PA_lim[1]] }
    psf_dict = {'name': "PointSourceRot", 'label': "psf", 'parameters': psf}"""
    # Gaussians
    gaussian = {'PA':[PA_ss, PA_lim[0],PA_lim[1]], 'ell':[ell_ss, ell_lim[0],ell_lim[1]], 
                'I_0':[I_ss, Iss_lim[0],Iss_lim[1]], 'sigma':[sigma, sigma_lim[0], sigma_lim[1]]}
    gaussian_dict = {'name': "Gaussian", 'label': "gaussian", 'parameters': gaussian}
    # Flat sky
    flatsky = {'I_sky': [Isky, Isky_lim[0], Isky_lim[1]]}
    flatsky_dict = {'name': "FlatSky", 'label': "flat_sky", 'parameters':flatsky}
    return sersic_dict, psf_dict, gaussian_dict, flatsky_dict


def galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, 
                    Xlim, Ylim, Xsslim, Ysslim,
                    PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                    PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                    sigma, sigma_lim, midf, Isky, Isky_lim):
    """Returns a function set dictionary with keys as model name, 
       values as model function set"""
    sersic_dict, psf_dict, gaussian_dict, flatsky_dict = makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                 sigma, sigma_lim, Isky, Isky_lim)
    #========function dictionary
    # psf
    funcset_dict_psf0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict]}
    funcset_dict_psf1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict]}
    # same center psf+sersic
    funcset_dict_psfser0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict,sersic_dict]}
    funcset_dict_psfser1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict,sersic_dict]}  
    # separate sersic
    funcset_dict_sersic0 = {'X0': [Xss0,Xsslim[0],Xsslim[1]], 'Y0': [Yss0,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict]}
    #========model dict
    funcset = {
        "1psf": [funcset_dict_psf0],
        "1psf+sersic,sameCenter": [funcset_dict_psfser0],
        "2psf": [funcset_dict_psf0, funcset_dict_psf1],
        "1psf+sersic,1psf": [funcset_dict_psf1,funcset_dict_psfser0],
        "2psf+sersic,sameCenter": [funcset_dict_psfser0,funcset_dict_psfser1],
        "1psf+sersic,diffCenter":[funcset_dict_psf0,funcset_dict_sersic0]
    }
    return funcset


def galaxy_model(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim, Xsslim, Ysslim,
                PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                sigma, sigma_lim,midf, Isky, Isky_lim):
    """return a dictionary of galaxy model with keys as model name"""
    funcset = galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim, Xsslim, Ysslim,
                    PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                    PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                    sigma, sigma_lim,midf, Isky, Isky_lim);
    models = {}
    for model in funcset:
        models[model]= pyimfit.ModelDescription.dict_to_ModelDescription({'function_sets':funcset[model]})
    return models


def get_dofit_val(imageFile):
    with fits.open(imageFile) as hdul:
        hdu = hdul[0]
    exptime= hdu.header['EXPTIME']
    noise = hdu.header['EFFRN']
    sky_level = hdu.header['BACKGND']
    numcom = hdu.header['NCOADD']
    return exptime, noise, sky_level,numcom


def dofit_no_oversp(modelName, dataImage, psf, readnoise=0.22, expT=1, skylevel = 654.63, ncom=4, solver="NM"):
    """do fit with not oversampled psf
       """
    fitter = pyimfit.Imfit(models_n1[modelName],psf=psf)
    fitter.loadData(dataImage, gain=9.942e-1,exp_time=expT, 
                    read_noise=readnoise, original_sky=skylevel,n_combined=ncom)
    fitter.doFit(solver)
    fitConfig = fitter.getModelDescription()
    fitModelImage = fitter.getModelImage()
    fitResult = fitter.getFitResult()
    param_names = fitter.numberedParameterNames
    return fitConfig, fitModelImage, fitResult, param_names


def fit_multi(models, efpsf, image,noise,exptime,skylev,numcom):
    """fit all models in models
       return lists of model config, model images, fit results, and parameter names"""
    models = list(models.keys())
    configs = []
    modelIms = []
    fitResults = []
    pnames= []

    logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    # fit all models
    for modelName in tqdm(models, desc="Fitting models"):
        try:
            config, modelIm, fitRes, pname = dofit_no_oversp(modelName, dataImage=image, psf=epsf, 
                                                             readnoise=noise, expT=exptime, skylevel = skylev, ncom=numcom, solver="LM")
            configs.append(config)
            modelIms.append(modelIm)
            fitResults.append(fitRes)
            pnames.append(pname)
        except Exception as e:
            error_message = f"An error occurred for {modelName}: {e}"
            logging.error(error_message)
            print(error_message) 
            continue  
    return configs, modelIms, fitResults, pnames


def save_data(image,models,configs,modelIms,fitResults,pnames,objectName):
    savedata = {}
    savedata['imageSS'] = image
    savedata['modelNames'] = models
    savedata['configs'] = configs
    savedata['modelImage'] = modelIms
    savedata['fitResults'] = fitResults
    savedata['paramNames'] = pnames
    filename = "fit_pkls/"+objectName+".pkl"
    pickle.dump(savedata,open(filename,"wb"))
    

if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to fit AGN cutouts
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--expPath", type=str, default="../cutouts/data", help="path to cut out directory")
    parser.add_argument("--psfPath", type=str, default="../psfConstruction/psf_pkls", help="path to psf directory")
    parser.add_argument("--inFile", type=str, help="cut out file")
    parser.add_argument("--outDir", type=str, default="fit_pkls", help="directory for fit results")
    parser.add_argument("--plotSkyHist", action="store_true")
    args = parser.parse_args()
    
    # load cut out and psf file
    cutoutPath = os.path.join(args.expPath, args.inFile)
    imageAGN = fits.getdata(cutoutPath)
    objectName = args.inFile.split(".")[0]
    psf_fileName = "psf_"+objectName+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (psfPath, "rb") as f:
        d = pickle.load(f)
    epsf = d['psf'].data
    
    # get do fit params
    exptime, noise, sky_level,numcom = get_dofit_val(cutoutPath)
    print(exptime, noise, sky_level,numcom)
    
    # cropping image and find centers for initial guess
    imageAGNcrop, ys, xs = crop_image(imageAGN)
    Imax = imageAGN.max()
    itot=1500
    framelim = imageAGNcrop.shape[0]
    midF=framelim//2
    # find background level and subtract
    sky = find_sky(imageAGNcrop, objectName, args)
    if len(sky) != 1:
        sky = sky[-1]
    print(sky)
    imageAGNcrop_bs = imageAGNcrop-sky
 
    # make models
    models_n1 = galaxy_model(X0=xs[0], Y0=ys[0], 
                         X1=xs[1], Y1=ys[1], 
                         Xss0=xs[0], Yss0=ys[0], 
                         Xss1=xs[1], Yss1=ys[1],
                         Xlim=[0,framelim], Ylim=[0,framelim], Xsslim = [0,framelim], Ysslim=[0,framelim],
                         PA_ss=200, ell_ss=0.1, n_ss=1, I_ss=1, r_ss=20, Itot=itot,
                         PA_lim=[0,360], ell_lim=[0.0,1.0], I_lim=[0.1,Imax],
                         Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
                         sigma = 5, sigma_lim = [1,20],midf=midF, Isky = 2.5, Isky_lim =[0,10])
    # fit and save results
    configs, modelIms, fitResults, pnames = fit_multi(models_n1, epsf, imageAGNcrop_bs,noise,exptime, sky_level,numcom)
    save_data(imageAGNcrop_bs,models_n1,configs,modelIms,fitResults,pnames,objectName)
    print('Done: ', args.inFile)