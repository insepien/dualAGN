import numpy as np
import pickle
import os
import pyimfit
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from modelComponents import makeModelDict

medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def find_highest_indices(arr):
    """returns a tuple of ys, xs - indices of pixels with highest intensity counts"""
    flattened_arr = np.array(arr).flatten()
    max_indices = np.unravel_index(np.argsort(flattened_arr)[-2:], arr.shape)
    return max_indices


def find_sky(image, objectName, args):
    """find background sky level as the peak of intensity histogram"""
    # plot intensity histograms
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(image.flatten(),bins=np.arange(np.min(image),np.max(image)))
    bgr = ax[1].hist(image.flatten(),bins=np.arange(np.min(image),20))
    # find bin of maximum number of pix and sky level
    max_ind = np.where(bgr[0]==np.max(bgr[0]))[0][0]
    sky = (bgr[1][max_ind]+bgr[1][max_ind+1])/2
    # option to save the histogram
    if args.plotSkyHist:
        [ax[i].set_title(['Whole intensity range',"min to 20"][i]) for i in range(2)]
        [ax[i].set_xlabel('intensity') for i in range(2)]
        ax[0].set_ylabel('number of agns') 
        fig.savefig("skyHist_"+objectName+".jpg")
    return sky


def galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, 
                    Xlim, Ylim, Xsslim, Ysslim,
                    PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                    PA_lim, ell_lim,  Iss_lim, rss_lim, Itot_lim,
                    midf, h1,h2,h_lim,alpha,alpha_lim):
    """Returns a function set dictionary with keys as model name, 
       values as model function set"""
    sersic_n1_dict, sersic2_dict, sersic1_dict, sersic_dict, psf_dict, flatbar_dict, exp_dict = makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                                                                    PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                                                                    h1,h2,h_lim,alpha,alpha_lim)
    #========function dictionary
    # psf
    funcset_dict_psf0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict]}
    funcset_dict_psf1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [psf_dict]}
    # same center psf+sersic
    funcset_dict_psfser0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,psf_dict]}
    funcset_dict_psfser1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,psf_dict]}  
    # separate sersic
    funcset_dict_sersic0 = {'X0': [Xss0,Xsslim[0],Xsslim[1]], 'Y0': [Yss0,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict]}
    funcset_dict_sersic1 = {'X0': [Xss1,Xsslim[0],Xsslim[1]], 'Y0': [Yss1,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic1_dict]}
    funcset_dict_serser0 = {'X0': [Xss0,Xsslim[0],Xsslim[1]], 'Y0': [Yss0,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict,sersic1_dict]}
    funcset_dict_serser1 = {'X0': [Xss1,Xsslim[0],Xsslim[1]], 'Y0': [Yss1,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic_dict,sersic1_dict]}
    funcset_dict_serserser = {'X0': [midf,Xsslim[0],Xsslim[1]], 'Y0': [midf,Ysslim[0],Ysslim[1]], 
                   'function_list': [sersic2_dict,sersic_dict,sersic1_dict]}

    # exponential
    funcset_dict_serexp= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,exp_dict]}
    funcset_dict_serexppsf= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,exp_dict,psf_dict]}
    # try sub exp with sersic n=1
    funcset_dict_sersern1= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,sersic_n1_dict]}
    funcset_dict_sersern1psf= {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [sersic_dict,sersic_n1_dict,psf_dict]}
    
    # bar
    funcset_dict_bar0 = {'X0': [Xss0,Xlim[0],Xlim[1]], 'Y0': [Yss0, Ylim[0],Ylim[1]], 
                    'function_list': [flatbar_dict]}
    funcset_dict_barser = {'X0': [midf,Xlim[0],Xlim[1]], 'Y0': [midf, Ylim[0],Ylim[1]], 
                    'function_list': [flatbar_dict,sersic_dict]}
    
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
        "bar+sersic":[funcset_dict_barser],
        "bar,sersic":[funcset_dict_bar0, funcset_dict_sersic1]
    }
    return funcset


def galaxy_model(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim, Xsslim, Ysslim,
                PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                midf, h1,h2,h_lim,alpha,alpha_lim):
    """return a dictionary of galaxy model with keys as model name"""
    funcset = galaxy_funcdict(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, 
                                Xlim, Ylim, Xsslim, Ysslim,
                                PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                                PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                                midf, h1,h2,h_lim,alpha,alpha_lim);
    models = {}
    for model in funcset:
        models[model]= pyimfit.ModelDescription.dict_to_ModelDescription({'function_sets':funcset[model]})
    return models


def get_dofit_val(objectName):
    """get image parameters for fitting"""
    mosfile = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+objectName+"*.mos.fits"))[0]
    expfile = glob.glob(os.path.expanduser("~/raw-data-agn/exp-fits-agn/*"+objectName+"*.exp.fits"))[0]
    with fits.open(mosfile) as hdul:
        hdu0 = hdul[0]
    sky_level = hdu0.header['BACKGND'] #[e-/s] native pixels, value should be in the same units as the data pixels
    with fits.open(expfile) as hdul:
        hdu = hdul[0]
    gain = hdu.header['EGAIN'] #[e-/DU] in header
    exptime= hdu.header['EXPOSURE'] # actual exp time
    noise = hdu.header['EFFRN'] #[e-] in header
    numcom = hdu.header['NCOADD'] #Number of Averaged Frames   
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
    for modelName in tqdm(models, desc="Fitting models"):
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


def save_data(image,models,configs,modelIms,fitResults,pnames,objectName):
    savedata = {}
    savedata['imageSS'] = image
    savedata['modelNames'] = models
    savedata['configs'] = configs
    savedata['modelImage'] = modelIms
    savedata['fitResults'] = fitResults
    savedata['paramNames'] = pnames
    filename = os.path.join(args.outDir, objectName+".pkl")
    pickle.dump(savedata,open(os.path.expanduser(filename),"wb"))
    

if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to fit AGN cutouts
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="~/research-data/agn-result/fit/final_fit_nb/", help="path to cut out directory")
    parser.add_argument("--oname", type=str, help="object name")
    parser.add_argument("--outDir", type=str, default="~/research-data/agn-result/fit/final_fit", help="output directory") 
    # fit guess params
    parser.add_argument("--PA", type=float, default=200., help='guess position angle')
    parser.add_argument("--ELL", type=float, default=200., help='guess ellipticity')
    parser.add_argument("--RE", type=float, default=200., help='guess effective radius')
    parser.add_argument("--X0", type=float, help='guess 1st x pos')
    parser.add_argument("--Y0", type=float, help='guess 1st y pos')
    parser.add_argument("--X1", type=float, help='guess 2nd x pos')
    parser.add_argument("--Y1", type=float, help='guess 2nd y pos')
    # use this if loading .fits files
    parser.add_argument("--inFile", type=str, help="cutout file")
    parser.add_argument("--original", action="store_true", default='use this flag if fitting original, not sky subtracted image')
    parser.add_argument("--psfPath", type=str, default="~/research-data/psf-results/psf_pkls", help="path to psf directory")
    parser.add_argument("--plotSkyHist", action="store_true")
    args = parser.parse_args()
    
    # load cutout file
    if args.inFile:
        cutoutPath = os.path.expanduser(args.inDir+args.inFile)
    else:
        cutoutPath = os.path.expanduser(args.inDir+args.oname+".pkl")
    if args.inFile and args.inFile[-4:] =="fits":
        imageAGN = fits.getdata(cutoutPath)
    else:
        with open(cutoutPath, 'rb') as f:
            d = pickle.load(f)
        imageAGN = d['imageSS']
    # load psf file
    psf_fileName = "psf_"+args.oname+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (os.path.expanduser(psfPath), "rb") as f:
        d = pickle.load(f)
    epsf = d['psf'].data
    # get do fit params
    exptime, noise, sky_level,numcom,gain = get_dofit_val(args.oname)
    # find centers for initial guess
    ys,xs = find_highest_indices(imageAGN)
    Imax = imageAGN.max()
    framelim = imageAGN.shape[0]
    midF=framelim//2
    # find background level and subtract if using original image
    if args.original:
        sky = find_sky(imageAGN, args.oname, args)
        imageAGN_bs = imageAGN-sky
    else:
        imageAGN_bs = imageAGN
    # make models
    if args.X1:
        if args.X0:
            xs[0] = args.X0
            ys[0] = args.Y0
        else:
            xs[0] = midF
            ys[0] = midF
            xs[1] = args.X1
            ys[1] = args.Y1
    models_n1 = galaxy_model(X0=xs[0], Y0=ys[0], 
                        X1=xs[1], Y1=ys[1], 
                        Xss0=xs[0], Yss0=ys[0], 
                        Xss1=xs[1], Yss1=ys[1],
                        Xlim=[0,framelim], Ylim=[0,framelim], Xsslim = [0,framelim], Ysslim=[0,framelim],
                        PA_ss=args.PA, ell_ss=args.ELL, n_ss=1, I_ss=1, r_ss=args.RE, Itot=1500,
                        PA_lim=[0,360], ell_lim=[0.0,1.0],
                        Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
                        midf=midF, 
                        h1=10,h2=10,h_lim=[0.1,10000],alpha=0.1,alpha_lim=[0.1,framelim])

    # fit and save results
    configs, modelIms, fitResults, pnames = fit_multi(models_n1, epsf, imageAGN_bs,noise,exptime, sky_level,numcom,gain)
    save_data(imageAGN_bs,models_n1,configs,modelIms,fitResults,pnames,args.oname)
    print('Done: ', args.oname)