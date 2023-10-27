import numpy as np 
import matplotlib.pyplot as plt
import pyimfit 
from astropy.io import fits
import tqdm
import pickle
import pathlib

plt.rcParams['image.cmap'] = 'Blues'
plt.rcParams['font.family'] = 'monospace'


def find_highest_indices(arr):
    """find the pixel with highest intensity
       return indices"""
    flattened_arr = np.array(arr).flatten()
    max_indices = np.unravel_index(np.argsort(flattened_arr)[-2:], arr.shape)
    return max_indices


def galaxy_model(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim,
                 PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim, sigma_lim, Iss_lim, rss_lim):
    """create galaxy models with 1/2 centers
       including: ps: 1 psf + 1 sersic
                  psps: 2 psf + 2 sersic
       return galaxy models"""
    # make model dictionaries
    sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 'fixed'],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}
    sersic_dict = {'name': "Sersic", 'label': "bulge", 'parameters': sersic}
    psf = {'I_tot' : Itot}
    psf_dict = {'name': "PointSource", 'label': "psf", 'parameters': psf}
    
    # initiate psf and sersics model functions
    funcset_dict_psf0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]],
                    'function_list': [psf_dict]}
    funcset_dict_psf1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]],
                    'function_list': [psf_dict]}
    funcset_dict_sersic0 = {'X0': [Xss0,Xlim[0],Xlim[1]], 'Y0': [Yss0,Ylim[0],Ylim[1]],
                   'function_list': [sersic_dict]}
    funcset_dict_sersic1 = {'X0': [Xss1,Xlim[0],Xlim[1]], 'Y0': [Yss1,Ylim[0],Ylim[1]],
                   'function_list': [sersic_dict]}

    funcset_dict_psfser0 = {'X0': [X0,Xlim[0],Xlim[1]], 'Y0': [Y0, Ylim[0],Ylim[1]],
                    'function_list': [psf_dict, sersic_dict]}
    funcset_dict_psfser1 = {'X0': [X1,Xlim[0],Xlim[1]], 'Y0': [Y1, Ylim[0],Ylim[1]],
                    'function_list': [psf_dict, sersic_dict]}

    # initiate models with same and separate centers
    model_dict_ps_1c = {'function_sets': [funcset_dict_psfser0]}
    model_dict_psps_1c = {'function_sets': [funcset_dict_psfser0,funcset_dict_psfser1]}
    model_dict_ps_2c = {'function_sets': [funcset_dict_psf0,funcset_dict_sersic0]}
    model_dict_psps_2c = {'function_sets': [funcset_dict_psf0, funcset_dict_psf1,funcset_dict_sersic0,funcset_dict_sersic1]}

    # build models
    model_ps1c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_ps_1c)
    model_psps1c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_psps_1c)
    model_ps2c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_ps_2c)
    model_psps2c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_psps_2c)
    return model_ps1c, model_psps1c, model_ps2c, model_psps2c


def buildModels(args, imageAGN, itot):
    """build models given limits and inital guesses for n=1 and n=4
       return models"""
    #find initial centroid and intensity guesses
    starpos = find_highest_indices(imageAGN)
    center = (starpos[0]+starpos[1])/2
    Imax = imageAGN.max()
    #create galaxy models with n=1
    models_n1 = galaxy_model(X0=starpos[0][0], Y0=starpos[0][1], X1=starpos[1][0], Y1=starpos[1][1], 
                                                     Xss0=starpos[0][0], Yss0=starpos[0][1], Xss1=starpos[1][0], Yss1=starpos[1][1],
                                                     Xlim=[0,100], Ylim=[0,100],
                                                     PA_ss=200, ell_ss=0.1, n_ss=1, I_ss=1, r_ss=20, Itot=itot,
                                                     PA_lim=[0,360], ell_lim=[0.0,1.0], I_lim=[0,Imax], sigma_lim=[0,15],
                                          Iss_lim=[0,Imax], rss_lim=[0,100])

    #create galaxy models with n=4
    models_n4 = galaxy_model(X0=starpos[0][0], Y0=starpos[0][1], X1=starpos[1][0], Y1=starpos[1][1], 
                                                     Xss0=starpos[0][0], Yss0=starpos[0][1], Xss1=starpos[1][0], Yss1=starpos[1][1],
                                                     Xlim=[0,100], Ylim=[0,100],
                                                     PA_ss=200, ell_ss=0.1, n_ss=4, I_ss=1, r_ss=20, Itot=itot,
                                                     PA_lim=[0,360], ell_lim=[0.0,1.0], I_lim=[0,Imax], sigma_lim=[0,15],
                                          Iss_lim=[0,Imax], rss_lim=[0,100])
    return models_n1, models_n4
 

def doFit(models,epsf,imageAGN): 
    """do fit with data using PSF oversampled by factor of 4 over whole image
       return fitters"""
    fitters = []
    # oversamped PSF
    for i in tqdm.tqdm(range(len(models)), desc="Fitting Models"):
        psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
        osampleList = [psfOsamp]
        imfit_fitter = pyimfit.Imfit(models[i],psf=epsf)
        imfit_fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
        imfit_fitter.doFit()
        fitters.append(imfit_fitter)
    return fitters


def plot_bestFit(fitters, imageAGN, title,suptit,kind):
    """plot and save bestfit images"""
    fig, ax = plt.subplots(len(fitters), 3,figsize=(12,16))
    for i in range(len(title)):
        im0 = ax[i,0].imshow(fitters[i].getModelImage())
        ax[i,0].set_title(title[i], fontweight='bold')

        im1 = ax[i,1].imshow(imageAGN-fitters[i].getModelImage())
        ax[i,1].set_title("residual")

        im2 = ax[i,2].imshow(imageAGN)
        ax[i,2].set_title("data")

        # Create a colorbar for each axis
        for j, im in zip(np.arange(3), [im0,im1,im2]):
            cbar = fig.colorbar(im, ax=ax[i,j], shrink=0.6)

        # Set the same colorbar scale for both axes
        vmin = min(im0.get_array().min(), im2.get_array().min())
        vmax = max(im0.get_array().max(), im2.get_array().max())
        [im.set_clim(vmin, vmax) for im in [im0,im2]]
    
    fig.suptitle(suptit)
    fig.tight_layout()
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), args.object+"_fitImage_"+kind+".jpg")
    fig.savefig(save_path)
   
    
    
    
if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to fit AGN image to galaxy models
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--imageFile", type=str, default="agn.fits",help="AGN image file name")
    parser.add_argument("--outdir", type=str, default='fitResults', help="directory for fit results")
    parser.add_argument("--object", type=str, help="name of observed system")
    parser.add_argument("--psfFile", type=str, default="epsf.fits", help="psf image file name")
    parser.add_argument("--twoMod", action="store_true")
    args = parser.parse_args()
    
    # load images
    imageAGN = fits.getdata(args.imageFile)
    epsf = fits.getdata("../psfConstruction/psfResults/"+args.psfFile)
    
    # build models and find best fits for n1
    models_n1, models_n4 = buildModels(args, imageAGN, itot = np.sum(epsf))
    fitters_n1 = doFit(models_n1,epsf,imageAGN)
    title_n1 = ["PSF+Sersic,n=1,same centers","2 PSF+Sersic,n=1,same centers",
         "PSF+Sersic,n=1,diff cenls ters","2 PSF+Sersic,n=1,diff centers"]
    plot_bestFit(fitters_n1, imageAGN, title_n1,"Fit results for n=1 using ePSF",'n1')
    
    # best fit for n4, can opt out if testing only n1
    if args.twoMod:
        fitters_n4 = doFit(models_n4,epsf,imageAGN)
        title_n4 = ["PSF+Sersic,n=4,same centers","2 PSF+Sersic,n=4,same centers",
         "PSF+Sersic,n=4,diff centers","2 PSF+Sersic,n=4,diff centers"]
        plot_bestFit(fitters_n4, imageAGN, title_n4,"Fit results for n=4 using ePSF",'n4')
    
    
    #save best fit values
    fitters = [fitters_n1]
    models = [models_n1]
    names = ['_n1']
    if args.twoMod:
        fitters.append(fitters_n4)
        models.append(models_n4)
        names.append('_n4')
        
    data_to_save = {}
    for fitters, models, kind in zip(fitters, models,names):
        data_to_save['bestfit'+kind] = [fitters[i].getFitResult() for i in range(4)]
        data_to_save['fitConfig'+kind] = [models[i] for i in range(4)]
        data_to_save['paramNames'] = [fitters[i].numberedParameterNames for i in range(4)]
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), args.object+"_fit.pkl")
    pickle.dump(data_to_save, open(save_path, 'wb'))
    

        
     
    
    
    
    

