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
    """find the pixel with highest intensity"""
    flattened_arr = np.array(arr).flatten()
    max_indices = np.unravel_index(np.argsort(flattened_arr)[-2:], arr.shape)
    return max_indices


def galaxy_model(X0, Y0, X1, Y1, Xss0, Yss0, Xss1, Yss1, Xlim, Ylim,
                 PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim, sigma_lim, Iss_lim, rss_lim):
    """create galaxy models with 1/2 centers
        including: ps: 1 psf + 1 sersic
                   psps: 2 psf + 2 sersic"""

    sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 'fixed'],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}
    sersic_dict = {'name': "Sersic", 'label': "bulge", 'parameters': sersic}

    psf = {'I_tot' : Itot}
    psf_dict = {'name': "PointSource", 'label': "psf", 'parameters': psf}

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


    model_dict_ps_1c = {'function_sets': [funcset_dict_psfser0]}
    model_dict_psps_1c = {'function_sets': [funcset_dict_psfser0,funcset_dict_psfser1]}

    model_dict_ps_2c = {'function_sets': [funcset_dict_psf0,funcset_dict_sersic0]}
    model_dict_psps_2c = {'function_sets': [funcset_dict_psf0, funcset_dict_psf1,funcset_dict_sersic0,funcset_dict_sersic1]}

    model_ps1c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_ps_1c)
    model_psps1c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_psps_1c)
    model_ps2c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_ps_2c)
    model_psps2c = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict_psps_2c)
    return model_ps1c, model_psps1c, model_ps2c, model_psps2c


def buildModels(args, imageAGN, itot):
    """build models with limits and inital guesses"""
    
    #find initial centroid and intensity guesses
    starpos = find_highest_indices(imageAGN)
    center = (starpos[0]+starpos[1])/2
    Imax = imageAGN.max()
    #initial psf intensity guess

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
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]
    for i in tqdm.tqdm(range(len(models)), desc="Fitting Models"):
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
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--imageFile", type=str, default="agn.fits", help="AGN image file name")
    parser.add_argument("--outdir", type=str, default='fitResults', help="directory for fit results")
    parser.add_argument("--object", type=str, default = "J1215+1344", help="name of observed system")
    parser.add_argument("--psfFile", type=str, default="../psfConstruction/epsf2.fits", help="psf image file name")
    parser.add_argument("--modelNum", type=int, default="1", help="model number, 0-3")
    args = parser.parse_args()
    
    # load images
    imageAGN = fits.getdata(args.imageFile)
    epsf = fits.getdata(args.psfFile)
    
    # build models and find best fits
    models_n1, models_n4 = buildModels(args, imageAGN, itot = 130)
    
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]
    
    # fit only 1 model
    model = models_n1[args.modelNum]
    fitter = pyimfit.Imfit(model,psf=epsf)
    fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
    fitter.doFit()
    print("finish fit")
    
    #save best fit values
    data_to_save={}
    data_to_save['fitResult'] = fitter.getFitResult()
    data_to_save['fitConfig'] = model
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), args.object+"_n1m"+str(args.modelNum)+"_fit.pkl")
    pickle.dump(data_to_save, open(save_path, 'wb'))
    print("finish save")
    
    
    
    
    
    
    

        
     
    
    
    
    

