import numpy as np
import pickle
import pathlib
import pyimfit
from astropy.io import fits
import corner
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'Blues'
plt.rcParams['font.family'] = 'monospace'

def Prior_func(params,imfitter,rmind):
    """Calculate prior for some parameters, assuming uniform prior
       return prior value"""
    parameterLimits = imfitter.getParameterLimits()
    parameterLimits = [element for indx, element in enumerate(parameterLimits) if indx not in rmind]
    parameterLimits = [(0,100000) if e is None else e for e in parameterLimits]
    nParams = len(params)
    for i in range(nParams):
        if params[i] < parameterLimits[i][0] or params[i] > parameterLimits[i][1]:
            return  0
    return 1


def Posterior_pf(params, imfitter, Prior_func, rmInd, insInd):
    """Calculate posterior for some parameters, using pyimfit to eval likelihood
       return posterior value"""
    prior = Prior_func(params, imfitter, rmInd)
    if not np.isfinite(prior):
        return 0
    params = np.insert(params,insInd,1)
    likelihood = np.exp(-0.5 * imfitter.computeFitStatistic(params))
    return prior*likelihood


def plot_bestFit(fitter,params, imageAGN):
    """plot and save bestfit images"""
    fig, ax = plt.subplots(1, 2,figsize=(8,4))
    im = fitter.getModelImage(newParameters=params)
    im0 = ax[0].imshow(im)
    ax[0].set_title("MCMC model")

    im1 = ax[1].imshow(imageAGN-im)
    ax[1].set_title("residual")

    for j, image in zip(np.arange(2), [im0,im1]):
        cbar = fig.colorbar(image, ax=ax[j], shrink=0.6)

    fig.suptitle("AGN model image and residuals from MCMC parameters, n="+str(args.modelnum))
    fig.tight_layout()
    save_path = pathlib.Path.joinpath(pathlib.Path("chainResults"), args.object+"_resi_"+str(args.modelnum)+".jpg")
    fig.savefig(save_path)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--chaindir", type=str,default="../chain/chainResults",help="chain directory")
    parser.add_argument("--fitdir", type=str,default="../fit/fitResults",help="fit directory")
    parser.add_argument("--object", type=str, default = "J1215+1344", help="name of observed system")
    
    parser.add_argument("--chainfile", type=str,help="chain file name")
    parser.add_argument("--fitfile", type=str,help="fit file name")
    parser.add_argument("--cutoff", type=int,help="cut off steps for convergence")
    parser.add_argument("--modelnum", type=int)
    args = parser.parse_args()
    
    #load chain
    chainpath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), args.chainfile)
    with open(chainpath, "rb") as file:
        d = pickle.load(file)   
        
    # get converged chain and flatten
    converged_sample = d['chain'][:,args.cutoff:,:]
    s = converged_sample.shape
    c = np.reshape(converged_sample, (s[0]*s[1], s[2]))
    
    # get posterior 
    pct = [np.percentile(c[:, i], [16, 50, 84]) for i in range(s[2])]
    q = np.diff(pct)
    
    
    # put back n, get posterior p-50th
    rm_ind = [[5],[5,13]]
    insert_ind = [[5],[5,12]]
    params = [np.percentile(c[:,i],50) for i in range(s[2])]
    paramsN = np.insert(params,insert_ind[args.modelnum],1)
    
    
    # CHECK: reconstruct fitter to calculate residue and fit results
    fitpath = pathlib.Path.joinpath(pathlib.Path(args.fitdir), args.fitfile)
    with open(fitpath, 'rb') as file:
        f = pickle.load(file)
    model = f['fitConfig']
    bestfits = f['fitResult']['params']
    epsf = fits.getdata("../psfConstruction/psfResults/epsf.fits")
    imageAGN = fits.getdata("../fit/agn.fits")
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]
    fitter = pyimfit.Imfit(model,psf=epsf)
    fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
    
    #calcualte BIC
    ml=fitter.computeFitStatistic(paramsN)
    ndims = len(params)
    bic = ndims*np.log(100*100)+ml
    
    # component to calculate bayes factor
    prior = Prior_func(params,fitter,rm_ind[args.modelnum])
    posterior = Posterior_pf(params, fitter, Prior_func, rm_ind[args.modelnum], insert_ind[args.modelnum])
    bayesf = prior*posterior
    
    # save model image and residual
    plot_bestFit(fitter,paramsN, imageAGN)
    
    
    # save posterior parameter, BIC, bayes calc
    data_save = {}
    data_save['paramNames'] = [element for indx, element in enumerate(fitter.numberedParameterNames) if indx not in rm_ind[args.modelnum]]
    data_save['params'] = params
    data_save['err'] = q
    data_save['fitStat'] = ml
    data_save['BIC'] = bic
    data_save['Bayes factor'] = [prior, posterior]
                 
    save_path = pathlib.Path.joinpath(pathlib.Path("chainResults"), args.object+"_MCstat_"+str(args.modelnum)+".pkl")
    pickle.dump(data_save, open(save_path, 'wb'))
    
    #corner plot and save
    #savepath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), "corner_"+args.object+"_"+str(args.modelnum)+".jpg")
    paramNames = np.delete(fitter.numberedParameterNames, rm_ind[args.modelnum])
    #fig = corner.corner(c, truths=np.delete(bestfits, rm_ind[args.modelnum]), labels=l)
    #fig.savefig(savepath)
