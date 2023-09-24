import emcee
import numpy as np
import pyimfit
import pathlib
import time
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits
import tqdm

plt.rcParams['image.cmap'] = 'Blues'
plt.rcParams['font.family'] = 'monospace'


def getFits(args):
    """get fitters and best fit values"""
    path = pathlib.Path.joinpath(pathlib.Path(args.inDir), args.fitFile)
    with open(path, 'rb') as file:
        d = pickle.load(file)
    # get model to create fitters
    model = d['fitConfig']
    bestfits = d['fitResult'].params
    
    epsf = fits.getdata(args.psfFile)
    imageAGN = fits.getdata(args.imageFile)
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]

    imfit_fitter = pyimfit.Imfit(model,psf=epsf)
    imfit_fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
    return imfit_fitter, bestfits


def get_rm_inds(fitter):
    names = fitter.numberedParameterNames
    rm_inds = [i for i, name in zip(range(len(names)),names) if "n_" in name]
    return rm_inds


def lnPrior_func(params,imfitter,rmind):
    parameterLimits = imfitter.getParameterLimits()
    parameterLimits = [element for indx, element in enumerate(parameterLimits) if indx not in rmind]
    parameterLimits = [(0,100000) if e is None else e for e in parameterLimits]
    nParams = len(params)
    for i in range(nParams):
        if params[i] < parameterLimits[i][0] or params[i] > parameterLimits[i][1]:
            return  -np.inf
    return 0.0


def lnPosterior_pf(params, imfitter, lnPrior_func, rmInd, insInd):
    lnPrior = lnPrior_func(params, imfitter, rmInd)
    if not np.isfinite(lnPrior):
        return -np.inf
    params = np.insert(params,insInd,1)
    
    lnLikelihood = -0.5 * imfitter.computeFitStatistic(params)
    return lnPrior + lnLikelihood


def run_emcee(args,p_bestfit, fitter,rmInd,insInd):
    p_bestfit = np.delete(p_bestfit, rmInd)
    ndims, nwalkers = len(p_bestfit), 50
    initial_pos = [p_bestfit + 0.001*np.random.randn(ndims) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndims, lnPosterior_pf, args=(fitter, lnPrior_func, rmInd, insInd))
    sampler.reset()
    final_state = sampler.run_mcmc(initial_pos,args.numsteps,progress=True)
    return sampler


def saveChain(args, sampler):
    attributes_and_methods = {}
    atts = ['acceptance_fraction','chain','flatchain', 'flatlnprobability', 'lnprobability']
    for attr in atts:
        attributes_and_methods[attr] = getattr(sampler, attr)
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), args.chainFile)
    with open(save_path, 'wb') as file:
        pickle.dump(attributes_and_methods,file)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="fitResults", help="name of fit result directory")
    parser.add_argument("--psfFile", type=str, default="../psfConstruction/epsf2.fits", help="psf image file name")
    parser.add_argument("--imageFile", type=str, default="agn.fits",help="AGN image file name")
    parser.add_argument("--fitFile", type=str, default = "J1215+1344_n1m1_fit.pkl", help="name of best fit result file")
    parser.add_argument("--numsteps", type=int, default=3, help="number of steps in chain")
    parser.add_argument("--outdir", type=str,default="chainResults",help="chain directory")
    parser.add_argument("--chainFile", type=str,help="chain file name")
    #parser.add_argument("--modelNum", type=int, help="model number, 0-3")
    args = parser.parse_args()
    
    # get fitters and bestfits
    fitter, bestfits = getFits(args)
    rm_inds = get_rm_inds(fitter)
   
    if len(rm_inds) == 2:
        ins_inds = [rm_inds[0], rm_inds[1]-1]
    else:
        ins_inds = rm_inds 
    # run mcmc
    sampler = run_emcee(args, bestfits,fitter,rm_inds, ins_inds)
    # save chain
    saveChain(args, sampler)
    
    
    
    
    
    
