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
    models_n1 = d['fitConfig_n1']
    models_n4 = d['fitConfig_n4']
    
    epsf = fits.getdata(args.psfFile)
    imageAGN = fits.getdata(args.imageFile)
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]
    fitters_n1, fitters_n4 =[], []
    for fitters, models in zip([fitters_n1, fitters_n4],[models_n1,models_n4]):
        for i in tqdm.tqdm(range(len(models)), desc="Fitting Models"):
            imfit_fitter = pyimfit.Imfit(models[i],psf=epsf)
            imfit_fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
            fitters.append(imfit_fitter)
    return fitters_n1, fitters_n4, d['bestfit_n1'], d['bestfit_n4']


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


def lnPosterior_pf(params, imfitter, lnPrior_func, rmInd):
    lnPrior = lnPrior_func(params, imfitter, rmInd)
    if not np.isfinite(lnPrior):
        return -np.inf
    params = np.insert(params,rmInd,1)
    
    lnLikelihood = -0.5 * imfitter.computeFitStatistic(params)
    return lnPrior + lnLikelihood


def run_emcee(args,p_bestfit, fitter,rmInd):
    p_bestfit = np.delete(p_bestfit, rmInd)
    ndims, nwalkers = len(p_bestfit), 50
    initial_pos = [p_bestfit + 0.001*np.random.randn(ndims) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndims, lnPosterior_pf, args=(fitter, lnPrior_func, rmInd))
    sampler.reset()
    final_state = sampler.run_mcmc(initial_pos,args.numsteps,progress=True)
    return sampler


def saveChain(args, sampler):
    attributes_and_methods = {}
    atts = ['acceptance_fraction','chain','flatchain', 'flatlnprobability', 'lnprobability']
    for attr in atts:
        attributes_and_methods[attr] = getattr(sampler, attr)
    with open(args.chainFile, 'wb') as file:
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
    parser.add_argument("--fitFile", type=str, default = "J1215+1344_fit_.pkl", help="name of best fit result file")
    parser.add_argument("--numsteps", type=int, help="number of steps in chain")
    parser.add_argument("--chainFile", type=str,help="chain file name")
    parser.add_argument("--modelNum", type=int, help="model number, 0-3")
    args = parser.parse_args()
    
    # get fitters and bestfits
    fitters_n1, fitters_n4, bestfits_n1, bestfits_n4 = getFits(args)
    # get n indices
    rm_inds = [get_rm_inds(fitter) for fitter in fitters_n1]
    # run mcmc
    sampler = run_emcee(args, bestfits_n1[args.modelNum],fitters_n1[args.modelNum],rm_inds[args.modelNum])
    # save chain
    saveChain(args, sampler)
    
    
    
    
    
    
