import numpy as np
import pickle
import pathlib
import pyimfit
from astropy.io import fits
import corner
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'Blues'
plt.rcParams['font.family'] = 'monospace'


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--chaindir", type=str,default="chainResults",help="chain directory")
    parser.add_argument("--fitdir", type=str,default="fitResults",help="fit directory")
    parser.add_argument("--object", type=str, default = "J1215+1344", help="name of observed system")
    
    parser.add_argument("--chainfile", type=str,help="chain file name")
    parser.add_argument("--fitfile", type=str,help="fit file name")
    parser.add_argument("--cutoff", type=int,help="cut off steps for convergence")
    parser.add_argument("--modelnum", type=int)
    
    args = parser.parse_args()
    
    #load chain and reshape
    chainpath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), args.chainfile)
    with open(chainpath, "rb") as file:
        d = pickle.load(file)    
    converged_sample = d['chain'][:,args.cutoff:,:]
    s = converged_sample.shape
    c = np.reshape(converged_sample, (s[0]*s[1], s[2]))
    rm_ind = [5,[5,13]]
    insert_ind = [5,[5,12]]
    params = np.insert([np.percentile(c[:,i],50) for i in range(s[2])],insert_ind[args.modelnum],1)
    
    
    #get fit infos
    fitpath = pathlib.Path.joinpath(pathlib.Path(args.fitdir), args.fitfile)
    with open(fitpath, 'rb') as file:
        f = pickle.load(file)
    model = f['fitConfig']
    bestfits = f['fitResult']['params']
    epsf = fits.getdata("../psfConstruction/epsf2.fits")
    imageAGN = fits.getdata("agn.fits")
    psfOsamp = pyimfit.MakePsfOversampler(epsf, 4, (0,100,0,100))
    osampleList = [psfOsamp]
    fitter = pyimfit.Imfit(model,psf=epsf)
    fitter.loadData(imageAGN, psf_oversampling_list=osampleList, gain=9.942e-1, read_noise=0.22, original_sky=15.683)
    l = np.delete(fitter.numberedParameterNames, rm_ind[args.modelnum])
    
    ml =fitter.computeFitStatistic(params)
    ndims = params.shape[0]
    print(f"BIC: {ndims*np.log(100*100)+ml:.4f}")
    
    savepath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), "corner_"+args.object+"_"+str(args.modelnum)+".jpg")
    fig = corner.corner(c, truths=np.delete(bestfits, rm_ind[args.modelnum]), labels=l)
    fig.savefig(savepath)
