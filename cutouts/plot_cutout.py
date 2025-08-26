import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import os
import numpy as np

plt.rcParams['font.family'] = 'monospace'
medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def plot_all(fileNames):
    n = len(fileNames)
    ncols=6
    nrows=int(np.ceil(n/6))
    fig,ax = plt.subplots(nrows,ncols,figsize=(nrows*3,ncols*3))
    ax = ax.ravel()
    for i in range(n):
        dpath = os.path.join(args.inDir,fileNames[i])
        ax[i].imshow(fits.getdata(dpath),norm='symlog')
        ax[i].set_title(fileNames[i][:10])
        # w = WCS(dpath)
        # ra,dec = w.all_pix2world(np.arange(100),np.arange(100),0)
        # xlabel = np.round(np.linspace(ra[0],ra[-1],3),2)
        # ylabel = np.round(np.linspace(dec[0],dec[-1],3),2)
        # ax[i].set_xticks([np.linspace(0, 99, 3)[1]], [f"{xlabel[1]}°"])
        # ax[i].set_yticks([np.linspace(0, 99, 3)[1]], [f"{ylabel[1]}°"])
    [ax[-i].axis("off") for i in range(1,len(ax)-n+1)]
    fig.tight_layout()
    return fig
    

if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outFile", type=str, help="file name for all object plot")
    parser.add_argument("--inDir", type=str)
    args = parser.parse_args()
    fileNames = os.listdir(args.inDir)
    objnames = [f for f in fileNames if f[0] =="J"]
    fig = plot_all(objnames)
    fig.savefig(args.outFile)