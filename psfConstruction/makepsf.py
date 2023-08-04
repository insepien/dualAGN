import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import emcee 
import pyimfit 
from astropy.io import fits
import corner
from IPython.display import Latex
import sys
from skimage.transform import resize

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from photutils import profiles

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['image.cmap'] = 'gray'


def find_peak_tbl(data, args):
    peaks_tbl = find_peaks(data, threshold=args.threshold)
    # remove duplicate coordinates
    df = peaks_tbl.to_pandas().sort_values(by='x_peak')
    dups = df.duplicated(subset="x_peak",keep="first")
    nodups = df[~dups]
    #elimiating overlapping sources
    proximity_threshold = 20
    tbl = nodups.groupby(nodups['x_peak'] // proximity_threshold * proximity_threshold).apply(lambda group: group.loc[group['peak_value'].idxmax()])
    return tbl


def get_star_list(data, peaks_tbl, args):
    hsize = (args.size - 1)/2
    x = peaks_tbl['x_peak']
    y = peaks_tbl['y_peak']
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] -1 - hsize)))
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]  
    stars_tbl['y'] = y[mask]  
    return stars_tbl

def remove_stars(stars_tbl, args):
    stars_tbl.remove_rows(args.remove)
    return stars_tbl

def get_star_stamp(data, stars_tbl, args):
    #subtract background
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
    data -= median_val  
    #extract star stamps
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=args.size) 
    return stars

def plot_stars(stars,args):
    ncols=5
    nrows=len(stars)//ncols+1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),squeeze=True)
    ax = ax.ravel()
    for i in range(len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        ax[i].set_title(i)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[i])
    if args.remove:
        fig.savefig("stars_filtered_"+args.object+".jpeg")
    else:
        fig.savefig("stars_"+args.object+".jpeg")


def plot_radials(args, stars, epsf_resized):    
    # plot normalized radial profiles of stars
    fig = plt.figure()
    for i in range(len(stars)):
        rp_star = profiles.RadialProfile(stars[i].data,xycen=stars[i].cutout_center,radii=np.arange(70))
        rp_star.normalize("max")
        plt.plot(rp_star.radius, rp_star.profile, color='darkseagreen')
    # plot normalized radial profiles of psf
    cen = args.size//2
    rp_psf = profiles.RadialProfile(epsf_resized,xycen=(cen,cen),radii = np.arange(70))
    rp_psf.normalize("max")
    plt.plot(rp_psf.radius, rp_psf.profile,"b")
    
    # cosmetics
    legend_handles = [plt.Line2D([], [], color='darkseagreen', label='stars'),plt.Line2D([], [], color='b', label='PSF')]
    plt.legend(handles=legend_handles)
    plt.title("Radial profile of effective PSF and Stars")
    fig.savefig("rp_"+args.object+".jpeg")


def makePSF(args):
    # getting exposure data
    data = fits.getdata(args.sourceFile)
    m, s = np.mean(data), np.std(data)
    # find stars and get their stamps
    tbl = find_peak_tbl(data, args)
    stars_tbl = get_star_list(data, tbl, args)
    if args.remove:
        stars_tbl = remove_stars(stars_tbl, args)
    stars = get_star_stamp(data, stars_tbl, args)
    plot_stars(stars, args)

    #construct psf from stars
    epsf_builder = EPSFBuilder()
    epsf, fitted_stars = epsf_builder(stars)
    
    #saving epsf to fits file
    hdu = fits.PrimaryHDU(epsf.data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(args.outfile, overwrite=True)

    #plot radial profiles
    epsf_resized = resize(epsf.data, (args.size, args.size))
    plot_radials(args, stars, epsf_resized)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outfile", type=str, default="epsf.fits", help="psf output file")
    parser.add_argument("--sourceFile", type=str, help="exposure file")
    parser.add_argument("--threshold", type=float, default=400.0, help="brightness threshold for stars used to construct PSF")
    parser.add_argument("--size", type=int, default=35, help="size of star cutout")
    parser.add_argument("--remove", nargs="+", type=int, help="list of stars to remove")
    parser.add_argument("--object", type=str, default="J0000+0000", help="name of object")
    #parser.add_argument("--starplot", type=str, help="name of star plots file")
    args = parser.parse_args()
    makePSF(args)
