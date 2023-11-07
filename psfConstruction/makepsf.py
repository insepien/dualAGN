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
import pathlib
from matplotlib.ticker import LogFormatter

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from photutils import profiles

#font settings
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['image.cmap'] = 'BuPu_r'
medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def find_peak_tbl(data, args):
    """find peaks, remove repeated/overlapping sources
       returns table of stars"""
    # find peaks in data
    peaks_tbl = find_peaks(data, threshold=args.threshold)
    # remove duplicate coordinates, keeping only 1 
    df = peaks_tbl.to_pandas().sort_values(by='x_peak')
    dups = df.duplicated(subset="x_peak",keep="first")
    nodups = df[~dups]
    # elimiating too close together sources
    proximity_threshold = 20
    tbl = nodups.groupby(nodups['x_peak'] // proximity_threshold * proximity_threshold).apply(lambda group: group.loc[group['peak_value'].idxmax()])
    return tbl


def get_star_list(data, peaks_tbl, args):
    """find stars from star table that don't lie too close to the image edge
       return star table"""
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
    """remove star using provided index list (on 2nd code run)
       return star table"""
    stars_tbl.remove_rows(args.remove)
    return stars_tbl

def get_star_stamp(data, stars_tbl, args):
    """subtract background from image and extract star stamps
       return star stamps"""
    #subtract background
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
    data -= median_val  
    #extract star stamps
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=args.size) 
    return stars

def plot_stars(stars,args):
    """plot star stamps and save figure"""
    # plotting
    ncols=5
    nrows=len(stars)//ncols+1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 17),squeeze=True,sharex=True)
    ax = ax.ravel()
    for i in range(len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        ax[i].set_title("Star "+str(i))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=ax[i], format=LogFormatter(labelOnlyBase=False), shrink=0.8)
        colorbar.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # turn of empty axis
    [ax[i].axis("off") if ax[i].has_data()==False else ax[i].axis("on") for i in range(len(ax))]
    # save figure
    if args.remove:
        fig.savefig("psfResults/stars_filtered_"+args.object+".jpeg")
    else:
        fig.savefig("psfResults/stars_"+args.object+".jpeg")

    
def plotPSF(args,epsf,epsf_resized):
    """plot original and resized psf"""
    fig, ax = plt.subplots(1,2)
    im1 = ax[0].imshow(epsf.data)
    im2 = ax[1].imshow(epsf_resized)
    for i in range(2):
        cbar = fig.colorbar([im1,im2][i], orientation="horizontal")
        cbar.set_label('Intensity')
        ax[i].set_title(['Original ePSF', 'Resized ePSF'][i])
        ax[i].set_xlabel("Pixels")
        ax[i].set_ylabel("Pixels")
    savepath = pathlib.Path.joinpath(pathlib.Path("psfResults"), args.psfImFile)
    fig.tight_layout()
    plt.savefig(savepath)
    
    
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
    legend_handles = [plt.Line2D([], [], color='darkseagreen', label='stars'),plt.Line2D([], [], color='b', label='resized PSF')]
    plt.legend(handles=legend_handles)
    plt.title("Radial profile of resized effective PSF and Stars")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Profile")
    fig.savefig("psfResults/rp_"+args.object+".jpeg")

    

def makePSF(args):
    # getting exposure data
    data = fits.getdata(args.sourceFile)
    # find star peaks
    tbl = find_peak_tbl(data, args)
    stars_tbl = get_star_list(data, tbl, args)
    # manually selecting some star
    if args.remove:
        stars_tbl = remove_stars(stars_tbl, args)
    # get star stamps
    stars = get_star_stamp(data, stars_tbl, args)
    # plot star stamps
    plot_stars(stars, args)

    #construct psf from stars
    epsf_builder = EPSFBuilder(maxiters=12, progress_bar=True)
    epsf, fitted_stars = epsf_builder(stars)
    
    #saving epsf to fits file
    hdu = fits.PrimaryHDU(epsf.data)
    hdulist = fits.HDUList([hdu])
    psfpath = pathlib.Path.joinpath(pathlib.Path("psfResults"), args.outfile)
    hdulist.writeto(psfpath, overwrite=True)

    #plot radial profiles
    epsf_resized = resize(epsf.data, (args.size, args.size))
    plot_radials(args, stars, epsf_resized)
    
    #plot psf
    plotPSF(args, epsf, epsf_resized)



if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outfile", type=str, default="epsf.fits", help="psf output file")
    parser.add_argument("--sourceFile", type=str, default = "2020-02-21_J_J1215+1344_c1-4_58900_32019.mos.fits", help="exposure file")
    parser.add_argument("--threshold", type=float, default=400.0, help="brightness threshold for stars used to construct PSF")
    parser.add_argument("--size", type=int, default=35, help="size of star cutout")
    parser.add_argument("--remove", nargs="+", type=int, help="list of stars to remove")
    parser.add_argument("--object", type=str, default="J0000+0000", help="name of object")
    parser.add_argument("--psfImFile", type=str, default="psf.png", help="name of psf image file")
    args = parser.parse_args()
    makePSF(args)
