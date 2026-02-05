import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pickle
import glob
from matplotlib.backends.backend_pdf import PdfPages

from photutils.detection import find_peaks
from astropy.visualization import simple_norm
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder


# plot settings
plt.rcParams['font.family'] = 'monospace'
medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def find_peaks_remove_dups(data,Imin):
    """Find peaks with pixel value > Imin
       remove duplicate values of x_peak
       find indices of detections that are too close/erred
       Return peak table with no duplicates and list of indices of close detections"""
    # FIND PEAKS
    peaks_tbl = find_peaks(data, threshold=Imin)   
    
    # Cleaning
    # search and remove sources with same x values, keeping only first finds
    df = peaks_tbl.to_pandas().sort_values(by='x_peak')
    dups = df.duplicated(subset="x_peak",keep="first")
    nodups = df[~dups]
    # list to remove sources with separation in x and y < 10 pixels 
    indices_to_drop = []
    for i in range(len(nodups)-1):
        if abs(nodups.iloc[i+1]['x_peak'] - nodups.iloc[i]['x_peak']) < 10 and abs(nodups.iloc[i+1]['y_peak'] - nodups.iloc[i]['y_peak']) < 10:
            indices_to_drop.append(nodups.iloc[i+1].name)
            indices_to_drop.append(nodups.iloc[i].name)

    indices_to_drop=list((set(indices_to_drop)))
    return nodups, indices_to_drop
    

def plot_erred_star_peaks(nodups,indices_to_drop,objectName,pdf):
    """save plot of duplicates detections that turns out to be erred stars"""
    nodups.loc[indices_to_drop].sort_values('x_peak')
    ncols = 6
    nrows = int(np.ceil(len(indices_to_drop)/ncols))
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*3))
    ax = ax.ravel()
    for i in range(len(indices_to_drop)):
        y=int(nodups.loc[indices_to_drop[i]]['y_peak'])
        x=int(nodups.loc[indices_to_drop[i]]['x_peak'])
        px=30
        ax[i].imshow(data[y-px:y+px,x-px:x+px], origin='lower', cmap='viridis')
        ax[i].set_title(f"{x},{y}",fontsize=10)
    [ax[-i].axis('off') for i in range(1,len(ax)-len(indices_to_drop)+1)]
    fig.suptitle("Erred and duplicate detections of peaks",y=1.0)
    fig.tight_layout()
    pdf.savefig(bbox_inches="tight")
    plt.close()

    
def drop_erred_peaks(nodups, indices_to_drop):
    """drop erred indices from indice_to_drop from nodups,  returns table of sources position and pixel counts"""
    peaks_tbl = nodups.drop(indices_to_drop)
    return peaks_tbl


def plot_stars(stars,pdf):
    """plotting a table of star stamps"""
    nrows = int(np.ceil(len(stars)/4))
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    ax = ax.ravel()
    for i in range(len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        im = ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        ax[i].set_title(i)
        colorbar = fig.colorbar(im, ax=ax[i], shrink=0.8)
        colorbar.ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    # turn off empty axes
    empty_axes = nrows*ncols-len(stars)
    [ax[-i].axis('off') for i in np.arange(1,empty_axes+1)]
    fig.suptitle("Star ensemble for ePSF construction",y=1)
    fig.tight_layout()
    if args.firstPass:
        pdf.savefig(bbox_inches="tight")
    else:
        fig.savefig(pdf)


def make_star_cutout(peaks_tbl, args):
    """make star cutouts from peaks_tbl"""
    # size of a star cutout
    size = args.size
    hsize = (size - 1) / 2
    x = peaks_tbl['x_peak']  
    y = peaks_tbl['y_peak']  
    # remove stars that are too close to edges
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] -1 - hsize))) 
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]  
    stars_tbl['y'] = y[mask]
    # extract stamps
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=args.size)
    return stars_tbl, stars


def drop_star_stamps(stars_tbl,remove):
    remove = [int(r) for r in remove]
    stars_tbl.remove_rows(remove)
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=35)
    return stars


def build_psf(stars,osamp,shp,k='quartic',norm_r=20,maxit=30):
    """build epsf from star ensemble"""
    epsf_builder = EPSFBuilder(oversampling=osamp,shape=shp,smoothing_kernel=k,norm_radius=norm_r,maxiters=maxit) 
    epsf, fitted_stars = epsf_builder(stars) 
    return epsf,fitted_stars


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure
        1. to do a first pass, i.e. just detect sources
                python makepsf.py --oname [obj name] --firstPass --outDir_starini [path to initially-selected star plots]
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--oname", type=str)
    parser.add_argument("--mosPath", help='optionalpath to dir with mos.fits file, used for re-reduction files')
    parser.add_argument("--firstPass", action='store_true')
    parser.add_argument("--outDir_starini", type=str, help='output directory to save plots of initially-selected stars')
    """optionally can change point source detection params"""
    parser.add_argument("--threshold", type=float, default=400.0, help="intensity threshold for peak detection")
    parser.add_argument("--size", type=int, default=35, help="size of star cutout")
    """2. for a second pass, remove bad stars after visual inspection and save star plot.
            bad stars' indices are in remove.txt
                python makepsf.py --oname [objname] --outDir_starpost [dir]
            After inspecting the post-selected star, can make psf by
                python makepsf.py --oname [objname] --outDir_PSF [dir]"""
    parser.add_argument("--outDir_starpost", type=str, help='output directory to save plots of final star ensemble')
    parser.add_argument("--makePSF", action='store_true')
    parser.add_argument("--outDir_PSF", type=str, help='output directory to save the PSF')
    """3. can also save a star cutout as a PSF. look at it first by
                python makepsf.py --plotStarPSF --starnum [#]
            then save by
                python makepsf.py --saveStarPSF --starnum [#] --outDir_PSF [dir]"""
    parser.add_argument("--starnum", type=int, help='star number to save as PSF')
    parser.add_argument("--plotStarPSF", action='store_true', help="option to plot a star")
    parser.add_argument("--saveStarPSF", action='store_true', help="option to save the star")
    args = parser.parse_args()

    # get array of source index to remove
    with open("remove.txt", "r") as f:
        d = f.read().splitlines()   
    r_dict = {}
    for dd in d:
        k,v = dd.split("__")
        r_dict[k]= v

    # read exposure data
    if args.mosPath:
        expPath = glob.glob(os.path.expanduser(args.mosPath+"/*"+args.oname+"*.mos.fits"))[0]
    else:
        expPath = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+args.oname+"*.mos.fits"))[0]
    data = fits.getdata(expPath)
    # find peaks and remove duplicates, mark erred detections
    nodups, indices_to_drop = find_peaks_remove_dups(data,args.threshold)
    # drop erred detections 
    peaks_tbl = drop_erred_peaks(nodups, indices_to_drop)
    # make star stamps
    stars_tbl, stars = make_star_cutout(peaks_tbl,args)
    # plot star stamps
    if args.firstPass: #first pass before visually inspecting and removing "bad" sources
        # pdf to save both erred peaks and detected sources
        with PdfPages(os.path.join(args.outDir_starini,args.oname+"_ini_.pdf")) as pdf:
            # save the auto-detected erred peaks for sanity check
            plot_erred_star_peaks(nodups,indices_to_drop,args.oname,pdf)
            # save star stamps
            plot_stars(stars,pdf)
        print("Done first pass: ", args.oname)
    else: #second pass, removing star with index from remove.txt
        stars = drop_star_stamps(stars_tbl,r_dict[args.oname].split(" "))
        if args.makePSF: # option to make the PSF
            data_to_save = {}
            data_to_save['stars'] = stars
            data_to_save['psf'], data_to_save['fitted_stars'] = build_psf(stars,1,shp=(args.size,args.size),k='quartic')
            psfPath = open(os.path.join(args.outDir_PSF,f"psf_"+args.oname+".pkl"),"wb")
            pickle.dump(data_to_save, psfPath)
            print("Done make psf: ", args.oname)
            plt.imshow(data_to_save['psf'].data)
            plt.show()
        elif args.plotStarPSF: # extract a star stamp and plot
            norm = simple_norm(stars[args.starnum], 'log', percent=99.0)
            plt.imshow(stars[args.starnum], norm=norm, origin='lower', cmap='viridis')
            plt.colorbar()
            plt.show()
        elif args.saveStarPSF: # save star stamp as psf
            pickle.dump(stars[args.starnum], open(os.path.join(args.outDir_PSF,f"psf_{args.oname}_star{args.starnum}.pkl"),"wb"))
            print(f"Done saving star {args.starnum} as psf: ", args.oname)
        else: # only remove stars and plot post-selected ones
            starPath = open(os.path.join(args.outDir_starpost,f"{args.oname}_post_.jpg"),"wb")
            plot_stars(stars,starPath)
            print("Done post star: ", args.oname)
        
            
    
