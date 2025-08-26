from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
import pandas as pd
import numpy as np
import os
import glob
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
from photutils.utils import circular_footprint
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
import pickle

def save_cutout_to_fits(data, header,hdu,cutoutfile):
    # Put the cutout image in the FITS HDU
    hdu.data = data
    # Update the FITS header with the cutout WCS
    hdu.header.update(header)
    # Write the cutout to a new FITS file
    hdu.writeto(cutoutfile, overwrite=True)

def make_cutout(originalfile, position, size):
    """Returns a cutout of agn from original exposure given cutout size and target position"""
    # Load the image and the WCS
    hdu = fits.open(os.path.expanduser(originalfile))[0]
    wcs = WCS(hdu.header)
    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    return hdu, cutout.data, cutout.wcs.to_header()
    

def make_seg_im(data,npix_=10,nsigma_=2):
    """return segmentation image for source detection in data"""
    # sigma clip outliers in data
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    # define a threshold for source detection, here set at 2-sigma
    threshold = detect_threshold(data, nsigma=nsigma_, sigma_clip=sigma_clip)
    # detect sources using image segmentation
    segment_img = detect_sources(data, threshold, npixels=npix_)
    return segment_img

def photsky(data):
    """return the data shape, mean, median, stddev for a 2D array after sigma clip and mask ext sources"""
    segment_img = make_seg_im(data)
    # make circular mask
    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    # mask detected sources and calculate stats
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    return np.array([data.shape[0], mean, median, std])

def cal_background(on):
    """calculate sky background as the median of the background level at different cutout sizes"""
    # load exposure
    exp_path = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+on+"*.fits"))[0]
    # create cutout of size 100 to 1000
    cutouts = [make_cutout(exp_path, catalog['coords'].loc[on], (i,i))[1] for i in np.arange(1,11)*100]
    # calculate background stats for each cutout size
    res = [photsky(i) for i in cutouts]
    # sky lev is the median
    skylev = np.median(np.array(res)[:,-2])
    return skylev, np.array(res)

def plot_sky(skylev, res):
    """plot skylev v. cutout sizes for sanity check"""
    fig,ax = plt.subplots()
    [ax.scatter(r[0],r[-2],c='k') for r in res]
    ax.axhline(skylev,c='k',alpha=0.5,label=f'sky={skylev:.3f}')
    pvar = 0.4
    ax.set_ylim(skylev*(1+pvar), skylev*(1-pvar))
    ax.axhline(skylev*(1+0.2), label='+/- 20%',alpha=0.5)
    ax.axhline(skylev*(1-0.2),alpha=0.5)
    ax.set_xlabel("Size [pix]")
    ax.set_ylabel(f"Estimated background [count]\n(ylim = {pvar*100:.0f}% sky)")
    ax.set_title(on)
    ax.legend()
    fig.tight_layout()
    fig.savefig("/home/insepien/research-data/agn-result/box/skylev/"+on+"_sky.png")
    plt.close();


def mask_img(data,sigm):
    """return masked image given data. use segmenatation image to detect sources and mask all but the center target"""
    # create a copy of segim
    segment_img = sigm.copy() 
    # drop the center target 
    midf = data.shape[0]//2
    segment_img.keep_label([c for c in segment_img.labels if c!= segment_img.data[midf,midf]])
    # mask all sources in segim
    imageAGN = np.where(segment_img.make_source_mask()==0,data,0)
    return imageAGN

def make_masked_cutout(on,deblendpix=20,npix=10,nsigma=2,save=True):
    """return masked cutout whose size correspond to some physical size in kpc"""
    # convert kpc to pix for 4star
    kpctopix = lambda kpc,z: ((kpc*u.kpc/cosmo.angular_diameter_distance(z)).to("")*u.rad).to(u.arcsec).value/0.16
    objmask = magel['Desig'] == on
    # determine frame size corresponding to some physical size
    framesize_pix = np.ceil(kpctopix(args.framesize_kpc,magel[objmask]['Z'].values).max())
    # make cutout
    exp_path = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+on+"*.fits"))[0]
    hdu, data, header = make_cutout(exp_path, catalog['coords'].loc[on], framesize_pix)
    # detect sources
    segment_img = make_seg_im(data,nsigma_=nsigma,npix_=npix)
    # deblend sources that might be blended into central target
    deblend_img = deblend_sources(data, segment_img, deblendpix)
    # mask every source detected in cutout except for center source
    masked_im = mask_img(data,deblend_img)
    if save:
        # save to fits
        cutfile = os.path.join(args.outDir, on+f"_{framesize_pix:.0f}_masked_{args.framesize_kpc:.0f}kpc.fits")
        save_cutout_to_fits(masked_im, header, hdu, cutfile)
    else:
        plt.imshow(masked_im, norm='symlog')
        plt.show()
    return masked_im

    
if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to make cutout
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outDir", default="~/research-data/agn-result/box/kpcbox", type=str, help="output directory")
    # args for creating single cutout
    parser.add_argument("--oname", type=str, help="object name")
    parser.add_argument("--cutSize", type=int, help="fits cutout size")
    parser.add_argument("--RA", type=float, help='cutout center RA')
    parser.add_argument("--DEC", type=float, help='cutout center DEC')
    # for calculating sky levels for all targets and save to df
    parser.add_argument("--calSky", action= 'store_true')
    parser.add_argument("--plotSky", action= 'store_true')
    # for save masked fits, not sky subtracted, same phys size for all targets
    parser.add_argument("--savefits", action= 'store_true')
    parser.add_argument("--single", action= 'store_true')
    parser.add_argument("--savesingle", action= 'store_true')
    parser.add_argument("--deblendpix", type=float)
    parser.add_argument("--sharedpix", type=float)
    parser.add_argument("--framesize_kpc", type=float)

    args = parser.parse_args()    

    # load sample data, will use name and Z
    magel = pd.read_pickle("/home/insepien/research-data/alpaka/magellan/alpaka_39fits.pkl")
    # load catalog of target coords in FK5
    catalog = pd.read_csv('../cutouts/catalog.txt', names=['name', 'ra', 'dec'], delimiter='\s+')
    coords = [SkyCoord(ra=catalog['ra'].loc[i]*u.deg, dec=catalog['dec'].loc[i]*u.deg, frame="fk5") for i in range(len(catalog))]
    catalog['coords'] = coords
    catalog.set_index("name",inplace=True)

    # option to calculate sky background: (1)generate cutout of size 100-1000, (2)sigma-clip and calculate backgrounds, (3) take median of all bckgr
    if args.calSky:
        # calculate background level for each target
        savedata = {}
        for on in magel['Desig'].values:
            skylev, res = cal_background(on)
            savedata[on] = [skylev, res]
        # save background data
        with open("/home/insepien/research-data/agn-result/box/skylev/sky.pkl","wb") as f:
            pickle.dump(savedata,f)
        print("Sky level saved")
    
    if args.plotSky:
        bg = pd.read_pickle("/home/insepien/research-data/agn-result/box/skylev/sky.pkl")
        for on in magel['Desig'].values:
            # save plot for sanity check later
            plot_sky(bg[on][0],bg[on][1])
            print(f"done: {on}")

    # option to generate a cutout (not masked) for a specific object
    if args.oname and args.cutSize:
        save_path = os.path.join(args.outDir,args.onam+'_'+str(args.cutSize)+'.fits')
        exp_path = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+args.oname+"*.fits"))[0]
        if args.RA:
            coords = SkyCoord(ra=args.RA*u.deg, dec=args.DEC*u.deg)
            hdu, data, header = make_cutout(exp_path, coords, (args.cutSize, args.cutSize), save_path)
            save_cutout_to_fits(data, header, hdu, save_path)
        else:
            hdu, data, header = make_cutout(exp_path, catalog['coords'].loc[args.oname], (args.cutSize, args.cutSize), save_path)
            save_cutout_to_fits(data, header, hdu, save_path)
        print(f"Saved cutout of size {args.cutSize} for {args.oname}")


    # save cutout fits for all objects with uniform physical size. Image is masked. backgrond is saved in header
    if args.savefits:
        if args.single:
            make_masked_cutout(args.oname, deblendpix=args.deblendpix, npix=args.sharedpix,save=args.savesingle)
        else:
            for on in magel['Desig'].values:
                make_masked_cutout(on)
                print(f"done {on}")

