from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
import pandas as pd
import numpy as np
import os
import glob

def download_image_save_cutout(originalfile, position, size, cutoutfile):
    """Returns a cutout of agn from original exp.fits"""
    # Load the image and the WCS
    hdu = fits.open(os.path.expanduser(originalfile))[0]
    wcs = WCS(hdu.header)
    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data
    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())
    # Write the cutout to a new FITS file
    hdu.writeto(cutoutfile, overwrite=True)

    
if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to make cutout
        """), formatter_class=RawDescriptionHelpFormatter)
    #parser.add_argument("--inDir", default="~/raw-data-agn/mos-fits-agn", type=str, help="output directory")
    parser.add_argument("--outDir", default="~/agn-result/box", type=str, help="output directory")
    parser.add_argument("--cutSize", type=int, help="fits cutout size")
    parser.add_argument("--objectName", type=str, help="object name")
    parser.add_argument("--makeMulti", action= 'store_true')
    args = parser.parse_args()    
    # reading coordinates from catalog
    catalog = pd.read_csv('catalog.txt', names=['name', 'ra', 'dec'], delimiter='\s+')
    coords = [SkyCoord(ra=catalog['ra'].loc[i]*u.deg, dec=catalog['dec'].loc[i]*u.deg) for i in range(len(catalog))]
    catalog['coords'] = coords
    catalog.set_index("name",inplace=True)
    exp_path = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/2020-02-22_J_"+args.objectName+"*.mos.fits"))[0]

    boxdirname = "boxsize_"+args.objectName
    if args.makeMulti:
        # make directory to store cutout of different sizes
        boxdir = os.mkdir(os.path.join(args.outDir,boxdirname))
        # make cutout of different sizes and save
        for i in np.arange(100,500,10):
            save_path = os.path.join(args.outDir,boxdirname,args.objectName+'_'+str(i)+'.fits') 
            download_image_save_cutout(exp_path, catalog['coords'].loc[args.objectName], (i,i), save_path)
    

    if args.cutSize:
        save_path = os.path.join(args.outDir,boxdirname,args.objectName+'_'+str(args.cutSize)+'.fits')
        download_image_save_cutout(exp_path, catalog['coords'].loc[args.objectName], (args.cutSize, args.cutSize), save_path)

    print("Finished")