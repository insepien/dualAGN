from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
import pandas as pd
import os

def download_image_save_cutout(originalfile, position, size, cutoutfile):
    """Returns a cutout of agn from original exp.fits"""
    # Load the image and the WCS
    hdu = fits.open(originalfile)[0]
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
        script to construct ePSF from an exposure
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--cutSize", type=int, default=100, help="fits cutout size")
    args = parser.parse_args()
    # reading coordinates from catalog
    catalog = pd.read_csv('catalog.txt', names=['name', 'ra', 'dec'], delimiter='\s+')
    coords = [SkyCoord(ra=catalog['ra'].loc[i]*u.deg, dec=catalog['dec'].loc[i]*u.deg) for i in range(len(catalog))]
    catalog['coords'] = coords
    catalog.set_index("name",inplace=True)
    # get gal names and file names
    dataPath = "../../agn-data"
    fileNames = os.listdir(dataPath)
    objectNames = [file.split("_")[2] for file in fileNames]
    # make cutouts
    for i in range(len(objectNames)):
        exp_path = "../../agn-data/"+fileNames[i]
        save_path = "data/"+objectNames[i]+'.fits'
        download_image_save_cutout(exp_path, catalog['coords'].loc[objectNames[i]], (args.cutSize,args.cutSize), save_path)