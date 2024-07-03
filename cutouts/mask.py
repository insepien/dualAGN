import os
from astropy.io import fits
from photutils.aperture import EllipticalAperture
from photutils.detection import find_peaks
import numpy as np
import matplotlib.pyplot as plt


def make_peak_tbl(image,intens,agnsize=35):
    """mask out agn and find peaks"""
    s = image.shape[0]
    midf = int(s//2)
    # mask out central agn
    peak_mask = np.zeros((s,s))
    peak_mask[midf-agnsize:midf+agnsize,midf-agnsize:midf+agnsize] = 1
    # convert to boolean
    peak_mask = peak_mask==1
    # detect peaks
    peak_tbl = find_peaks(image,threshold=intens,mask=peak_mask)
    return peak_tbl


def make_mask(image,pos,aper_radius,pa=180):
    """make a mask provided position and aperture radius"""
    aper0 = EllipticalAperture(pos,aper_radius,aper_radius,pa)
    aper_mask0 = aper0.to_mask()
    mask0 = aper_mask0.to_image(image.shape)
    return mask0

def mask_image(image,peak_tbl,rad):
    """create mask and masked images"""
    mask=[]
    # make masks
    for i in range(len(peak_tbl)):
        mask.append(make_mask(image,pos = [peak_tbl[i]['x_peak'],peak_tbl[i]['y_peak']],aper_radius=rad))
    # sum all masks
    mask = np.sum(mask,axis=0)
    # make masked image
    masked_im = np.where(mask==0,image,0)
    return masked_im


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to mask cutout
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", default="~/agn-result/box", type=str, help="input directory")
    parser.add_argument("--inFile", type=str)
    parser.add_argument("--showPlot", action='store_true')
    args = parser.parse_args()    
    objectName = args.inFile[:10]
    boxDir = "boxsize_"+ objectName
    inPath = os.path.join(args.inDir, boxDir, args.inFile)
    image = fits.getdata(os.path.expanduser(inPath))
    peak_tbl = make_peak_tbl(image,intens=7)
    masked_image = mask_image(image,peak_tbl,rad=23)
    if args.showPlot:
        plt.imshow(masked_image,norm='symlog')
        plt.show()
    savepath = os.path.join(args.inDir,args.inFile[:-5]+"_masked.fits")
    fits.writeto(filename=savepath, data=masked_image)
