import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import CubicSpline

def get_wise_mags(wise_):
    """get wise mags and mag errors from data frame of ipac search results"""
    ## get wise mags and errors
    w1mag = wise_['w1mpro']
    w2mag = wise_['w2mpro']
    w3mag = wise_['w3mpro']
    w4mag = wise_['w4mpro']
    wmags_ = np.array([w1mag, w2mag, w3mag, w4mag])
    wmags_err_ = np.array([wise_['w1sigmpro'], wise_['w2sigmpro'], wise_['w3sigmpro'], wise_['w4sigmpro']])
    return wmags_, wmags_err_


def wise_lum_from_mag(wmags_, wmags_err_, obs_wavelength_, redshift_):
    """calculate wise luminosity from magnitude at some observed wavelength"""
    ## change mags to fluxes -- http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#example
    zeromagflux = np.array([309.540, 171.787, 31.674, 8.363])*u.Jy
    fluxdens = zeromagflux*10**(-wmags_/2.5) # in Jy
    # now either interpolate flux dens to some wavelength or use a band from wise
    wise_wavelengths = np.array([3.4, 4.6, 12., 22.]) # 1e-6 m
    if np.isin(obs_wavelength_, wise_wavelengths): # check if need to interpolate to non-wise wl
        obs_flux = fluxdens[wise_wavelengths==obs_wavelength_][0]
    else: # interpolate
        fluxdens_err = zeromagflux*10**(-wmags_err_/2.5)
        ## interpolate - use straight line
        wiseflux = np.polyfit(wise_wavelengths, fluxdens.value,1, w=1./fluxdens_err)
        ## get flux at obs wavelength, i.e. just a straight line here
        obs_flux = (wiseflux[0]*obs_wavelength_+wiseflux[1])*u.Jy      
    ## change to luminosity
    obs_hz = (const.c/(obs_wavelength_*u.micron)).to(u.Hz)
    lum = (obs_flux*obs_hz*4*np.pi*
           cosmo.luminosity_distance(redshift_)**2).to(u.erg/u.s)
    return lum


def correct_ir():
    """correct IR luminosity at 15 microns rest frame based on Hopkins+20"""
    # load hopkins bolometric correction
    with open("/home/insepien/research-data/pop-result/bc.txt","r") as f:
        d = f.read().splitlines()
    hopkins = pd.DataFrame([d[1:][i].split(' ') for i in range(len(d[1:]))],columns=d[0].split(' '))
    Lbol = np.array(list(hopkins['Lbols'].values), dtype=float)
    LIR = np.array(list(hopkins['LIRs'].values), dtype=float)
    spl = CubicSpline(LIR, Lbol)
    return spl


def get_wise_ir_lums(cat,wl_=22):
    """calculate wise IR luminosity and bolometric lum, 
        default keys (variants of 'desig') are for magellan sample"""
    ## get wise mags and errors
    wmags, wmags_err_nan = get_wise_mags(cat)
    # replace nan values in mag error with median
    wmags_err = np.nan_to_num(wmags_err_nan,np.median(wmags_err_nan))
    # calculate luminosity
    wise_lums = np.zeros((len(cat)))
    for i in range(0, len(cat)):
        z = cat.loc[i,'Z']
        wise_lums[i] = wise_lum_from_mag(wmags[:,i], wmags_err[:,i], wl_, z).value
    # check wavelength to see how to do bolo correction
    if wl_==15: # use Hopkins+2020 if at 15 microns
        spl = correct_ir()
        irbol = 10**(spl(np.log10(wise_lums)))
    else: # else correct by 12%
        irbol = wise_lums*10**1.12
    return wmags, wise_lums,irbol


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=("analysis of fit result script"),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--savedf", action="store_true")
    args = parser.parse_args()
    # load Alpaka
    alpaka = Table(fits.getdata('/home/insepien/research-data/alpaka/ALPAKA_v1_withDes.fits')).to_pandas()
    # cut z=0.1-0.5 and type 2 
    mask = (alpaka['Z']>0.1) & (alpaka['Z']<0.5) & (alpaka['AGN_TYPE']==2)
    alpaka_z05 = alpaka[mask]
    # load WISE query result
    wsearch = pd.read_csv('/home/insepien/research-data/alpaka/wise-cat/wise_z05_result.csv')
    # Check: Wise entries should be unique, i.e. 1 coord should have 1 search output
    wise_num_dup = (wsearch['ra_01'].duplicated(keep=False) & wsearch['dec_01'].duplicated(keep=False)).sum()
    print(f"Check 0:\n*****There are {wise_num_dup} duplicated RA and DEC in WISE catalog")
    # sort both DF by RA in Alpaka
    wsearch.sort_values(by='ra_01') # input RA is saved as ra_01 in query result
    wsearch_ord = wsearch.reset_index(drop=True)
    alpaka_z05_ord = alpaka_z05.sort_values(by='RA').reset_index(drop=True)
    # merge catalogs
    mdf = pd.concat([alpaka_z05_ord,wsearch_ord],axis=1)
    # Check: dimensions. wsearch and alpaka should have same dimensions since there are no duplicates
    print(f"Check 1:\n*****Wise cat dim: {wsearch_ord.shape} + Alpaka cat dim {alpaka_z05_ord.shape} = Merged cat dim {mdf.shape}")
    # Check: merged df merges alpaka and wise in right order. check by comparing RA and DEC
    sameRA = (np.abs(mdf['RA']- mdf['ra_01'])<1e-3).sum()
    sameDEC = (np.abs(mdf['DEC']- mdf['dec_01'])<1e-3).sum()
    print(f"Check 2:\n*****{sameRA}/{len(mdf)} have similar RA, {sameDEC}/{len(mdf)} have similar DEC in merged df")

    # calculate luminosities
    wmag,wlum,wbol = get_wise_ir_lums(mdf,wl_=22)
    # Check: magnitudes should be extracted correctly for each row
    unequal_mag_mask = ~(wmag[-1] == mdf['w4mpro'])
    print(f"Check 3:\n*****{unequal_mag_mask.sum()} targets have different wise mags")
    is_finite = np.isfinite(mdf[unequal_mag_mask]['w4mpro']).sum()
    print(f"\n*****{is_finite}/{unequal_mag_mask.sum()} mismatches is finite. (if this is 0, all mismatches are NaN)")
    
    # add luminosities to DF
    mdf['wiseLum'] = wlum
    mdf['irbol'] = wbol
    # save
    if args.savedf:
        mdf.to_pickle("/home/insepien/research-data/alpaka/alpaka_z05_merged_wise.pkl")
        print("Done saving merged DF")
