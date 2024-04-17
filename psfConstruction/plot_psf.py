import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy.io import fits
from photutils import profiles

medium_font_size = 10
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size
plt.rcParams['font.family'] = 'monospace'

    
def plot_psf_rProfile(epsf1,stars,objectName):
    """plot psf and stars radial profiles"""
    fig,ax = plt.subplots(1,3,figsize=(12,4))

    # plot x-sum in first axis
    for n in range(len(stars)):
        ax[0].plot(np.sum(stars[n].data, axis=0), "lightblue", alpha=0.5)
    ax[0].plot(np.sum(epsf1.data, axis=0), "steelblue")
    ax[0].axvline(x=stars[0].cutout_center[0],alpha=0.5)

    # plot radial profile in 2nd axis
    rp_psf1 = profiles.RadialProfile(epsf1.data,xycen=stars[0].cutout_center,radii = np.arange(25))
    rp_psf1.normalize("max")
    
    s = 0
    for i in range(len(stars)):
        rp_star = profiles.RadialProfile(stars[i].data,xycen=stars[i].cutout_center,radii=np.arange(25))
        rp_star.normalize("max")
        m = rp_star.area*rp_star.profile
        s+=np.sum(m[~np.isnan(m)])
        ax[1].plot(rp_star.radius, rp_star.profile, color="lightblue", alpha=0.5)
        ax[1].plot(rp_star.radius, rp_star.profile-rp_psf1.profile, "g", alpha=0.1)
    ax[1].plot(rp_psf1.radius, rp_psf1.profile,"steelblue")

    ax[0].set_title("x-axis profile")
    ax[0].set_xlabel("pixels")
    ax[0].set_ylabel("intensity(counts)")
    ax[1].set_title("Radial profile of stars and non-oversampled psf")
    ax[1].set_xlabel("pixels")
    ax[1].set_ylabel("intensity(counts)")
    legend_handles = [plt.Line2D([], [], color='lightblue', label='stars'),
                      plt.Line2D([], [], color='steelblue', label='PSF'),
                      plt.Line2D([], [], color='green', label='residual')]
    ax[1].legend(handles=legend_handles)
    psfsum = rp_psf1.area*rp_psf1.profile
    psf_integrated = np.sum(psfsum[~np.isnan(psfsum)])
    star_integrated = s/len(stars)
    ax[1].text(0.3, 0.5, f"Integrated PSF: {psf_integrated:.2f}", transform=ax[1].transAxes, fontsize=10, color='k')
    ax[1].text(0.3, 0.4, f"Ave.intg. star: {star_integrated:.2f}", transform=ax[1].transAxes, fontsize=10, color='k')
    
    im = ax[2].imshow(epsf1.data)
    fig.colorbar(im, ax=ax[2], shrink=0.7)
    ax[2].set_title(objectName + " PSF")
    fig.tight_layout()
    #fig.savefig("psf_plots/radial_profile_"+objectName+".pdf",bbox_inches='tight')
    fig.savefig("psf_plots/"+objectName+"_4_radial_profile.jpg",bbox_inches="tight",dpi=300)

    
def plot_point_subtraction(epsf1,stars,objectName):
    """plot point source subtraction"""
    ncols = 5
    nrows = int(np.ceil(len(stars)/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols*2), int(nrows*2)),
                           squeeze=True)
    ax = ax.ravel()
    for n in range(len(stars)):
        resi = stars[n].compute_residual_image(epsf1)
        err = np.sum(resi)/stars[n].estimate_flux()*100
        im = ax[n].imshow(resi)
        fig.colorbar(im,ax=ax[n],shrink=0.4)
        ax[n].set_title(f"flux overshoot:{err:.2f}%")
    # turn off empty axes
    empty_axes = nrows*ncols-len(stars)
    [ax[-i].axis('off') for i in np.arange(1,empty_axes+1)]
    fig.tight_layout()
    fig.suptitle(objectName, y=1)
    #fig.savefig("psf_plots/point_subtract_"+objectName+".pdf",bbox_inches='tight')
    fig.savefig("psf_plots/"+objectName+"_3_point_subtract.jpg",bbox_inches="tight",dpi=300)
    
    
if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        Script to plot and save PSF, it's radial profile, and point-source subtraction
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inFile", type=str, help="name of file with psf")
    args = parser.parse_args()
    # open psf pickle file
    objectName = args.inFile.split("_")[1].split(".")[0]
    with open('psf_pkls/'+args.inFile,"rb") as f:
        d = pickle.load(f)
    epsf1 = d['psf']
    stars = d['stars']
    # plot psf
    
    # plot radial profile
    plot_psf_rProfile(epsf1,stars,objectName)
    # point-source subtraction
    plot_point_subtraction(epsf1,stars,objectName)
