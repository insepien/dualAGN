import numpy as np
import pickle
import pyimfit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from photutils import profiles
from matplotlib.backends.backend_pdf import PdfPages

medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                 sigma, sigma_lim, Isky, Isky_lim):
    """Return Sersic, PSF, and Gaussian model parameter dictionary"""
    # Sersic
    """sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 'fixed'],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}"""
    sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 0, 10],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}
    sersic_dict = {'name': "Sersic", 'label': "bulge", 'parameters': sersic}
    # PSF
    psf = {'I_tot' : [Itot, Itot_lim[0], Itot_lim[1]]}
    psf_dict = {'name': "PointSource", 'label': "psf", 'parameters': psf}
    """psf = {'I_tot' : [Itot, Itot_lim[0], Itot_lim[1]], 'PA':[PA_ss, PA_lim[0],PA_lim[1]] }
    psf_dict = {'name': "PointSourceRot", 'label': "psf", 'parameters': psf}"""
    # Gaussians
    gaussian = {'PA':[PA_ss, PA_lim[0],PA_lim[1]], 'ell':[ell_ss, ell_lim[0],ell_lim[1]], 
                'I_0':[I_ss, Iss_lim[0],Iss_lim[1]], 'sigma':[sigma, sigma_lim[0], sigma_lim[1]]}
    gaussian_dict = {'name': "Gaussian", 'label': "gaussian", 'parameters': gaussian}
    # Flat sky
    flatsky = {'I_sky': [Isky, Isky_lim[0], Isky_lim[1]]}
    flatsky_dict = {'name': "FlatSky", 'label': "flat_sky", 'parameters':flatsky}
    return sersic_dict, psf_dict, gaussian_dict, flatsky_dict


def make1model(xpos,ypos,function_dict,psf,dataImage):
    """make 1 image model"""
    func_set_dict = {'X0': xpos, 'Y0': ypos, 
                    'function_list': [function_dict]}
    funcset = [func_set_dict]
    model= pyimfit.ModelDescription.dict_to_ModelDescription({'function_sets':funcset})
    fitter = pyimfit.Imfit(model,psf=psf)
    #fitter.loadData(dataImage, gain=9.942e-1,exp_time=1, 
                    #read_noise=0.22, original_sky=654.63,n_combined=4)
    return fitter


def plot_fit_result(d,figtitle,pdf,plot2best=False):
    image = d['imageSS']
    modelNum = len(d['fitResults'])
    fitStats = [d['fitResults'][i].fitStatReduced for i in range(modelNum)]
    best_2ind = np.argsort([d['fitResults'][i].fitStatReduced for i in range(modelNum)])[:2]
    if plot2best:
        nrows = 2
        ff = [best_2ind, np.arange(2)] 
    else:
        nrows = modelNum
        ff = [np.argsort(fitStats),np.arange(nrows)]   
    ncols=3 
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*3))  
    for i, j in zip(ff[0],ff[1]):
        resi = image-d['modelImage'][i]
        im0 = ax[j,0].imshow(resi)
        im1 = ax[j,2].imshow(d['modelImage'][i])
        im2 = ax[j,1].imshow(image)
        [fig.colorbar([im0,im1,im2][k], ax=ax[j,k], shrink=0.7) for k in range(ncols)]
        rmsNoise = np.sqrt(np.sum(resi**2)/image.shape[0]**2)
        iRatio = np.sum(resi)/np.sum(image)*100
        ax[j,2].text(0.05, 0.25, f"$\chi^2$:{fitStats[i]:.2f}", transform=ax[j,2].transAxes, fontsize=10, color='w')
        ax[j,2].text(0.05, 0.15, f"noise RMS:{rmsNoise:.2f}", transform=ax[j,2].transAxes, fontsize=10, color='w')
        ax[j,2].text(0.05, 0.05, f"% I_res: {np.abs(iRatio):.2f}", transform=ax[j,2].transAxes, fontsize=10, color='w')
        title = list(d['modelNames'].keys())[i]
        title = '\n'.join(title.split(',', 1)) if len(title) > 15 else title
        ax[j,2].set_title(title, fontsize=10)
        ax[j,0].set_title("residual", fontsize=10)
        ax[j,1].set_title("data", fontsize=10)
    fig.suptitle(figtitle,y=0.98)
    fig.tight_layout(pad=0.9, h_pad=0, w_pad=1)
    pdf.savefig()
    plt.close()
    
    
def mcomps_dict(d, psf_dict,sersic_dict):
    """make model components"""
    params = [d["fitResults"][i].params for i in range(len(d["fitResults"]))]
    mcomps = {}
    mcomps['1psf'] = [[psf_dict], [params[0]]]
    mcomps['1psf+sersic'] = [[psf_dict, sersic_dict], [params[1][:3], np.delete(params[1],2)]]
    mcomps['2psf'] = [[psf_dict, psf_dict], [params[2][:3], params[2][3:]]]
    mcomps['1psf+sersic,1psf'] = [[psf_dict, psf_dict, sersic_dict], [params[3][:3], params[3][3:6], np.delete(params[3][3:],2)]]
    mcomps['2psf+sersic,sameCenter'] = [[psf_dict,psf_dict,sersic_dict,sersic_dict], [params[4][:3],params[4][8:11], np.delete(params[4][:8],2), np.delete(params[4][8:],2)]]
    mcomps['1psf+sersic,diffCenter'] = [[psf_dict, sersic_dict],[params[5][:3], params[5][3:]]]
    return mcomps


def get_comps(model_dicts,newparams,epsf,image):
    """make model component images"""
    n_comps = len(model_dicts)
    model_images = []
    for i in range(len(model_dicts)):
        fitter = make1model(20,20,model_dicts[i],epsf,image)
        mi = fitter.getModelImage(shape=[40,40],newParameters=newparams[i])
        model_images.append(mi) 
    return n_comps,model_images


def make_plot_comps(image,n_comps,model_images,mnum,pdf):
    """make plot of model components"""
    fig,ax = plt.subplots(3,4,figsize=(16,9))
    modelcomps = np.sum(model_images, axis=0)
    comp_resi = modelcomps-d['modelImage'][mnum]
    for i in range(len(model_images)):
        im = ax[0,i].imshow(model_images[i])
        im1 = ax[1,i].imshow(image-model_images[i])
        im2 = ax[2,i].imshow(image-np.sum(model_images[:i+1], axis=0))
        fig.colorbar(im, ax=ax[0,i], shrink=0.5)
        fig.colorbar(im1, ax=ax[1,i], shrink=0.5)
        fig.colorbar(im2, ax=ax[2,i], shrink=0.5)
    full_title_list = [['psf1'],['psf','sersic'],['psf1','psf2'],
                       ['psf1','psf2','sersic1'],['psf1','psf2','sersic1','sersic2'],
                       ['psf','sersic']]
    [ax[0,i].set_title(full_title_list[mnum][i]) for i in range(n_comps)]
    [ax[1,i].set_title("image $-$ "+full_title_list[mnum][i]) for i in range(n_comps)]
    ax[0,0].text(0.05,0.05, f"\% I_res: {np.sum(comp_resi):.2f}", transform=ax[0,0].transAxes,fontsize=14, color='w')
    [[ax[j,-i].axis('off') for i in range(1,4-n_comps+1)] for j in range(3)]
    fig.suptitle(list(d['modelNames'].keys())[mnum],y=0.98)
    fig.subplots_adjust(hspace=0.4,wspace=0.2)
    pdf.savefig()
    plt.close()
    
    
def plot_comps(d, psf_dict, sersic_dict,pdf):
    """plot model components"""
    mcomps = mcomps_dict(d, psf_dict,sersic_dict)
    for mname,n in zip(mcomps,np.arange(len(d['modelNames']))):
        n_comps,model_images = get_comps(mcomps[mname][0],mcomps[mname][1],epsf,d['imageSS'])
        make_plot_comps(d['imageSS'],n_comps,model_images,n,pdf)

if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to plot fit results
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--psfPath", type=str, default="../psfConstruction/psf_pkls", help="path to psf directory")
    parser.add_argument("--inDir", type=str, default="fit_pkls", help="cut out file")
    parser.add_argument("--inFile", type=str, help="cut out file")
    parser.add_argument("--outDir", type=str, default="fit_pkls", help="directory for fit results")
    parser.add_argument("--plotComps", action="store_true", help="call this to plot components of models")
    args = parser.parse_args()
    
    # load psf
    objectName = args.inFile.split(".")[0]
    psf_fileName = "psf_"+objectName+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (psfPath, "rb") as fp:
        p = pickle.load(fp)
    epsf = p['psf'].data
    # load fit
    fitPath = os.path.join(args.inDir, args.inFile)
    with open (fitPath, "rb") as fd:
        d = pickle.load(fd)
    # make model comps
    Imax = d['imageSS'].max()
    itot=1500
    framelim = d['imageSS'].shape[0]
    midF=framelim//2
    sersic_dict, psf_dict, gaussian_dict, flatsky_dict = makeModelDict(PA_ss=200, ell_ss=0.1, n_ss=1, I_ss=1, r_ss=20, Itot=itot,
                                                                     PA_lim=[0,360], ell_lim=[0.0,1.0], I_lim=[0.1,Imax],
                                                                     Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
                                                                     sigma = 5, sigma_lim = [1,20], Isky = 2.5, Isky_lim =[0,10])
    # save to 1 pdf
    pdf = PdfPages("fit_plots/"+objectName+".pdf")
    with PdfPages("fit_plots/"+objectName+".pdf") as pdf:
        # main plot of model and residual
        plot_fit_result(d,objectName,pdf)
        # radial profile plot

        # plotting components
        if args.plotComps:
            plot_comps(d, psf_dict, sersic_dict,pdf)
        print("Done: ", objectName)