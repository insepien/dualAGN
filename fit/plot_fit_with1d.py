import numpy as np
import pickle
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from modelComponents import modelComps
import matplotlib.gridspec as gridspec
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import seaborn as sns
import glob
from astropy.wcs import WCS
from astropy.coordinates import angular_separation


medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'monospace'


def plot_isophotes(ax,isolist,num_aper=10):
    """plot aperatures on image"""
    for sma in np.linspace(isolist.sma[0],isolist.sma[-1],num_aper):
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax.plot(x, y, color='white',linewidth="0.5")


def plot_model_components(pdf,comp_ims,comp_names,comp_pos,isolist_comps,serinds,args):
    """plot 2D model components and check residual with model image"""
    clmap = sns.color_palette(args.cmap, as_cmap=True).reversed()
    ncom = len(comp_names)
    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(ncom*4,3))
    im = [ax[i].imshow(comp_ims[i],norm='symlog',cmap=clmap) for i in range(ncom)]
    [ax[i].text(0.05, 0.05, f"(x,y)=({comp_pos[i][0]:.1f},{comp_pos[i][1]:.1f})", transform=ax[i].transAxes, fontsize=8, color='w') for i in range(ncom-1)]
    count_ind  = 0
    for k in range(ncom):
        if "bulge" in comp_names[k]:
            axtit = f"{comp_names[k][:6]} {count_ind}, n={serinds[count_ind]:.2f}"
            count_ind+=1
        else:
            axtit = comp_names[k]
        ax[k].set_title(axtit)
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],norm='symlog',cmap=clmap))
    ax[-1].set_title("model-comps")
    [fig.colorbar(im[i], ax=ax[i], shrink=0.5).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
    if args.plotIso:
        for i in range(len(isolist_comps)):
             plot_isophotes(ax[i],isolist_comps[i],num_aper=5)
    fig.tight_layout()
    pdf.savefig();


def fit_stat_1d(iso_data,iso_model,nparams,cut=0.8):
    """calculate 1d chi squared, cut to 80% of data"""
    cut_index = int(np.round(len(iso_data.sma)*cut))
    sma_cut = iso_data.sma[cut_index]
    diff = iso_data.intens[1:cut_index]-iso_model.intens[1:cut_index]
    chi_square1d_reduced = np.sum((diff/iso_data.rms[1:cut_index])**2)/(cut_index-nparams)
    diff_full = iso_data.intens[1:]-iso_model.intens[1:]
    chi_square1d_full_reduced = np.sum((diff_full/iso_data.rms[1:])**2)/(len(iso_data)-1-nparams)
    return chi_square1d_reduced,sma_cut,chi_square1d_full_reduced


def plot_1isophote(ax,sma,isolist):
    """plot aperatures on image"""
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax.plot(x, y, color='white',linewidth="0.3")


def pix_to_arcsec(imageFile,framelim):
    """convert pixel to arcsec"""
    w = WCS(imageFile)
    # convert pixel to degree
    ra1,dec1 = w.pixel_to_world_values(0,0)
    ra2,dec2 = w.pixel_to_world_values(framelim,framelim)
    framelim_deg = angular_separation(ra1*u.degree,dec1*u.degree,ra2*u.degree,dec2*u.degree)
    # find pixel-arsec scale
    framelim_arcsec = framelim_deg.to('arcsec')
    arcsec_per_pix = framelim_arcsec/framelim
    return arcsec_per_pix, [ra1,dec1,ra2,dec2]

def surface_brightness(intensity, area, magZPT):
    """calculate surface brightness"""
    return -2.5*np.log10(intensity/area)+ magZPT


def radial_plot_params(imageFile, framelim, isolist_data,isolist_comps,hdu_exp,z=0.2):
    """calculate sma and surface brightness in physical units for 1d plot"""
    # convert pixel to arcsec and kpc
    arcsec_per_pix, skycoords = pix_to_arcsec(imageFile,framelim)
    sma_arcsec = isolist_data.sma*arcsec_per_pix
    sma_kpc = (cosmo.angular_diameter_distance(z)*sma_arcsec.to('rad').value).to('kpc')
    # calculate isophote areas and find surface brightness
    areas = (np.sqrt((1-isolist_data.eps**2)*sma_arcsec**2)*np.pi*sma_arcsec).value
    magZPT = hdu_exp.header['MAGZP']
    mu_data = [surface_brightness(i,areas,magZPT) for i in [isolist_data.intens,isolist_data.intens-isolist_data.int_err,isolist_data.intens+isolist_data.int_err]]
    mu_models = [[surface_brightness(i,areas,magZPT) for i in [isolist_comps[j].intens,isolist_comps[j].intens-isolist_comps[j].int_err,isolist_comps[j].intens+isolist_comps[j].int_err]] for j in range(len(isolist_comps))]
    return sma_arcsec, sma_kpc, mu_data, mu_models, skycoords
    

def plot_everything(pdf,on,image,m,modelname,comp_names,fs,fsr,sma_arcsec,sma_kpc,mu_data,mu_models,skycoords):
    colors = sns.color_palette("colorblind", len(comp_names))
    ls = ['-', '--', '-.', ':']
    cmapp = sns.color_palette(args.cmap, as_cmap=True).reversed()
    if len(modelname) > 16 and ',' in modelname:
        modelname.replace('\n','')
        modelname = modelname.split(",")[0]+',\n'+modelname.split(",")[1]
    # Create grid and add subplots
    fig = plt.figure(figsize=(14, 4),layout='tight')
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1], width_ratios=[1.25,1.25,1,1.5],hspace=0.05,wspace=0.05)
    ax1 = fig.add_subplot(gs[:, 0],xlabel='RA (deg)',ylabel='DEC (deg)') 
    ax2 = fig.add_subplot(gs[:, 1],xticks=[],yticks=[])
    ax3 = fig.add_subplot(gs[:, 2],xticks=[],yticks=[])
    ax4a = fig.add_subplot(gs[0, 3]) 
    ax4b = fig.add_subplot(gs[1, 3])   
    # formatting ticks
    xticks = np.linspace(skycoords[0],skycoords[2],4)
    yticks = np.linspace(skycoords[1],skycoords[3],4)
    ax1.set_xticks(np.linspace(0,image.shape[0],4))
    ax1.set_yticks(np.linspace(0,image.shape[0],4))
    ax1.set_xticklabels([f'{x:.3f}' for x in xticks])
    ax1.set_yticklabels([f'{y:.3f}' for y in yticks],rotation=90)
    ax1.tick_params(direction='in')
    # plot 2d and colorbars
    ax = [ax1,ax2,ax3,ax4a,ax4b]
    im = [ax[i].imshow([image, m][i], norm='symlog',cmap=cmapp) for i in range(2)]
    im2 = ax[2].imshow(image-m,cmap=cmapp)
    fig.colorbar(im2,ax=ax[2],orientation='horizontal',location='bottom',pad=0.05)
    fig.colorbar(im[1],ax=[ax[0],ax[1]],orientation='vertical',location='right',shrink=0.5)
    [ax[i].set_title([on,f"Model:\n{modelname}",f'Residual,\n$\chi^2_r$={fsr:.3f}\n$\chi^2$={fs:.0f}'][i]) for i in range(3)]
    # radial plot data
    ax[3].plot(sma_arcsec[1:],mu_data[0][1:],label="data",c="k")
    ax[3].fill_between(sma_arcsec[1:].value,mu_data[1][1:],mu_data[2][1:],color="k",alpha=0.2)
    # radial plot model
    #ax[3].plot(sma_arcsec[1:],mu_models[-1][0][1:],label="model",c=colors[-2],linestyle="dashdot")
    #ax[3].fill_between(sma_arcsec[1:].value, mu_models[-1][1][1:],mu_models[-1][2][1:],color=colors[-2],alpha=0.5)
    # radial plot components
    [ax[3].plot(sma_arcsec[1:],mu_models[i][0][1:],label=comp_names[i],linestyle=ls[i],c=colors[i]) for i in range(len(comp_names)-1)]
    [ax[3].fill_between(sma_arcsec[1:].value,mu_models[i][1][1:],mu_models[i][2][1:],color=colors[i],alpha=0.5) for i in range(len(comp_names)-1)]
    ax[4].plot(sma_kpc[1:],mu_data[0][1:]-mu_models[-1][0][1:],c='rebeccapurple',linestyle="dashdot")
    ax[4].fill_between(sma_kpc[1:].value,mu_data[1][1:]-mu_models[-1][1][1:],mu_data[2][1:]-mu_models[-1][2][1:],color='rebeccapurple',alpha=0.5)
    ax[4].axhline(y=0,linestyle='--',c="k",alpha=0.5,lw=1)
    # format ticks
    ax[3].invert_yaxis()
    ax[3].set_xlabel("R[arcsec]")
    ax[3].set_ylabel("$\mu$ [mag arcsec$^{-2}$]")
    ax[3].xaxis.set_label_position('top') 
    ax[3].xaxis.set_ticks_position('top') 
    ax[3].legend(fontsize=10,loc='upper right')
    
    ax[4].set_xlabel("R[kpc]")
    ax[4].set_ylabel("$\Delta \mu$") 
    ax[4].set_ylim((-0.5,0.5))
    [ax.yaxis.set_label_position('right') for ax in [ax4a,ax4b]]
    [ax.yaxis.set_ticks_position('right') for ax in [ax4a,ax4b]]
    pdf.savefig(bbox_inches='tight', pad_inches=0.2);


if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to plot fit results
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="~/agn-result/fit/final_comps", help="input directory of fit result files")
    parser.add_argument("--oname", type=str, help="object name")
    parser.add_argument("--outDir", type=str, default="~/agn-result/fit/final_plots", help="directory for fit results")
    parser.add_argument("--outFile", type=str, help="output file")
    parser.add_argument("--plotIso", action='store_true')
    parser.add_argument("--useAIC", action='store_true')
    parser.add_argument('--cmap', type=str, default="ch:s=-.3,r=.6")
    args = parser.parse_args()
    
    # load fit file to get sersic index
    fitPath = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit/"+args.oname+".pkl")
    with open (fitPath, "rb") as f:
        d_fit = pickle.load(f)
    param_vals = [d_fit['fitResults'][i].params for i in range(len(d_fit['fitResults']))]
    param_names = d_fit['paramNames']
    ns = []
    for i in range(len(param_names)):
        dic = dict(zip(param_names[i],param_vals[i]))
        ns.append([dic[i] for i in dic if i[0]=="n"])
    # load fit component file
    compPath = os.path.expanduser(os.path.join(args.inDir, args.oname+"_comp.pkl"))
    with open (compPath, "rb") as f:
        d = pickle.load(f)
    imageAGN = d['agn']
    isolist_agn= d['agn-iso']
    # load stuffs for coordinate calculations
    imageAGNFile = glob.glob(os.path.expanduser("~/research-data/agn-result/box/final_cut/"+args.oname+"*"))[0]
    mosfile = glob.glob(os.path.expanduser("~/raw-data-agn/mos-fits-agn/*"+args.oname+"*.mos.fits"))[0]
    with fits.open(mosfile) as hdul:
        hdu0 = hdul[0]
    # create output file
    if args.outFile:
        savepath = os.path.expanduser(os.path.join(args.outDir,args.outFile))
    else:
        savepath = os.path.expanduser(os.path.join(args.outDir,args.oname+".pdf"))
    pdf = PdfPages(savepath)

    # get model names
    model_names = list(d.keys())[:-2]
    if args.useAIC:
        sorted_ind = np.argsort([d[m].fit_result.aic for m in model_names])
        fs_lab = "AIC"
        fstat='aic'
    else:
        sorted_ind = np.argsort([d[m].fit_result.fitStatReduced for m in model_names])
        fs_lab = "$\chi^2$"
        fstat = 'fitStat'
    
    with PdfPages(savepath) as pdf:
        for i in sorted_ind:
            model = d[model_names[i]]
            sma_Arcsec, sma_Kpc, mu_Data,mu_Models,skyCoords = radial_plot_params(imageFile=imageAGNFile, framelim=imageAGN.shape[0],
                                                                                  isolist_data=isolist_agn,isolist_comps=model.iso_comp,
                                                                                  hdu_exp=hdu0,z=0.2)
            plot_everything(pdf,on=args.oname,image=imageAGN,m=model.comp_im[-1],modelname= model.model_name,
                            comp_names=model.comp_name, fs=model.fit_result.fitStat, fsr=model.fit_result.fitStatReduced, 
                            sma_arcsec=sma_Arcsec,sma_kpc=sma_Kpc,mu_data=mu_Data,mu_models=mu_Models,skycoords=skyCoords)
            plot_model_components(pdf,comp_ims=model.comp_im,comp_names=model.comp_name, comp_pos=model.comp_pos,
                                  isolist_comps=model.iso_comp,serinds=ns[i],args=args)
    print("Done: ", args.oname)