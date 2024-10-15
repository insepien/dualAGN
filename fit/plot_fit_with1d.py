import numpy as np
import pickle
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from modelComponents import modelComps
import matplotlib.gridspec as gridspec


medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


def plot_isophotes(ax,isolist,num_aper=10):
    """plot aperatures on image"""
    for sma in np.linspace(isolist.sma[0],isolist.sma[-1],num_aper):
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax.plot(x, y, color='white',linewidth="0.5")


def plot_model_components(pdf,comp_ims,comp_names,comp_pos,isolist_comps,args):
    """plot 2D model components and check residual with model image"""
    ncom = len(comp_names)
    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(14,3))
    im = [ax[i].imshow(comp_ims[i],norm='symlog') for i in range(ncom)]
    [ax[i].text(0.05, 0.05, f"(x,y)=({comp_pos[i][0]:.1f},{comp_pos[i][1]:.1f})", transform=ax[i].transAxes, fontsize=8, color='w') for i in range(ncom-1)]
    [ax[i].set_title(comp_names[i]) for i in range(ncom)]
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],norm='symlog'))
    ax[-1].set_title("model-comps")
    [fig.colorbar(im[i], ax=ax[i], shrink=0.7).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
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


def fit_stat_2d(image,model_im,sky_level,args):
    """calculate 2d weighted by uncertainty chi squared"""
    midf = image.shape[0]//2
    imcrop = image[midf-args.cut2d:midf+args.cut2d,midf-args.cut2d:midf+args.cut2d]+sky_level
    modelcrop = model_im[midf-args.cut2d:midf+args.cut2d,midf-args.cut2d:midf+args.cut2d]+sky_level

    diff2d = imcrop-modelcrop
    chi2dw = np.sum(diff2d**2/np.sqrt(imcrop))
    return chi2dw

def plot_1isophote(ax,sma,isolist):
    """plot aperatures on image"""
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax.plot(x, y, color='white',linewidth="0.3")

def plot_everything(pdf,image,m,modelname,isolist_data,isolist_comps,comp_names,fs,fsr,nParams,args):
    chi_1d,sma_cut,chi_square1d_full = fit_stat_1d(iso_data=isolist_data,iso_model=isolist_comps[-1],nparams=nParams)
    fig = plt.figure(figsize=(14, 4))
    # Create grid and add subplots
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1], width_ratios=[1,1,1,1.5])
    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])
    ax4a = fig.add_subplot(gs[0, 3]) 
    ax4b = fig.add_subplot(gs[1, 3],sharex=ax4a)  
    plt.setp(ax4a.get_xticklabels(), visible=False)

    #plotting
    ax = [ax1,ax2,ax3,ax4a,ax4b]
    im = [ax[i].imshow([image, m, image-m][i], norm='symlog') for i in range(3)]
    [fig.colorbar(im[i],ax=ax[i],shrink=0.5) for i in range(3)]

    [ax[i].set_title([args.oname,f"model: {modelname}",f'residual,\n $\chi^2$={fs:.0f}, $\chi^2_r$={fsr:.2f}'][i]) for i in range(3)]
    ax[3].plot(isolist_data.sma,isolist_data.intens,label="data",c="k",alpha=0.3)
    ax[3].plot(isolist_comps[-1].sma,isolist_comps[-1].intens,c='k',linestyle="",marker="o",markersize=1,label="model")
    [ax[3].plot(isolist_comps[i].sma,isolist_comps[i].intens,label=comp_names[i],linestyle="--") for i in range(len(comp_names)-1)]
    ax[3].axvline(x=sma_cut,label=f"$sma_\\chi$={sma_cut:.1f}",lw=1,alpha=0.5)
    ax[4].plot(isolist_comps[-1].sma,isolist_data.intens-isolist_comps[-1].intens,c='magenta',alpha=0.5,lw=1,label="residual")

    plot_1isophote(ax[0],sma=sma_cut,isolist=isolist_data)
    ax[3].set_yscale('log')
    ax[3].set_ylim(ymin=1)
    ax[3].set_xlim(xmin=1)
    ax[3].set_title(f"1D profile, $\chi^2_c$={chi_1d:.2f},$\chi^2_f$={chi_square1d_full:.2f}")
    ax[3].set_ylabel("I")
    ax[4].set_ylabel("$\Delta$ I")
    ax[4].set_xlabel("sma")
    ax[3].legend(fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    gs.update(hspace=0.1)
    pdf.savefig();


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
    args = parser.parse_args()
    
    # load fit component file
    compPath = os.path.expanduser(os.path.join(args.inDir, args.oname+"_comp.pkl"))
    with open (compPath, "rb") as f:
        d = pickle.load(f)
    imageAGN = d['agn']
    isolist_agn= d['agn-iso']
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
        sorted_ind = np.argsort([d[m].fit_result.fitStat for m in model_names])
        fs_lab = "$\chi^2$"
        fstat = 'fitStat'
    
    with PdfPages(savepath) as pdf:
        for i in sorted_ind:
            model = d[model_names[i]]
            plot_everything(pdf,image=imageAGN,m=model.comp_im[-1],modelname= model.model_name,
                            isolist_data=isolist_agn,isolist_comps=model.iso_comp,
                            comp_names=model.comp_name, fs=model.fit_result[fstat], fsr=model.fit_result.fitStatReduced, nParams=len(model.fit_result['params']),args=args)
            plot_model_components(pdf,comp_ims=model.comp_im,comp_names=model.comp_name, comp_pos=model.comp_pos,
                                  isolist_comps=model.iso_comp,args=args)
    print("Done: ", args.oname)