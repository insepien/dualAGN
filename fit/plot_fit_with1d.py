import numpy as np
import pickle
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from modelComponents import modelComps


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


def plot_model_components(pdf,comp_ims,comp_names,comp_pos,isolist_comps,plot_iso=False):
    """plot 2D model components and check residual with model image"""
    ncom = len(comp_names)
    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(14,3))
    im = [ax[i].imshow(comp_ims[i],norm='symlog') for i in range(ncom)]
    [ax[i].text(0.05, 0.05, f"(x,y)=({comp_pos[i][0]:.1f},{comp_pos[i][1]:.1f})", transform=ax[i].transAxes, fontsize=8, color='w') for i in range(ncom-1)]
    [ax[i].set_title(comp_names[i]) for i in range(ncom)]
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],norm='symlog'))
    ax[-1].set_title("model-comps")
    [fig.colorbar(im[i], ax=ax[i], shrink=0.7).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
    if plot_iso:
        for i in range(len(isolist_comps)):
             plot_isophotes(ax[i],isolist_comps[i],num_aper=5)
    fig.tight_layout()
    pdf.savefig();


def fit_stat_1d(iso_data,iso_model,cut=35):
    """calculate 1d chi squared"""
    sma_cut = iso_data.sma[cut-1]
    diff = iso_data.intens[:cut]-iso_model.intens[:cut]
    chi_square1d = np.sum(diff**2 / np.sqrt(iso_data.intens[:cut]))
    return chi_square1d,sma_cut

def fit_stat_2d(image,model_im,sky_level,args):
    """calculate 2d weighted by uncertainty chi squared"""
    midf = image.shape[0]//2
    imcrop = image[midf-args.cut2d:midf+args.cut2d,midf-args.cut2d:midf+args.cut2d]+sky_level
    modelcrop = model_im[midf-args.cut2d:midf+args.cut2d,midf-args.cut2d:midf+args.cut2d]+sky_level

    diff2d = imcrop-modelcrop
    chi2dw = np.sum(diff2d**2/np.sqrt(imcrop))
    return chi2dw

def plot_everything(pdf,image,m,modelname,isolist_data,isolist_comps,comp_names,fs):
    chi_1d,sma_cut = fit_stat_1d(iso_data=isolist_data,iso_model=isolist_comps[-1])
    # plot everything 
    fig,ax = plt.subplots(1,4,figsize=(14,3),gridspec_kw={'width_ratios': [1, 1, 1, 1.5]})
    #norms = [simple_norm([image, m, image-m][i],'log') for i in range(3)]
    im = [ax[i].imshow([image, m, image-m][i], norm='symlog') for i in range(3)]
    cbar = [fig.colorbar(im[i],ax=ax[i],shrink=0.5) for i in range(3)]
    #[ax[i].set_title(['data',f"model: {modelname}",f'residual, $\chi^2$={chi_2d:.1f} ({args.cut2d}x{args.cut2d} pix)'][i]) for i in range(3)]
    [ax[i].set_title(['data',f"model: {modelname}",f'residual, $\chi^2$={fs:.1f}'][i]) for i in range(3)]

    ax[-1].plot(isolist_data.sma**0.25,isolist_data.intens,label="data",c="k",alpha=0.3)
    ax[-1].plot(isolist_comps[-1].sma**0.25,isolist_comps[-1].intens,c='k',linestyle="",marker="o",markersize=1,label="model")
    ax[-1].plot(isolist_comps[-1].sma**0.25,isolist_data.intens-isolist_comps[-1].intens,c='magenta',alpha=0.5,lw=1,label="residual")
    [ax[-1].plot(isolist_comps[i].sma**0.25,isolist_comps[i].intens,label=comp_names[i],linestyle="--") for i in range(len(comp_names)-1)]
    ax[-1].axvline(x=sma_cut**0.25,label=f"$sma_\\chi$={sma_cut:.1f}",lw=1,alpha=0.5)
    ax[-1].set_yscale('log')
    ax[-1].set_ylim(ymin=1)
    ax[-1].set_title(f"1D profile, $\chi^2$={chi_1d:.1f}")
    ax[-1].set_ylabel("log(I)")
    ax[-1].set_xlabel("sma$^{0.25}$")
    ax[-1].legend(fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5));

    fig.tight_layout()
    pdf.savefig();


if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to plot fit results
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--inDir", type=str, default="~/agn-result/fit", help="input directory of fit result files")
    parser.add_argument("--inFile", type=str, help="fit result file")
    parser.add_argument("--outDir", type=str, default="~/agn-result/fit", help="directory for fit results")
    parser.add_argument("--outFile", type=str, help="output file")
    args = parser.parse_args()
    
    # load fit component file
    objectName = args.inFile[:10]
    fitPath = os.path.expanduser(os.path.join(args.inDir, args.inFile))
    with open (fitPath, "rb") as f:
        d = pickle.load(f)
    imageAGN = d['agn']
    isolist_agn= d['agn-iso']
    # create output file
    if args.outFile:
        savepath = os.path.expanduser(os.path.join(args.outDir,args.outFile))
    else:
        savepath = os.path.expanduser(os.path.join(args.outDir,objectName+".pdf"))
    pdf = PdfPages(savepath)

    # get model names
    model_names = list(d.keys())[:-2]
    sorted_ind = np.argsort([d[m].fit_result.fitStat for m in model_names])
    
    with PdfPages(savepath) as pdf:
        for i in sorted_ind:
            model = d[model_names[i]]
            plot_everything(pdf,image=imageAGN,m=model.comp_im[-1],modelname= model.model_name,
                            isolist_data=isolist_agn,isolist_comps=model.iso_comp,
                            comp_names=model.comp_name, fs=model.fit_result.fitStat)
            plot_model_components(pdf,comp_ims=model.comp_im,comp_names=model.comp_name, comp_pos=model.comp_pos,
                                  isolist_comps=model.iso_comp,plot_iso=True)
    print("Done: ", objectName)