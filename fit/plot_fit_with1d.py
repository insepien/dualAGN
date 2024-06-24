import numpy as np
import pickle
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from photutils.isophote import Ellipse, EllipseGeometry, IsophoteList
from photutils.isophote import EllipseSample, Isophote
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter
from matplotlib.ticker import FormatStrFormatter


medium_font_size = 14 
plt.rcParams['font.size'] = medium_font_size
plt.rcParams['axes.labelsize'] = medium_font_size
plt.rcParams['axes.titlesize'] = medium_font_size
plt.rcParams['xtick.labelsize'] = medium_font_size
plt.rcParams['ytick.labelsize'] = medium_font_size


class modelComps:
    def __init__(self, modelname, comp_im, comp_pos, comp_name,fit_result):
        self.model_name = modelname
        self.comp_im = comp_im
        self.comp_pos = comp_pos
        self.comp_name = comp_name
        self.fit_result = fit_result
        self.iso_comp = None

    def make_model_isophotes(self,isolist_data):
        isolist_comps=[]
        circ = [self.comp_name[i]=='psf' for i in range(len(self.comp_name))]
        for i in range(len(self.comp_name)):
            isolist_ = []
            for iso in isolist_data[1:]:
                g = iso.sample.geometry
                ell = 0 if circ[i] else g.eps
                gn = EllipseGeometry(g.x0,g.y0, g.sma, ell, g.pa)
                sample = EllipseSample(self.comp_im[i],g.sma,geometry=gn)
                sample.update()
                iso_ = Isophote(sample,0,True,0)
                isolist_.append(iso_)
            isolist = IsophoteList(isolist_)
            g = EllipseGeometry(self.fit_result.params[0],self.fit_result.params[1], 0.0, 0., 0.)
            sample = CentralEllipseSample(self.comp_im[i], 0., geometry=g)
            fitter = CentralEllipseFitter(sample)
            center = fitter.fit()
            isolist.append(center)
            isolist.sort()
            isolist_comps.append(isolist)
        self.iso_comp = isolist_comps


def plot_isophotes(ax,isolist,num_aper=10):
    """plot aperatures on image"""
    for sma in np.linspace(isolist.sma[0],isolist.sma[-1],num_aper):
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax.plot(x, y, color='white',linewidth="0.5")


def plot_model_components(pdf,comp_ims,comp_names,isolist_comps,plot_iso=False):
    """plot 2D model components and check residual with model image"""
    ncom = len(comp_names)
    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(14,3))
    im = [ax[i].imshow(comp_ims[i],norm='symlog') for i in range(ncom)]
    [ax[i].set_title(comp_names[i]) for i in range(ncom)]
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],norm='symlog'))
    ax[-1].set_title("model-comps")
    [fig.colorbar(im[i], ax=ax[i], shrink=0.7).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
    if plot_iso:
        for i in range(len(isolist_comps)):
             plot_isophotes(ax[i],isolist_comps[i],num_aper=5)
    fig.tight_layout()
    pdf.savefig();


def plot_everything(pdf,image,m,modelname,isolist_data,isolist_comps,comp_names,fs):
    # plot everything 
    fig,ax = plt.subplots(1,4,figsize=(14,3),gridspec_kw={'width_ratios': [1, 1, 1, 1.5]})
    #norms = [simple_norm([image, m, image-m][i],'log') for i in range(3)]
    im = [ax[i].imshow([image, m, image-m][i], norm='symlog') for i in range(3)]
    cbar = [fig.colorbar(im[i],ax=ax[i],shrink=0.5) for i in range(3)]
    [ax[i].set_title(['data',f"model: {modelname}",f'residual, $\chi^2$={fs:.3f}'][i]) for i in range(3)]

    ax[-1].plot(isolist_data.sma**0.25,isolist_data.intens,label="data",c="k",alpha=0.3)
    ax[-1].plot(isolist_comps[-1].sma**0.25,isolist_comps[-1].intens,c='k',linestyle="",marker="o",markersize=1,label="model")
    ax[-1].plot(isolist_comps[-1].sma**0.25,isolist_data.intens-isolist_comps[-1].intens,c='magenta',alpha=0.5,lw=1,label="residual")
    [ax[-1].plot(isolist_comps[i].sma**0.25,isolist_comps[i].intens,label=comp_names[i],linestyle="--") for i in range(len(comp_names)-1)]
    ax[-1].set_yscale('log')
    ax[-1].set_ylim(ymin=1)
    ax[-1].set_title("1D profile")
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
    parser.add_argument("--inDir", type=str, default="fit_pkls", help="input directory of fit result files")
    parser.add_argument("--inFile", type=str, help="fit result file")
    parser.add_argument("--outDir", type=str, default="fit_plots", help="directory for fit results")
    parser.add_argument("--outFile", type=str, help="output file")
    args = parser.parse_args()
    
    # load fit component file
    objectName = args.inFile[:10]
    fitPath = os.path.join(args.inDir, args.inFile)
    with open (fitPath, "rb") as f:
        d = pickle.load(f)
    imageAGN = d['agn']
    isolist_agn= d['agn-iso']
    # create output file
    if args.outFile:
        savepath = os.path.join(args.outDir,args.outFile)
    else:
        savepath = os.path.join(args.outDir,objectName+".pdf")
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
            plot_model_components(pdf,comp_ims=model.comp_im,comp_names=model.comp_name,
                                  isolist_comps=model.iso_comp,plot_iso=True)
    print("Done: ", objectName)