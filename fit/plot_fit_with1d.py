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
import pandas as pd
from scipy.stats import chi2
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable

# medium_font_size = 22
# plt.rcParams['font.size'] = medium_font_size
# plt.rcParams['axes.labelsize'] = medium_font_size
# plt.rcParams['axes.titlesize'] = medium_font_size
# plt.rcParams['xtick.labelsize'] = medium_font_size
# plt.rcParams['ytick.labelsize'] = medium_font_size
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.family'] = 'monospace'

plt.rcParams["text.usetex"] = True
sns.set_context("paper",font_scale=1.75)
figparams = {'font.family': 'DejaVu Sans',
        'font.serif':'Times',
        'text.latex.preamble': [r'\usepackage{amsmath}'],
        'text.usetex':True,
            'hatch.linewidth' : 3.0}

def cal_pval(df,i,j,sig_lev=0.05):
    """check if models fit equally well (null hypothesis)
        calculate p-value given 2 row indices in summary df using difference in chi2 and dof
        compare with significance level, default 0.05, to reject null"""
    del_dof = np.abs(df.loc[i,"dof"] - df.loc[j,"dof"])
    del_chi = np.abs(df.loc[i,'chi2'] - df.loc[j,'chi2'])
    df.at[i,"del chi"] = del_chi
    df.at[i,"reject null"] = (1 - chi2.cdf(del_chi,del_dof)) < sig_lev

def add_model(d_fit,nest_dict,args):
    """add a new model to the chi difference test given the nested model name"""
    new_model = list(d_fit['modelNames'].keys())[-1]
    nest_dict[new_model] = args.nestModelName
    return nest_dict

def test_chi_diff(d_fit):
    """perform chi difference test to compare 2 nested models"""
    n = len(d_fit['fitResults'])
    chisq = np.round([d_fit['fitResults'][i].fitStat for i in range(n)])
    modelnames = list(d_fit['modelNames'].keys())
    dofs = [len(d_fit['fitResults'][i].params) for i in range(n)]
    df = pd.DataFrame(data=[modelnames,dofs,chisq], index=['model', "dof",'chi2']).T
    # get index of the nested model, i.e the 11th model is nested in model of row 12, 9 nested in 11, etc.
    nest_dict = {'sersic'                    : None,
                'sersic+sersic'              : 'sersic',
                'sersic,sersic'              : 'sersic+sersic',
                'psf+sersic'                 : 'sersic',
                'psf,sersic'                 : 'psf+sersic',
                'psf,sersic+exp'             : 'exp+sersic+psf',
                'psf,sersic+sersic(n1)'      : 'sersic+sersic(n1)+psf',
                'exp+sersic+psf'             : 'psf+sersic',
                'sersic+sersic(n1)+psf'      : 'psf+sersic',
                'sersic+psf,psf'             : 'psf+sersic',
                'sersic+psf,sersic+psf'      : 'sersic+psf,psf',
                'sersic+sersic+sersic'       : 'sersic+sersic',
                'sersic+psf,sersic'          : 'psf+sersic',
                'sersic+sersic,sersic'       : 'sersic+sersic+sersic',
                'sersic+sersic,sersic+sersic': 'sersic+sersic,sersic',
                'bar+sersic'                 : 'sersic',
                'bar,sersic'                 : 'bar+sersic'}
    # if plotting an extra model
    if args.addModel:
        nest_dict = add_model(d_fit, nest_dict, args)
    # get indices of the nests
    df['nests ind'] = [np.where(df==modname)[0][0] if modname!= None else None for modname in list(nest_dict.values())]
    # check if model at index k fit equally well as model with index in "nests ind"
    df['del chi'] = None
    df['reject null'] = None
    for k in range(1,n):
        cal_pval(df, k, df.loc[k,"nests ind"])
    return df


def pix_to_arcsec(imageFile,framelim):
    """convert pixel to arcsec"""
    w = WCS(imageFile)
    # convert pixel to degree
    ra1,dec1 = w.pixel_to_world_values(0,0)
    ra2,dec2 = w.pixel_to_world_values(framelim,framelim)
    # plate scale from data fits files
    arcsec_per_pix = 0.16*u.arcsec
    return arcsec_per_pix, [ra1,dec1,ra2,dec2]

def surface_brightness(intensity, area, magZPT):
    """calculate surface brightness. area must be in arcsec square
        following Eq. 35 in galfit manual 
        https://users.obs.carnegiescience.edu/peng/work/galfit/README.pdf """
    return -2.5*np.log10(intensity/area)+ magZPT


def radial_plot_params(imageFile, framelim, isolist_data,isolist_comps,hdu_exp,z=0.2):
    """calculate sma and surface brightness [mag/arcsec square] for 1d plot"""
    # convert pixel to arcsec and kpc
    arcsec_per_pix, skycoords = pix_to_arcsec(imageFile,framelim)
    sma_arcsec = isolist_data.sma*arcsec_per_pix
    sma_kpc = (cosmo.angular_diameter_distance(z)*sma_arcsec.to('rad').value).to('kpc')
    sma_15kpc_to_arcsec = (((15*u.kpc/cosmo.angular_diameter_distance(z)).to("").value)*u.rad).to('arcsec').value
    # calculate isophote areas
    semi_minor_arcsec = np.sqrt((1-isolist_data.eps**2)*sma_arcsec**2)
    areas = (np.pi*sma_arcsec*semi_minor_arcsec).value
    # find surface brightness and associated errors
    magZPT = hdu_exp.header['MAGZP']
    mu_data = [surface_brightness(i,areas,magZPT) for i in [isolist_data.intens,isolist_data.intens-isolist_data.int_err,isolist_data.intens+isolist_data.int_err]]
    mu_models = [[surface_brightness(i,areas,magZPT) for i in [isolist_comps[j].intens,isolist_comps[j].intens-isolist_comps[j].int_err,isolist_comps[j].intens+isolist_comps[j].int_err]] for j in range(len(isolist_comps))]
    # find difference in surface brightness
    del_fluxes = [isolist_data.intens/isolist_comps[-1].intens,
                  (isolist_data.intens-isolist_data.int_err)/(isolist_comps[-1].intens-isolist_comps[-1].int_err), 
                  (isolist_data.intens+isolist_data.int_err)/(isolist_comps[-1].intens+isolist_comps[-1].int_err)]
    del_mu = [-2.5*np.log10(i) for i in del_fluxes]
    return (sma_arcsec, sma_kpc, mu_data, mu_models, del_mu, skycoords, sma_15kpc_to_arcsec)
    

def plot_everything(pdf, image, model_, rp_params_, model_index, isolist_data, rank, args):
    # getting 2D plot params
    m = model_.comp_im[-1]
    modelname = model_.model_name
    comp_names = model_.comp_name

    # change name so bulge number doesn't repeat in 1d plot
    bulge_count  = 0
    psf_count = 0
    better_comp_names = []
    for k in range(len(comp_names)):
        if "bulge" in comp_names[k]:
            better_comp_names.append(rf"S\'ersic {bulge_count}")
            bulge_count += 1
        elif "psf" in comp_names[k]:
            better_comp_names.append(f"PSF {psf_count}")
            psf_count += 1
        else:
            better_comp_names.append(comp_names[k])

    comp_pos = model_.comp_pos
    fs = model_.fit_result.fitStat
    fsr = model_.fit_result.fitStatReduced
    # getting 1D plot params
    sma_arcsec, sma_kpc, mu_data, mu_models, del_mu, skycoords, sma_15kpc_to_arcsec = rp_params_
    # choose colormap
    colors = sns.color_palette("colorblind", len(comp_names))
    ls = ['-', '--', '-.', ':']
    cmapp = sns.color_palette(args.cmap, as_cmap=True).reversed()

    # Create grid and add subplots
    fig = plt.figure(figsize=(14, 4),layout='tight')
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1], width_ratios=[1,1,1,1.5],hspace=0.1,wspace=0.05)
    ax1 = fig.add_subplot(gs[:, 0],xlabel='RA (deg)',ylabel='DEC (deg)') 
    ax2 = fig.add_subplot(gs[:, 1],xticks=[],yticks=[])
    ax3 = fig.add_subplot(gs[:, 2],xticks=[],yticks=[])
    ax4a = fig.add_subplot(gs[0, 3]) 
    ax4b = fig.add_subplot(gs[1, 3])   
    # formatting ticks as RA and DEC in degree
    xticks = np.linspace(skycoords[0],skycoords[2],4)
    yticks = np.linspace(skycoords[1],skycoords[3],4)
    ax1.set_xticks(np.linspace(0,image.shape[0],4))
    ax1.set_yticks(np.linspace(0,image.shape[0],4))
    ax1.set_xticklabels([f'{x:.3f}' for x in xticks])
    ax1.set_yticklabels([f'{y:.3f}' for y in yticks],rotation=90)
    ax1.tick_params(direction='in',colors='white', labelcolor='black')

    # plot 2d and colorbars
    ax = [ax1,ax2,ax3,ax4a,ax4b]
    # try to put colorbar for ax0 and ax1
    vmin, vmax = min(image.min(), m.min()), max(image.max(), m.max())
    sm = plt.cm.ScalarMappable(cmap=cmapp, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    im = [ax[i].imshow([image, m][i], 
                       norm='symlog',cmap=cmapp,vmin=vmin, vmax=vmax) for i in range(2)]
    im2 = ax[2].imshow(image-m,
                       cmap=cmapp)

    #################testing colorbars
    div0 = make_axes_locatable(ax[0])
    cax0 = div0.append_axes("bottom", size='5%', pad=0.1)
    cax0.axis('off')

    div1 = make_axes_locatable(ax[1])
    cax1 = div1.append_axes("bottom", size='5%', pad=0.1)
    cbr1 = plt.colorbar(sm,cax=cax1,orientation='horizontal')
    cax1.xaxis.set_ticks_position("bottom")

    div2 = make_axes_locatable(ax[2])
    cax2 = div2.append_axes("bottom", size='5%', pad=0.1)
    cbr2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cax2.xaxis.set_ticks_position("bottom")
    ###############################

    #fig.colorbar(im2,ax=ax[2], orientation='horizontal',location='bottom',pad=0.05)
    #fig.colorbar(sm, ax=ax[1], orientation='vertical',location='right',shrink=0.5)
    # radial plot data
    ax[3].plot(sma_arcsec[1:], mu_data[0][1:],
               label="Data", c="k")
    ax[3].fill_between(sma_arcsec[1:].value, mu_data[1][1:],
                       mu_data[2][1:], color="k", alpha=0.2)
    # radial plot components
    # plot start from [1:] since first point is a single point, so area is ~ 0, so mu~inf
    [ax[3].plot(sma_arcsec[1:], mu_models[i][0][1:], 
                label=better_comp_names[i], linestyle=ls[i], c=colors[i]) for i in range(len(comp_names)-1)]
    [ax[3].fill_between(sma_arcsec[1:].value, mu_models[i][1][1:], mu_models[i][2][1:],
                        color=colors[i], alpha=0.5) for i in range(len(comp_names)-1)]
    ax[4].plot(sma_kpc[1:], del_mu[0][1:],
               c='rebeccapurple', linestyle="dashdot")
    ax[4].fill_between(sma_kpc[1:].value, del_mu[1][1:], del_mu[2][1:],
                       color='rebeccapurple',alpha=0.5)
    ax[4].axhline(y=0,linestyle='--',c="k",alpha=0.5,lw=1)
    # format ticks
    ax[3].set_ylim(ymax=30)
    ax[3].invert_yaxis()
    ax[3].set_xlabel("R[arcsec]")
    ax[3].set_ylabel("$\mu$ [mag arcsec$^{-2}$]")
    ax[3].set_xscale('log')
    ax[3].xaxis.set_label_position('top') 
    ax[3].xaxis.set_ticks_position('top') 
    ax[3].legend(fontsize=12,loc='upper right')
    
    
    ax[4].set_xlabel("R[kpc]")
    ax[4].set_ylabel("$\Delta$m") 
    ax[4].set_ylim((-0.5,0.5))
    ax[4].set_xscale('log')
    [ax.yaxis.set_label_position('right') for ax in [ax4a,ax4b]]
    [ax.yaxis.set_ticks_position('right') for ax in [ax4a,ax4b]]

    # paper or big plot
    if args.paper:
        [ax[i].set_title([args.oname,
                    f"Model",
                    'Residual'][i]) for i in range(3)]
        fig.savefig(os.path.expanduser(os.path.join(args.outDir,args.outFile)),bbox_inches='tight', pad_inches=0.2)
    else:
        # formatting model name
        if len(modelname) > 16 and ',' in modelname:
            modelname.replace('\n','')
            modelname = modelname.split(",")[0]+',\n'+modelname.split(",")[1]
        [ax[i].set_title([args.oname,
                    f"Model {model_index}:\n{modelname}",
                    f'Residual,\n$\chi^2_r$={fsr:.3f}\n$\chi^2$={fs:.0f}'][i]) for i in range(3)]
        fig.colorbar(im[1],ax=[ax[0],ax[1]],
                 orientation='vertical',location='right',shrink=0.5)
        ax[2].legend(loc='upper left', fontsize='x-small')
        # option to put midframe point on 2D
        if args.plot00:
            midF = image.shape[0]//2
            ax[0].plot(midF,midF,
                    marker='x',color="k",lw=2,alpha=0.3)
        # putting model positions on
        [[ax[j].plot(comp_pos[i][0]-1, comp_pos[i][1]-1, 
                marker='x',markersize=5, color=["k","","w"][j],alpha=[0.2,"",0.2][j]) for i in range(len(comp_pos))] for j in (0,2)]
        # putting 10kpc isophote on
        sma5_pix = sma_15kpc_to_arcsec/0.16 #plate scale of 0.16 arcsec/pix
        plot_1isophote(ax=ax[2],sma=sma5_pix,isolist=isolist_data,label_="15kpc")
        # add chi2 test stats
        rejectNull = str(df_chi.loc[model_index,'reject null'])
        delChi = df_chi.loc[model_index,"del chi"]
        compare_ind = df_chi.loc[model_index, "nests ind"]
        fig.text(0.01, 0.99, 
                f"Compare model: {compare_ind}\nReject null: {rejectNull}\n $\Delta \chi^2$={delChi}", 
                verticalalignment='top', horizontalalignment='left')
        
        # add model rank by chi 2
        fig.text(0.01, 0.01, 
                f"Model rank: {rank:.0f}", 
                verticalalignment='bottom', horizontalalignment='left')
        pdf.savefig(bbox_inches='tight', pad_inches=0.2);


def plot_model_components(pdf, model_, serinds, args):
    """plot 2D model components and check residual with model image"""
    # load plotting params
    comp_ims = model_.comp_im
    comp_names = model_.comp_name
    comp_pos = model_.comp_pos
    isolist_comps = model_.iso_comp
    # choose colormap
    clmap = sns.color_palette(args.cmap, as_cmap=True).reversed()
    ncom = len(comp_names)

    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(ncom*4,3))
    im = [ax[i].imshow(comp_ims[i],
                       norm='symlog', cmap=clmap) for i in range(ncom)]
    # add positions
    [ax[i].text(0.05, 0.05, 
                f"(x,y)=({comp_pos[i][0]:.1f},{comp_pos[i][1]:.1f})", 
                transform=ax[i].transAxes, fontsize=8, color='w') for i in range(ncom-1)]
    # add sersic indices if sersic
    bulge_count  = 0
    psf_count = 0
    for k in range(ncom):
        if "bulge" in comp_names[k]:
            axtit = f"sersic {bulge_count}, n={serinds[bulge_count]:.2f}"
            bulge_count += 1
        elif "psf" in comp_names[k]:
            axtit = f"psf {psf_count}"
            psf_count += 1
        else:
            axtit = comp_names[k]
        ax[k].set_title(axtit)
    # check if model-comps is 0
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],
                            norm='symlog', cmap=clmap))
    ax[-1].set_title("model-comps")
    # add colorbars
    [fig.colorbar(im[i], ax=ax[i], shrink=0.5).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
    # option to plot isophote contours 
    if args.plotIso:
        for i in range(len(isolist_comps)):
            plot_isophotes(ax[i], isolist_comps[i], num_aper=5)
    fig.tight_layout()
    pdf.savefig();


def plot_isophotes(ax,isolist,num_aper=10):
    """plot aperatures on image"""
    for sma in np.linspace(isolist.sma[0],isolist.sma[-1],num_aper):
        # get isophote closest to some sma value
        iso = isolist.get_closest(sma)
        # sample the x,y coords of that isphote
        x, y, = iso.sampled_coordinates()
        ax.plot(x, y, 
                color='white', linewidth="0.5", alpha=0.3)
        

def plot_1isophote(ax,sma,isolist,label_):
    """plot aperatures on image"""
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax.plot(x, y, color='white',linewidth="0.3",alpha=0.5,label=label_)
        

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
    # args to add a model to big pdf
    parser.add_argument("--addModel", action='store_true', help='flag if plotting an extra model')
    parser.add_argument("--nestModelName", type=str, help='name of model nesting/nested by extra model')
    # args for everything plot
    parser.add_argument("--plotIso", action='store_true', help="option to plot isophotes in component plots")
    parser.add_argument("--sorted", action='store_true', help="option to sort model by best chi")
    parser.add_argument("--plot00", action='store_true', help = 'flag for plot the frame center')
    parser.add_argument('--cmap', type=str, default="ch:s=-.3,r=.6")
    # args for paper plot
    parser.add_argument("--paper", action='store_true', help="flag to make plot for paper")
    parser.add_argument("--modelName", type=str, help='model name for paper plot')
    args = parser.parse_args()
    
    # load fit file to get sersic index for component plots
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
    
    # get model names and sort index from best to worst chi reduced
    model_names = list(d.keys())[:-2]
    sorted_ind = np.argsort([d[m].fit_result.fitStat for m in model_names])
    sorted_model_num = [np.where(sorted_ind==i)[0][0] for i in range(len(sorted_ind))]

    # do chi2 diff test
    df_chi = test_chi_diff(d_fit)
    # find redshift
    alpaka = pd.read_pickle("/home/insepien/research-data/alpaka/alpaka_39fits.pkl")
    redshift = np.mean(alpaka[alpaka['Desig']==args.oname]['Z'])

    # choose paper plot or analysis plot
    if args.paper:
        model_ind = np.where(np.array(model_names)==args.modelName)[0][0]
        print(model_ind)
        model = d[model_names[model_ind]]
        model_rank = ""
        rp_params = radial_plot_params(imageFile=imageAGNFile, framelim=imageAGN.shape[0],
                                                isolist_data=isolist_agn,isolist_comps=model.iso_comp,
                                                hdu_exp=hdu0,z=redshift)
        plot_everything(pdf="", image=imageAGN, model_=model, rp_params_ = rp_params,
                        model_index = model_ind, isolist_data=isolist_agn, rank=model_rank, args=args)
    else:
        # create output file
        if args.outFile:
            savepath = os.path.expanduser(os.path.join(args.outDir,args.outFile))
        else:
            savepath = os.path.expanduser(os.path.join(args.outDir,args.oname+".pdf"))
        pdf = PdfPages(savepath)
        with PdfPages(savepath) as pdf:
            if args.sorted:
                order = sorted_ind
            else:
                order = range(len(sorted_ind))
            for model_ind,model_rank in zip (order,sorted_model_num):
                model = d[model_names[model_ind]]
                rp_params = radial_plot_params(imageFile=imageAGNFile, framelim=imageAGN.shape[0],
                                                isolist_data=isolist_agn,isolist_comps=model.iso_comp,
                                                hdu_exp=hdu0,z=redshift)
                plot_everything(pdf, image=imageAGN, model_=model, rp_params_ = rp_params,
                                model_index = model_ind, isolist_data=isolist_agn, rank=model_rank, args=args)
                plot_model_components(pdf, model_=model, serinds=ns[model_ind], args=args)
        print("Done: ", args.oname)