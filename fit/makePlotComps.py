import numpy as np
import pickle
import os
import pyimfit
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from photutils.isophote import Ellipse, EllipseGeometry, IsophoteList
from photutils.isophote import EllipseSample, Isophote
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages


def makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, I_lim,  Iss_lim, rss_lim, Itot_lim,
                 sigma, sigma_lim,
                 h1,h2,h_lim,alpha,alpha_lim):
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
    # flat bar
    flatbar = {'PA':[PA_ss, PA_lim[0],PA_lim[1]], 'ell':[ell_ss, ell_lim[0],ell_lim[1]],
               'deltaPA_max':[PA_ss, PA_lim[0],PA_lim[1]], 'I_0':[I_ss, Iss_lim[0],Iss_lim[1]],
               'h1':[h1, h_lim[0],h_lim[1]], 'h2':[h2, h_lim[0],h_lim[1]], 
               'r_break':[r_ss, rss_lim[0],rss_lim[1]], 'alpha':[alpha,alpha_lim[0],alpha_lim[1]]}
    flatbar_dict = {'name': "FlatBar", 'label': "flat_bar", 'parameters':flatbar}
    # Exponential
    exponential = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell': [ell_ss, ell_lim[0],ell_lim[1]], 
                   'I_0': [I_ss, Iss_lim[0],Iss_lim[1]], 'h': [h1, h_lim[0],h_lim[1]]}
    exp_dict = {'name': "Exponential", 'label': "disk", 'parameters':exponential}
    return sersic_dict, psf_dict, flatbar_dict, exp_dict


def make_model_components(config,imshape):
    """make model component images from best fit config"""
    comp_names = config.functionLabelList()
    comp_ims=[]
    comp_pos = []
    for i in range(len(config.getModelAsDict()['function_sets'])):
        posX = config.getModelAsDict()['function_sets'][i]['X0']
        posY = config.getModelAsDict()['function_sets'][i]['Y0']
        functions = config.getModelAsDict()['function_sets'][i]['function_list']
        for j in range(len(functions)):
            funcset_dict = {'X0': posX, 'Y0': posY, 'function_list': [functions[j]]}
            model_dict = {'function_sets': [funcset_dict]}
            model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
            imfit_fitter = pyimfit.Imfit(model,epsf)
            comp_ims.append(imfit_fitter.getModelImage(shape=(imshape,imshape)))
            comp_pos.append([posX[0],posY[0]])
    return comp_ims, comp_pos, comp_names


def profile_1D(semiA,image,PA=180,ell=0.5):
        """make 1D elliptical profiles"""
        # create guess ellipse
        pos0 = image.shape[0]//2
        geometry = EllipseGeometry(x0=pos0, y0=pos0, sma=semiA, eps=ell,
                                pa=PA * np.pi / 180.0)
        # load image and geometry
        ellipse = Ellipse(image, geometry)
        # do isophote fit
        isolist = ellipse.fit_image()
        return isolist

def make_data_isophotes(data,sma,midFrame):
    isolist_data = profile_1D(semiA=sma,image=data)
    # discard first isophote and make new
    isolist_data = isolist_data[1:]
    g = EllipseGeometry(midFrame,midFrame, 0.0, 0., 0.)
    sample = CentralEllipseSample(data, 0., geometry=g)
    fitter = CentralEllipseFitter(sample)
    center = fitter.fit()
    isolist_data.append(center)
    isolist_data.sort()
    return isolist_data


def plot_isophotes(ax,isolist,num_aper=10):
    """plot aperatures on image"""
    for sma in np.linspace(isolist.sma[0],isolist.sma[-1],num_aper):
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax.plot(x, y, color='white',linewidth="0.5")


def plot_model_components(comp_ims,comp_names,isolist_comps,plot_iso=False):
    """plot 2D model components and check residual with model image"""
    ncom = len(comp_names)
    fig,ax = plt.subplots(nrows=1,ncols=ncom+1, figsize=(ncom*3+3,3))
    im = [ax[i].imshow(comp_ims[i],norm='symlog') for i in range(ncom)]
    [ax[i].set_title(comp_names[i]) for i in range(ncom)]
    im.append(ax[-1].imshow(np.sum(comp_ims[:-1],axis=0)-comp_ims[-1],norm='symlog'))
    ax[-1].set_title("model-comps")
    [fig.colorbar(im[i], ax=ax[i], shrink=0.7).ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(len(ax))]
    if plot_iso:
        for i in range(len(isolist_comps)):
             plot_isophotes(ax[i],isolist_comps[i],num_aper=5)
    fig.tight_layout();


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
        midf = self.comp_im[0].shape[0]//2
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
            g = EllipseGeometry(midf,midf, 0.0, 0., 0.)
            sample = CentralEllipseSample(self.comp_im[i], 0., geometry=g)
            fitter = CentralEllipseFitter(sample)
            center = fitter.fit()
            isolist.append(center)
            isolist.sort()
            isolist_comps.append(isolist)
        self.iso_comp = isolist_comps


if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to make fit result components and isophotes
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--psfPath", type=str, default="../psfConstruction/psf_pkls", help="path to psf directory")
    parser.add_argument("--inDir", type=str, default="fit_pkls", help="path to fit result directory")
    parser.add_argument("--inFile", type=str, help="fit result pickle file")
    parser.add_argument("--outDir", type=str, default="fit_pkls/", help="output directory")
    parser.add_argument("--outFile", type=str,  help="output file name")
    args = parser.parse_args()

    # load psf
    objectName = args.inFile[:10]
    psf_fileName = "psf_"+objectName+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (psfPath, "rb") as fp:
        p = pickle.load(fp)
    epsf = p['psf'].data

    # load fit results
    fitPath = os.path.join(args.inDir, args.inFile)
    with open (fitPath, "rb") as fd:
        d = pickle.load(fd)
    image = d['imageSS']
    model_names = list(d['modelNames'].keys())
    configs = d['configs'] 
    model_images = d['modelImage']
    fit_results = d['fitResults']
    param_names = d['paramNames'] 


    # make model functions    
    Imax = image.max()
    framelim = image.shape[0]
    midF=framelim//2
    sersic_dict, psf_dict, flatbar_dict, exp_dict = makeModelDict(PA_ss=200, ell_ss=0.1, n_ss=1, I_ss=1, r_ss=20, Itot=1500,
                                                                     PA_lim=[0,360], ell_lim=[0.0,1.0], I_lim=[0.1,Imax],
                                                                     Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
                                                                     sigma = 5, sigma_lim = [1,20], 
                                                                     h1=10,h2=10,h_lim=[0.1,framelim],alpha=0.1,alpha_lim=[0.1,framelim])
    
    isolist_data = make_data_isophotes(data=image,sma=30,midFrame=midF)
    data_to_save = {}
    for i in range(len(model_names)):
        comp_ims, comp_pos, comp_names = make_model_components(configs[i],imshape=image.shape[0])
        comp_ims.append(model_images[i])
        comp_names.append("model")
        c = modelComps(model_names[i],comp_ims,comp_pos, comp_names,fit_results[i])
        c.make_model_isophotes(isolist_data)
        data_to_save[c.model_name] = c

    data_to_save['agn'] = image
    data_to_save['agn-iso'] = isolist_data
    if args.outFile:
        filename = os.path.join(args.outDir, args.outFile)
    else:
        filename = os.path.join(args.outDir, objectName+"_comp.pkl")
    pickle.dump(data_to_save,open(filename,"wb"))
    print("Done: "+objectName)



    
    