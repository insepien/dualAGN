import numpy as np
import pickle
import os
import pyimfit
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter
from modelComponents import modelComps, makeModelDict


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
    """make 1d profile of data"""
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


if __name__=="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to make fit result components and isophotes
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--psfPath", type=str, default="~/research-data/psf-results/psf_pkls", help="path to psf directory")
    parser.add_argument("--inDir", type=str, default="~/agn-result/fit", help="path to fit result directory")
    parser.add_argument("--inFile", type=str, help="fit result pickle file")
    parser.add_argument("--outDir", type=str, default="~/agn-result/fit/components", help="output directory")
    parser.add_argument("--outFile", type=str,  help="output file name")
    args = parser.parse_args()

    # load psf
    objectName = args.inFile[:10]
    psf_fileName = "psf_"+objectName+".pkl"
    psfPath = os.path.join(args.psfPath, psf_fileName)
    with open (os.path.expanduser(psfPath), "rb") as fp:
        p = pickle.load(fp)
    epsf = p['psf'].data

    # load fit results
    fitPath = os.path.join(args.inDir, args.inFile)
    with open (os.path.expanduser(fitPath), "rb") as fd:
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
                                                                     PA_lim=[0,360], ell_lim=[0.0,1.0],
                                                                     Iss_lim=[0.1,Imax], rss_lim=[0.1,framelim], Itot_lim=[0.1,1e4],
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
    pickle.dump(data_to_save,open(os.path.expanduser(filename),"wb"))
    print("Done: "+objectName)



    
    