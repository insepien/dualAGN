import numpy as np
from modelComponents import modelComps
import os
import pickle


def replace_fit_info(d_fit, d_nb):
    """replace a model in mass fit file with results in notebook fit file"""
    modelname_nb = list(d_nb['modelNames'].keys())[0]
    modelnames = list(d_fit['modelNames'].keys())
    # find index of model to replace
    ind = np.where(np.array(list(d_fit['modelNames'].keys()))==modelname_nb)[0][0]
    # replace all items but the image
    for key in d_fit: 
        if key != 'imageSS':
            if isinstance(d_fit[key],dict):
                d_fit[key][modelnames[ind]] = d_nb[key][modelname_nb][0]
            else:
                d_fit[key][ind] = d_nb[key][0]
    return d_fit

def add_fit_info(d_fit, d_nb):
    """add a model in mass fit file with results in notebook fit file"""
    modelname_nb = list(d_nb['modelNames'].keys())[0]
    for key in d_fit: 
        if key != 'imageSS':
            if isinstance(d_fit[key],dict):
                d_fit[key][modelname_nb] = d_nb[key][modelname_nb][0]
            else:
                d_fit[key].append(d_nb[key][0])
    return d_fit

if __name__ =="__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to replace a fit result with a nb fit
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--on", type=str, help='object name')
    parser.add_argument("--nbDir", type=str, default="~/research-data/agn-result/fit/fit_masked_n.3to6/nb_fit/", help="path to nb fit")
    parser.add_argument("--massDir", type=str, default="~/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit/", help="path to mass fit")
    parser.add_argument("--otherDir", type=str, default="~/research-data/agn-result/fit/fit_masked_n.3to6/other/", help="path to other folder")
    parser.add_argument("--replace", action='store_true', help='flag to replace model')
    parser.add_argument("--add", action='store_true', help='flag to add model')

    args = parser.parse_args()
    current_mass_fit_filename = os.path.expanduser(args.massDir+args.on+".pkl")
    # load nb fit
    with open(os.path.expanduser(args.nbDir+args.on+"_nb.pkl"),"rb") as f:
        d_nb = pickle.load(f)
    # load mass fit file
    with open(current_mass_fit_filename,"rb") as f:
        d_fit = pickle.load(f)
    # replace 1 model infos
    if args.replace:
        d_fit_new = replace_fit_info(d_fit,d_nb)
        # rename old mass fit file
        new_name = os.path.expanduser(args.otherDir+args.on+"_bad_guess.pkl")
        new_pdf_name = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/other/"+args.on+"_bad_guess.pdf")
    # add 1 model infos
    if args.add:
        d_fit_new = add_fit_info(d_fit,d_nb)
        # rename old mass fit file
        new_name = os.path.expanduser(args.otherDir+args.on+"_no_extra_model.pkl")
        new_pdf_name = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/other/"+args.on+"_no_extra_model.pdf")
    # rename old fit file and move out of fit directory
    os.rename(current_mass_fit_filename, new_name)
    # rename and move pdf
    current_pdf_filename = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/"+args.on+".pdf")
    os.rename(current_pdf_filename, new_pdf_name)
    # save new mass fit file
    pickle.dump(d_fit,open(os.path.expanduser(current_mass_fit_filename),"wb"))
    print("new file saved")
