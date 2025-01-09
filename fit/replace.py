import numpy as np
from modelComponents import modelComps
import os
import pickle


def replace_fit_info(d_fit, d_nb):
    modelname_nb = list(d_nb['modelNames'].keys())[0]
    modelnames = list(d_fit['modelNames'].keys())
    ind = np.where(np.array(list(d_fit['modelNames'].keys()))==modelname_nb)[0][0]
    for key in d_fit: 
        if key != 'imageSS':
            if isinstance(d_fit[key],dict):
                d_fit[key][modelnames[ind]] = d_nb[key][modelname_nb]
            else:
                d_fit[key][ind] = d_nb[key][0]
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
    args = parser.parse_args()
    current_mass_fit_filename = os.path.expanduser(args.massDir+args.on+".pkl")
    # load nb fit
    with open(os.path.expanduser(args.nbDir+args.on+"_nb.pkl"),"rb") as f:
        d_nb = pickle.load(f)
    # load mass fit file
    with open(current_mass_fit_filename,"rb") as f:
        d_fit = pickle.load(f)
    # replace 1 model infos
    d_fit_new = replace_fit_info(d_fit,d_nb)
    # rename old mass fit file
    new_name = os.path.expanduser(args.otherDir+args.on+"_bad_guess.pkl")
    try:
        # Rename the file
        os.rename(current_mass_fit_filename, new_name)
        print(f"Fit file renamed")
    except FileNotFoundError:
            print(f"The file {current_mass_fit_filename} does not exist.")
    except Exception as e:
            print(f"An error occurred: {e}")
    # save new mass file
    pickle.dump(d_fit,open(os.path.expanduser(current_mass_fit_filename),"wb"))
    # rename pdf
    try:
        current_pdf_filename = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/"+args.on+".pdf")
        new_pdf_name = os.path.expanduser("~/research-data/agn-result/fit/fit_masked_n.3to6/other/"+args.on+"_bad_guess.pdf")
        os.rename(current_mass_fit_filename, new_name)
    except FileNotFoundError:
            print(f"The file {current_pdf_filename} does not exist.")
    print("Finished replacing")
