from photutils.isophote import EllipseGeometry, IsophoteList, EllipseSample, Isophote
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter


def makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
                 h1,h2,h_lim,alpha,alpha_lim,sky):
    """Return Sersic, PSF, and Gaussian model parameter dictionary"""
    # Sersic
    """sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 'fixed'],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}"""
    sersic = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 0.3, 6],
    'I_e': [I_ss, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss, rss_lim[0],rss_lim[1]]}
    sersic_dict = {'name': "Sersic", 'label': "bulge", 'parameters': sersic}

    sersic1 = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 0.3, 6],
    'I_e': [I_ss*4, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss/4, rss_lim[0],rss_lim[1]]}
    sersic1_dict = {'name': "Sersic", 'label': "bulge 1", 'parameters': sersic1}

    sersic2 = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [n_ss, 0.3, 6],
    'I_e': [I_ss*4, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss/4, rss_lim[0],rss_lim[1]]}
    sersic2_dict = {'name': "Sersic", 'label': "bulge 2", 'parameters': sersic2}
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
    # sersic with n=1 (exponential)
    sersic_n1 = {'PA': [PA_ss, PA_lim[0],PA_lim[1]], 'ell_bulge': [ell_ss, ell_lim[0],ell_lim[1]], 'n': [1, "fixed"],
    'I_e': [I_ss*4, Iss_lim[0],Iss_lim[1]], 'r_e': [r_ss/4, rss_lim[0],rss_lim[1]]}
    sersic_n1_dict = {'name': "Sersic", 'label': "bulge n=1", 'parameters': sersic_n1}
    # fixed flat sky
    flatsky = {'I_sky': [sky,'fixed']}
    flatsky_dict = {'name': "FlatSky", 'label': "flat_sky", 'parameters':flatsky}

    return sersic_n1_dict, sersic2_dict, sersic1_dict, sersic_dict, psf_dict, flatbar_dict, exp_dict, flatsky_dict


class modelComps:
    def __init__(self, modelname, comp_im, comp_pos, comp_name,fit_result):
        self.model_name = modelname
        self.comp_im = comp_im
        self.comp_pos = comp_pos
        self.comp_name = comp_name
        self.fit_result = fit_result
        self.iso_comp = None

    def make_model_isophotes(self,isolist_data):
        """sample model components along isophotes from data fit
            center point is midframe"""
        isolist_comps=[]
        midf = self.comp_im[0].shape[0]//2
        # flag: if psf, use circular annulus
        circ = [self.comp_name[i]=='psf' for i in range(len(self.comp_name))]
        for i in range(len(self.comp_name)):
            # list to store one component's isophote
            isolist_ = []
            for iso in isolist_data[1:]:
                # get config of each data isophote
                g = iso.sample.geometry
                ell = 0 if circ[i] else g.eps
                # make isophote
                gn = EllipseGeometry(g.x0,g.y0, g.sma, ell, g.pa)
                # sample the image component along isophote
                sample = EllipseSample(self.comp_im[i],g.sma,geometry=gn)
                sample.update()
                iso_ = Isophote(sample,0,True,0)
                isolist_.append(iso_)
            # convert to isophote list object
            isolist = IsophoteList(isolist_)
            # add central brightness
            g = EllipseGeometry(midf,midf, 0.0, 0., 0.)
            sample = CentralEllipseSample(self.comp_im[i], 0., geometry=g)
            fitter = CentralEllipseFitter(sample)
            center = fitter.fit()
            isolist.append(center)
            isolist.sort()
            isolist_comps.append(isolist)
        self.iso_comp = isolist_comps