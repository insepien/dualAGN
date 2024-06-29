from photutils.isophote import EllipseGeometry, IsophoteList, EllipseSample, Isophote
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter


def makeModelDict(PA_ss, ell_ss, n_ss, I_ss, r_ss, Itot,
                 PA_lim, ell_lim, Iss_lim, rss_lim, Itot_lim,
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