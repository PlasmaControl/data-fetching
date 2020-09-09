import numpy as np
from scipy import interpolate

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'astrolibpy/mpfit/'))
from mpfit import mpfit

from OMFITlib_fit import fit_rbf

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__),'..','lib'))

from transport_helpers import my_interp

def linear_interp_1d(in_x, in_t, value, uncertainty, out_x, t_out):
    final_sig=[]

    for time_ind in range(len(in_t)):
        excluded_inds=np.isnan(value[time_ind,:])
        y=value[time_ind,~excluded_inds]
        x=in_x[time_ind,~excluded_inds]
        err=uncertainty[time_ind,~excluded_inds]

        try:
            get_value=my_interp(x,y,kind='linear')
            final_sig.append(get_value(out_x))
        except:
            final_sig.append(np.zeros(len(out_x)))
            
    final_sig=np.array(final_sig)
    return final_sig

def nn_interp_2d(in_x, in_t, value, uncertainty, out_x, out_t):
    final_sig=[]

    excluded_inds=np.isnan(value)
    y=value[~excluded_inds]
    x=in_x[~excluded_inds]
    t=in_t[~excluded_inds]
    err=uncertainty[~excluded_inds]
    
    def scale(signal, basis):
        return (signal-min(basis))/(max(basis)-min(basis))
    
    t_scaled=scale(t,out_t) 
    out_t_scaled=scale(out_t,out_t)

    x_scaled=scale(x,[0,1]) #out_x)
    out_x_scaled=scale(out_x,[0,1]) #out_x)
    
    get_value=interpolate.NearestNDInterpolator(np.stack((t_scaled,x_scaled)).T,y)
    import itertools
    coords=np.array(list(itertools.product(out_t_scaled,out_x_scaled))).T
    final_sig=get_value(coords[0],coords[1])
    final_sig=final_sig.reshape((len(out_t_scaled),len(out_x_scaled)))
    
    return final_sig

def linear_interp_2d(in_x, in_t, value, uncertainty, out_x, out_t):
    final_sig=[]

    excluded_inds=np.isnan(value)
    y=value[~excluded_inds]
    x=in_x[~excluded_inds]
    t=in_t[~excluded_inds]

    inds=np.random.permutation(list(range(len(y))))
    y=y[inds[:100]]
    x=x[inds[:100]]
    t=t[inds[:100]]
    
    err=uncertainty[~excluded_inds]

    def scale(signal, basis):
        return (signal-min(basis))/(max(basis)-min(basis))
    
    t_scaled=scale(t,out_t) 
    out_t_scaled=scale(out_t,out_t)

    x_scaled=scale(x,[0,1]) #out_x)
    out_x_scaled=scale(out_x,[0,1]) #out_x)
    
    get_value=interpolate.interp2d(t_scaled,x_scaled,y,
                                   bounds_error=False,
                                   kind='linear')
    final_sig=get_value(out_t_scaled,out_x_scaled).T
    
    return final_sig

def rbf_interp_2d_new(in_x, in_t, value, uncertainty, out_x, out_t, debug=False):
    return fit_rbf(in_x, in_t, value, uncertainty, out_x, out_t)


def rbf_interp_2d(in_x, in_t, value, uncertainty, out_x, out_t, debug=False):
    final_sig=[]

    excluded_inds=np.isnan(value)
    y=value[~excluded_inds]
    x=in_x[~excluded_inds]
    t=in_t[~excluded_inds]
    err=uncertainty[~excluded_inds]

    n=100
    inds=np.random.permutation(list(range(len(y))))
    y=y[inds[:n]]
    x=x[inds[:n]]
    t=t[inds[:n]]
    
    def scale(signal, basis):
        return (signal-min(basis))/(max(basis)-min(basis))
    
    t_scaled=scale(t,out_t) 
    out_t_scaled=scale(out_t,out_t)

    x_scaled=scale(x,[0,1]) #out_x)
    out_x_scaled=scale(out_x,[0,1]) #out_x)
    
    get_value=interpolate.Rbf(t_scaled,x_scaled,y, function='Gaussian', epsilon=.1) #,metric='seuclidean')
    import itertools
    coords=np.array(list(itertools.product(out_t_scaled,out_x_scaled))).T
    final_sig=get_value(coords[0],coords[1])
    final_sig=final_sig.reshape((len(out_t_scaled),len(out_x_scaled)))

    return final_sig

# def real_to_psi_profile(psi, t0, value, uncertainty, standard_psi, t_out):
#     final_sig=[]

#     for ind in range(value.shape[1]):
#         excluded_inds=np.isnan(value[:,ind])
#         y=value[~excluded_inds,ind]
#         x=psi[~excluded_inds,ind]
#         err=uncertainty[~excluded_inds,ind]
#         try:
#             get_value=interpolate.interp1d(x,y,bounds_error=False)
#             final_sig.append(get_value(standard_psi))
#         except:
#             final_sig.append(np.zeros(len(standard_psi)))
            
#     final_sig=np.array(final_sig)
#     return final_sig

# regular 1D interpolation
# def real_to_psi_profile(psi, t0, value, uncertainty, standard_psi, t_out):
#     final_sig=[]
#     for ind in range(value.shape[1]):
#         get_value=interpolate.interp1d(psi[:,ind],value[:,ind],bounds_error=False,fill_value=1)
#         final_sig.append(get_value(standard_psi))
#     final_sig=np.array(final_sig)
#     return final_sig

def mtanh_1d(psi, t0, value, uncertainty, standard_psi, t_out):
    p0=np.array([1.0, 3.0, 0.01, 1.0, 0.01],dtype='float64')  #initial conditions

    parinfo=[]

    parinfo.append({'value':p0[0],
                    'fixed':0,
                    'limited':[1,0], #ped height positive
                    'limits':[0.1,0]}) #ped height positive
    parinfo.append({'value':p0[1],
                    'fixed':0,
                    'limited':[1,0], #offset positive
                    'limits':[0.001,0]}) #offset positive
    parinfo.append({'value':p0[2],
                    'fixed':0,
                    'limited':[1,0], #core slope positive
                    'limits':[0.0001,0]}) #core slope positive
    parinfo.append({'value':p0[3],
                    'fixed':0,
                    'limited':[1,1], #sym point pos between rho=[0.85,1.15]
                    'limits':[.85,1.15]}) #sym point pos between rho=[0.85,1.15]
    parinfo.append({'value':p0[4],
                    'fixed':0,
                    'limited':[1,0], #make width pos and >=0.01 to avoid inf from exp(z)
                    'limits':[0.01,0]}) #make width pos and >=0.01 to avoid inf from exp(z)
    
    # if len(error)!=0:
    #     error=np.array(error)

    # uncertainty=np.array(uncertainty)
    # value=np.array(value)
    # psi=np.array(psi)
    # max_uncertainty=np.nanmax(uncertainty)
    # uncertainty[np.isclose(value,0)]=1e30
    # uncertainty[np.isclose(uncertainty,0)]=1e30

    final_sig=[]
    
    for ind in range(value.shape[1]):
        excluded_inds=np.isnan(value[:,ind])

        # if len(error)!=0:
        #     excluded_inds|=(error[:,ind]==1)

    #    psi_to_rho=interpolate.interp1d(standard_psi,rho_grid[ind],fill_value='extrapolate')
    #    rho=psi_to_rho(psi[~excluded_inds,ind])
        x=psi[~excluded_inds,ind]
        fa = {'x': x, 'y': value[~excluded_inds,ind], 'err': uncertainty[~excluded_inds,ind]}

        m = mpfit(my_shifted_mtanh, p0, parinfo=parinfo, functkw=fa, quiet=1)

        final_sig.append(shifted_mtanh(standard_psi,m.params))
        
    return np.array(final_sig)

def mtanh(alpha,z):
    return ((1+alpha*z)*np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
def shifted_mtanh(x,p):
    a=p[0]
    b=p[1]
    alpha=p[2]
    xsym=p[3]
    hwid=p[4]
    
    z = (xsym-x)/hwid
    y = a * mtanh(alpha,z) + b

    return y

def my_shifted_mtanh(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = shifted_mtanh(x, p)
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return [status, (y-model)/err]
