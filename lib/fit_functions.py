import numpy as np
from scipy import interpolate

import sys
import os
from OMFITlib_fit import fit_rbf
from splines.pcs_fit_helpers import calculate_mhat, spline_eval

from transport_helpers import my_interp

def linear_interp_1d(in_x, in_t, value, uncertainty, out_x):
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


def csaps_1d(in_x, in_t, value, uncertainty, out_x):
    from csaps import csaps
    final_sig=[]

    for time_ind in range(len(in_t)):
        excluded_inds=np.isnan(value[time_ind,:])
        y=value[time_ind,~excluded_inds]
        x=in_x[time_ind,~excluded_inds]
        err=uncertainty[time_ind,~excluded_inds]

        inds=np.argsort(x)
        x=x[inds]
        y=y[inds]

        try:
            final_sig.append(csaps(x,y,out_x,smooth=0.99995)) #get_value(out_x))
        except:
            final_sig.append(np.zeros(len(out_x)))

    final_sig=np.array(final_sig)
    return final_sig

def spline_1d(in_x, in_t, value, uncertainty, out_x):
    final_sig=[]

    for time_ind in range(len(in_t)):
        excluded_inds=np.isnan(value[time_ind,:])
        y=value[time_ind,~excluded_inds]
        x=in_x[time_ind,~excluded_inds]
        err=uncertainty[time_ind,~excluded_inds]
        ordered_inds=np.argsort(x)
        x=x[ordered_inds]
        y=y[ordered_inds]
        try:
            get_value=interpolate.UnivariateSpline(x,y)
            final_sig.append(get_value(out_x))
        except:
            final_sig.append(np.zeros(len(out_x)))

    final_sig=np.array(final_sig)
    return final_sig

def pcs_spline_1d(in_x, in_t, value, uncertainty, out_x):
    final_sig=[]

    for time_ind in range(len(in_t)):
        excluded_inds=np.isnan(value[time_ind,:])
        y=value[time_ind,~excluded_inds]
        x=in_x[time_ind,~excluded_inds]
        err=uncertainty[time_ind,~excluded_inds]

        (mPsi,mHat)=calculate_mhat(x,y,p=0.5,dxMin=0.01)
        mPsi=mPsi[:-1] # still not sure why the last index is always 0, should talk to ricardo (TODO)
        mHat=mHat[:-1]
        splined_rot=spline_eval(mPsi,mHat,len(mHat))
        get_rot=my_interp(np.linspace(0,1.2,121),splined_rot,kind='linear')
        final_sig.append(get_rot(out_x))

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

def mtanh_1d(in_x, in_t, value, uncertainty, out_x):
    sys.path.append(os.path.join(os.path.dirname(__file__),'astrolibpy/mpfit/'))
    from mpfit import mpfit

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
    
    for ind in range(len(in_t)):
        excluded_inds=np.isnan(value[ind,:])

        fa = {'x': in_x[ind,~excluded_inds], 'y': value[ind,~excluded_inds], 'err': uncertainty[ind,~excluded_inds]}

        m = mpfit(my_shifted_mtanh, p0, parinfo=parinfo, functkw=fa, quiet=1)

        final_sig.append(shifted_mtanh(out_x,m.params))
        
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
