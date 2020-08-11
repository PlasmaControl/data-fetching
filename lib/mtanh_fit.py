import numpy as np
import sys
sys.path.append('/Users/josephabbate/Documents/research/fitting/astrolibpy/mpfit')
from mpfit import mpfit

import matplotlib.pyplot as plt

def real_to_psi_profile(psi, t0, value, uncertainty, standard_psi, t_out):
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
        
#         if debug:
#             plt.plot(x,shifted_mtanh(x,m.params),c='r')
#             plt.errorbar(x,
#                          value[~excluded_inds,ind],
#                          np.clip(uncertainty[~excluded_inds,ind],0,max_uncertainty),
#                          ls='None')
#             plt.xlabel('psi')
# #            plt.ylabel(signal)
#             plt.title('{}ms'.format(t_out[ind]))
#             plt.show()

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
