import numpy as np
from scipy import interpolate
import time

def fill_value(arr2d):
    # dv is differential so will be missing the last volume element
    # this is a hack, but we add the last psi value for each time
    # to the end to reshape it to match basis_psi
    last_val=np.atleast_2d(arr2d[:,-1])
    #transpose and untraspose to match what np.concatentate expects
    return np.concatenate((arr2d.T,last_val),axis=0).T

def interp_ND_rectangular(coords,data):
    return interpolate.RegularGridInterpolator(coords,data,
                                               bounds_error=False,
                                               fill_value=None)

# Context manager that we can use to time execution
class Timer(object):
    def __init__(self):
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print('---> Ran in {0:.2f} s'.format(elapsed))

def standardize_time(old_signal,old_timebase,standard_times,
                     causal=True, window_size=50,
                     exponential_falloff=False, numpy_smoothing_fxn=np.mean, falloff_rate=100):
    """ Rebase >=1D array signal to a new timebase

    Keyword arguments:
    old_signal -- the signal you want to rebase
    old_timebase -- the old_signal's timebase
    standard_times -- the new (desired) timebase
    causal -- whether to start the window backwards from the specified time, else window around timepoint
    window_size -- how large the window
    exponential_falloff -- whether to count points close to time more heavily (boxcar smoothing if False)
    numpy_smoothing_fxn -- some function that takes in an array and an array axis along which to apply (e.g. mean, mode, etc)
    falloff_rate -- (only used if exponential_falloff True) 1/e decay of importance in time
    """
    new_signal=[]
    for i in range(len(standard_times)):
        if causal:
            inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-window_size,old_timebase<standard_times[i]))[0]
        else:
            inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-window_size,old_timebase<standard_times[i]+window_size))[0]
        if len(inds_in_range)==0:
            if len(old_signal.shape)==1:
                new_signal.append(np.nan)
            else:
                new_signal.append(np.full(old_signal.shape[1:],np.nan))
        else:
            if exponential_falloff:
                weights=np.array([np.exp(- np.abs(standard_times[i]-old_timebase[ind]) / falloff_rate) for ind in inds_in_range])
                weights/=sum(weights)
                new_signal.append( np.array( np.sum( [old_signal[ind]*weights[j] for j,ind in enumerate(inds_in_range)], axis=0) ) )
            else:
                new_signal.append(numpy_smoothing_fxn(old_signal[inds_in_range],axis=0))
    return np.array(new_signal)

def my_interp(x,y,kind='linear'):
    return interpolate.interp1d(x,y,
                                kind=kind,
                                bounds_error=False,
                                fill_value=(y[np.argmin(x)],
                                            y[np.argmax(x)]))

def get_volume(r,z,psi_grid,basis,
               fit_psi=True, rho_grid=None):
    efit_psi=np.linspace(0,1,65)
    num_psi=len(basis)
    dr=np.diff(r)
    dz=np.diff(z)
    dv=np.zeros((psi_grid.shape[0],num_psi-1))
    #G1=np.zeros(num_psi)
    
    for time_ind in range(psi_grid.shape[0]):
        if fit_psi:
            basis_psi=basis
        else:
            rho_to_psi = interpolate.interp1d(rho_grid[time_ind],
                                              efit_psi)
            basis_psi=rho_to_psi(basis)
        dpsi=np.diff(basis_psi)

        for psi_ind in range(num_psi-1):
            gradient=np.gradient(psi_grid[time_ind])

            psi=basis_psi[psi_ind] #dpsi[psi_ind]*psi_ind
            hull=np.where(np.logical_and(psi_grid[time_ind]<psi+dpsi[psi_ind],
                          psi_grid[time_ind]>=psi))

            volume=0
            for (z_ind,r_ind) in zip(*hull):
                # if z_ind==num_psi-1:
                #     z_ind=num_psi-2
                # if r_ind==num_psi-1:
                #     r_ind=num_psi-2
                volume+=r[r_ind]*dr[r_ind]*dz[z_ind]
            volume*=2*np.pi
            dv[time_ind,psi_ind]=volume

    # dv is differential so will be missing the last volume element
    # this is a hack, but we add the last psi value for each time
    # to the end to reshape it to match basis_psi
    last_dv=np.atleast_2d(dv[:,-1])
    #transpose and untraspose to match what np.concatentate expects
    dv=np.concatenate((dv.T,last_dv),axis=0).T

    return dv
        # psi_gradient=0
        # for (z_ind,r_ind) in zip(*hull):
        #     psi_gradient+=np.square(gradient[0][r_ind][z_ind])+np.square(gradient[1][r_ind][z_ind])
        # G1[i]=psi_gradient/len(hull[0])
        
# from Janev's 1989 "Penetration of energetic neutral beams into fusion plasmas"
# ne is to be in 10^19 m^-3, Te in keV, E in keV/u
def get_sigma(ne,Te,E=75,Z=2):
    
    def S1(E,ne,Te):
        A=[[[4.4,-2.49e-2],[7.46e-2,2.27e-3],[3.16e-3,-2.78e-5]],
           [[2.3e-1,-1.15e-2],[-2.55e-3,-6.2e-4],[1.32e-3,3.38e-5]]]
        # 10^13 cm^-3 in paper, so (since we deal in 10^19 m^-3) 
        # we just have n0=1

        ret=0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    ret+=A[i][j][k] \
                        *np.power(np.log(E),i) \
                        *np.power(np.log(ne),j) \
                        *np.power(np.log(Te),k)
        return ret

    def Sz(E,ne,Te,impurity='C'):
        # we just have the numbers for C impurity here
        B={'C':[[[-1.49,-1.54e-2],[-1.19e-1,-1.5e-2]],
                [[5.18e-1,7.18e-3],[2.92e-2,3.66e-3]],
                [[-3.36e-2,3.41e-4],[-1.79e-3,-2.04e-4]]]}

        ret=0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    ret+=B[impurity][i][j][k] \
                        *np.power(np.log(E),i) \
                        *np.power(np.log(ne),j) \
                        *np.power(np.log(Te),k)
        return ret


    return np.exp(S1(E,ne,Te))/E * ( 1 + (Z-1)*Sz(E,ne,Te) ) * 1e-16

if __name__ == "__main__":
    '''
    import matplotlib.pyplot as plt
    # these are the values from the paper, this plots a comparison to double check
    # we implemented the right equation 
    ne=1e14
    Te=10
    Z=2
    E_arr=np.logspace(2,4)
    sigma=[get_sigma(E,ne,Te,Z) for E in E_arr]
    plt.loglog([1e2,2e3,1e4],[2.8e-16,4e-17,1e-17],label='estimated paper values for ne=1e14')
    plt.loglog(E_arr,sigma,label='this code for ne={:.0e}'.format(ne))
    plt.xlabel('E')
    plt.ylabel('sigma')
    plt.legend()
    plt.show()
    '''
    import pickle
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from plot_tools import plot_comparison_over_time
    
    data_dir=os.path.join(os.path.dirname(__file__),'..','data')
    with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'rb') as f:
        data=pickle.load(f)

    shot=163303
    efit_type='EFIT01'
    standard_psi=np.linspace(0,1,65)
    sigma=np.zeros((len(data[shot]['time']), len(standard_psi)))
    for time_ind in range(len(data[shot]['time'])):
        for psi_ind in range(len(standard_psi)):
            sigma[time_ind][psi_ind]=get_sigma(ne=data[shot]['thomson_dens_{}'.format(efit_type)][time_ind][psi_ind],
                                               Te=data[shot]['thomson_temp_{}'.format(efit_type)][time_ind][psi_ind]) * 1e-4

    plot_comparison_over_time(xlist=[standard_psi],
                              ylist=[sigma],
                              time=data[shot]['time'],
                              ylabel='sigma',
                              xlabel='psi',
                              uncertaintylist=None,
                              labels=None)
