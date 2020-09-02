import os
import pickle
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from plot_tools import plot_comparison_over_time
from transport_helpers import fill_value

shot=163303
efit_type='EFIT01'

debug=False

standard_rho=np.linspace(.025,.975,20)
# standard_psi=np.linspace(0,1,65)
# transp_psi=np.linspace(0,1,20)

data_dir=os.path.join(os.path.dirname(__file__),'data')
with open(os.path.join(data_dir,'final_data_full_batch_0.pkl'),'rb') as f:
    full_data=pickle.load(f)
with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'rb') as f:
    data=pickle.load(f)

dt=np.diff(data[shot]['time'])[0] / 1000

#beam_deposition=data[shot]['total_deposition']
beam_deposition=data[shot]['transp_PBI']

#J to eV, then eV to keV, then keV to particles (all per s*cm^3)
#then from cm^-3 to m^-3
particle_energy=75
source=beam_deposition /1.6e-19 /1000 /particle_energy *1e6
particles_from_source = np.multiply(data[shot]['dv'], source)

density=data[shot]['thomson_dens_{}'.format(efit_type)] *1e19 

particle_time_change=np.diff(np.multiply(data[shot]['dv'],
                                         density),axis=0) / dt
last_val=np.atleast_2d(particle_time_change[-1,:])
particle_time_change=np.concatenate((particle_time_change,last_val),axis=0)

if debug:
    # if DIII-D confinement time is 100ms and there are a few 10^19 * 40, i.e. a few 10^20, particles at any given
    # time, then there should be about 10^21 particles coming in per second. I'd guess most of these are from beams?
    #
    # a more direct way is we're dumping a few MW into the plasma, i.e. a few 10^20 particles per second
    print('Average particles from source per second: {:.2e}'.format(np.mean(np.sum(particles_from_source,axis=-1))))
    plot_comparison_over_time(xlist=[standard_rho, standard_rho],
                              ylist=[particle_time_change, particles_from_source],
                              time=data[shot]['time'],
                              ylabel='particles / s',
                              xlabel='rho',
                              uncertaintylist=None,
                              labels=('particle time change','particles from source'))

dGamma_dRho = particles_from_source - particle_time_change

# from ASTRA paper
# gamma = -dv * G1 * De * dn/drho

gamma=np.cumsum(np.multiply(dGamma_dRho,standard_rho),axis=-1)

dn_dRho=np.diff(density,axis=-1)
last_val=np.atleast_2d(dn_dRho[:,-1])
dn_dRho=np.concatenate((dn_dRho.T,last_val),axis=0).T

dn_dRho=np.clip(dn_dRho,.1e19/len(standard_rho),None)

De = - np.divide(gamma,
                 np.multiply(data[shot]['dv'],
                                    np.multiply(data[shot]['G1'], dn_dRho) ))

if True:
    plot_comparison_over_time(xlist=[standard_rho],
                              ylist=[De],
                              time=data[shot]['time'],
                              ylabel='De',
                              xlabel='rho',
                              uncertaintylist=None,
                              labels=None)
