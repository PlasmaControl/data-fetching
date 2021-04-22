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

sig_name_dens='zipfit_dens_{}'.format(efit_type)
sig_name_etemp='zipfit_temp_{}'.format(efit_type)

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

# J/ (s cm^3) --> eV --> keV --> /m^3 to keV/s
power_from_source = np.multiply(data[shot]['dv'],beam_deposition) /1.6e-19 /1000 *1e6
power_from_source_to_electrons = power_from_source*.5

particle_energy=75 #in keV!
particles_from_source=power_from_source / particle_energy

etemp=data[shot][sig_name_etemp]
density=data[shot][sig_name_dens] *1e19


# particle flux equation
particle_time_change=np.diff(np.multiply(data[shot]['dv'],
                                         density),axis=0) / dt
last_val=np.atleast_2d(particle_time_change[-1,:])
particle_time_change=np.concatenate((particle_time_change,last_val),axis=0)

if False: #debug:
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

# for debugging purposes, plot gamma due to source and due to time change separately
if True:
    gamma_source=np.cumsum(np.multiply(particles_from_source,standard_rho),axis=-1)
    gamma_time=-np.cumsum(np.multiply(particle_time_change,standard_rho),axis=-1)

dn_dRho=np.diff(density,axis=-1)
last_val=np.atleast_2d(dn_dRho[:,-1])
dn_dRho=np.concatenate((dn_dRho.T,last_val),axis=0).T

#dn_dRho=np.clip(dn_dRho,.1e19/len(standard_rho),None)

De = - np.divide(gamma,
                 np.multiply(data[shot]['dv'],
                                    np.multiply(data[shot]['G1'], dn_dRho) ))

clip_value=10000

De = np.clip(De,-clip_value,clip_value)

if True:
    De_time = - np.divide(gamma_time,
                          np.multiply(data[shot]['dv'],
                                      np.multiply(data[shot]['G1'], dn_dRho) ))
    De_time = np.clip(De_time,-clip_value,clip_value)    

    De_source = - np.divide(gamma_source,
                          np.multiply(data[shot]['dv'],
                                      np.multiply(data[shot]['G1'], dn_dRho) ))
    De_source = np.clip(De_source,-clip_value,clip_value)    

    
# electron heat flux equation
etemp_time_change=np.diff(np.multiply(np.multiply(np.power(data[shot]['dv'],5/3.),
                                                 density),
                                     etemp),axis=0) / dt
last_val=np.atleast_2d(etemp_time_change[-1,:])
etemp_time_change=np.concatenate((etemp_time_change,last_val),axis=0)
etemp_time_change=(3/2.) * np.multiply(etemp_time_change,np.power(data[shot]['dv'],-2/3.))

etemp_from_particle_transport=np.diff((5/2.)*np.multiply(etemp,
                                                        gamma),axis=-1)
last_val=np.atleast_2d(etemp_from_particle_transport[:,-1])
etemp_from_particle_transport=np.concatenate((etemp_from_particle_transport.T,
                                             last_val),axis=0).T


dqe_dRho = power_from_source_to_electrons - etemp_from_particle_transport - etemp_time_change
qe=np.cumsum(np.multiply(dqe_dRho,standard_rho),axis=-1)

# from ASTRA paper
# q = -dv * G1 * De * ne * dn/drho

dTe_dRho=np.diff(etemp,axis=-1)
last_val=np.atleast_2d(dTe_dRho[:,-1])
dTe_dRho=np.concatenate((dTe_dRho.T,last_val),axis=0).T

chie = - np.divide(qe,
                   np.multiply(np.multiply(data[shot]['dv'],
                                           np.multiply(data[shot]['G1'], dTe_dRho)),
                               density))
chie = np.clip(chie,-clip_value,clip_value*5)

if True:
    plot_comparison_over_time(xlist=[standard_rho for i in range(2)],
                              ylist=[data[shot]['transp_DIFFE'], De],
                              time=data[shot]['time'],
                              ylabel='chi',
                              xlabel='rho',
                              uncertaintylist=None,
                              labels=['TRANSP', 'me'])
if False:
    plot_comparison_over_time(xlist=[standard_rho],
                              ylist=[De],
                              time=data[shot]['time'],
                              ylabel='diffusivity',
                              xlabel='rho',
                              uncertaintylist=None,
                              labels=['me'])
