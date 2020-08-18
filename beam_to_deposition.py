import os
import pickle
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from plot_tools import plot_comparison_over_time
from transport_helpers import get_volume

data_dir=os.path.join(os.path.dirname(__file__),'fitting')
with open(os.path.join(data_dir,'final_data_full_batch_0.pkl'),'rb') as f:
    full_data=pickle.load(f)
with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'rb') as f:
    data=pickle.load(f)

# note 15 --> 150, 21 --> 210, 33 --> 330
# see Heidbrink "Initial measurements of the DIII-D off-axis neutral beams" for details
# on geometry

# https://ieeexplore.ieee.org/document/6052323
# "The goal of the off-axis injection is to have the center of the ion 
# sources aimed at a position 40 cm below the geometric center of the plasma. 
# To achieve this off-axis injection, the beamline requires a mechanical lifting 
# system that can elevate the beamline up to 16.5 deg from horizontal."

beams=['30L', '30R', '15L', '15R', '21L', '21R', '33L', '33R']

angles={'30L': 0, 
            '30R': 0, 
            '15L': -16.5, 
            '15R': -16.5, 
            '21L': 0, 
            '21R': 0, 
            '33L': 0, 
            '33R': 0}

# from Tim Scoville
tangencies={'30L': 1.149, 
            '30R': .749, 
            '15L': 1.149, 
            '15R': .749, 
            '21L': -.749, 
            '21R': -1.149, 
            '33L': 1.149, 
            '33R': .749}
# = 
perveances={'30L': 3, 
            '30R': 2.5, 
            '15L': 2.1, 
            '15R': 2.3, 
            '21L': 2.8, 
            '21R': 2.9, 
            '33L': 2.75, 
            '33R': 2.8}
# in keV
voltages={'30L': 75, 
            '30R': 81, 
            '15L': 75, 
            '15R': 75, 
            '21L': 75, 
            '21R': 75, 
            '33L': 75, 
            '33R': 75}

def rt_and_angle_to_r_and_z(rt, angle, num_points=100, R=1.67):
    angle=angle*np.pi/180
    perp_length=2*np.sqrt(np.square(R)-np.square(rt))
    start_point=np.array([rt,-perp_length/2,0])
    end_point=np.array([rt,perp_length/2,perp_length*np.tan(angle)])
    line=np.linspace(start_point,end_point,num_points).T
    r=np.sqrt(np.square(line[0])+np.square(line[1]))
    z=line[2]
    return (np.stack((r,z),axis=-1),np.linalg.norm(end_point-start_point) / num_points)

shot=163303
efit_type='EFIT01'
beam='30L'
(points,volume)=rt_and_angle_to_r_and_z(tangencies[beam],angles[beam])

#$\sigma$ is a factor of $10^{-16}$ $cm^2$, i.e. $10^{-20}$ $m^3$.
# And our density is in units of $10^{19}$ $m^3$.
# So ultimately $\sigma$ in the code is a factor of $.1$

standard_psi=np.linspace(0,1,65)
deposition_psi=np.linspace(0,1,20)
depositions={}
for beam in beams:
    power=data[shot]['bmspinj{}'.format(beam)]
    iBeam0=power
    
    deposition=np.zeros((len(data[shot]['time']),len(deposition_psi)))
    for time_ind in range(len(data[shot]['time'])):
        # see pencil beam theory, Rome's 1974
        # "Neutral-beam injection into a tokamak, 
        # part I: fast- ion spatial distribution for tangential injection"
        # dNb / ds = - Nb / \lambda(s)
        # \lambda(s) = 1 / n\sigma
        r_z_to_psi=interpolate.interp2d(full_data[shot][efit_type]['R'],
                                        full_data[shot][efit_type]['Z'],
                                        data[shot]['psirz'][time_ind])
        psi_to_density=interpolate.interp1d(standard_psi,
                                            data[shot]['thomson_dens_{}'.format(efit_type)][time_ind])
        iBeam=np.zeros(len(points))
        iBeam[0]=iBeam0[time_ind]

        for point_ind in range(len(points)-1):
            r=points[point_ind][0]
            z=points[point_ind][1]
            psi=r_z_to_psi(r,z)
            if psi<1:
                psi_index=np.searchsorted(deposition_psi,psi)
                sigma=.2 #1e-15/1e3
                lambdaP=1/(psi_to_density(psi)*sigma)
                diBeam=( -iBeam[point_ind] / lambdaP ) * volume
                deposition[time_ind][psi_index] -= diBeam
                iBeam[point_ind+1] = iBeam[point_ind] + diBeam
            
    depositions[beam]=deposition

total_depositions=np.zeros(depositions[beams[0]].shape)
for beam in beams:
    total_depositions+=depositions[beam]

r=full_data[shot]['EFIT01']['R']
z=full_data[shot]['EFIT01']['Z']
psi_grid=np.array(data[shot]['psirz'])
# the 1e6 is because we want volume in cm^3 instead of m^3
dv=get_volume(r=r,z=z,psi_grid=psi_grid,basis_psi=deposition_psi)*1e6

total_depositions=np.divide(total_depositions,dv)

plot_comparison_over_time(xlist=(standard_psi,deposition_psi),
                          ylist=(data[shot]['transp_PBI'],total_depositions),
                          time=data[shot]['time'],
                          ylabel='PBI (W/cm^3)',
                          xlabel='psi',
                          uncertaintylist=None,
                          labels=('transp_PBI','joe_PBI'))
