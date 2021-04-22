import matplotlib.pyplot as plt
import pickle
import sys
import os
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from plot_tools import plot_comparison_over_time
from scipy import interpolate

shot=175970
standard_x=np.linspace(0,1,65)
with open('data/aza.pkl','rb') as f:
    data=pickle.load(f)

const_e=1.6e-19
const_m=9.1e-31
const_eps0=8.85e-12
const_scale=2*np.pi*1e9 #omega to (f in GHz)

z_ind=np.argmin(np.abs(data[shot]['psirz_z']))
R=data[shot]['psirz_r']
psi=data[shot]['psirz'][:,:,z_ind]

bt=[np.divide(data[shot]['bt'][time_ind]*data[shot]['R0'][time_ind],R) for time_ind in range(len(data[shot]['time']))]
bt=np.abs(np.array(bt))

get_density=[interpolate.interp1d(standard_x,
                                  data[shot]['thomson_density_linear_interp_1d'][time_ind],
                                  bounds_error=False,
                                  fill_value=0) for time_ind in range(len(data[shot]['time']))]

density=[get_density[time_ind](psi[time_ind]) for time_ind in range(len(data[shot]['time']))]
density=np.array(density)

w_ce=const_e*bt/const_m
w_pe=np.sqrt(density*1e19*const_e**2/(const_m*const_eps0))
w_cutoff=(w_ce+np.sqrt(np.square(w_ce)+4*np.square(w_pe)) )/2

channel_freqs=[]
channel_freqs.extend(np.arange(83.5,98.5,1))
channel_freqs.extend(np.arange(98.5,113.5,1))
channel_freqs.extend(np.arange(115.5,129.5,2))

time_ind=60
plt.plot(R,w_ce[time_ind] / (const_scale),label=r'$f_{ce}$')
plt.plot(R,2*w_ce[time_ind] / (const_scale),label=r'2 $f_{ce}$')
plt.plot(R,3*w_ce[time_ind] / (const_scale),label=r'3 $f_{ce}$')
plt.plot(R,w_cutoff[time_ind] / (const_scale),label=r'$f_{cutoff}$')
plt.xlabel('R (m)')
plt.ylabel('frequency (GHz)')
plt.title('Shot {}, time {}ms'.format(shot,data[shot]['time'][time_ind]))
for channel_freq in channel_freqs:
    plt.axhline(channel_freq,linewidth=1,alpha=.5)
plt.legend()
plt.show()

xlist=[]
ylist=[]
uncertaintylist=[]
labels=[]

if False:
    #psi
    xlist.append(R)
    ylist.append(psi)
    uncertaintylist.append(None)
    labels.append('psi normalized')

    #density
    xlist.append(R)
    ylist.append(density)
    uncertaintylist.append(None)
    labels.append('density (10^19 m^-3)')

    #Bt
    xlist.append(R)
    ylist.append(bt)
    uncertaintylist.append(None)
    labels.append('Bt (1/R way) (T)')

    plot_comparison_over_time(xlist=xlist,
                              ylist=ylist,
                              time=data[shot]['time'],
                              ylabel='various quantities',
                              xlabel='R (m)',
                              uncertaintylist=uncertaintylist,
                              labels=labels)

if False:
    xlist.append(R)
    ylist.append(w_ce / (const_scale))
    uncertaintylist.append(None)
    labels.append(r'$f_{ce}$')

    xlist.append(R)
    ylist.append(2*w_ce / (const_scale))
    uncertaintylist.append(None)
    labels.append(r'$2f_{ce}$')

    xlist.append(R)
    ylist.append(3*w_ce / (const_scale))
    uncertaintylist.append(None)
    labels.append(r'$3f_{ce}$')

    # xlist.append(R)
    # ylist.append(w_pe / (const_scale))
    # uncertaintylist.append(None)
    # labels.append(r'$f_{ne}$')

    xlist.append(R)
    ylist.append(w_cutoff / (const_scale))
    uncertaintylist.append(None)
    labels.append(r'$f_{cutoff}$')    

    plot_comparison_over_time(xlist=xlist,
                              ylist=ylist,
                              time=data[shot]['time'],
                              ylabel='frequency (GHz)',
                              xlabel='R (m)',
                              uncertaintylist=uncertaintylist,
                              labels=labels)

