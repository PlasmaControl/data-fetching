import pickle
from scipy import interpolate
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/josephabbate/Documents/research/fitting/lib')
from mtanh_fit import *

debug=True
debug_sig='dens'
debug_shot=163303

batch_num=0

time_base=50
time_step=time_base

cer_type='cerquick'
efit_type='EFIT01'

standard_psi=np.linspace(0,1,65)

with open('final_data_full_batch_{}.pkl'.format(batch_num),'rb') as f:
    data=pickle.load(f)

def standardize_time(old_signal,old_timebase):
    new_signal=[]
    for i in range(len(standard_times)):
        inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-time_base,old_timebase<standard_times[i]))[0]
        if len(inds_in_range)==0:
            new_signal.append(np.nan)
        else:
            new_signal.append(np.mean(old_signal[inds_in_range],axis=0))
    return new_signal

final_data={}
for shot in data.keys():
    min_time=max(data[shot]['t_ip_flat'],
                 min(data[shot][efit_type]['time']))
    min_time+=time_base
    max_time=min(data[shot]['t_ip_flat']+data[shot]['ip_flat_duration'],
                 max(data[shot][efit_type]['time']))
    standard_times=np.arange(min_time,max_time,time_step)
    final_data[shot]={}
    
    final_data[shot]['time']=standard_times

    final_data[shot]['topology']=data[shot]['topology']
    final_data[shot]['t_ip_flat']=data[shot]['t_ip_flat']
    final_data[shot]['ip_flat_duration']=data[shot]['ip_flat_duration']
    
    for sig in ['curr']: #data[shot]['scalars']:
        final_data[shot][sig]=standardize_time(data[shot]['scalars'][sig]['data'],
                                              data[shot]['scalars'][sig]['time'])
        if debug and sig==debug_sig and shot==debug_shot:
            plt.plot(data[shot]['scalars'][sig]['time'],data[shot]['scalars'][sig]['data'])
            plt.scatter(standard_times,final_data[shot][sig],c='r',zorder=10)
            plt.show()

    psirz=standardize_time(data[shot][efit_type]['psi_grid'],data[shot][efit_type]['time'])
    get_psi=[interpolate.interp2d(data[shot][efit_type]['R'],data[shot][efit_type]['Z'],psirz[time_ind]) for time_ind in range(len(standard_times))]

    # CER
    for signal in ['temp','rotation']:
        r=[]
        z=[]
        psi=[]
        value=[]
        error=[]

        scale={'temp': 1e3, 'rotation': 1}
        for system in ['VERTICAL','TANGENTIAL']:
            for channel in data[shot]['cer'][system]:
                if cer_type in data[shot]['cer'][system][channel]:
                    value.append(standardize_time(data[shot]['cer'][system][channel][cer_type][signal]/scale[signal],
                                                  data[shot]['cer'][system][channel][cer_type]['time']))
                    try:
                        error.append(standardize_time(data[shot]['cer'][system][channel][cer_type]['errors'],
                                                      data[shot]['cer'][system][channel][cer_type]['time'],
                                                      integer_value=True))
                    except:
                        pass
                    r.append(standardize_time(data[shot]['cer'][system][channel][cer_type]['R'],
                                              data[shot]['cer'][system][channel][cer_type]['time']))
                    z.append(standardize_time(data[shot]['cer'][system][channel][cer_type]['Z'],
                                              data[shot]['cer'][system][channel][cer_type]['time']))
                    psi.append([get_psi[time_ind](r[-1][time_ind],z[-1][time_ind])[0] for time_ind in range(len(standard_times))])

        value=np.array(value)
        psi=np.array(psi)
        error=np.array(error)
        uncertainty=np.full(np.shape(value),np.nanmean(np.abs(value))*.1)
        max_uncertainty=np.nanmax(uncertainty)*5
        value[np.where(error==1)]=np.nan
        
        final_data[shot]['cer_{}_{}'.format(signal,efit_type)] = real_to_psi_profile(psi,standard_times,value,uncertainty, standard_psi,standard_times,debug= (debug and debug_sig==signal and debug_shot==shot), max_uncertainty=max_uncertainty)

    # Thomson
    for signal in ['temp','dens']:
        r=[]
        z=[]
        psi=[]
        value=[]
        uncertainty=[]

        scale={'temp': 1e3, 'dens': 1e19}
        for system in ['CORE','TANGENTIAL']:
            for channel in range(data[shot]['thomson'][system][signal].shape[1]):
                value.append(standardize_time(data[shot]['thomson'][system][signal].T[channel]/scale[signal],
                                              data[shot]['thomson'][system]['time']))
                #uncertainty.append(data[shot]['cer'][system][channel][cer_type][signal]

                if system=='TANGENTIAL':
                    r=data[shot]['thomson'][system]['R'].T[channel]
                    z=0
                if system=='CORE':
                    z=data[shot]['thomson'][system]['Z'].T[channel]
                    r=1.94
                psi.append([get_psi[time_ind](r,z)[0] for time_ind in range(len(standard_times))])
                uncertainty.append(standardize_time(data[shot]['thomson'][system][signal+'_uncertainty'].T[channel]/scale[signal],
                                            data[shot]['thomson'][system]['time']))

        value=np.array(value)
        psi=np.array(psi)
        uncertainty=np.array(uncertainty)
        max_uncertainty=np.nanmax(uncertainty)
        uncertainty[np.isclose(value,0)]=1e30
        uncertainty[np.isclose(uncertainty,0)]=1e30
        
        final_data[shot]['thomson_{}_{}'.format(signal,efit_type)] = real_to_psi_profile(psi,standard_times,value,uncertainty, standard_psi,standard_times,debug= (debug and debug_sig==signal and debug_shot==shot),max_uncertainty=max_uncertainty) #real_to_psi_profile(psi,value,uncertainty,error, standard_psi,debug_profile= (debug and debug_sig==signal and debug_shot==shot))

with open('final_data_batch_{}.pkl'.format(batch_num),'wb') as f:
    pickle.dump(final_data,f)
