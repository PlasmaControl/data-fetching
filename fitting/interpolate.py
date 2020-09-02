import pickle
from scipy import interpolate
import numpy as np

import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'..','lib'))
#from mtanh_fit import real_to_psi_profile
from transport_helpers import my_interp
from rbf_fit import real_to_psi_profile
from plot_tools import plot_comparison_over_time

data_dir=os.path.join(os.path.dirname(__file__),'..','data')

cer_type='cerquick'
efit_type='EFIT01'

thomson_sigs=['temp','dens']
cer_sigs=['temp','rotation']
scalar_sigs=['bmspinj15L', 'bmspinj21L', 'bmspinj30L',
             'bmspinj33L', 'bmspinj15R', 'bmspinj21R',
             'bmspinj30R', 'bmspinj33R', 'bmstinj15L',
             'bmstinj21L', 'bmstinj30L', 'bmstinj33L',
             'bmstinj15R', 'bmstinj21R', 'bmstinj30R',
             'bmstinj33R']
transp_sigs=['PBI']
include_psirz=True
include_rho_grid=True

debug=True
debug_sig='thomson_dens_{}'.format(efit_type) 
debug_shot=163303

batch_num=0

time_base=200
#falloff=50
time_step=50

fit_psi=False
efit_psi=np.linspace(0,1,65)
standard_psi=np.linspace(0,1,65)
standard_rho=np.linspace(.025,.975,20)

with open(os.path.join(data_dir,'final_data_full_batch_{}.pkl'.format(batch_num)),'rb') as f:
    data=pickle.load(f)

def standardize_time(old_signal,old_timebase):
    new_signal=[]
    for i in range(len(standard_times)):
        inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-time_base,old_timebase<standard_times[i]))[0]
        if len(inds_in_range)==0:
            new_signal.append(np.nan)
        else:
            # weights=np.array([np.exp(- (standard_times[i]-old_timebase[ind]) / falloff) for ind in inds_in_range])
            # plt.plot(old_timebase[inds_in_range],weights)
            # plt.show()
            # weights/=sum(weights)
            # new_signal.append(np.sum(np.tensordot(old_signal[inds_in_range],weights,axes=0)[0],axis=0))
            new_signal.append(np.mean(old_signal[inds_in_range],axis=0))
    return new_signal

final_data={}
for shot in data.keys():
    min_time=max(data[shot]['t_ip_flat'],
                 min(data[shot][efit_type]['time']))
    #min_time+=time_base
    max_time=min(data[shot]['t_ip_flat']+data[shot]['ip_flat_duration'],
                 max(data[shot][efit_type]['time']))
    standard_times=np.arange(min_time,max_time,time_step)
    final_data[shot]={}
    
    final_data[shot]['time']=standard_times

    final_data[shot]['topology']=data[shot]['topology']
    final_data[shot]['t_ip_flat']=data[shot]['t_ip_flat']
    final_data[shot]['ip_flat_duration']=data[shot]['ip_flat_duration']
    
    for sig in scalar_sigs:
        final_data[shot][sig]=standardize_time(data[shot]['scalars'][sig]['data'],
                                              data[shot]['scalars'][sig]['time'])
        if debug and sig==debug_sig and shot==debug_shot:
            plt.plot(data[shot]['scalars'][sig]['time'],data[shot]['scalars'][sig]['data'])
            plt.scatter(standard_times,final_data[shot][sig],c='r',zorder=10)
            plt.show()

    psirz=standardize_time(data[shot][efit_type]['psi_grid'],data[shot][efit_type]['time'])
    rho=standardize_time(data[shot][efit_type]['rho_grid'],data[shot][efit_type]['time'])

    if include_psirz:
        final_data[shot]['psirz']=psirz
    if include_rho_grid:
        final_data[shot]['rho_grid']=rho
        
    r_z_to_psi=[interpolate.interp2d(data[shot][efit_type]['R'],data[shot][efit_type]['Z'],psirz[time_ind]) for time_ind in range(len(standard_times))]
    psi_to_rho=[my_interp(efit_psi,
                          data[shot][efit_type]['rho_grid'][time_ind]) for time_ind in range(len(standard_times))]

    # CER
    for signal in cer_sigs:
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
                    psi.append([r_z_to_psi[time_ind](r[-1][time_ind],z[-1][time_ind])[0] for time_ind in range(len(standard_times))])
                    
        value=np.array(value).T
        psi=np.array(psi).T
        error=np.array(error).T
        uncertainty=np.full(np.shape(value),np.nanmean(np.abs(value))*.1)
        max_uncertainty=np.nanmax(uncertainty)*5
        value[np.where(error==1)]=np.nan

        final_sig_name='cer_{}_{}'.format(signal,efit_type)

        if fit_psi:
            in_x=psi
            out_x=standard_psi
            x_label='psi'
        else:
            in_x=[]
            for time_ind in range(len(standard_times)):
                in_x.append(psi_to_rho[time_ind](psi[time_ind]))
            in_x=np.array(in_x)
            out_x=standard_rho
            x_label='rho'

        final_data[shot][final_sig_name] = real_to_psi_profile(in_x,standard_times,value,uncertainty, out_x,standard_times)

        if debug and debug_sig==final_sig_name and debug_shot==shot:
            uncertainty=np.clip(uncertainty,0,max_uncertainty)
            plot_comparison_over_time(xlist=(in_x,out_x),
                                      ylist=(value,final_data[shot][final_sig_name]), 
		                      time=standard_times,
                	              ylabel=final_sig_name,
                	              xlabel=x_label,
                	              uncertaintylist=(uncertainty,None),
                                      labels=('data','original'))

    # Thomson
    for signal in thomson_sigs:
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

                if system=='TANGENTIAL':
                    r=data[shot]['thomson'][system]['R'].T[channel]
                    z=0
                if system=='CORE':
                    z=data[shot]['thomson'][system]['Z'].T[channel]
                    r=1.94
                psi.append([r_z_to_psi[time_ind](r,z)[0] for time_ind in range(len(standard_times))])
                uncertainty.append(standardize_time(data[shot]['thomson'][system][signal+'_uncertainty'].T[channel]/scale[signal],
                                            data[shot]['thomson'][system]['time']))

        value=np.array(value).T
        psi=np.array(psi).T
        uncertainty=np.array(uncertainty).T
        max_uncertainty=np.nanmax(uncertainty)
        value[np.isclose(value,0)]=np.nan
        value[np.isclose(uncertainty,0)]=np.nan

        final_sig_name='thomson_{}_{}'.format(signal,efit_type)
        if fit_psi:
            in_x=psi
            out_x=standard_psi
            x_label='psi'
        else:
            in_x=[]
            for time_ind in range(len(standard_times)):
                in_x.append(psi_to_rho[time_ind](psi[time_ind]))
            in_x=np.array(in_x)
            out_x=standard_rho
            x_label='rho'
            
        final_data[shot][final_sig_name] = real_to_psi_profile(in_x,standard_times,value,uncertainty,out_x,standard_times)

        if debug and debug_sig==final_sig_name and debug_shot==shot:
            plot_comparison_over_time(xlist=(in_x,out_x),
                                      ylist=(value,final_data[shot][final_sig_name]), 
		                      time=standard_times,
                	              ylabel=final_sig_name,
                	              xlabel=x_label,
                	              uncertaintylist=(uncertainty,None),
                                      labels=('data','original'))

    if 'transp' in data[shot]:
        transp_rho=data[shot]['transp']['beam_rho']
        for signal in transp_sigs:
            final_sig_name='transp_{}'.format(signal)
            final_data[shot][final_sig_name]=[]
            transp_data=standardize_time(data[shot]['transp'][signal],data[shot]['transp']['beam_time']*1000)
            if fit_psi:
                for time_ind in range(len(standard_times)):
                    rho_to_transp=my_interp(transp_rho,transp_data[time_ind])
                    final_data[shot][final_sig_name].append(rho_to_transp(rho[time_ind]))
                final_data[shot][final_sig_name]=np.array(final_data[shot][final_sig_name])
                x_out=standard_psi
                x_label='psi'
            else:
                final_data[shot][final_sig_name]=np.array(transp_data)
                x_out=transp_rho
                x_label='rho'

            if debug and debug_sig==final_sig_name and debug_shot==shot:
                plot_comparison_over_time(xlist=[x_out],
                                          ylist=[final_data[shot][final_sig_name]],
		                          time=standard_times,
                	                  ylabel=final_sig_name,
                	                  xlabel=x_label,
                	                  uncertaintylist=None,
                                          labels=None)
                                
with open(os.path.join(data_dir,'final_data_batch_{}.pkl'.format(batch_num)),'wb') as f:
    pickle.dump(final_data,f)
