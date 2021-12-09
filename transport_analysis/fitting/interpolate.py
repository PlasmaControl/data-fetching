import pickle
from scipy import interpolate
import numpy as np

import time

import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__),'..','lib'))
from transport_helpers import my_interp, interp_ND_rectangular
import fit_functions
from plot_tools import plot_comparison_over_time, plot_2d_comparison

data_dir=os.path.join(os.path.dirname(__file__),'..','data')

fit_function_dict={'linear_interp_1d': fit_functions.linear_interp_1d,
                   'spline_1d': fit_functions.spline_1d,
                   'nn_interp_2d': fit_functions.nn_interp_2d,
                   'linear_interp_2d': fit_functions.linear_interp_2d,
                   'mtanh_1d': fit_functions.mtanh_1d,
                   'rbf_interp_2d': fit_functions.rbf_interp_2d}
fit_functions_1d=['linear_interp_1d', 'mtanh_1d','spline_1d']
fit_functions_2d=['nn_interp_2d','linear_interp_2d','rbf_interp_2d']

cer_type='cerquick'
efit_type='EFIT01'

trial_fits=['mtanh_1d','spline_1d']
zipfit_interp=fit_function_dict['linear_interp_1d']

thomson_sigs=['temp'] #['temp','dens']
cer_sigs=[] #['temp','rotation']
scalar_sigs=[] #['bmspinj15L', 'bmspinj21L', 'bmspinj30L',
             # 'bmspinj33L', 'bmspinj15R', 'bmspinj21R',
             # 'bmspinj30R', 'bmspinj33R', 'bmstinj15L',
             # 'bmstinj21L', 'bmstinj30L', 'bmstinj33L',
             # 'bmstinj15R', 'bmstinj21R', 'bmstinj30R',
             # 'bmstinj33R',
             # 'bt']

transp_sigs=[] #['PBI',
             # 'DIFFE','DIFFI','DIFFX',
             # 'CONDE','CONDI',
             # 'CHPHI']

zipfit_sigs=['temp', 'dens', 'rotation','idens', 'itemp']

include_psirz=False
include_rho_grid=False

debug=True
debug_sig='thomson_temp'
debug_shot=163303

batch_num=0

causal=True #False
time_base=300
exponential_falloff=True
falloff=50
time_step=50

fit_psi=True #False
efit_psi=np.linspace(0,1,65)
standard_psi=np.linspace(0,1,65)
standard_rho=np.linspace(.025,.975,20)

with open(os.path.join(data_dir,'final_data_full_batch_{}.pkl'.format(batch_num)),'rb') as f:
    data=pickle.load(f)

def fit_all_and_save(prefix, signal,
                     in_x_1d, value,  uncertainty, 
                     in_x_2d, time_2d_fit, value_2d_fit, uncertainty_2d_fit,
                     out_x, standard_times,
                     x_label, trial_fits, debug):

    # first exclude nan indices for 2d

    '''
    excluded_inds=np.isnan(value_2d_fit)
    value_2d_fit=value_2d_fit[~excluded_inds]
    in_x_2d=in_x_2d[~excluded_inds]
    time_2d_fit=time_2d_fit[~excluded_inds]
    uncertainty_2d_fit=uncertainty_2d_fit[~excluded_inds]
    '''
    
    # now fit
    for trial_fit in trial_fits:
        final_sig_name='{}_{}_{}'.format(prefix,
                                         signal,
                                         trial_fit)
        if trial_fit in fit_functions_1d:
            final_data[shot][final_sig_name] = fit_function_dict[trial_fit](in_x_1d,standard_times,value,uncertainty, out_x,standard_times)
        elif trial_fit in fit_functions_2d:
            final_data[shot][final_sig_name] = fit_function_dict[trial_fit](in_x_2d,time_2d_fit,value_2d_fit,uncertainty_2d_fit, out_x,standard_times)
            if debug:
                plot_2d_comparison(out_x, standard_times, final_data[shot][final_sig_name],
                                   in_x_2d, time_2d_fit, value_2d_fit,
                                   xlims=[0,1], ylims=[min(standard_times),max(standard_times)],
                                   xlabel='position', ylabel='time (ms)',title=trial_fit)

    if debug: #debug_sig==final_sig_name and debug_shot==shot:
        xlist=[in_x_1d]
        ylist=[value]
        #uncertainty_2d_fit=np.clip(uncertainty_2d_fit,0,max_uncertainty_2d_fit)
        #uncertainty=np.clip(uncertainty,0,max_uncertainty)
        uncertaintylist=[uncertainty]
        labels=['data']
        for trial_fit in trial_fits:
            final_sig_name='{}_{}_{}'.format(prefix,
                                             signal,
                                             trial_fit)
            xlist.append(out_x)
            ylist.append(final_data[shot][final_sig_name])
            uncertaintylist.append(None)
            labels.append(trial_fit)
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=standard_times,
                                  ylabel=final_sig_name,
                                  xlabel=x_label,
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

    
def standardize_time(old_signal,old_timebase):
    new_signal=[]
    for i in range(len(standard_times)):
        if causal:
            inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-time_base,old_timebase<standard_times[i]))[0]
        else:
            inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-time_base,old_timebase<standard_times[i]+time_base))[0]
        if len(inds_in_range)==0:
            if len(old_signal.shape)==1:
                new_signal.append(np.nan)
            else: 
                new_signal.append(np.full(old_signal.shape[-1],np.nan))
        else:
            if exponential_falloff:
                weights=np.array([np.exp(- np.abs(standard_times[i]-old_timebase[ind]) / falloff) for ind in inds_in_range])
                weights/=sum(weights)
                new_signal.append( np.array( np.sum( [old_signal[ind]*weights[j] for j,ind in enumerate(inds_in_range)], axis=0) ) )
            else:     
                new_signal.append(np.mean(old_signal[inds_in_range],axis=0))
    return new_signal

final_data={}
for shot in data.keys():
    min_time=max(data[shot]['t_ip_flat'],
                 min(data[shot][efit_type]['time']))
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
    efit_rho=standardize_time(data[shot][efit_type]['rho_grid'],data[shot][efit_type]['time'])

    if include_psirz:
        final_data[shot]['psirz']=psirz
    if include_rho_grid:
        final_data[shot]['rho_grid']=efit_rho
        
    r_z_to_psi=[interpolate.interp2d(data[shot][efit_type]['R'],data[shot][efit_type]['Z'],psirz[time_ind]) for time_ind in range(len(standard_times))]
    
    r_z_to_psi_2d_fit=interp_ND_rectangular((data[shot][efit_type]['time'],
                                              data[shot][efit_type]['R'],
                                              data[shot][efit_type]['Z']),
                                             data[shot][efit_type]['psi_grid'])
    def r_z_to_psi_2d_fit(entries):
        channel_times, channel_rs, channel_zs = entries
        psis=np.zeros(len(channel_times))
        for i in range(len(channel_times)):
            time_ind=np.searchsorted(standard_times, channel_times[i])
            if time_ind==len(standard_times):
                time_ind-=1
            try:
                psis[i] = r_z_to_psi[time_ind](channel_rs[i], channel_zs[i])
            except:
                psis[i] = r_z_to_psi[time_ind](channel_rs, channel_zs)
        return np.array(psis)
    
    rho_to_psi=[my_interp(data[shot][efit_type]['rho_grid'][time_ind],
                          efit_psi) for time_ind in range(len(standard_times))]

    psi_to_rho=[my_interp(efit_psi,
                          data[shot][efit_type]['rho_grid'][time_ind]) for time_ind in range(len(standard_times))]
    
    psi_to_rho_2d_fit=interp_ND_rectangular((data[shot][efit_type]['time'],
                                             efit_psi),
                                            data[shot][efit_type]['rho_grid'])
    def psi_to_rho_2d_fit(entries):
        time_arr, psi_arr = entries
        rhos=np.zeros(len(time_arr))
        for i in range(len(time_arr)):
            time_ind=np.searchsorted(standard_times, time_arr[i])
            if time_ind==len(standard_times):
                time_ind-=1
            try:
                rhos[i] = psi_to_rho[time_ind](psi_arr[i])
            except:
                rhos[i] = psi_to_rho[time_ind](psi_arr)
        return np.array(rhos)

    # CER
    for signal in cer_sigs:
        psi=[]
        value=[]
        error=[]
        
        psi_2d_fit=[]
        time_2d_fit=[]
        value_2d_fit=[]
        error_2d_fit=[]
        
        scale={'temp': 1e3, 'rotation': 1}
        for system in ['VERTICAL','TANGENTIAL']:
            for channel in data[shot]['cer'][system]:
                if cer_type in data[shot]['cer'][system][channel]:

                    # for regular fits
                    value.append(standardize_time(data[shot]['cer'][system][channel][cer_type][signal]/scale[signal],
                                                  data[shot]['cer'][system][channel][cer_type]['time']))
                    try:
                        error.append(standardize_time(data[shot]['cer'][system][channel][cer_type]['errors'],
                                                      data[shot]['cer'][system][channel][cer_type]['time'],
                                                      integer_value=True))
                    except:
                        pass
                    r=standardize_time(data[shot]['cer'][system][channel][cer_type]['R'],
                                              data[shot]['cer'][system][channel][cer_type]['time'])
                    z=standardize_time(data[shot]['cer'][system][channel][cer_type]['Z'],
                                              data[shot]['cer'][system][channel][cer_type]['time'])
                    psi.append([r_z_to_psi[time_ind](r[time_ind],z[time_ind])[0] for time_ind in range(len(standard_times))])

                    # for 2d fits (space and time)
                    value_2d_fit.extend(data[shot]['cer'][system][channel][cer_type][signal]/scale[signal])
                    channel_times=data[shot]['cer'][system][channel][cer_type]['time']
                    time_2d_fit.extend(channel_times)
                    try:
                        error_2d_fit.append(data[shot]['cer'][system][channel][cer_type]['errors'])
                    except:
                        pass
                    channel_rs=data[shot]['cer'][system][channel][cer_type]['R']
                    channel_zs=data[shot]['cer'][system][channel][cer_type]['Z']
                    psi_2d_fit.extend(r_z_to_psi_2d_fit((channel_times, channel_rs, channel_zs)))

        # for regular fits
        value=np.array(value).T
        psi=np.array(psi).T
        error=np.array(error).T
        uncertainty=np.full(np.shape(value),np.nanmean(np.abs(value))*.1)
        max_uncertainty=np.nanmax(uncertainty)*5
        value[np.where(error==1)]=np.nan
        final_sig_name='cer_{}_{}'.format(signal,efit_type)

        # for 2d fits
        value_2d_fit=np.array(value_2d_fit)
        psi_2d_fit=np.array(psi_2d_fit)
        time_2d_fit=np.array(time_2d_fit)
        error_2d_fit=np.array(error_2d_fit)
        uncertainty_2d_fit=np.full(np.shape(value_2d_fit),np.nanmean(np.abs(value_2d_fit))*.1)
        value_2d_fit[np.where(error_2d_fit==1)]=np.nan

        if fit_psi:
            in_x_1d=psi
            in_x_2d=psi_2d_fit
            out_x=standard_psi
            x_label='psi'
        else:
            in_x_1d=[]
            for time_ind in range(len(standard_times)):
                in_x_1d.append(psi_to_rho[time_ind](psi[time_ind]))
            in_x_1d=np.array(in_x_1d)
            in_x_2d=psi_to_rho_2d_fit((time_2d_fit,psi_2d_fit))
            out_x=standard_rho
            x_label='rho'
        fit_all_and_save('cer', signal,
                         in_x_1d, value,  uncertainty, 
                         in_x_2d, time_2d_fit, value_2d_fit, uncertainty_2d_fit,
                         out_x, standard_times,
                         x_label, trial_fits, debug and debug_shot==shot and debug_sig=='cer_{}'.format(signal))

    # Thomson
    for signal in thomson_sigs:
        psi=[]
        value=[]
        uncertainty=[]

        psi_2d_fit=[]
        time_2d_fit=[]
        value_2d_fit=[]
        uncertainty_2d_fit=[]

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

                value_2d_fit.extend(data[shot]['thomson'][system][signal].T[channel]/scale[signal])
                uncertainty_2d_fit.extend(data[shot]['thomson'][system][signal+'_uncertainty'].T[channel]/scale[signal])
                psi_2d_fit.extend(r_z_to_psi_2d_fit((data[shot]['thomson'][system]['time'], r, z)))
                time_2d_fit.extend(data[shot]['thomson'][system]['time'])

        value=np.array(value).T
        psi=np.array(psi).T
        uncertainty=np.array(uncertainty).T
        value[np.isclose(value,0)]=np.nan
        value[np.isclose(uncertainty,0)]=np.nan

        value_2d_fit=np.array(value_2d_fit)
        value_2d_fit[np.isclose(value_2d_fit,0)]=np.nan
        value_2d_fit[np.isclose(uncertainty_2d_fit,0)]=np.nan        
        psi_2d_fit=np.array(psi_2d_fit)
        time_2d_fit=np.array(time_2d_fit)
        uncertainty_2d_fit=np.array(uncertainty_2d_fit)

        if fit_psi:
            in_x_1d=psi
            in_x_2d=psi_2d_fit
            out_x=standard_psi
            x_label='psi'
        else:
            in_x_1d=[]
            for time_ind in range(len(standard_times)):
                in_x_1d.append(psi_to_rho[time_ind](psi[time_ind]))
            in_x_1d=np.array(in_x_1d)
            in_x_2d=psi_to_rho_2d_fit((time_2d_fit,psi_2d_fit))
            out_x=standard_rho
            x_label='rho'
        fit_all_and_save('thomson', signal,
                         in_x_1d, value,  uncertainty, 
                         in_x_2d, time_2d_fit, value_2d_fit, uncertainty_2d_fit,
                         out_x, standard_times,
                         x_label, trial_fits, debug and debug_shot==shot and debug_sig=='thomson_{}'.format(signal))

    if 'transp' in data[shot]:
        transp_rho=data[shot]['transp']['beam_rho']
        for signal in transp_sigs:
            final_sig_name='transp_{}'.format(signal)
            final_data[shot][final_sig_name]=[]
            transp_data=standardize_time(data[shot]['transp'][signal],data[shot]['transp']['beam_time']*1000)
            if fit_psi:
                for time_ind in range(len(standard_times)):
                    rho_to_transp=my_interp(transp_rho,transp_data[time_ind])
                    final_data[shot][final_sig_name].append(rho_to_transp(psi_to_rho[time_ind](standard_psi)))
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
            
            # if debug and debug_sig==final_sig_name and debug_shot==shot:
            #     plot_comparison_over_time(xlist=[x_out],
            #                               ylist=[final_data[shot][final_sig_name]],
	    #                               time=standard_times,
            #     	                  ylabel=final_sig_name,
            #     	                  xlabel=x_label,
            #     	                  uncertaintylist=None,
            #                               labels=None)

    # Zipfit
    for signal in zipfit_sigs:
        value=standardize_time(data[shot]['zipfit_and_aot'][signal]['value'],
                               data[shot]['zipfit_and_aot'][signal]['time'])
        uncertainty=standardize_time(data[shot]['zipfit_and_aot'][signal]['uncertainty'],
                                     data[shot]['zipfit_and_aot'][signal]['time'])

        value=np.array(value)
        uncertainty=np.array(uncertainty)

        final_sig_name='zipfit_{}_{}'.format(signal,efit_type)
        if fit_psi:
            in_x=[]
            for time_ind in range(len(standard_times)):
                in_x.append(rho_to_psi[time_ind](data[shot]['zipfit_and_aot'][signal]['rho']))
            in_x=np.array(in_x)
            out_x=standard_psi
            x_label='psi'
        else:
            in_x=data[shot]['zipfit_and_aot'][signal]['rho']
            in_x=np.tile(in_x,(len(standard_times),1))
            out_x=standard_rho
            x_label='rho'

        #import pdb; pdb.set_trace()
        final_data[shot][final_sig_name]=zipfit_interp(in_x,standard_times,value,uncertainty,out_x,standard_times)

        if debug and debug_sig==final_sig_name and debug_shot==shot:
            plot_comparison_over_time(xlist=(in_x,out_x),
                                      ylist=(value,final_data[shot][final_sig_name]), 
		                      time=standard_times,
                	              ylabel=final_sig_name,
                	              xlabel=x_label,
                	              uncertaintylist=(uncertainty,None),
                                      labels=('data','original'))
            
# with open(os.path.join(data_dir,'final_data_batch_{}.pkl'.format(batch_num)),'wb') as f:
#     pickle.dump(final_data,f)
