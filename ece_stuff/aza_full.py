#!/usr/bin/env python
'''
Simple script to utilize toksearch capabilities to build a Thomson ne database
Saves data to pickle files with a single time array

Author: Joe Abbate & Oak Nelson (Nov 22 2020)

INPUTS: 
 - shots      (list)    | a list of shots to include
 - tmin       (float)   | minimum time to include in output
 - tmax       (float)   | maximum time to include in output
 - outdir     (string)  | output directory for pkl files

OUTPUTS:
 - <shot>.pkl (pkl)     | a pickle file with the ECE output
'''


from toksearch import PtDataSignal, MdsSignal, Pipeline
import numpy as np
import collections
import pprint
import pickle

import sys
import os

from scipy import interpolate

sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from transport_helpers import my_interp, interp_ND_rectangular
import fit_functions
from plot_tools import plot_comparison_over_time, plot_2d_comparison

import matplotlib.pyplot as plt

###### USER INPUTS ######
#                       #

shots=[]
with open('ece_shots.txt','rb') as f:
    for line in f:
        shots.append(int(line))
shots = [163303] #175970 is Colin's good modulation shot; 154764 is David Eldon's

tmin = 0
tmax = 5000

output_file='/mnt/beegfs/users/abbatej/ece_cutoffs.pkl'

time_step=50

debug=True
debug_sig_name='' #'thomson_DENSITY'

scalar_sig_names=['bt'] #['dstdenp','dssdenest','iptipp',
                  #'bt','DUSTRIPPED','ip','N1ICWMTH','N1IIWMTH','echpwr','dsifbonoff']
stability_sig_names=[] #['n1rms']
nb_sig_names=[] # ['pinj','tinj']

efit_profile_sig_names=['fpol'] #['fpol','qpsi','pres']
efit_scalar_sig_names=['R0','BCENTR','RMAXIS'] #['li','aminor','kappa','tritop','tribot','volume']

efit_type='EFIT02'

include_psirz=True
include_rhovn=True
include_ece=True

thomson_sig_names=['density','temp']
thomson_scale={'density': 1e19, 'temp': 1}
include_thomson_uncertainty=True
thomson_areas=['CORE','TANGENTIAL']

cer_sig_names=[] #temp
cer_scale={'temp': 1e3, 'rot': 1}
cer_type='cerquick'
cer_areas=['TANGENTIAL', 'VERTICAL']
cer_channels={'TANGENTIAL': np.arange(1,33),
              'VERTICAL': np.arange(1,49)}

zipfit_sig_names=[] #['trotfit','edensfit','etempfit','itempfit','idensfit']

zipfit_pairs={'cer_temp': 'itempfit',
              'cer_rot': 'trotfit',
              'thomson_temp': 'etempfit',
              'thomson_density': 'edensfit'}

# psi
standard_x=np.linspace(0,1,65)

fit_function_dict={'linear_interp_1d': fit_functions.linear_interp_1d,
                   'spline_1d': fit_functions.spline_1d,
                   'nn_interp_2d': fit_functions.nn_interp_2d,
                   'linear_interp_2d': fit_functions.linear_interp_2d,
                   'mtanh_1d': fit_functions.mtanh_1d,
                   'rbf_interp_2d': fit_functions.rbf_interp_2d}
fit_functions_1d=['linear_interp_1d', 'mtanh_1d','spline_1d']
fit_functions_2d=['nn_interp_2d','linear_interp_2d','rbf_interp_2d']

trial_fits=['linear_interp_1d'] #,'mtanh_1d']

#                       #
#### END USER INPUTS ####


#### START OF SCRIPT ####
#                       #

#### GATHER DATA ####

pipeline = Pipeline(shots) 

def standardize_time(old_signal,old_timebase,standard_times,
                     causal=True, window_size=50,
                     exponential_falloff=False, falloff_rate=100):
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
                new_signal.append(np.mean(old_signal[inds_in_range],axis=0))
    return np.array(new_signal)

######## FETCH SCALARS #############
for sig_name in scalar_sig_names:
    signal=PtDataSignal(sig_name)
    pipeline.fetch('{}_full'.format(sig_name),signal)

######## FETCH SCALARS #############
for sig_name in nb_sig_names:
    signal=MdsSignal(sig_name,
                     'NB',
                     location='remote://atlas.gat.com')
    pipeline.fetch('{}_full'.format(sig_name),signal)

######## FETCH EFIT PROFILES #############
for sig_name in efit_profile_sig_names:
    signal=MdsSignal('RESULTS.GEQDSK.{}'.format(sig_name),
                     efit_type,
                     location='remote://atlas.gat.com',
                     dims=['psi','times'])
    pipeline.fetch('{}_full'.format(sig_name),signal)

######## FETCH EFIT PROFILES #############
for sig_name in efit_scalar_sig_names:
    signal=MdsSignal('RESULTS.GEQDSK.{}'.format(sig_name),
                     efit_type,
                     location='remote://atlas.gat.com')
    pipeline.fetch('{}_full'.format(sig_name),signal)


######## FETCH PSIRZ   #############
if include_psirz:
    psirz_sig = MdsSignal(r'\psirz',
                          efit_type, 
                          location='remote://atlas.gat.com',
                          dims=['r','z','times'])
    pipeline.fetch('psirz_full',psirz_sig)
    ssimag_sig = MdsSignal(r'\ssimag',
                          efit_type, 
                          location='remote://atlas.gat.com')
    pipeline.fetch('ssimag_full',ssimag_sig)
    ssibry_sig = MdsSignal(r'\ssibry',
                          efit_type, 
                          location='remote://atlas.gat.com')
    pipeline.fetch('ssibry_full',ssibry_sig)

######## FETCH RHOVN ###############
if include_rhovn:
    rhovn_sig = MdsSignal(r'\rhovn',
                          efit_type, 
                          location='remote://atlas.gat.com',
                          dims=['psi','times'])
    pipeline.fetch('rhovn_full',rhovn_sig)

######## FETCH THOMSON #############
for sig_name in thomson_sig_names:
    for thomson_area in thomson_areas:
        thomson_sig = MdsSignal(r'TS.BLESSED.{}.{}'.format(thomson_area,sig_name),
                                'ELECTRONS',
                                location='remote://atlas.gat.com', 
                                dims=('times','position'))
        pipeline.fetch('thomson_{}_{}_full'.format(thomson_area,sig_name),thomson_sig)
        if include_thomson_uncertainty:
            thomson_error_sig = MdsSignal(r'TS.BLESSED.{}.{}_E'.format(thomson_area,sig_name),
                                          'ELECTRONS',
                                          location='remote://atlas.gat.com')
            pipeline.fetch('thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name),thomson_error_sig)

######## FETCH ECE     #############
if include_ece:
    signal=MdsSignal('.PROFS.ECEPROF',
                     'ECE',
                     location='remote://atlas.gat.com',
                     dims=['times','channels'])
    pipeline.fetch('ece_full',signal)

@pipeline.map
def add_timebase(record):
    standard_times=np.arange(tmin,tmax,time_step)
    record['standard_time']=standard_times

@pipeline.map
def change_timebase(record):
    all_sig_names=efit_profile_sig_names+efit_scalar_sig_names+nb_sig_names+scalar_sig_names+stability_sig_names
    for sig_name in all_sig_names:
        try:
            record[sig_name]=standardize_time(record['{}_full'.format(sig_name)]['data'],
                                              record['{}_full'.format(sig_name)]['times'],
                                              record['standard_time'])
        except:
            print('missing {}'.format(sig_name))

@pipeline.map
def add_psin(record):
    psi_norm_f = record['ssibry_full']['data'] - record['ssimag_full']['data']
    # Prevent divide by 0 error by replacing 0s in the denominator
    problems = psi_norm_f == 0
    psi_norm_f[problems] = 1.
    record['psirz'] = (record['psirz_full']['data'] - record['ssimag_full']['data'][:, np.newaxis, np.newaxis]) / psi_norm_f[:, np.newaxis, np.newaxis]
    record['psirz'][problems] = 0
    record['psirz']=standardize_time(record['psirz'],
                                          record['psirz_full']['times'],
                                          record['standard_time'])

    #############
    record['psirz_unnorm']=standardize_time(record['psirz_full']['data'],
                                            record['psirz_full']['times'],
                                            record['standard_time'])
    #############

    record['psirz_r']=record['psirz_full']['r']
    record['psirz_z']=record['psirz_full']['z']
        
@pipeline.map
def map_thomson_1d(record):
    # an rz interpolator for each standard time
    r_z_to_psi=[interpolate.interp2d(record['psirz_r'],
                                     record['psirz_z'],
                                     record['psirz'][time_ind]) for time_ind in range(len(record['standard_time']))]

    for sig_name in thomson_sig_names:
        value=[]
        psi=[]
        uncertainty=[]
        for thomson_area in thomson_areas:
            for channel in range(len(record['thomson_{}_{}_full'.format(thomson_area,sig_name)]['position'])):
                value.append(standardize_time(record['thomson_{}_{}_full'.format(thomson_area,sig_name)]['data'][channel],
                                              record['thomson_{}_{}_full'.format(thomson_area,sig_name)]['times'],
                                              record['standard_time']))
                if thomson_area=='TANGENTIAL':
                    r=record['thomson_{}_{}_full'.format(thomson_area,sig_name)]['position'][channel]
                    z=0
                elif thomson_area=='CORE':
                    z=record['thomson_{}_{}_full'.format(thomson_area,sig_name)]['position'][channel]
                    r=1.94
                psi.append([r_z_to_psi[time_ind](r,z)[0] for time_ind in range(len(record['standard_time']))])
                if include_thomson_uncertainty:
                    uncertainty.append(standardize_time(record['thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name)]['data'][channel],
                                              record['thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name)]['times'],
                                              record['standard_time']))
                
        value=np.array(value).T/thomson_scale[sig_name]
        psi=np.array(psi).T
        value[np.isclose(value,0)]=np.nan
        if include_thomson_uncertainty:
            uncertainty=np.array(uncertainty).T/thomson_scale[sig_name]
            value[np.isclose(uncertainty,0)]=np.nan
        else:
            uncertainty=np.ones(np.shape(value))
        record['thomson_{}_raw_1d'.format(sig_name)]=value
        record['thomson_{}_uncertainty_raw_1d'.format(sig_name)]=uncertainty
        record['thomson_{}_psi_raw_1d'.format(sig_name)]=psi
        for trial_fit in trial_fits:
            if trial_fit in fit_functions_1d:
                record['thomson_{}_{}'.format(sig_name,trial_fit)] = fit_function_dict[trial_fit](psi,record['standard_time'],value,uncertainty,standard_x)

@pipeline.map
def get_ece_info(record):
    const_e=1.6e-19
    const_m=9.1e-31
    const_eps0=8.85e-12
    const_scale=2*np.pi*1e9 #omega to (f in GHz)
    z_ind=np.argmin(np.abs(record['psirz_z']))
    R=record['psirz_r']
    psi=record['psirz'][:,:,z_ind]

    #bt=[np.divide(record['bt'][time_ind]*record['R0'][time_ind],R) for time_ind in range(len(record['standard_time']))]
    bt=[np.divide(record['BCENTR'][time_ind]*record['RMAXIS'][time_ind],R) for time_ind in range(len(record['standard_time']))]
    bt=np.abs(np.array(bt))

    get_density=[interpolate.interp1d(standard_x,
                                      record['thomson_density_linear_interp_1d'][time_ind],
                                      bounds_error=False,
                                      fill_value=0) for time_ind in range(len(record['standard_time']))]

    density=[get_density[time_ind](psi[time_ind]) for time_ind in range(len(record['standard_time']))]
    density=np.array(density)

    w_ce=const_e*bt/const_m
    w_pe=np.sqrt(density*1e19*const_e**2/(const_m*const_eps0))
    w_cutoff=(w_ce+np.sqrt(np.square(w_ce)+4*np.square(w_pe)) )/2

    ############
    
    bp_2d=[]
    for time_ind in range(len(record['standard_time'])):
        [dPSIdR, dPSIdZ] = np.gradient(record['psirz_unnorm'][time_ind,:,:],
                                          record['psirz_r'],
                                          record['psirz_z'])
        bp_2d.append(np.sqrt(np.square(dPSIdR)+np.square(dPSIdZ)))
    bp_2d=np.array(bp_2d)
    bp=bp_2d[:,:,z_ind]

    get_fpol=[interpolate.interp1d(record['fpol_full']['psi'],
                                   record['fpol'][time_ind],
                                   bounds_error=False,
                                   fill_value=0) for time_ind in range(len(record['standard_time']))]
    fpol=[get_fpol[time_ind](psi[time_ind]) for time_ind in range(len(record['standard_time']))]
    fpol=np.abs(np.array(fpol))
    
    bt_fpol=np.divide(fpol,R)
    bt_fpol_and_bp=np.sqrt(np.square(bt_fpol)+np.square(bp))
    w_ce_fpol=const_e*bt_fpol/const_m
    w_ce_fpol_and_bp=const_e*bt_fpol_and_bp/const_m
    record['ece_fce_2_fpol']=2*w_ce_fpol/const_scale

    record['ece_fce_2_fpol_and_bp']=2*w_ce_fpol_and_bp/const_scale
    #############

    record['ece_fce_2']=2*w_ce/const_scale
    record['ece_fcutoff']=w_cutoff/const_scale
    record['ece_R']=R
    record['ece_psi']=psi

    channel_freqs=[]
    channel_freqs.extend(np.arange(83.5,98.5,1))
    channel_freqs.extend(np.arange(98.5,114.5,1))
    channel_freqs.extend(np.arange(115.5,131.5,2))
    
    psi_channels=[]
    r_channels=[]
    fce_2_channels=[]
    fcutoff_channels=[]
    for channel_freq in channel_freqs:
        psi_inds=np.argmin(np.abs(channel_freq-record['ece_fce_2']),axis=1)
        psi_channels.append(np.array([psi[i,psi_inds[i]] for i in range(len(record['standard_time']))]))
        r_channels.append(np.array([R[psi_inds[i]] for i in range(len(record['standard_time']))]))
        fce_2_channels.append(np.array([record['ece_fce_2'][i,psi_inds[i]] for i in range(len(record['standard_time']))]))
        fcutoff_channels.append(np.array([record['ece_fcutoff'][i,psi_inds[i]] for i in range(len(record['standard_time']))]))

    record['ece_psi_channel']=np.array(psi_channels).T
    record['ece_r_channel']=np.array(r_channels).T
    record['ece_fce_2_channel']=np.array(fce_2_channels).T
    record['ece_fcutoff_channel']=np.array(fcutoff_channels).T
    
    if include_ece:
        record['ece']=standardize_time(record['ece_full']['data'].T,
                                       record['ece_full']['times'],
                                       record['standard_time'])

#Needed_sigs=[sig_name for sig_name in all_sig_names]
needed_sigs=[]
needed_sigs+=['ece_fce_2','ece_fce_2_fpol','ece_fce_2_fpol_and_bp','ece_fcutoff','ece_R','ece_psi']
needed_sigs+=['ece_r_channel','ece_psi_channel','ece_fce_2_channel','ece_fcutoff_channel']
needed_sigs+=['thomson_density_linear_interp_1d','thomson_temp_linear_interp_1d']
if include_ece:
    needed_sigs+=['ece']

needed_sigs.append('standard_time')

pipeline.keep(needed_sigs)

records=pipeline.compute_serial()

final_data={}
for i in range(len(records)):
    record=records[i]
    shot=int(record['shot'])
    final_data[shot]={}
    for sig in needed_sigs:
        final_data[shot][sig]=np.array(record[sig])

with open(output_file,'wb') as f:
    pass #pickle.dump(final_data, f)

if debug:        
    shot=list(final_data.keys())[0]

    plt.contourf(final_data[shot]['standard_time'],
                 np.arange(len(final_data[shot]['ece'][0])),
                 final_data[shot]['ece'].T)
    plt.axvline(3000)
    plt.show()
    
    # plot ECE full info at single timestep
    if False:
        channel_freqs=[]
        channel_freqs.extend(np.arange(83.5,98.5,1))
        channel_freqs.extend(np.arange(98.5,114.5,1))
        channel_freqs.extend(np.arange(115.5,131.5,2))
        
        time_ind=np.searchsorted(final_data[shot]['standard_time'],3000) #60
        plt.plot(final_data[shot]['ece_R'],
                 final_data[shot]['ece_fce_2'][time_ind],
                 label=r'2 $f_{ce}$ from bt/R0')
        plt.plot(final_data[shot]['ece_R'],
                 final_data[shot]['ece_fce_2_fpol'][time_ind],
                 label=r'2 $f_{ce}$ from $|F_p/R|$')
        plt.plot(final_data[shot]['ece_R'],
                 final_data[shot]['ece_fce_2_fpol_and_bp'][time_ind],
                 label=r'2 $f_{ce}$ from $|F_p/R + \nabla \psi|$')
        plt.plot(final_data[shot]['ece_R'],
                 final_data[shot]['ece_fcutoff'][time_ind],
                 label=r'$f_{cutoff}$')
        plt.xlabel('R (m)')
        plt.ylabel('frequency (GHz)')
        plt.title('Shot {}, time {}ms'.format(shot,final_data[shot]['standard_time'][time_ind]))
        for i,channel_freq in enumerate(channel_freqs):
            plt.axhline(channel_freq,linewidth=1,alpha=.5)
            if final_data[shot]['ece_fce_2'][time_ind,i]<final_data[shot]['ece_fcutoff'][time_ind,i]:
                this_color='r'
            else:
                this_color='g'
            #plt.axvline(final_data[shot]['ece_r_channel'][time_ind,i],linewidth=1,alpha=.5,c=this_color)
        plt.legend()
        plt.show()

    # plot ECE full info over time
    if False: 
        xlist=[final_data[shot]['ece_R'],final_data[shot]['ece_R']]
        ylist=[final_data[shot]['ece_fce_2'],
               final_data[shot]['ece_fcutoff']]
        uncertaintylist=[None,None]
        labels=[r'2 $f_{ce}$',r'$f_{cutoff}$']
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=final_data[shot]['standard_time'],
                                  ylabel=debug_sig_name,
                                  xlabel='R (m)',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)
    
    if False:
        xlist=[np.arange(1,40),
               np.arange(1,40)]
        ylist=[final_data[shot]['ece_fcutoff_channel'],
               final_data[shot]['ece_fce_2_channel']]
        uncertaintylist=[None,None]
        labels=['cutoff','2nd cyclotron']
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=final_data[shot]['standard_time'],
                                  ylabel='Frequency (GHz)',
                                  xlabel='Channel',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

    if False:
        xlist=[np.arange(1,40),np.arange(1,40)]
        ylist=[final_data[shot]['ece_psi_channel'],
               final_data[shot]['ece_r_channel']]
        uncertaintylist=[None,None]
        labels=['psi','R']
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=final_data[shot]['standard_time'],
                                  ylabel=None,
                                  xlabel='Channel',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

    if False:
        xlist=[final_data[shot]['ece_psi_channel'],
               standard_x]
        ylist=[final_data[shot]['ece'],
               final_data[shot]['thomson_temp_linear_interp_1d']]
        uncertaintylist=[None,None]
        labels=['ece','thomson']
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=final_data[shot]['standard_time'],
                                  ylabel='Temp (eV)',
                                  xlabel='psi',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)
