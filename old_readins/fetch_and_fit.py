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

shots = [175970] #,165249,166348,167303,167449,167605,169605,170090,170256,170338,170503,170783,171169]

tmin = 0
tmax = 5000

output_file='joe_test.pkl'

time_step=50

debug=False
debug_sig_name='cer_rot' #'thomson_DENSITY'

scalar_sig_names=[]
include_psirz=True
include_rhovn=True

thomson_sig_names=[] #['density']
thomson_scale={'density': 1e19, 'temp': 1e3}
include_thomson_uncertainty=True
thomson_areas=['CORE','TANGENTIAL']

cer_sig_names=[] #temp
cer_scale={'temp': 1e3, 'rot': 1}
cer_type='cerquick'
cer_areas=['TANGENTIAL', 'VERTICAL']
cer_channels={'TANGENTIAL': np.arange(1,33),
              'VERTICAL': np.arange(1,49)}

zipfit_sig_names=['trotfit','edensfit','etempfit']

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
                     causal=True, window_size=200,
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
####################################

######## FETCH PSIRZ   #############
if include_psirz:
    psirz_sig = MdsSignal(r'\psirz',
                          'efit01', 
                          location='remote://atlas.gat.com',
                          dims=['r','z','times'])
    pipeline.fetch('psirz_full',psirz_sig)
    ssimag_sig = MdsSignal(r'\ssimag',
                          'efit01', 
                          location='remote://atlas.gat.com')
    pipeline.fetch('ssimag_full',ssimag_sig)
    ssibry_sig = MdsSignal(r'\ssibry',
                          'efit01', 
                          location='remote://atlas.gat.com')
    pipeline.fetch('ssibry_full',ssibry_sig)
####################################

######## FETCH RHOVN ###############
if include_rhovn:
    rhovn_sig = MdsSignal(r'\rhovn',
                          'efit01', 
                          location='remote://atlas.gat.com',
                          dims=['psi','times'])
    pipeline.fetch('rhovn_full',rhovn_sig)
####################################

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
####################################

######## FETCH CER     #############

for cer_area in cer_areas:
    for channel in cer_channels[cer_area]:
        cer_R_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.R'.format(cer_type,
                                                                 cer_area,
                                                                 channel),
                              'IONS',
                              location='remote://atlas.gat.com')
        pipeline.fetch('cer_{}_{}_R_full'.format(cer_area,channel),cer_R_sig)
        cer_Z_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.Z'.format(cer_type,
                                                                 cer_area,
                                                                 channel),
                              'IONS',
                              location='remote://atlas.gat.com')
        pipeline.fetch('cer_{}_{}_Z_full'.format(cer_area,channel),cer_Z_sig)

        for sig_name in cer_sig_names:
            correction=''
            if sig_name=='rot':
                correction='c'
            cer_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.{}'.format(cer_type,
                                                                    cer_area,
                                                                    channel,
                                                                    sig_name+correction),
                                'IONS',
                                location='remote://atlas.gat.com')
            pipeline.fetch('cer_{}_{}_{}_full'.format(cer_area,sig_name,channel),cer_sig)
            cer_error_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.{}_ERR'.format(cer_type,
                                                                              cer_area,
                                                                              channel,
                                                                              sig_name),
                                      'IONS',
                                      location='remote://atlas.gat.com')
            pipeline.fetch('cer_{}_{}_{}_error_full'.format(cer_area,sig_name,channel),cer_error_sig)
            
####################################


######## FETCH ZIPFIT ##############
for sig_name in zipfit_sig_names:
    zipfit_sig = MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times'])
    pipeline.fetch('zipfit_{}_full'.format(sig_name),zipfit_sig)
####################################

@pipeline.map
def add_timebase(record):
    standard_times=np.arange(tmin,tmax,time_step)
    record['standard_time']=standard_times

@pipeline.map
def change_scalar_timebase(record):
    for sig_name in scalar_sig_names:
        record[sig_name]=standardize_time(record['{}_full'.format(sig_name)]['data'],
                                          record['{}_full'.format(sig_name)]['times'],
                                          record['standard_time'])
if include_psirz:
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
        record['psirz_r']=record['psirz_full']['r']
        record['psirz_z']=record['psirz_full']['z']

if include_rhovn:
    @pipeline.map
    def add_rhovn(record):
        record['rhovn']=standardize_time(record['rhovn_full']['data'],
                                         record['rhovn_full']['times'],
                                         record['standard_time'])
@pipeline.map
def zipfit_rhovn_to_psin(record):
    for sig_name in zipfit_sig_names:
        record['zipfit_{}_rhon_basis'.format(sig_name)]=standardize_time(record['zipfit_{}_full'.format(sig_name)]['data'],
                                                                   record['zipfit_{}_full'.format(sig_name)]['times'],
                                                                   record['standard_time'])

        rho_to_psi=[my_interp(record['rhovn'][time_ind], 
                              record['rhovn_full']['psi']) for time_ind in range(len(record['standard_time']))]
        record['zipfit_{}_psi'.format(sig_name)]=[]
        for time_ind in range(len(record['standard_time'])):
            record['zipfit_{}_psi'.format(sig_name)].append(rho_to_psi[time_ind](record['zipfit_{}_full'.format(sig_name)]['rhon']))
        record['zipfit_{}_psi'.format(sig_name)]=np.array(record['zipfit_{}_psi'.format(sig_name)])

        zipfit_interp=fit_function_dict['linear_interp_1d']
        record['zipfit_{}'.format(sig_name)]=zipfit_interp(record['zipfit_{}_psi'.format(sig_name)],
                                                           record['standard_time'],
                                                           record['zipfit_{}_rhon_basis'.format(sig_name)],
                                                           np.ones(record['zipfit_{}_rhon_basis'.format(sig_name)].shape),
                                                           standard_x)
#        record['zipfit_{}'.format(sig_name)]=record['zipfit_{}_full'.format(sig_name)]
        
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
def map_cer_1d(record):
    # an rz interpolator for each standard time
    r_z_to_psi=[interpolate.interp2d(record['psirz_r'],
                                     record['psirz_z'],
                                     record['psirz'][time_ind]) for time_ind in range(len(record['standard_time']))]

    for sig_name in cer_sig_names:
        value=[]
        psi=[]
        error=[]
        for cer_area in cer_areas:
            for channel in cer_channels[cer_area]:
                if record['cer_{}_{}_{}_full'.format(cer_area,sig_name,channel)] is not None:
                    r=standardize_time(record['cer_{}_{}_R_full'.format(cer_area,channel)]['data'],
                                       record['cer_{}_{}_{}_full'.format(cer_area,sig_name,channel)]['times'],
                                       record['standard_time'])
                    z=standardize_time(record['cer_{}_{}_Z_full'.format(cer_area,channel)]['data'],
                                       record['cer_{}_{}_{}_full'.format(cer_area,sig_name,channel)]['times'],
                                       record['standard_time'])

                    value.append(standardize_time(record['cer_{}_{}_{}_full'.format(cer_area,sig_name,channel)]['data'],
                                                  record['cer_{}_{}_{}_full'.format(cer_area,sig_name,channel)]['times'],
                                                  record['standard_time']))
                    if sig_name=='rot':
                        value[-1]=np.divide(value[-1],r)
                    psi.append([r_z_to_psi[time_ind](r[time_ind],z[time_ind])[0] \
                                for time_ind in range(len(record['standard_time']))])
                    error.append(standardize_time(record['cer_{}_{}_{}_error_full'.format(cer_area,sig_name,channel)]['data'],
                                                  record['cer_{}_{}_{}_error_full'.format(cer_area,sig_name,channel)]['times'],
                                                  record['standard_time']))
        value=np.array(value).T/cer_scale[sig_name]
        psi=np.array(psi).T
        error=np.array(error).T
        value[np.where(error==1)]=np.nan
        uncertainty=np.ones(np.shape(value))
        record['cer_{}_raw_1d'.format(sig_name)]=value
        record['cer_{}_uncertainty_raw_1d'.format(sig_name)]=uncertainty
        record['cer_{}_psi_raw_1d'.format(sig_name)]=psi
        for trial_fit in trial_fits:
            if trial_fit in fit_functions_1d:
                record['cer_{}_{}'.format(sig_name,trial_fit)] = fit_function_dict[trial_fit](psi,record['standard_time'],value,uncertainty,standard_x)

#needed_sigs=[sig_name for sig_name in all_sig_names]
needed_sigs=[]
for trial_fit in trial_fits:
    needed_sigs+=['cer_{}_{}'.format(sig_name,trial_fit) for sig_name in cer_sig_names]
    needed_sigs+=['thomson_{}_{}'.format(sig_name,trial_fit) for sig_name in thomson_sig_names]
needed_sigs+=['zipfit_{}'.format(sig_name) for sig_name in zipfit_sig_names]
####### TAKE THIS FOR NEWER MODELS ############
needed_sigs+=['zipfit_{}_rhon_basis'.format(sig_name) for sig_name in zipfit_sig_names]
###############################################
needed_sigs.append('standard_time')
if debug:
    needed_sigs.append('{}_psi_raw_1d'.format(debug_sig_name))
    needed_sigs.append('{}_raw_1d'.format(debug_sig_name))
    needed_sigs.append('{}_uncertainty_raw_1d'.format(debug_sig_name))
pipeline.keep(needed_sigs)

records=pipeline.compute_serial()

with open(output_file,'wb') as f:
    pickle.dump(records, f)

if debug:
    for elem in records[0].keys():
        if elem is not None:
            print(elem)
#    print(records[0]['cer_VERTICAL_temp_20_full'])
    #print('errors: ')
    #print(records[0]['errors'])
    if 'thomson' in debug_sig_name or 'cer' in debug_sig_name:
        xlist=[records[0]['{}_psi_raw_1d'.format(debug_sig_name)]]
        ylist=[records[0]['{}_raw_1d'.format(debug_sig_name)]]
        uncertaintylist=[records[0]['{}_uncertainty_raw_1d'.format(debug_sig_name)]]
        labels=['raw']
        for trial_fit in trial_fits:
            xlist.append(standard_x)
            ylist.append(records[0]['{}_{}'.format(debug_sig_name,trial_fit)])
            uncertaintylist.append(None)
            labels.append(trial_fit)
        xlist.append(standard_x)
        ylist.append(records[0]['zipfit_{}'.format(zipfit_pairs[debug_sig_name])])
        uncertaintylist.append(None)
        labels.append('zipfit_{}'.format(zipfit_pairs[debug_sig_name]))
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=records[0]['standard_time'],
                                  ylabel=debug_sig_name,
                                  xlabel='psi',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

    if debug_sig_name in scalar_sig_names:
        plt.scatter(records[0]['{}_full'.format(debug_sig_name)]['times'][::100],
                    records[0]['{}_full'.format(debug_sig_name)]['data'][::100],
                    c='b',
                    label='original')
        plt.plot(records[0]['standard_time'],
                 records[0][debug_sig_name],
                 c='r',
                 label='interpolated')
        plt.legend()
        plt.xlabel('time (unit: {})'.format(records[0]['{}_full'.format(debug_sig_name)]['units']['times']))
        plt.ylabel('{} (unit: {})'.format(debug_sig_name, records[0]['{}_full'.format(debug_sig_name)]['units']['data']))
        plt.show()
