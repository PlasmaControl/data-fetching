#!/usr/bin/env python
'''
Requires
1) git clone https://github.com/segasai/astrolibpy
   into the lib/ dir (this is for mtanh fits for temperature)
3) pip install csaps
   this is for smoothing spline fits for rotation
3) cd to lib/splines/, module load gcc-9.2.0, and type "make"
   this is to make libspline.o, called by pcs_fit_helpers.py
   which is in turn called by pcs_spline_1d (pcs spline for
   rotation)
4) module purge, then module load toksearch
Run as python new_database_maker.py configs/etemp.yaml
'''

from toksearch import PtDataSignal, MdsSignal, Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np
import collections
import pprint
import sys
import os
import time
from scipy import interpolate
sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from transport_helpers import my_interp, standardize_time, Timer
import fit_functions
from plot_tools import plot_comparison_over_time, plot_2d_comparison
import matplotlib.pyplot as plt
import yaml
import argparse
import h5py

parser = argparse.ArgumentParser(description='Read tokamak data via toksearch.')
parser.add_argument('config_filename', type=str,
                    help='configuration file (e.g. configs/autoencoder.yaml)')
args = parser.parse_args()

with open(args.config_filename,"r") as f:
    cfg=yaml.safe_load(f)

from database_settings import pcs_length, name_map, zipfit_pairs, cer_scale, cer_areas, cer_channels_realtime, cer_channels_all, thomson_areas, thomson_scale
for sig in cfg['data']['efit_scalar_sig_names']+cfg['data']['efit_profile_sig_names']:
    if sig in name_map:
        name_map[sig]=name_map[sig]+f"_{cfg['data']['efit_type']}"
    else:
        name_map[sig]=sig+f"_{cfg['data']['efit_type']}"
if cfg['data']['include_rhovn']:
    name_map['rhovn']=f"rhovn_{cfg['data']['efit_type']}"


##########################

if cfg['logistics']['num_processes']>1:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

# first_shot_of_year=[0,    140838,143703,148158,152159,156197,160938,164773,168439,174574,177976,181675,183948,200000]
# campaign_names=    ['old','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
if isinstance(cfg['data']['shots'],str):
    all_shots=np.load(cfg['data']['shots'])
else:
    all_shots=cfg['data']['shots']
all_shots=sorted(all_shots,reverse=True)

# psi / rho
standard_x=np.linspace(0,1,cfg['data']['num_x_points'])
psirz_needed=(len(cfg['data']['cer_sig_names'])>0 or len(cfg['data']['thomson_sig_names'])>0)
fit_function_dict={'linear_interp_1d': fit_functions.linear_interp_1d,
                   'spline_1d': fit_functions.spline_1d,
                   'pcs_spline_1d': fit_functions.pcs_spline_1d,
                   'nn_interp_2d': fit_functions.nn_interp_2d,
                   'linear_interp_2d': fit_functions.linear_interp_2d,
                   'mtanh_1d': fit_functions.mtanh_1d,
                   'csaps_1d': fit_functions.csaps_1d,
                   'rbf_interp_2d': fit_functions.rbf_interp_2d}

fit_functions_1d=['linear_interp_1d', 'spline_1d', 'pcs_spline_1d', 'mtanh_1d','csaps_1d']
fit_functions_2d=['nn_interp_2d','linear_interp_2d','rbf_interp_2d']

filename=cfg['logistics']['output_file']

standard_times=np.arange(cfg['data']['tmin'],cfg['data']['tmax'],cfg['data']['time_step'])
with h5py.File(filename,'a') as final_data:
    if 'times' in final_data:
        assert np.all(final_data['times']==standard_times), f"Time in existing h5 file {filename} different from the one you attempt to read (based on config file's tmin, tmax, time_step)"
    else:
        final_data['times']=standard_times

if cfg['logistics']['overwrite_shots']:
    with h5py.File(filename,'a') as final_data:
        for shot in all_shots:
            if str(shot) in final_data:
                del final_data[str(shot)]

subshots=[]
num_files=int(len(all_shots)/(cfg['logistics']['max_shots_per_run']+1)) + 1
for i in range(num_files):
    subshots.append(all_shots[i*cfg['logistics']['max_shots_per_run']:min((i+1)*cfg['logistics']['max_shots_per_run'],
                                                                           len(all_shots))])

for which_shot,shots in enumerate(subshots):
    print(f'Starting shot {shots[0]}-{shots[-1]}')
    sys.stdout.flush()

    # pipeline for SQL signals
    if len(cfg['data']['sql_sig_names'])>0:
        query="select shot,{} from summaries where shot in {}".format(
            ','.join(cfg['data']['sql_sig_names']),
            '({})'.format(','.join([str(elem) for elem in shots]))
            )
        conn = connect_d3drdb()
        pipeline = Pipeline.from_sql(conn, query)
        records=pipeline.compute_serial()
        with h5py.File(filename,'a') as final_data:
            for record in records:
                shot=str(record['shot'])
                final_data.require_group(shot)
                for sig in cfg['data']['sql_sig_names']:
                    # if we get None it throws an error...
                    if record[sig]==None:
                        final_data[shot][sig]=np.nan
                    else:
                        final_data[shot][sig]=record[sig]
    # pipeline for regular signals
    pipeline = Pipeline(shots)

    ######## FETCH SCALARS #############
    for sig_name in cfg['data']['scalar_sig_names']:
        signal=PtDataSignal(sig_name)
        pipeline.fetch('{}_full'.format(sig_name),signal)

    ######## FETCH STABILITY #############
    for sig_name in cfg['data']['stability_sig_names']:
        signal=MdsSignal('.MIRNOV.{}'.format(sig_name),
                         'MHD',
                         location='remote://atlas.gat.com')
        pipeline.fetch('{}_full'.format(sig_name),signal)

    ######## FETCH SCALARS #############
    for sig_name in cfg['data']['nb_sig_names']:
        signal=MdsSignal(sig_name,
                         'NB',
                         location='remote://atlas.gat.com')
        pipeline.fetch('{}_full'.format(sig_name),signal)

    ######## FETCH EFIT PROFILES #############
    for sig_name in cfg['data']['efit_profile_sig_names']:
        signal=MdsSignal('RESULTS.GEQDSK.{}'.format(sig_name),
                         cfg['data']['efit_type'],
                         location='remote://atlas.gat.com',
                         dims=['psi','times'])
        pipeline.fetch('{}_full'.format(sig_name),signal)

    ######## FETCH EFIT PROFILES #############
    for sig_name in cfg['data']['efit_scalar_sig_names'] :
        signal=MdsSignal(r'\{}'.format(sig_name.upper()),
                         cfg['data']['efit_type'],
                         location='remote://atlas.gat.com')
        pipeline.fetch('{}_full'.format(sig_name),signal)


    ######## FETCH PSIRZ   #############
    if cfg['data']['include_psirz'] or psirz_needed:
        psirz_sig = MdsSignal(r'\psirz',
                              cfg['data']['efit_type'],
                              location='remote://atlas.gat.com',
                              dims=['r','z','times'])
        pipeline.fetch('psirz_full',psirz_sig)
        ssimag_sig = MdsSignal(r'\ssimag',
                              cfg['data']['efit_type'],
                              location='remote://atlas.gat.com')
        pipeline.fetch('ssimag_full',ssimag_sig)
        ssibry_sig = MdsSignal(r'\ssibry',
                              cfg['data']['efit_type'],
                              location='remote://atlas.gat.com')
        pipeline.fetch('ssibry_full',ssibry_sig)

    ######## FETCH RHOVN ###############
    if cfg['data']['include_rhovn']:
        rhovn_sig = MdsSignal(r'\rhovn',
                              cfg['data']['efit_type'],
                              location='remote://atlas.gat.com',
                              dims=['psi','times'])
        pipeline.fetch('rhovn_full',rhovn_sig)

    ######## FETCH THOMSON #############
    for sig_name in cfg['data']['thomson_sig_names']:
        for thomson_area in thomson_areas:
            thomson_sig = MdsSignal(r'TS.BLESSED.{}.{}'.format(thomson_area,sig_name),
                                    'ELECTRONS',
                                    location='remote://atlas.gat.com',
                                    dims=('times','position'))
            pipeline.fetch('thomson_{}_{}_full'.format(thomson_area,sig_name),thomson_sig)
            if cfg['data']['include_thomson_uncertainty']:
                thomson_error_sig = MdsSignal(r'TS.BLESSED.{}.{}_E'.format(thomson_area,sig_name),
                                              'ELECTRONS',
                                              location='remote://atlas.gat.com')
                pipeline.fetch('thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name),thomson_error_sig)

    ######## FETCH CER     #############
    if len(cfg['data']['cer_sig_names'])>0:
        if cfg['data']['cer_realtime_channels']:
            cer_channels=cer_channels_realtime
        else:
            cer_channels=cer_channels_all
        for cer_area in cer_areas:
            for channel in cer_channels[cer_area]:
                cer_R_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.R'.format(cfg['data']['cer_type'],
                                                                         cer_area,
                                                                         channel),
                                      'IONS',
                                      location='remote://atlas.gat.com')
                pipeline.fetch('cer_{}_{}_R_full'.format(cer_area,channel),cer_R_sig)
                cer_Z_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.Z'.format(cfg['data']['cer_type'],
                                                                         cer_area,
                                                                         channel),
                                      'IONS',
                                      location='remote://atlas.gat.com')
                pipeline.fetch('cer_{}_{}_Z_full'.format(cer_area,channel),cer_Z_sig)

                for sig_name in cfg['data']['cer_sig_names']:
                    correction=''
                    if sig_name=='rot':
                        correction='c'
                    cer_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.{}'.format(cfg['data']['cer_type'],
                                                                            cer_area,
                                                                            channel,
                                                                            sig_name+correction),
                                        'IONS',
                                        location='remote://atlas.gat.com')
                    pipeline.fetch('cer_{}_{}_{}_full'.format(cer_area,sig_name,channel),cer_sig)
                    cer_error_sig = MdsSignal('CER.{}.{}.CHANNEL{:02d}.{}_ERR'.format(cfg['data']['cer_type'],
                                                                                      cer_area,
                                                                                      channel,
                                                                                      sig_name),
                                              'IONS',
                                              location='remote://atlas.gat.com')
                    pipeline.fetch('cer_{}_{}_{}_error_full'.format(cer_area,sig_name,channel),cer_error_sig)


    ######## FETCH ZIPFIT ##############
    for sig_name in cfg['data']['zipfit_sig_names']:
        zipfit_sig = MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times'])
        pipeline.fetch('zipfit_{}_full'.format(sig_name),zipfit_sig)

    ######## FETCH OUR PCS ALGO STUFF #############
    for sig_name in cfg['data']['pcs_sig_names']:
        for i in pcs_length[sig_name]:
            pcs_sig = PtDataSignal('{}{}'.format(sig_name,i))
            pipeline.fetch('{}{}_full'.format(sig_name,i),pcs_sig)

    ######## FETCH BOLOMETRY STUFF #############
    if cfg['data']['include_radiation']:
        for i in range(1,25):
            for position in ['L','U']:
                radiation_sig=MdsSignal(f'\\SPECTROSCOPY::TOP.PRAD.BOLOM.PRAD_01.POWER.BOL_{position}{i:02d}_P',
                                        'SPECTROSCOPY')
                pipeline.fetch(f'prad{position}{i}_full',radiation_sig)
        for key in ['KAPPA','PRAD_DIVL','PRAD_DIVU','PRAD_TOT']:
            radiation_sig=MdsSignal(f'\\SPECTROSCOPY::TOP.PRAD.BOLOM.PRAD_01.PRAD.{key}',
                                    'SPECTROSCOPY')
            pipeline.fetch(f'prad{key}_full',radiation_sig)
    @pipeline.map
    def add_timebase(record):
        standard_times=np.arange(cfg['data']['tmin'],cfg['data']['tmax'],cfg['data']['time_step'])
        record['standard_time']=standard_times

    @pipeline.map
    def change_timebase(record):
        all_sig_names=name_map.keys()
        for sig_name in all_sig_names:
            try:
                record[sig_name]=standardize_time(record['{}_full'.format(sig_name)]['data'],
                                                  record['{}_full'.format(sig_name)]['times'],
                                                  record['standard_time'])
                # kludged fix to exclude 'zipfit_' from the name here
                if (sig_name[7:] in cfg['data']['zipfit_sig_names']):
                    data=[]
                    for time_ind in range(len(record[sig_name])):
                        interpolator=interpolate.interp1d(record[f'{sig_name}_full']['rhon'],
                                                          record[sig_name][time_ind,:])
                        data.append(interpolator(standard_x))
                    record[sig_name]=np.array(data)
                if (sig_name in cfg['data']['efit_profile_sig_names']):
                    data=[]
                    for time_ind in range(len(record[sig_name])):
                        interpolator=interpolate.interp1d(record[f'{sig_name}_full']['psi'],
                                                          record[sig_name][time_ind,:])
                        data.append(interpolator(standard_x))
                    record[sig_name]=np.array(data)
            except:
                pass
                #print('missing {}'.format(sig_name))

    if cfg['data']['include_psirz'] or psirz_needed:
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

    if cfg['data']['include_rhovn']:
        @pipeline.map
        def add_rhovn(record):
            record['rhovn']=standardize_time(record['rhovn_full']['data'],
                                             record['rhovn_full']['times'],
                                             record['standard_time'])
#        @pipeline.map
        def zipfit_rhovn_to_psin(record):
            for sig_name in cfg['data']['zipfit_sig_names']:
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

        for sig_name in cfg['data']['thomson_sig_names']:
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
                    if cfg['data']['include_thomson_uncertainty']:
                        uncertainty.append(standardize_time(record['thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name)]['data'][channel],
                                                  record['thomson_{}_{}_uncertainty_full'.format(thomson_area,sig_name)]['times'],
                                                  record['standard_time']))

            value=np.array(value).T/thomson_scale[sig_name]
            psi=np.array(psi).T
            value[np.isclose(value,0)]=np.nan
            if cfg['data']['include_thomson_uncertainty']:
                uncertainty=np.array(uncertainty).T/thomson_scale[sig_name]
                value[np.isclose(uncertainty,0)]=np.nan
            else:
                uncertainty=np.ones(np.shape(value))
            record['thomson_{}_raw_1d'.format(sig_name)]=value
            record['thomson_{}_uncertainty_raw_1d'.format(sig_name)]=uncertainty
            record['thomson_{}_psi_raw_1d'.format(sig_name)]=psi
            for trial_fit in cfg['data']['trial_fits']:
                if trial_fit in fit_functions_1d:
                    record['thomson_{}_{}'.format(sig_name,trial_fit)] = fit_function_dict[trial_fit](psi,record['standard_time'],value,uncertainty,standard_x)

    @pipeline.map
    def map_cer_1d(record):
        # an rz interpolator for each standard time
        r_z_to_psi=[interpolate.interp2d(record['psirz_r'],
                                         record['psirz_z'],
                                         record['psirz'][time_ind]) for time_ind in range(len(record['standard_time']))]

        for sig_name in cfg['data']['cer_sig_names']:
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
                        # set to true for rotation if we want to convert km/s to kHz/s
                        if (sig_name=='rot' and cfg['data']['cer_rotation_units_of_kHz']):
                            value[-1]=np.divide(value[-1],2*np.pi*r)
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
            record['cer_{}_r_raw_1d'.format(sig_name)]=r
            for trial_fit in cfg['data']['trial_fits']:
                if trial_fit in fit_functions_1d:
                    record['cer_{}_{}'.format(sig_name,trial_fit)] = fit_function_dict[trial_fit](psi,record['standard_time'],value,uncertainty,standard_x)

    @pipeline.map
    def pcs_processing(record):
        for sig_name in cfg['data']['pcs_sig_names']:
            record['{}'.format(sig_name)]=[]
            for i in pcs_length[sig_name]:
                nonzero_inds=np.nonzero(record['{}{}_full'.format(sig_name,i)]['data'])
                if 'fts' in sig_name:
                    record['{}'.format(sig_name)].append(standardize_time(record['{}{}_full'.format(sig_name,i)]['data'][nonzero_inds],
                                                                          record['{}{}_full'.format(sig_name,i)]['times'][nonzero_inds],
                                                                          record['standard_time'],
                                                                          numpy_smoothing_fxn=np.max))
                else:
                    record['{}'.format(sig_name)].append(standardize_time(record['{}{}_full'.format(sig_name,i)]['data'][nonzero_inds],
                                                                          record['{}{}_full'.format(sig_name,i)]['times'][nonzero_inds],
                                                                          record['standard_time']))

    if not cfg['data']['gather_raw']:
        needed_sigs=[]
        needed_sigs+=[sig_name for sig_name in cfg['data']['scalar_sig_names']]
        needed_sigs+=[sig_name for sig_name in cfg['data']['nb_sig_names']]
        needed_sigs+=[sig_name for sig_name in cfg['data']['efit_profile_sig_names']]
        needed_sigs+=[sig_name for sig_name in cfg['data']['efit_scalar_sig_names'] ]
        needed_sigs+=[sig_name for sig_name in cfg['data']['stability_sig_names']]
        needed_sigs+=[sig_name for sig_name in cfg['data']['pcs_sig_names']]

        if cfg['data']['include_psirz']:
            needed_sigs+=['psirz','psirz_r','psirz_z']
        if cfg['data']['include_rhovn']:
            needed_sigs+=['rhovn']
        for sig_name in cfg['data']['cer_sig_names']:
            needed_sigs+=[f'cer_{sig_name}_raw_1d',
                          #f'cer_{sig_name}_uncertainty_raw_1d', no real uncertainty for CER
                          f'cer_{sig_name}_psi_raw_1d',
                          f'cer_{sig_name}_r_raw_1d']
        for sig_name in cfg['data']['thomson_sig_names']:
            needed_sigs+=[f'thomson_{sig_name}_raw_1d',
                          f'thomson_{sig_name}_uncertainty_raw_1d',
                          f'thomson_{sig_name}_psi_raw_1d']
        if cfg['data']['include_radiation']:
            for i in range(1,25):
                for position in ['L','U']:
                    needed_sigs+=[f'prad{position}{i}']
            for key in ['KAPPA','PRAD_DIVL','PRAD_DIVU','PRAD_TOT']:
                needed_sigs+=[f'prad{key}']

        for trial_fit in cfg['data']['trial_fits']:
            needed_sigs+=['cer_{}_{}'.format(sig_name,trial_fit) for sig_name in cfg['data']['cer_sig_names']]
            needed_sigs+=['thomson_{}_{}'.format(sig_name,trial_fit) for sig_name in cfg['data']['thomson_sig_names']]
        needed_sigs+=['zipfit_{}_rhon_basis'.format(sig_name) for sig_name in cfg['data']['zipfit_sig_names']]
        needed_sigs+=['zipfit_{}'.format(sig_name) for sig_name in cfg['data']['zipfit_sig_names']]
        # use below to discard unneeded info
        pipeline.keep(needed_sigs)

    ####### TAKE THIS OUT FOR NEWER MODELS, UNCOMMENT ABOVE ############
    #needed_sigs+=['zipfit_{}_full'.format(sig_name) for sig_name in cfg['data']['zipfit_sig_names']]
    #needed_sigs+=['pinj_full','dstdenp_full','iptipp_full','volume_full','tinj_full']
    #needed_sigs+=['n1rms_full']
    ###############################################
    # if cfg['logistics']['debug']:
    #     needed_sigs.append('{}_psi_raw_1d'.format(cfg['logistics']['debug_sig_name']))
    #     needed_sigs.append('{}_raw_1d'.format(cfg['logistics']['debug_sig_name']))
    #     needed_sigs.append('{}_uncertainty_raw_1d'.format(cfg['logistics']['debug_sig_name']))

    with Timer():
        if cfg['logistics']['num_processes']>1:
            records=pipeline.compute_spark(numparts=cfg['logistics']['num_processes'])
        else:
            records=pipeline.compute_serial()
    with h5py.File(filename,'a') as final_data:
        for record in records:
            shot=str(record['shot'])
            final_data.require_group(shot)
            for sig in record.keys():
                if sig=='shot' or sig=='errors':
                    continue
                if sig in name_map:
                    # for handling zipfit
                    if 'rhon_basis' in sig:
                        tmp=[]
                        rhon=record['zipfit_{}_full'.format(cfg['data']['zipfit_sig_names'][0])]['rhon']
                        for time_ind in range(len(record['standard_time'])):
                            rho_to_zipfit=my_interp(rhon,
                                                    record[sig][time_ind])
                            tmp.append(rho_to_zipfit(standard_x))
                        final_data[shot][name_map[sig]]=np.array(final_data[shot][name_map[sig]])
                    else:
                        scale_factor=1
                        if name_map[sig]=='curr_target':
                            scale_factor=0.5e6
                        final_data[shot][name_map[sig]]=np.array(record[sig])*scale_factor
                else:
                    final_data[shot][sig]=record[sig]
            print(final_data[shot].keys())
    # for i in range(len(records)):
    #     record=records[i]
        # to accomodate the old code's bug of flipping top and bottom
    #     if False:
    #         tmp=final_data[shot]['triangularity_top_EFIT01'].copy()
    #         final_data[shot]['triangularity_top_EFIT01']=final_data[shot]['triangularity_bot_EFIT01'].copy()
    #         final_data[shot]['triangularity_bot_EFIT01']=tmp
    #         final_data[shot]['pinj_full']=record['pinj_full']
    #         final_data[shot]['tinj_full']=record['tinj_full']
    #         final_data[shot]['iptipp_full']=record['iptipp_full']
    #         final_data[shot]['iptipp_full']['data']*=.5e6
    #         final_data[shot]['dstdenp_full']=record['dstdenp_full']
    #         final_data[shot]['volume_full']=record['volume_full']
    #         final_data[shot]['n1rms_full']=record['n1rms_full']
    # with open(filename,'wb') as f:
    #     pickle.dump(final_data, f)

if cfg['logistics']['debug']:
    for shot in final_data:
        print(shot)
        print(final_data[shot].keys())
#    print(records[0]['cer_VERTICAL_temp_20_full'])
    #print('errors: ')
    #print(records[0]['errors'])
    if 'thomson' in cfg['logistics']['debug_sig_name'] or 'cer' in cfg['logistics']['debug_sig_name']:
        xlist=[records[0]['{}_psi_raw_1d'.format(cfg['logistics']['debug_sig_name'])]]
        ylist=[records[0]['{}_raw_1d'.format(cfg['logistics']['debug_sig_name'])]]
        uncertaintylist=[records[0]['{}_uncertainty_raw_1d'.format(cfg['logistics']['debug_sig_name'])]]
        labels=['raw']
        for trial_fit in cfg['data']['trial_fits']:
            xlist.append(standard_x)
            ylist.append(records[0]['{}_{}'.format(cfg['logistics']['debug_sig_name'],trial_fit)])
            uncertaintylist.append(None)
            labels.append(trial_fit)
        xlist.append(standard_x)
        ylist.append(records[0]['zipfit_{}'.format(zipfit_pairs[cfg['logistics']['debug_sig_name']])])
        uncertaintylist.append(None)
        labels.append('zipfit_{}'.format(zipfit_pairs[cfg['logistics']['debug_sig_name']]))
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=records[0]['standard_time'],
                                  ylabel=cfg['logistics']['debug_sig_name'],
                                  xlabel='psi',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

    if cfg['logistics']['debug_sig_name'] in cfg['data']['scalar_sig_names']+cfg['data']['efit_scalar_sig_names'] +cfg['data']['nb_sig_names']:
        plt.scatter(records[0]['{}_full'.format(cfg['logistics']['debug_sig_name'])]['times'][::100],
                    records[0]['{}_full'.format(cfg['logistics']['debug_sig_name'])]['data'][::100],
                    c='b',
                    label='original')
        plt.plot(records[0]['standard_time'],
                 records[0][cfg['logistics']['debug_sig_name']],
                 c='r',
                 label='interpolated')
        plt.legend()
        plt.xlabel('time (unit: {})'.format(records[0]['{}_full'.format(cfg['logistics']['debug_sig_name'])]['units']['times']))
        plt.ylabel('{} (unit: {})'.format(cfg['logistics']['debug_sig_name'], records[0]['{}_full'.format(cfg['logistics']['debug_sig_name'])]['units']['data']))
        plt.show()

    if cfg['logistics']['debug_sig_name'] in cfg['data']['efit_profile_sig_names']:
        xlist=[standard_x]
        ylist=[records[0]['{}'.format(cfg['logistics']['debug_sig_name'])]]
        uncertaintylist=[None]
        labels=[cfg['logistics']['debug_sig_name']]
        plot_comparison_over_time(xlist=xlist,
                                  ylist=ylist,
                                  time=records[0]['standard_time'],
                                  ylabel=cfg['logistics']['debug_sig_name'],
                                  xlabel='psi',
                                  uncertaintylist=uncertaintylist,
                                  labels=labels)

