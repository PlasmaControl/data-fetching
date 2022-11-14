#!/usr/bin/env python
'''
Requires module purge, then module load toksearch
Run as python new_database_maker.py configs/quick_test.yaml
If you ever get an issue related to the D3DRDB.sybase_login
  try copying this file from your iris home directory to 
  your saga home directory or vice versa

A few other dependencies for specific fits (ignore
if you're not fitting cer and thomson yourself):
1) git clone https://github.com/segasai/astrolibpy
   into the lib/ dir (this is for mtanh fits for temperature)
3) pip install csaps
   this is for smoothing spline fits for rotation
3) cd to lib/splines/, module load gcc-9.2.0, and type "make"
   this is to make libspline.o, called by pcs_fit_helpers.py
   which is in turn called by pcs_spline_1d (pcs spline for
   rotation)
'''

from toksearch import PtDataSignal, MdsSignal, Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np
import collections
import pprint
import sys
import os
import time
from scipy import interpolate, stats
sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from transport_helpers import my_interp, standardize_time, Timer
import fit_functions
from plot_tools import plot_comparison_over_time, plot_2d_comparison
import matplotlib.pyplot as plt
import yaml
import argparse
import h5py
import datetime # for dealing with getting the datetime from the summaries table

parser = argparse.ArgumentParser(description='Read tokamak data via toksearch.')
parser.add_argument('config_filename', type=str,
                    help='configuration file (e.g. configs/autoencoder.yaml)')
args = parser.parse_args()

with open(args.config_filename,"r") as f:
    cfg=yaml.safe_load(f)

from database_settings import pcs_length, zipfit_pairs, cer_scale, cer_areas, cer_channels_realtime, cer_channels_all, thomson_areas, thomson_scale, modal_sig_names

needed_sigs=[]
needed_sigs+=[sig_name for sig_name in cfg['data']['scalar_sig_names']]
needed_sigs+=[sig_name for sig_name in cfg['data']['nb_sig_names']]
needed_sigs+=[sig_name for sig_name in cfg['data']['stability_sig_names']]
needed_sigs+=[sig_name for sig_name in cfg['data']['pcs_sig_names']]
needed_sigs+=[sig_name for sig_name in cfg['data']['aot_scalar_sig_names']]
for efit_type in cfg['data']['efit_types']:
    needed_sigs+=[f'{sig_name}_{efit_type}' for sig_name in cfg['data']['efit_profile_sig_names']]
    needed_sigs+=[f'{sig_name}_{efit_type}' for sig_name in cfg['data']['efit_scalar_sig_names'] ]
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
if cfg['data']['include_full_ech_data']:
    needed_sigs+=['ech_names','ech_frequency','ech_R','ech_Z',
                  'ech_pwr','ech_aziang','ech_polang','ech_pwr_total']
if cfg['data']['include_full_nb_data']:
    needed_sigs+=['nb_pinj','nb_tinj','nb_vinj','nb_vinj_scalar','nb_210_rtan','nb_150_tilt']
for trial_fit in cfg['data']['trial_fits']:
    needed_sigs+=['cer_{}_{}'.format(sig_name,trial_fit) for sig_name in cfg['data']['cer_sig_names']]
    needed_sigs+=['thomson_{}_{}'.format(sig_name,trial_fit) for sig_name in cfg['data']['thomson_sig_names']]
needed_sigs+=['zipfit_{}_rho'.format(sig_name) for sig_name in cfg['data']['zipfit_sig_names']]
needed_sigs+=['zipfit_{}_psi'.format(sig_name) for sig_name in cfg['data']['zipfit_sig_names']]

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

filename=os.path.expandvars(cfg['logistics']['output_file'])

standard_times=np.arange(cfg['data']['tmin'],cfg['data']['tmax'],cfg['data']['time_step'])
with h5py.File(filename,'a') as final_data:
    if 'times' in final_data:
        assert np.all(final_data['times']==standard_times), f"Time in existing h5 file {filename} different from the one you attempt to read (based on config file's tmin, tmax, time_step)"
    else:
        final_data['times']=standard_times
    if 'spatial_coordinates' in final_data:
        assert np.all(final_data['spatial_coordinates']==standard_x), f"Time in existing h5 file {filename} different from the one you attempt to read (based on config file's tmin, tmax, time_step)"
    else:
        final_data['spatial_coordinates']=standard_x

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
        conn = connect_d3drdb()
        # you can continue adding joins to make sure all signals get collected
        query="""SELECT summaries.shot,{}
                 FROM summaries
                 INNER JOIN shots ON summaries.shot=shots.shot
                 WHERE summaries.shot in {}
              """.format(
            ','.join(cfg['data']['sql_sig_names']),
            '({})'.format(','.join([str(elem) for elem in shots]))
            )
        pipeline = Pipeline.from_sql(conn, query)
        records=pipeline.compute_serial()
        with h5py.File(filename,'a') as final_data:
            for record in records:
                shot=str(record['shot'])
                final_data.require_group(shot)
                for sig in cfg['data']['sql_sig_names']:
                    sig_name=sig+'_sql'
                    # if we get None it throws an error...
                    if record[sig]==None:
                        final_data[shot][sig_name]=np.nan
                    # primarily for dealing with time_of_shot in summaries table
                    elif isinstance(record[sig],datetime.datetime):
                        final_data[shot][sig_name]=str(record[sig])
                    else:
                        final_data[shot][sig_name]=record[sig]

    # pipeline for GAS
    if cfg['data']['include_gas_valve_info']:
        gas_sigs=['gas','valve']
        conn = connect_d3drdb()
        # you can continue adding joins to make sure all signals get collected
        query="""SELECT shot,{}
                 FROM gasvalves
                 WHERE shot in {}
              """.format(
            ','.join(gas_sigs),
            '({})'.format(','.join([str(elem) for elem in shots]))
            )
        pipeline = Pipeline.from_sql(conn, query)
        records=pipeline.compute_serial()
        tmp_dic={str(shot): {sig: [] for sig in gas_sigs} for shot in shots}
        for record in records:
            for sig in gas_sigs:
                shot=str(record['shot'])
                tmp_dic[shot][sig].append(record[sig])
        with h5py.File(filename,'a') as final_data:
            for shot in tmp_dic:
                final_data.require_group(shot)
                for sig in gas_sigs:
                    sig_name=sig+'_sql'
                    final_data[shot][sig_name]=tmp_dic[shot][sig]

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
    for efit_type in cfg['data']['efit_types']:
        for sig_name in cfg['data']['efit_profile_sig_names']:
            signal=MdsSignal('RESULTS.GEQDSK.{}'.format(sig_name),
                             efit_type,
                             location='remote://atlas.gat.com',
                             dims=['psi','times'])
            pipeline.fetch('{}_{}_full'.format(sig_name,efit_type),
                           signal)
        ######## FETCH EFIT PROFILES #############
        for sig_name in cfg['data']['efit_scalar_sig_names'] :
            signal=MdsSignal(r'\{}'.format(sig_name.upper()),
                             efit_type,
                             location='remote://atlas.gat.com')
            pipeline.fetch('{}_{}_full'.format(sig_name,efit_type),
                           signal)

    ######## FETCH AOT SCALARS #############
    for sig_name in cfg['data']['aot_scalar_sig_names'] :
        signal=MdsSignal('{}'.format(sig_name.upper()),
                         'AOT',
                         location='remote://atlas.gat.com')
        pipeline.fetch('{}_full'.format(sig_name),
                       signal)

    ######## FETCH PSIRZ (FIRST EFIT ONLY)  #############
    if cfg['data']['include_psirz'] or psirz_needed:
        psirz_sig = MdsSignal(r'\psirz',
                              cfg['data']['efit_types'][0],
                              location='remote://atlas.gat.com',
                              dims=['r','z','times'])
        pipeline.fetch('psirz_full',psirz_sig)
        ssimag_sig = MdsSignal(r'\ssimag',
                              cfg['data']['efit_types'][0],
                              location='remote://atlas.gat.com')
        pipeline.fetch('ssimag_full',ssimag_sig)
        ssibry_sig = MdsSignal(r'\ssibry',
                              cfg['data']['efit_types'][0],
                              location='remote://atlas.gat.com')
        pipeline.fetch('ssibry_full',ssibry_sig)

    ######## FETCH RHOVN (FIRST EFIT ONLY) ###############
    if cfg['data']['include_rhovn'] or len(cfg['data']['zipfit_sig_names'])>0:
        rhovn_sig = MdsSignal(r'\rhovn',
                              cfg['data']['efit_types'][0],
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
                                        'SPECTROSCOPY',
                                        location='remote://atlas.gat.com')
                pipeline.fetch(f'prad{position}{i}_full',radiation_sig)
        for key in ['KAPPA','PRAD_DIVL','PRAD_DIVU','PRAD_TOT']:
            radiation_sig=MdsSignal(f'\\SPECTROSCOPY::TOP.PRAD.BOLOM.PRAD_01.PRAD.{key}',
                                    'SPECTROSCOPY',
                                    location='remote://atlas.gat.com')
            pipeline.fetch(f'prad{key}_full',radiation_sig)

    ######## ECH DETAILED INFO #########
    # Note, I'd love to include rho as theoretically AOT does automatically
    # (see https://diii-d.gat.com/d3d-wiki/images/1/12/Autoonetwo_pointnames_by_function_20150518.pdf)
    # but it seems for older shots the data isn't available...
    if cfg['data']['include_full_ech_data']:
        num_systems=MdsSignal('ECH.NUM_SYSTEMS','RF',dims=())
        pipeline.fetch('ech_num_systems',num_systems)
        for i in range(1,7):
            signal=MdsSignal(f'ECH.SYSTEM_{i}.GYROTRON.NAME','RF',dims=(),
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_name_{i}',signal)
            signal=MdsSignal(f'ECH.SYSTEM_{i}.GYROTRON.FREQUENCY','RF',dims=(),
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_frequency_{i}',signal)
            signal=MdsSignal(f'ECH.SYSTEM_{i}.ANTENNA.LAUNCH_R','RF',dims=(),
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_R_{i}',signal)
            signal=MdsSignal(f'ECH.SYSTEM_{i}.ANTENNA.LAUNCH_Z','RF',dims=(),
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_Z_{i}',signal)

        #https://diii-d.gat.com/diii-d/ECHStatus
        signal=MdsSignal(r'\echpwrc','RF',
                         location='remote://atlas.gat.com')
        pipeline.fetch(f'ech_pwr_total_full',signal)
        for gyro in ['LEIA', 'LUKE', 'R2D2', #active
                     'YODA', #starting up
                     'SCARECROW', 'TINMAN', 'CHEWBACCA', #retired
                     'TOTO', 'NATASHA', 'KATYA', #not on website but in tree
                     'LION', 'HAN', 'NASA', 'VADER']: #not operational
            signal=MdsSignal(f'ECH.{gyro}.EC{gyro[:3]}AZIANG','RF',
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_aziang_{gyro}',signal)
            signal=MdsSignal(f'ECH.{gyro}.EC{gyro[:3]}POLANG','RF',
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_polang_{gyro}',signal)
            signal=MdsSignal(f'ECH.{gyro}.EC{gyro[:3]}FPWRC','RF',
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_pwr_{gyro}',signal)
            signal=MdsSignal(f'ECH.{gyro}.EC{gyro[:3]}XMFRAC','RF',
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_xmfrac_{gyro}',signal)
            signal=MdsSignal(f'ECH.{gyro}.EC{gyro[:3]}STAT','RF',dims=(),
                             location='remote://atlas.gat.com')
            pipeline.fetch(f'ech_stat_{gyro}',signal)

    ######## NB DETAILED INFO #########
    if cfg['data']['include_full_nb_data']:
        for beam in [30,150,210,330]:
            beam_name=str(beam)[:2]
            for location in ['L','R']:
                # PINJ_ is not there for older shots, which is incredibly annoying
                # DIIID-BEAMS script (see OMFIT-source/modules/DIIID-BEAMS/SCRIPTS/LIB/OMFITlib_utilities)
                # handles this by taking the scalar value and multiplying by BEAMSTAT
                signal=MdsSignal(f'NB{beam_name}{location}.PINJ_{beam_name}{location}','NB',
                                 location='remote://atlas.gat.com')
                pipeline.fetch(f'nb_{beam}{location}_pinj',signal)
                signal=MdsSignal(f'NB{beam_name}{location}.TINJ_{beam_name}{location}','NB',
                                 location='remote://atlas.gat.com')
                pipeline.fetch(f'nb_{beam}{location}_tinj',signal)
                signal=MdsSignal(f'NB{beam_name}{location}.VBEAM','NB',
                                 location='remote://atlas.gat.com')
                pipeline.fetch(f'nb_{beam}{location}_vinj',signal)
                signal=MdsSignal(f'NB{beam_name}{location}.NBVAC_SCALAR','NB',dims=(),
                                 location='remote://atlas.gat.com')
                pipeline.fetch(f'nb_{beam}{location}_vinj_scalar',signal)
        signal=MdsSignal(f'NB15L.OANB.BLPTCH_CAD','NB',dims=(),
                         location='remote://atlas.gat.com')
        pipeline.fetch(f'nb_150_tilt',signal)
        signal=MdsSignal(f'NB21L.CCOANB.BLROT','NB',dims=(),
                         location='remote://atlas.gat.com')
        pipeline.fetch(f'nb_210_rtan',signal)

    @pipeline.map
    def add_timebase(record):
        standard_times=np.arange(cfg['data']['tmin'],cfg['data']['tmax'],cfg['data']['time_step'])
        record['standard_time']=standard_times

    if cfg['data']['include_full_ech_data']:
        @pipeline.map
        def add_ech_info(record):
            record['ech_pwr_total']=standardize_time(record[f'ech_pwr_total_full']['data'],
                                                     record[f'ech_pwr_total_full']['times'],
                                                     record['standard_time'])
            num_systems=record['ech_num_systems']['data']
            record['ech_names']=[]
            sigs_0d=['frequency','R','Z']
            sigs_1d=['pwr','aziang','polang']
            for key in sigs_0d+sigs_1d:
                record[f'ech_{key}']=[]
            for i in range(1,num_systems+1):
                gyro=record[f'ech_name_{i}']['data'].upper()
                record['ech_names'].append(gyro)
                for key in sigs_0d:
                    record[f'ech_{key}'].append(record[f'ech_{key}_{i}']['data'])
                for key in sigs_1d:
                    record[f'ech_{key}'].append(standardize_time(record[f'ech_{key}_{gyro}']['data'],
                                                                record[f'ech_{key}_{gyro}']['times'],
                                                                record['standard_time']))

    if cfg['data']['include_full_nb_data']:
        @pipeline.map
        def add_nb_info(record):
            sigs_1d=['pinj','tinj','vinj'] #make sure vinj is last since it fails on shots without time-dependent v
            for sig in sigs_1d:
                record[f'nb_{sig}']=[]
            record['nb_vinj_scalar']=[]
            for sig in ['nb_210_rtan','nb_150_tilt']:
                try:
                    assert(record[sig]['data'] is not None)
                    record[sig]=record[sig]['data']
                except:
                    record[sig]=np.nan
            for sig in sigs_1d:
                for beam in [30,150,210,330]:
                    for location in ['L','R']:
                        try:
                            record[f'nb_{sig}'].append(standardize_time(record[f'nb_{beam}{location}_{sig}']['data'],
                                                                        record[f'nb_{beam}{location}_{sig}']['times'],
                                                                        record['standard_time']))
                        except:
                            pass
            for beam in [30,150,210,330]:
                for location in ['L','R']:
                    record['nb_vinj_scalar'].append(record[f'nb_{beam}{location}_vinj_scalar']['data'])

    @pipeline.map
    def change_timebase(record):
        all_sig_names=needed_sigs
        for sig_name in all_sig_names:
            try:
                numpy_smoothing_fxn=np.mean
                if sig_name.casefold() in modal_sig_names:
                    def get_mode(arr, axis=0):
                        return np.squeeze(stats.mode(a=arr, axis=axis)[0])
                    numpy_smoothing_fxn=get_mode
                record[sig_name]=standardize_time(record['{}_full'.format(sig_name)]['data'],
                                                  record['{}_full'.format(sig_name)]['times'],
                                                  record['standard_time'],
                                                  numpy_smoothing_fxn=numpy_smoothing_fxn)
            except:
                pass
        for efit_type in cfg['data']['efit_types']:
            for base_sig in cfg['data']['efit_profile_sig_names']:
                sig_name=f'{base_sig}_{efit_type}'
                data=[]
                for time_ind in range(len(record[sig_name])):
                    interpolator=interpolate.interp1d(record[f'{sig_name}_full']['psi'],
                                                      record[sig_name][time_ind,:])
                    data.append(interpolator(standard_x))
                record[sig_name]=np.array(data)

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

    @pipeline.map
    def zipfit_rho(record):
        for sig_name in cfg['data']['zipfit_sig_names']:
            record['zipfit_{}_rhon_basis'.format(sig_name)]=standardize_time(record['zipfit_{}_full'.format(sig_name)]['data'],
                                                                             record['zipfit_{}_full'.format(sig_name)]['times'],
                                                                             record['standard_time'])
            tmp=[]
            rhon=record['zipfit_{}_full'.format(sig_name)]['rhon']
            for time_ind in range(len(record['standard_time'])):
                rho_to_zipfit=my_interp(rhon,
                                        record['zipfit_{}_rhon_basis'.format(sig_name)][time_ind])
                tmp.append(rho_to_zipfit(standard_x))
            record['zipfit_{}_rho'.format(sig_name)]=np.array(tmp)

    if cfg['data']['include_rhovn'] or len(cfg['data']['zipfit_sig_names'])>0:
        @pipeline.map
        def add_rhovn(record):
            record['rhovn']=standardize_time(record['rhovn_full']['data'],
                                             record['rhovn_full']['times'],
                                             record['standard_time'])
        @pipeline.map
        def zipfit_psi(record):
            for sig_name in cfg['data']['zipfit_sig_names']:
                rho_to_psi=[my_interp(record['rhovn'][time_ind],
                                      record['rhovn_full']['psi']) for time_ind in range(len(record['standard_time']))]
                record['zipfit_{}_psi_full'.format(sig_name)]=[]
                for time_ind in range(len(record['standard_time'])):
                    record['zipfit_{}_psi_full'.format(sig_name)].append(rho_to_psi[time_ind](record['zipfit_{}_full'.format(sig_name)]['rhon']))
                record['zipfit_{}_psi_full'.format(sig_name)]=np.array(record['zipfit_{}_psi_full'.format(sig_name)])

                zipfit_interp=fit_function_dict['linear_interp_1d']
                record['zipfit_{}_psi'.format(sig_name)]=zipfit_interp(record['zipfit_{}_psi_full'.format(sig_name)],
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
                        # set to true for rotation if we want to convert km/s to krad/s
                        if (sig_name=='rot' and cfg['data']['cer_rotation_units_of_krad']):
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

    if True: #not cfg['data']['gather_raw']: <-- deprecated (annoying to gather random datatypes into h5)
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
                if sig in final_data[shot]:
                    del final_data[shot][sig]
                final_data[shot][sig]=record[sig]
            # for key in record['errors']:
            #     print(key)
            #     print(record['errors'][key]['traceback'].replace('\\n','\n'))
