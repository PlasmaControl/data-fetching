# used to create data for paper_plot*.py
# specify the single shot number

# requires module load toksearch

from pcs_normalizations import pcs_normalizations

import toksearch
import numpy as np
import pickle

shot=142107

cache_file='pcs_data.pkl'

def standardize_time(old_signal,old_timebase,standard_times,offline=False):
    if offline:
        window_size=50
        new_signal=[]
        for i in range(len(standard_times)):
            inds_in_range=np.where(np.logical_and(old_timebase>=standard_times[i]-window_size,old_timebase<standard_times[i]))[0]
            if len(inds_in_range)==0:
                if len(old_signal.shape)==1:
                    new_signal.append(np.nan)
                else:
                    new_signal.append(np.full(old_signal.shape[1:],np.nan))
            else:
                new_signal.append(np.mean(old_signal[inds_in_range],axis=0))
        return np.array(new_signal)
    else:
        inds=np.searchsorted(old_timebase,standard_times)
        return old_signal[inds]

pcs_name_map={'temp': 'etstein',
              'dens': 'etsnein',
              'itemp': 'etsctin',
              'rotation': 'etscrin',
              'q_EFIT01': 'etsinq',
              'press_EFIT01': 'etsinprs',
              'temp_out': 'etsteout',
              'dens_out': 'etsneout',
              'q_EFIT01_out': 'etsqout',
              'rotation_out': 'etsrout',
              'press_EFIT01_out': 'etspout',
              'density_estimate': 'etsinneb',
              'triangularity_bot_EFIT01': 'etsintrb',
              'triangularity_top_EFIT01': 'etsintrt',
              'kappa_EFIT01': 'etsinkap',
              'density_estimate': 'etsinneb',
              'target_density': 'etsinden',
              'pinj': 'etsinpwr',
              'tinj': 'etsintor',
              'curr_target': 'etsincur'
              }
non_pcs_name_map={'volume_EFIT01': r'\volume',
                  'li_EFIT01': r'\li'}

standard_times=np.arange(0,6000,50)

final_data={}
final_data[shot]={}

profiles=['temp','dens','rotation','itemp','q_EFIT01','press_EFIT01',
          'temp_out','dens_out', 'q_EFIT01_out', 'rotation_out', 'press_EFIT01_out']
scalars_and_actuators=['density_estimate',
                       'triangularity_bot_EFIT01',
                       'triangularity_top_EFIT01',
                       'kappa_EFIT01',
                       'density_estimate',
                       'target_density',
                       'pinj',
                       'tinj',
                       'curr_target',
                       'volume_EFIT01',
                       'li_EFIT01']

for elem in profiles:
    print(f'gathering {elem}')
    data=[]
    for i in range(33):
        toksig=toksearch.PtDataSignal(f'{pcs_name_map[elem]}{i}').fetch(shot)
        data.append(toksig['data'])
    data=np.array(data).T
    final_data[shot][elem]=standardize_time(data,
                                      toksig['times'],
                                      standard_times)

for elem in scalars_and_actuators:
    print(f'gathering {elem}')
    if elem in pcs_name_map:
        # 6th entry is the "present" value
        toksig=toksearch.PtDataSignal(f'{pcs_name_map[elem]}6').fetch(shot)
        final_data[shot][elem]=standardize_time(toksig['data'],
                                                toksig['times'],
                                                standard_times)
    else:
        toksig=toksearch.MdsSignal(f'{non_pcs_name_map[elem]}',
                                   'EFIT01',
                                   location='remote://atlas.gat.com').fetch(shot)
        final_data[shot][elem]=standardize_time(toksig['data'],
                                                toksig['times'],
                                                standard_times,
                                                offline=True)

for key in final_data[shot]:
    if key not in non_pcs_name_map:
        final_data[shot][key]=(final_data[shot][key]*pcs_normalizations[key]['iqr'])+pcs_normalizations[key]['median']

final_data[shot]['q_EFIT01']=1./final_data[shot]['q_EFIT01']

final_data[shot]['time']=standard_times

with open(cache_file,'wb') as f:
    pickle.dump(final_data,f)
