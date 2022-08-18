# for generating plot used in the experiment paper showing single timeslice:
# the actuator proposal timetrace, and the predicted profiles

# for offline data:
# just module load toksearch, it self-caches
# for pcs data:
# run "python dump_pcs_data.py" first to create "pcs_data.pkl"

# in both cases:
# note you have to use separate terminal, with "module load toksearch"

# for the offline model:
# need to make keras2c_tests library (just type make in that repo)
# that relies on k2c repo, which you also have to make - being sure
# you have -fPIC as a compiler flag

# for the online model:
# requires source activate-ing the env specified by etemp_requirements.txt
# in the astra-control repository

import numpy as np
import copy
import scipy.interpolate
import pickle
import os
import sys
import toksearch

import matplotlib_parameters

sys.path.append(os.path.expanduser('~/keras2c_tests/'))
from realtime_predictor_tests import test_predictor

from pcs_normalizations import pcs_normalizations

shot=187076
# make the below single-valued to plot profiles at that time
time=1450 #np.arange(1000,3000,50) #[1650.]
# make the below None to plot all
profiles_to_plot=['press_EFIT01','temp']
actuators_to_plot=['pinj']
scalars_to_plot=[] #None
profiles=['temp','dens','rotation','press_EFIT01','q_EFIT01']
scalars=['density_estimate']
actuators=['pinj', 'tinj', 'curr_target', 'target_density']

ylimits={key: (None, None) for key in actuators+profiles+scalars}
ylimits['temp']= (0,2.5) #3400),
ylimits['dens']= (0,7)
ylimits['rotation']= (20,300)
ylimits['q_EFIT01']= (0,4)
ylimits['press_EFIT01']= (0,70)
ylimits['pinj']= (0, 6)
ylimits['tinj']= (-2,7)
ylimits['curr_target']= (0,1.5e6)
ylimits['target_density']= (0,5)
ylimits['density_estimate']= (0,5)
ylimits['triangularity_top_EFIT01']=(0.2,0.8)
ylimits['triangularity_bot_EFIT01']=(0.2,0.8)
ylimits['volume_EFIT01']=(15,25)
ylimits['li_EFIT01']=(0,1.5)
ylimits['aminor_EFIT01']=(0,1)
ylimits['kappa_EFIT01']=(1,2)

ylabels={key: key for key in actuators+profiles+scalars}
ylabels['temp']= 'Temperature (keV)'
ylabels['dens']= r'Density ($10^{19}/m^3$)'
ylabels['rotation']= 'Rotation'
ylabels['q_EFIT01']= 'Safety factor (q)'
ylabels['press_EFIT01']= 'Pressure (kPa)'
ylabels['pinj']= r'$pinj$ (MW)'
ylabels['tinj']= r'$tinj$ (N m)'
ylabels['curr_target']= 'current setpoint (A)'
ylabels['target_density']= 'density setpoint ($10^{19}/m^3$)'

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

normalizations=pcs_normalizations
with open('pcs_data.pkl','rb') as f:
    data=pickle.load(f)

data=data[shot]

standard_x=np.linspace(0,1,33)

final_info={}

def normalize(arr,elem):
    arr=arr.copy()
    if 'q_EFIT01' in elem:
        arr=1./arr
    arr=(arr-normalizations[elem]['median'])/normalizations[elem]['iqr']
    return arr
def denormalize(arr,elem):
    arr=arr.copy()
    arr=arr*normalizations[elem]['iqr']+normalizations[elem]['median']
    if 'q_EFIT01' in elem:
        arr=1./arr
    if 'press_EFIT01' in elem:
        arr=arr*1e-3
    if 'pinj' in elem:
        arr=arr*1e-3
    return arr

profiles=['temp','dens','rotation','itemp','q_EFIT01','press_EFIT01',
          'temp_out','dens_out', 'q_EFIT01_out', 'rotation_out', 'press_EFIT01_out']
scalars_and_actuators=['density_estimate',
                       'target_density',
                       'pinj',
                       'tinj',
                       'curr_target']

cache_file='cache_paper_plot_final_data.pkl'
if os.path.exists(cache_file):
    with open(cache_file,'rb') as f:
        full_data=pickle.load(f)
else:
    full_data={}
    for elem in profiles:
        print(f'gathering {elem}')
        data=[]
        for i in range(33):
            toksig=toksearch.PtDataSignal(f'{pcs_name_map[elem]}{i}').fetch(shot)
            data.append(toksig['data'])
        data=np.array(data).T
        full_data[elem]={}
        full_data[elem]['times']=toksig['times']
        full_data[elem]['data']=data
    for elem in scalars_and_actuators:
        print(f'gathering {elem}')
        data=[]
        # 6th entry is the "present" value, 0th is oldest
        for i in range(7):
            toksig=toksearch.PtDataSignal(f'{pcs_name_map[elem]}{i}').fetch(shot)
            data.append(toksig['data'])
        data=np.array(data).T
        full_data[elem]={}
        full_data[elem]['times']=toksig['times']
        full_data[elem]['data']=data
    with open(cache_file,'wb') as f:
        pickle.dump(full_data,f)
input_data={}
input_data['input_profiles']={}
input_data['input_scalars']={}
input_data['input_actuators']={}
for elem in profiles:
    time_ind=np.searchsorted(full_data[elem]['times'],time)
    input_data['input_profiles'][elem]=full_data[elem]['data'][time_ind]
for elem in scalars_and_actuators:
    time_ind=np.searchsorted(full_data[elem]['times'],time)
    input_data['input_scalars'][f'past_{elem}']=full_data[elem]['data'][time_ind]
for elem in actuators:
    current_value=input_data['input_scalars'][f'past_{elem}'][-1]
    input_data['input_actuators'][f'future_{elem}']=np.ones(4)*current_value

targets={key: [None for i in range(len(standard_x))] for key in profiles}
if time<2000:
    targets['temp'][:3]=[2.00000, 1.91419, 1.82874]
    targets['press_EFIT01'][:3]=np.array([47493,45691,43908.6])*1e-3
elif time<2500:
    targets['temp'][:3]=[3.00000, 2.87128, 2.74311]
else:
    targets['temp'][:3]=[6.00000, 5.74257, 5.48622]

proposals=[{'target_density': [0,0,0,0],
            'curr_target': [0,0,0,0],
            'pinj': [0,0,0,0],
            'tinj': [0,0,0,0]
            } for i in range(5)]

ramp=np.array([0.1,0.2,0.3,0.4])

proposals[0]['pinj']=-ramp
proposals[1]['pinj']=-ramp
proposals[3]['pinj']=ramp
proposals[4]['pinj']=ramp

if time<2000:
    proposals[0]['tinj']=-ramp
    proposals[1]['tinj']=ramp
    proposals[3]['tinj']=-ramp
    proposals[4]['tinj']=ramp
else:
    proposals[0]['target_density']=-ramp
    proposals[1]['target_density']=ramp
    proposals[3]['target_density']=-ramp
    proposals[4]['target_density']=ramp

import matplotlib.pyplot as plt
fig,axes=plt.subplots(len(profiles_to_plot)+len(actuators_to_plot),figsize=(7,10))
axes=np.atleast_1d(axes)
proposal_colors=[matplotlib_parameters.colorblind_colors[i] for i in [1,2,0]]
proposal_labels=[r'$\uparrow$ $pinj$','constant',r'$\downarrow$ $pinj$']

for j,profile in enumerate(profiles_to_plot):
    axes[j].plot(standard_x,
                 denormalize(input_data['input_profiles'][profile],profile),
                 c='k',linewidth=8,alpha=0.4,
                 label='input')

# proposal is always a delta away from the present value
input_data_copy=copy.deepcopy(input_data)
for i,proposal in enumerate([proposals[i] for i in [4,2,0]]):
    for j,elem in enumerate(actuators):
        input_data_copy['input_actuators'][f'future_{elem}']=input_data['input_actuators'][f'future_{elem}']+proposal[elem]
        if elem in actuators_to_plot:
            actuator_times=np.arange(time-50,time+50*4,50)
            actuator_values=np.append([input_data_copy['input_scalars'][f'past_{elem}'][-1]],
                                      input_data_copy['input_actuators'][f'future_{elem}'])
            axes[len(profiles_to_plot)+actuators_to_plot.index(elem)].plot(actuator_times*1e-3,
                                                                           denormalize(actuator_values,elem),
                                                                           c=proposal_colors[i])
    output_dic=test_predictor(**input_data_copy,
                              prediction_type='pcs')
    for j,profile in enumerate(profiles):
        if profile in profiles_to_plot:
            axes[profiles_to_plot.index(profile)].plot(standard_x,
                                                       denormalize(output_dic[f'{profile}_out'],profile),
                                                       c=proposal_colors[i],
                                                       label=proposal_labels[i])
for j,profile in enumerate(profiles_to_plot):
    axes[j].plot(standard_x[:3],
                 targets[profile][:3],
                 marker='x', markersize=13,
                 label='target',linestyle='--',c='k')
for j,actuator in enumerate(actuators_to_plot):
    axes[len(profiles_to_plot)+j].plot(np.arange(time-50*7,time,50)*1e-3,
                                       denormalize(input_data['input_scalars'][f'past_{actuator}'],actuator),
                                       c='k',
                                       linewidth=5,
                                       alpha=0.4)

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
for i in range(len(profiles_to_plot)):
    ax=axes[i]
    ax.set_ylabel(ylabels[profiles_to_plot[i]])
    ax.set_xlim(0,1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.set_ylim(ylimits[profiles_to_plot[i]])
for i in range(len(actuators_to_plot)):
    ax=axes[len(profiles_to_plot)+i]
    ax.set_ylabel(ylabels[actuators_to_plot[i]])
    ax.set_xlim(1.050,1.650)
    ax.xaxis.set_major_locator(MultipleLocator(.100))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(.010))
    ax.set_ylim(ylimits[actuators_to_plot[i]])

axes[0].yaxis.set_major_locator(MultipleLocator(20))
axes[0].yaxis.set_minor_locator(MultipleLocator(4))
axes[1].yaxis.set_major_locator(MultipleLocator(1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.2))
axes[2].yaxis.set_major_locator(MultipleLocator(2))
axes[2].yaxis.set_minor_locator(MultipleLocator(0.5))

axes[0].xaxis.set_ticklabels([])

axes[len(profiles_to_plot)-1].set_xlabel('$\psi_N$')
axes[-1].set_xlabel('Time (s)')
#fig.suptitle(f'Shot {shot}, 1450ms')
plt.subplots_adjust(hspace=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.7,0.485))

for ax in axes:
    ax.tick_params('both',direction="in",which='both')
    ax.tick_params('both',length=3,which='minor')
    ax.tick_params('both',length=8,which='major')
plt.savefig('paper_plot_final.png')
plt.savefig('paper_plot_final.pdf')
#plt.show()


#make sure the online matches the offline implementation (just to test)
if False:
    import matplotlib.pyplot as plt
    profile='press_EFIT01'
    time_ind=np.searchsorted(full_data[f'{profile}_out']['times'],time)
    plt.plot(denormalize(full_data[f'{profile}_out']['data'][time_ind],profile),linewidth=10,alpha=0.2,c='g')
    for i,proposal in enumerate(proposals):
        input_data_copy=copy.deepcopy(input_data)
        for elem in actuators:
            input_data_copy['input_actuators'][f'future_{elem}']=input_data['input_actuators'][f'future_{elem}']+proposal[elem]
        output_dic=test_predictor(**input_data_copy,
                                  prediction_type='pcs')
        plt.plot(denormalize(output_dic[f'{profile}_out'],profile),label=i)
    for i in range(5):
        toksig=toksearch.PtDataSignal(f'etspprop{i}').fetch(shot)
        time_ind=np.searchsorted(toksig['times'],time)
        plt.scatter([0],[denormalize(toksig['data'][time_ind],profile)],label=i)
    plt.legend()
    plt.show()

