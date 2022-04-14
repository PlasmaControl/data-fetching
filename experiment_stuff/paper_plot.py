# for generic plotting of the timetrace with the actuators that truly
# happened during the experiment

# for offline data:
# run "python new_database_maker.py configs/etemp.yaml" to create "data.pkl"
# (in the headnode above this directory, in the tokamak-transport repo)
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

sys.path.append(os.path.expanduser('~/keras2c_tests/'))
from realtime_predictor_tests import test_predictor

from pcs_normalizations import pcs_normalizations

model_type='offline' # "offline" for keras, "pcs" for pcs
data_type='offline'  # "offline" for offline, anything else for pcs
shot=142110
# make the below single-valued to plot profiles at that time
times=np.arange(500,4000,50)
# make the below None to plot all
profiles_to_plot=['temp', 'dens'] #None #['press_EFIT01','temp']
actuators_to_plot=['pinj', 'tinj'] #None #['tinj','pinj']
scalars_to_plot=[] #None
cache_file=f'cache_for_paper_plot_{shot}_{model_type}_{data_type}.pkl'

if model_type=='offline':
    profiles=['temp','dens','rotation','press_EFIT01','q_EFIT01']
    scalars=['density_estimate',
    'triangularity_bot_EFIT01','triangularity_top_EFIT01','kappa_EFIT01', 'li_EFIT01', 'volume_EFIT01']
    actuators=['pinj', 'tinj', 'curr_target', 'target_density']
else:
    profiles=['temp','dens','rotation','press_EFIT01','q_EFIT01']
    scalars=['density_estimate']
    actuators=['pinj', 'tinj', 'curr_target', 'target_density']

ylimits={key: (None, None) for key in actuators+profiles+scalars}
ylimits['temp']= (0,4.5) #3400),
ylimits['dens']= (0,7)
ylimits['rotation']= (20,300)
ylimits['q_EFIT01']= (0,4)
ylimits['press_EFIT01']= (0,1.3e5)
ylimits['pinj']= (0, 11.e3)
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
ylabels['temp']= 'Temperature (eV)'
ylabels['dens']= r'Density ($10^{19}/m^3$)'
ylabels['rotation']= 'Rotation'
ylabels['q_EFIT01']= 'Safety factor (q)'
ylabels['pinj']= 'pinj (kW)'
ylabels['tinj']= 'tinj (N m)'
ylabels['curr_target']= 'current setpoint (A)'
ylabels['target_density']= 'density setpoint ($10^{19}/m^3$)'

if model_type=='offline':
    with open('/cscratch/abbatej/astra-control-files/normalizations.pkl','rb') as f:
        normalizations=pickle.load(f)
else:
    normalizations=pcs_normalizations

if data_type=='offline':
    with open('../data.pkl','rb') as f:
        data=pickle.load(f)
    for profile in ['temp','dens','q_EFIT01','press_EFIT01','rotation']:
        #downsample from 65 to 33 spatial points
        data[shot][profile]=data[shot][profile][:,::2]
else:
    with open('pcs_data.pkl','rb') as f:
        data=pickle.load(f)
data=data[shot]

standard_x=np.linspace(0,1,33)

cache_exists=os.path.exists(cache_file)
if cache_exists:
    with open(cache_file,'rb') as f:
        final_info=pickle.load(f)
    if (min(final_info['times'])>min(times)) or (max(final_info['times'])<max(times)):
        cache_exists=False
if cache_exists:
    pass #final_info is already set
else:
    final_info={}
    for profile in profiles:
        final_info[f'{profile}_prediction']=[]
        final_info[f'{profile}_target']=[]
        final_info[f'{profile}_input']=[]
        final_info[f'{profile}_expt_prediction']=[]
    for actuator in actuators:
        final_info[f'{actuator}_proposals']=[]
    for elem in scalars+actuators:
        final_info[f'{elem}_input']=[]

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
        return arr

    for time in times:
        input_data={}
        input_data['input_profiles']={}
        input_data['input_scalars']={}
        input_data['input_actuators']={}

        print(f'Starting {time}ms')
        time_ind=np.searchsorted(data['time'],time)
        for elem in profiles:
            input_data['input_profiles'][elem]=normalize(data[elem][time_ind],elem)
        for elem in scalars+actuators:
            input_data['input_scalars'][f'past_{elem}']=normalize(data[elem][time_ind-7:time_ind],elem)
        for elem in actuators:
            current_value=input_data['input_scalars'][f'past_{elem}'][-1]
            input_data['input_actuators'][f'future_{elem}']=np.ones(4)*current_value

        targets={key: [None for i in range(len(standard_x))] for key in profiles}
        if time<2000:
            targets['temp'][:3]=[2.00000, 1.91419, 1.82874]
            targets['press_EFIT01'][:3]=[47493,45691,43908.6]
        elif time<2500:
            targets['temp'][:3]=[3.00000, 2.87128, 2.74311]
        else:
            targets['temp'][:3]=[6.00000, 5.74257, 5.48622]

        '''
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
        '''
        proposals=[{'target_density': [0,0,0,0],
                    'curr_target': [0,0,0,0],
                    'pinj': [0,0,0,0],
                    'tinj': [0,0,0,0]
                    }, #float the previous values
                   {elem: normalize(data[elem][time_ind:time_ind+4],elem)-normalize(data[elem][time_ind-1],elem) for elem in actuators}
                   # ^ propose deltas to match the actual future values in the experiment to see true predictions
                   ]

        # add a new timestep to final_info for proposal stuff
        for actuator in actuators:
            final_info[f'{actuator}_proposals'].append([])
        for profile in profiles:
            final_info[f'{profile}_prediction'].append([])

        # proposal is always a delta away from the present value
        actuator_base=copy.deepcopy(input_data['input_actuators'])
        for i,proposal in enumerate(proposals):
            for j,elem in enumerate(actuators):
                input_data['input_actuators'][f'future_{elem}']=actuator_base[f'future_{elem}']+proposal[elem]
                final_info[f'{elem}_proposals'][-1].append(denormalize(input_data['input_actuators'][f'future_{elem}'],elem))
            output_dic=test_predictor(**input_data,
                                      prediction_type=model_type)
            for profile in profiles:
                final_info[f'{profile}_prediction'][-1].append(denormalize(output_dic[f'{profile}_out'],profile))

        for profile in profiles:
            final_info[f'{profile}_input'].append(denormalize(input_data['input_profiles'][profile],profile))
            final_info[f'{profile}_target'].append(targets[profile])
            if data_type!='offline':
                expt_data=data[f'{profile}_out'][time_ind]
                if 'q_EFIT01' in profile:
                    expt_data=1./expt_data

                final_info[f'{profile}_expt_prediction'].append(expt_data)
        for elem in scalars+actuators:
            final_info[f'{elem}_input'].append(denormalize(input_data['input_scalars'][f'past_{elem}'],elem))
    final_info['times']=times
    with open(cache_file,'wb') as f:
        pickle.dump(final_info,f)


import matplotlib.pyplot as plt

#redefined here for sake of plotting, kind of kludgy
if profiles_to_plot is None:
    profiles_to_plot=profiles
if actuators_to_plot is None:
    actuators_to_plot=actuators
if scalars_to_plot is None:
    scalars_to_plot=scalars

#### FOR PROFILE PLOTTING
if len(times)==1:
    time_ind=np.searchsorted(final_info['times'],times[0])
    time=final_info['times'][time_ind]
    num_rows=max(len(profiles_to_plot),len(actuators_to_plot)+len(scalars_to_plot))
    fig,axes=plt.subplots(nrows=num_rows,ncols=2,sharex='col')
    axes=np.atleast_2d(axes)

    num_proposals=len(final_info[f'{actuators[0]}_proposals'][time_ind])
    cmap=list(reversed(plt.cm.viridis(np.linspace(0,1,num_proposals))))
    linestyles=['dotted']*num_proposals
    proposal_labels=[f'prop {i}' for i in range(num_proposals)] #[r'$\downarrow$ pinj','constant',r'$\uparrow$ pinj']

    for i,profile in enumerate(profiles_to_plot):
        axes[i,0].plot(standard_x,
                       final_info[f'{profile}_input'][time_ind],
                       label='input',c='k')
        # axes[i,0].plot(standard_x,
        #                final_info[f'{profile}_target'][time_ind],
        #                label='targets',marker='x',linestyle='--',c='k')
        axes[i,0].set_ylim(ylimits[profile])
    #    axes[i,0].plot(auxiliary_data[profile],label='0th entry PCS output')
        axes[i,0].set_ylabel(ylabels[profile])
    for i,elem in enumerate(actuators_to_plot+scalars_to_plot):
        #### to fix the values of past actuators ####
        #current_value=input_data['input_scalars'][f'past_{elem}'][-1]
        #input_data['input_scalars'][f'past_{elem}']=current_value*np.ones(7)
        #############################################
        axes[i,1].plot(np.arange(time-50*7,time,50),
                       final_info[f'{elem}_input'][time_ind],
                       c='k')
        axes[i,1].set_ylim(ylimits[elem])
    #    axes[i,0].plot(auxiliary_data[profile],label='0th entry PCS output')
        axes[i,1].set_ylabel(ylabels[elem])

    for i in range(num_proposals):
        # plot future actuators for each proposal
        for j,elem in enumerate(actuators_to_plot):
            axes[j,1].plot(np.arange(time,time+50*4,50),
                           final_info[f'{elem}_proposals'][time_ind][i],
                           linestyle=linestyles[i],
                           c=cmap[i],
                           alpha=1)

        # plot profile predictions for each proposal
        for j,profile in enumerate(profiles_to_plot):
            axes[j,0].plot(standard_x,
                           final_info[f'{profile}_prediction'][time_ind][i],
                           label=proposal_labels[i],c=cmap[i],alpha=0.7,
                           linestyle=linestyles[i])
            print(f'{profile}:')
            print(final_info[f'{profile}_prediction'][time_ind][i])

    if False:
        for j,profile in enumerate(profiles_to_plot):
            axes[j,0].plot(standard_x,
                           final_info[f'{profile}_expt_prediction'][time_ind],
                           label='expt output',
                           c='k',
                           alpha=0.5)
    fig.suptitle(f'{shot}, {time}-ish ms\n{model_type} model, {data_type} data')
    axes[0,0].legend()
    plt.show()
### FOR TIME PLOTTING
else:
    time_inds=np.searchsorted(times,final_info['times'])
    final_times=final_info['times'][time_inds]
    rho_ind=0

    num_rows=max(len(profiles_to_plot),len(actuators_to_plot)+len(scalars_to_plot))
    fig,axes=plt.subplots(nrows=num_rows,ncols=2,sharex=True)
    axes=np.atleast_2d(axes)

    num_proposals=len(final_info[f'{actuators[0]}_proposals'][0])
    cmap=list(reversed(plt.cm.viridis(np.linspace(0,1,num_proposals))))
    linestyles=['dotted']*num_proposals
    proposal_labels=[f'prop {i}' for i in range(num_proposals)] #[r'$\downarrow$ pinj','constant',r'$\uparrow$ pinj']
    for i,profile in enumerate(profiles_to_plot):
        axes[i,0].plot(final_times,
                       np.array(final_info[f'{profile}_input'])[time_inds,rho_ind],
                       label='input',c='k')
        # ytargets=np.array(final_info[f'{profile}_target'])[time_inds,rho_ind]
        # axes[i,0].plot(final_times,
        #                ytargets,
        #                label='targets',marker='x',linestyle='--',c='k')
        axes[i,0].set_ylim(ylimits[profile])
        axes[i,0].set_ylabel(ylabels[profile])
    for i,elem in enumerate(actuators_to_plot+scalars_to_plot):
        #### to fix the values of past actuators ####
        #current_value=input_data['input_scalars'][f'past_{elem}'][-1]
        #input_data['input_scalars'][f'past_{elem}']=current_value*np.ones(7)
        #############################################
        axes[i,1].plot(final_times,
                       np.array(final_info[f'{elem}_input'])[time_inds,-1],
                       c='k')
        axes[i,1].set_ylim(ylimits[elem])
    #    axes[i,0].plot(auxiliary_data[profile],label='0th entry PCS output')
        axes[i,1].set_ylabel(ylabels[elem])
    if True:
        i=1 # choose the proposal corresponding to the real actuators
        # plot profile predictions for each proposal
        for j,profile in enumerate(profiles_to_plot):
            axes[j,0].plot(final_times+200, # offset by 200
                           np.array(final_info[f'{profile}_prediction'])[time_inds,i,rho_ind],
                           label=proposal_labels[i],c=cmap[i],alpha=0.7,
                           linestyle=linestyles[i])
    if False:
        for i in range(num_proposals):
            # plot future actuators for each proposal
            for j,elem in enumerate(actuators_to_plot):
                axes[j,1].plot(final_times,
                               np.array(final_info[f'{elem}_proposals'])[time_inds,i,rho_ind],
                               linestyle=linestyles[i],
                               c=cmap[i],
                               alpha=1)
            # plot profile predictions for each proposal
            for j,profile in enumerate(profiles_to_plot):
                axes[j,0].plot(final_times,
                               np.array(final_info[f'{profile}_prediction'])[time_inds,i,rho_ind],
                               label=proposal_labels[i],c=cmap[i],alpha=0.7,
                               linestyle=linestyles[i])
    # plot the keras2c output from the experiment
    if False:
        for j,profile in enumerate(profiles_to_plot):
            axes[j,0].plot(final_times,
                           np.array(final_info[f'{profile}_expt_prediction'])[time_inds,rho_ind],
                           label='expt output',
                           c='k',
                           alpha=0.5)

    fig.suptitle(f'{shot}, rho={standard_x[rho_ind]}\n{model_type} model, {data_type} data')
    axes[0,0].legend()
    plt.show()
