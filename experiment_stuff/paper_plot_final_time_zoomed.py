# for generating plot used in the experiment paper showing full timetrace
# of experiment, include the targets, the proposal chosen, etc.

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
import toksearch

import matplotlib_parameters
import matplotlib

sys.path.append(os.path.expanduser('~/keras2c_tests/'))
from realtime_predictor_tests import test_predictor

from pcs_normalizations import pcs_normalizations

shot=187076
times=np.arange(1000,3000,10)
# make the below single-valued to plot profiles at that time
# make the below None to plot all
profiles_to_plot=['temp']
actuators_to_plot=['pinj']
scalars_to_plot=[] #['density_estimate'] #None
profiles=['temp','dens','rotation','press_EFIT01','q_EFIT01']
scalars=['density_estimate']
actuators=['pinj', 'tinj', 'curr_target', 'target_density']

ylimits={key: (None, None) for key in actuators+profiles+scalars}
ylimits['temp']= (2.5,3.2) #3400),
ylimits['dens']= (0,7)
ylimits['rotation']= (20,300)
ylimits['q_EFIT01']= (0,4)
ylimits['press_EFIT01']= (0,110)
ylimits['pinj']= (2, 9)
ylimits['tinj']= (-2,7)
ylimits['curr_target']= (0.7e6,1.1e6)
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
ylabels['press_EFIT01']='Pressure (kPa)'
ylabels['q_EFIT01']= 'Safety factor (q)'
ylabels['pinj']= r'$pinj$ (MW)'
ylabels['tinj']= r'$tinj$ (N m)'
ylabels['curr_target']= 'Plasma Current (A)'
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

standard_x=np.linspace(0,1,33)

final_info={}

def normalize(arr,elem):
    arr=arr.copy()
    if 'q_EFIT01' in elem:
        arr=1./arr
    if 'press_EFIT01' in elem:
        arr=arr/1e-3
    if 'pinj' in elem:
        arr=arr/1e-3
    arr=(arr-normalizations[elem]['median'])/normalizations[elem]['iqr']
    return arr
def denormalize(arr,elem):
    try:
        arr=arr.copy()
    except:
        pass
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

cache_file=f'cache_paper_plot_final_data_{shot}.pkl'
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

cache_file=f'cache_paper_plot_final_predictions_{shot}.pkl'
if os.path.exists(cache_file):
    with open(cache_file,'rb') as f:
        final_info=pickle.load(f)
else:
    final_info={}
    for profile in profiles:
        final_info[f'{profile}_prediction']=[]
        final_info[f'{profile}_target']=[]
        final_info[f'{profile}_input']=[]
    for actuator in actuators:
        final_info[f'{actuator}_proposals']=[]
    for elem in scalars+actuators:
        final_info[f'{elem}_input']=[]
    for time in times:
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

        if True: #time<2000:
            proposals[0]['tinj']=-ramp
            proposals[1]['tinj']=ramp
            proposals[3]['tinj']=-ramp
            proposals[4]['tinj']=ramp
        else:
            proposals[0]['target_density']=-ramp
            proposals[1]['target_density']=ramp
            proposals[3]['target_density']=-ramp
            proposals[4]['target_density']=ramp

        # add a new timestep to final_info for proposal stuff
        for actuator in actuators:
            final_info[f'{actuator}_proposals'].append([])
        for profile in profiles:
            final_info[f'{profile}_prediction'].append([])
        for i,proposal in enumerate([proposals[i] for i in [4,2,0]]):
            input_data_copy=copy.deepcopy(input_data)
            for j,elem in enumerate(actuators):
                input_data_copy['input_actuators'][f'future_{elem}']=input_data['input_actuators'][f'future_{elem}']+proposal[elem]
                final_info[f'{elem}_proposals'][-1].append(denormalize(input_data_copy['input_actuators'][f'future_{elem}'],elem))
            output_dic=test_predictor(**input_data_copy,
                                      prediction_type='pcs')
            for profile in profiles:
                final_info[f'{profile}_prediction'][-1].append(denormalize(output_dic[f'{profile}_out'],profile))
        for profile in profiles:
            final_info[f'{profile}_input'].append(denormalize(input_data['input_profiles'][profile],profile))
            final_info[f'{profile}_target'].append(targets[profile])
        for elem in scalars+actuators:
            final_info[f'{elem}_input'].append(denormalize(input_data['input_scalars'][f'past_{elem}'],elem))

    final_info['times']=times
    with open(cache_file,'wb') as f:
        pickle.dump(final_info,f)

import matplotlib.pyplot as plt

time_inds=np.searchsorted(times,final_info['times'])
final_times=final_info['times'][time_inds]*1e-3
rho_ind=1

nplots=len(profiles_to_plot)+1
fig,axes=plt.subplots(nplots, sharex=True, figsize=(7,10))
axes=np.atleast_1d(axes)
for ax in axes:
    ax.axvline(2.42,linestyle='--',c='k',zorder=-1)

num_proposals=len(final_info[f'{actuators[0]}_proposals'][0])
cmap=[matplotlib_parameters.colorblind_colors[i] for i in [1,2,0]]
proposal_labels=[r'$\uparrow$ $pinj$','constant',r'$\downarrow$ $pinj$']
for i,profile in enumerate(profiles_to_plot):
    axes[i].plot(final_times,
                   np.array(final_info[f'{profile}_input'])[time_inds,rho_ind],
                   label='input',c='0.45',linewidth=8)
for i,elem in enumerate(actuators_to_plot+scalars_to_plot):
    #### to fix the values of past actuators ####
    #current_value=input_data['input_scalars'][f'past_{elem}'][-1]
    #input_data['input_scalars'][f'past_{elem}']=current_value*np.ones(7)
    #############################################
    axes[len(profiles_to_plot)+i].plot(final_times,
                                       np.array(final_info[f'{elem}_input'])[time_inds,-1],
                                       c='0.45',linewidth=8)
# simshots don't have processed pinj
# if shot<900000:
#     toksig=toksearch.MdsSignal(r'\pinj',
#                      'NB',
#                      location='remote://atlas.gat.com').fetch(shot)
#     j=actuators_to_plot.index('pinj')
#     axes[len(profiles_to_plot)+j].plot(toksig['times']*1e-3,
#                                        toksig['data']*1e-3,
#                                        c='0.45',
#                                        linewidth=0.8)
for i in range(num_proposals):
    # plot future actuators for each proposal
    # for j,elem in enumerate(actuators_to_plot):
    #     #plot proposal corresponding to 200ms into future
    #     axes[len(profiles_to_plot)+j].plot(final_times,
    #                                        np.array(final_info[f'{elem}_proposals'])[time_inds,i,-1],
    #                                        c=cmap[i])
    # plot profile predictions for each proposal
    for j,profile in enumerate(profiles_to_plot):
        axes[j].plot(final_times,
                     np.array(final_info[f'{profile}_prediction'])[time_inds,i,rho_ind],
                     label=proposal_labels[i],
                     c=cmap[i])
target_time_inds_each_phase=[
    np.where(final_info['times']<2000)[0],
    np.where(np.logical_and(final_info['times']>2000,
                            final_info['times']<2500))[0],
    np.where(np.logical_and(final_info['times']>2500,
                            final_info['times']<3000))[0]
]
for i,profile in enumerate(profiles_to_plot):
    # for each phase
    for j in range(3):
        if j==0:
            label='target'
        else:
            label=None
        target_times=final_info['times'][target_time_inds_each_phase[j]]*1e-3
        axes[i].plot(target_times,
                     np.array(final_info[f'{profile}_target'])[target_time_inds_each_phase[j],rho_ind],
                     label=label,linestyle='--',c='k')

which_proposal=toksearch.PtDataSignal('etswchprop').fetch(shot)
proposals={}
proposals['pinj']=[{'times': None, 'ycoords': None} for i in range(3)]
decreasing_mask=np.where(which_proposal['data']<2)
const_mask=np.where(which_proposal['data']==2)
increasing_mask=np.where(which_proposal['data']>2)
#rate=(denormalize(0.1,'pinj')-denormalize(0,'pinj'))/50 #kW/ms == MW/s
rate=0.31
proposals['pinj'][0]['times']=which_proposal['times'][decreasing_mask]
proposals['pinj'][0]['ycoords']=[-rate]*len(proposals['pinj'][0]['times'])
proposals['pinj'][1]['times']=which_proposal['times'][const_mask]
proposals['pinj'][1]['ycoords']=[0.]*len(proposals['pinj'][1]['times'])
proposals['pinj'][2]['times']=which_proposal['times'][increasing_mask]
proposals['pinj'][2]['ycoords']=[rate]*len(proposals['pinj'][2]['times'])

for i in range(3):
    axes[-1].scatter(proposals['pinj'][i]['times']*1e-3,
                     np.array(proposals['pinj'][i]['ycoords']),
                     c=cmap[(len(cmap)-1)-i])
axes[-1].set_ylabel(r'$\Delta$ target $pinj$ (MW)')
axes[-1].set_ylim(-0.4,0.4)

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
for i in range(len(profiles_to_plot)):
    ax=axes[i]
    ax.set_ylabel(ylabels[profiles_to_plot[i]])
    ax.set_ylim(ylimits[profiles_to_plot[i]])
# for i in range(len(actuators_to_plot)):
#     ax=axes[len(profiles_to_plot)+i]
#     ax.set_ylabel(ylabels[actuators_to_plot[i]])
#     ax.set_ylim(ylimits[actuators_to_plot[i]])

axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))
#axes[1].yaxis.set_major_locator(MultipleLocator(20))
#axes[1].yaxis.set_minor_locator(MultipleLocator(5))
#axes[1].yaxis.set_major_locator(MultipleLocator(2))
#axes[1].yaxis.set_minor_locator(MultipleLocator(0.5))
axes[-1].yaxis.set_major_locator(MultipleLocator(0.3))
axes[-1].yaxis.set_minor_locator(MultipleLocator(0.1))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right',bbox_to_anchor=(1,0.475))

# for i in range(len(axes)):
#     axes[i].axvspan(1.350,1.730,color=matplotlib_parameters.matlab_colors[-1],alpha=0.3)
# axes[1].text((1.350+1.730)/2,85,'rtEFIT\nError',ha='center',va='center')
# axes[1].annotate('Estimate\nrecovers',xy=(1.73,60), xycoords='data', xytext=(1.87,100),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right',verticalalignment='top',ha='center')
# axes[1].axvspan(2.,3.,color='k',alpha=0.2)
# t=axes[1].text(2.5,35,'Pressure\nControl Off',ha='center',va='center')

axes[-1].set_xlim(2.325,2.495)
axes[-1].xaxis.set_major_locator(MultipleLocator(0.05))
axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[-1].xaxis.set_minor_locator(MultipleLocator(0.01))
axes[-1].set_xlabel('Time (s)')

# show that the target temp goes up
# to 6keV (beyond ylimits of plot) at 2.49s
'''
time_ind=np.searchsorted(final_info['times'],2490)
target=final_info['temp_target'][time_ind][rho_ind]
axes[0].arrow(2.49,
              target,
              (0.010/(6.-target))*(ylimits['temp'][-1]-target),
              ylimits['temp'][-1]-target,
              head_width=.06,
              head_length=.28,
              length_includes_head=True,
              clip_on=False,
              color='k',
              zorder=10)
'''
#axes[0].text(2.53,ylimits['temp'][-1],'6keV target',va='top')

# add Phase labels
# for i in range(len(axes)):
#     for phase_location in [2,2.5]:
#         # for all except pressure, which is not being controlled at 2.5
#         # and has text we don't want to cover...
#         if not (i==1 and phase_location==2.5):
#             axes[i].axvline(phase_location,c='k',zorder=-1,
#                             linewidth=2*matplotlib.rcParams['lines.linewidth'])
# phase_text_locs=[(2-1.35)/2+1.35,2.25,2.75]
# phase_labels=['Phase I','Phase II','Phase III']
# for i in range(3):
#     axes[0].text(phase_text_locs[i],1.1,phase_labels[i],
#                  va='top',ha='center',
#                  fontsize=1.6*matplotlib.rcParams['font.size'],bbox=dict(facecolor='white'))

for ax in axes:
    ax.tick_params('both',direction="in",which='both')
    ax.tick_params('both',length=3,which='minor')
    ax.tick_params('both',length=8,which='major')
plt.savefig('paper_plot_final_time_zoomed.png')
plt.savefig('paper_plot_final_time_zoomed.pdf')
#plt.show()
