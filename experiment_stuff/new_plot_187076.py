# plot showing what actually happened during shot 187076 (the one
# used for our experiment paper), using only signals saved in
# realtime. A prettier / 187076-specific version of
# plot_any_shot_fully.py

# requires module load toksearch

from toksearch import PtDataSignal, MdsSignal, Pipeline
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
#plt.rcParams.update({'text.usetex': True})
import numpy as np
from pcs_normalizations import pcs_normalizations

shot=187076

profiles=['temp','press_EFIT01']
actuators=['pinj']

# targets
targets={key: {'times': [None], 'ycoords': [None]} for key in profiles}
targets['temp']['times']=[1000,2000,2500,3000]
targets['temp']['ycoords']=[2.0,3.0,6.0]
targets['press_EFIT01']['times']=[1000,2000,2500,3000]
targets['press_EFIT01']['ycoords']=[47493,None,None]

req_scaling={key: 1 for key in actuators}
req_scaling['pinj']=1

ylimits={key: (None, None) for key in actuators+profiles}
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

ylabels={key: key for key in actuators+profiles}
ylabels['temp']= 'Temperature (eV)'
ylabels['dens']= r'Density ($10^{19}/m^3$)'
ylabels['rotation']= 'Rotation'
ylabels['q_EFIT01']= 'Safety factor (q)'
ylabels['pinj']= 'pinj (kW)'
ylabels['tinj']= 'tinj (N m)'
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
proposal_name_map={'temp': 'etsteprop',
                   'press_EFIT01': 'etspprop'}

true_sig={}
algo_sig={}
for actuator in actuators:
    true_sig[actuator]=MdsSignal(f'\{actuator}',
                                 'NB',
                                 location='remote://atlas.gat.com').fetch(shot)
for elem in actuators+profiles:
    if elem in actuators:
        algo_sig[elem]=PtDataSignal(f'{pcs_name_map[elem]}6').fetch(shot)
    else:
        algo_sig[elem]=PtDataSignal(f'{pcs_name_map[elem]}1').fetch(shot)
    algo_sig[elem]['data']=algo_sig[elem]['data']*pcs_normalizations[elem]['iqr']+pcs_normalizations[elem]['median']
predictions={}
for elem in profiles:
    predictions[elem]=[PtDataSignal(f'{proposal_name_map[elem]}{i}').fetch(shot) for i in [0,2,4]]
    for i in range(3):
        predictions[elem][i]['data']=predictions[elem][i]['data']*pcs_normalizations[elem]['iqr']+pcs_normalizations[elem]['median']

which_proposal=PtDataSignal('etswchprop').fetch(shot)
proposals={}
proposals['pinj']=[{'times': None, 'ycoords': None} for i in range(3)]
decreasing_mask=np.where(which_proposal['data']<2)
const_mask=np.where(which_proposal['data']==2)
increasing_mask=np.where(which_proposal['data']>2)
proposals['pinj'][0]['times']=which_proposal['times'][decreasing_mask]
proposals['pinj'][0]['ycoords']=[-.009]*len(proposals['pinj'][0]['times'])
proposals['pinj'][1]['times']=which_proposal['times'][const_mask]
proposals['pinj'][1]['ycoords']=[0.]*len(proposals['pinj'][1]['times'])
proposals['pinj'][2]['times']=which_proposal['times'][increasing_mask]
proposals['pinj'][2]['ycoords']=[.009]*len(proposals['pinj'][2]['times'])
# proposals['tinj']=np.zeros(np.shape(which_proposal['data']))
# proposals['tinj'][np.logical_and(which_proposal['data']==2,which_proposal['times']<2000)]=0
# proposals['tinj'][np.logical_and(which_proposal['data']==0,which_proposal['times']<2000)]=-0.27
# proposals['tinj'][np.logical_and(which_proposal['data']==3,which_proposal['times']<2000)]=-0.27
# proposals['tinj'][np.logical_and(which_proposal['data']==1,which_proposal['times']<2000)]=0.27
# proposals['tinj'][np.logical_and(which_proposal['data']==4,which_proposal['times']<2000)]=0.27

fig,axes=plt.subplots(len(profiles)+2*len(actuators),sharex=True,figsize=(8,16))
axes=np.atleast_1d(axes)

colors=['b','orange','r']
for j,profile in enumerate(profiles):
    axes[j].plot(algo_sig[profile]['times'],
                 algo_sig[profile]['data'],
                 c='k')
    for i in range(3):
        axes[j].plot(predictions[profile][i]['times'],
                     predictions[profile][i]['data'],
                     c=colors[i], alpha=0.6)
    for i in range(len(targets[profile]['times'])-1):
        axes[j].plot([targets[profile]['times'][i],targets[profile]['times'][i+1]],
                     [targets[profile]['ycoords'][i],targets[profile]['ycoords'][i]],
                     c='k',linestyle='--')
    axes[j].set_ylabel(ylabels[profile])

for j,actuator in enumerate(actuators):
    for i in range(3):
        axes[len(profiles)+j].scatter(proposals[actuator][i]['times'],proposals[actuator][i]['ycoords'],
                                                     label=r'$\Delta$pinj chosen (MW/s)',c=colors[i])
    axes[len(profiles)+len(actuators)+j].plot(algo_sig[actuator]['times'],algo_sig[actuator]['data']*req_scaling[actuator],
                                              c='k')
    axes[len(profiles)+len(actuators)+j].plot(true_sig[actuator]['times'],true_sig[actuator]['data'], 
                                              alpha=.4,linewidth=.8,c='b')

for ax in axes:
    ax.set_xlim(900,3000)
    for time in [1000,2000,2500]:
        ax.axvline(time,c='r')

axes[-1].set_xlabel('Time (ms)')
axes[1].legend(loc=(.05,.6))
#plt.savefig('187076_figure.png')
plt.show()

