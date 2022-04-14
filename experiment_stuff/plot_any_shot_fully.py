# run as python plot_any_shot_fully.py 187076 (or any of 187065-187080)
# gives high-level plots for the shot from our experiments in June 2021
# showing the targets, the finite set of actuator proposals and which
# choice was made, etc.

# requires running pcs_dump.py with the shot you care about
# to generate data_experiment.pkl

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=6)
#matplotlib.rcParams['text.usetex'] = True

sig_names={'etste': 'Te',
           'etstein': 'Te',
           'etsne': 'ne',
           'etsnein': 'ne',
           'etsinprs': 'P',
           'etsintor6': 'tinj',
           'etsinden6': 'dens',
           'etsincur6': 'ip',
           'etsinpwr6': 'pinj'}

with open('data_experiment.pkl','rb') as f:
    data=pickle.load(f)

shot=sys.argv[1] #187078
shot=list(data.keys())[0]
print(shot)
sigs=['etste','etsinprs','etsne'] #['etste','etsne','etscr','etsct','etsinq']

targets={'etste': np.ones((len(data[shot]['time']),33)),
         'etsne': np.ones((len(data[shot]['time']),33)),
         'etsinprs': np.ones((len(data[shot]['time']),33))}
start_time_ind=np.searchsorted(data[shot]['time'],1000)
switch_time_ind=np.searchsorted(data[shot]['time'],2000)
switch_time_ind_2=np.searchsorted(data[shot]['time'],2500)

# get these values with e.g. this script
# shot=187074
# OMFIT['ipsip']=OMFITmdsValue(server='DIII-D',shot=shot,treename=None,TDI='ipsip')
# OMFIT['etsincur5']=OMFITmdsValue(server='DIII-D',shot=shot,treename=None,TDI='etsincur5')

# plt.plot(OMFIT['ipsip'].dim_of(0),OMFIT['ipsip'].data()*1e6,label='ipsip')
# plt.plot((OMFIT['etsincur5'].dim_of(0)+1966)*1.3,OMFIT['etsincur5'].data()*389672+989468,label='etsincur5')
# plt.legend()
# plt.title(shot)

time_offset=0 #2200
time_crunch=1 #1.2

if shot in [187070,187071,187072,187073,187074]:
#ipsip vs etsincur5 shows 800 to 5000 vs -1300 to 1900
    time_offset=1966
    time_crunch=1.3
    data[shot]['time']=(data[shot]['time']+time_offset)*time_crunch
if shot in [187075]:
#800 to 5000 vs -4650 to -2300
    time_offset=4650
    time_crunch=2.1
    data[shot]['time']=(data[shot]['time']+time_offset)*time_crunch

if shot in [187070,187071,187072,187073,187074]:
    times=[1950,2450,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=np.searchsorted(data[shot]['time'],2500)
    actuator_choices={1950: [r'$\downarrow$ pinj',
                             'const pinj',
                             r'$\uparrow$ pinj'],
                      2450: [r'$\downarrow$ pinj',
                             'const pinj',
                             r'$\uparrow$ pinj'],
                      2950: [r'$\downarrow$ pinj',
                             'const pinj',
                             r'$\uparrow$ pinj']}

if shot in [187075]:
    times=[1950,2450,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=np.searchsorted(data[shot]['time'],2500)
    actuator_choices={1950: [r'$\downarrow$ pinj, $\downarrow$ dens',
                              r'$\downarrow$ pinj, $\uparrow$ dens',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ dens',
                              r'$\uparrow$ pinj, $\uparrow$ dens'],
                      2450: [r'$\downarrow$ pinj, $\downarrow$ dens',
                              r'$\downarrow$ pinj, $\uparrow$ dens',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ dens',
                              r'$\uparrow$ pinj, $\uparrow$ dens'],
                      2950: [r'$\downarrow$ pinj, $\downarrow$ dens',
                             r'$\downarrow$ pinj, $\uparrow$ dens',
                             'const pinj',
                             r'$\uparrow$ pinj, $\downarrow$ dens',
                             r'$\uparrow$ pinj, $\uparrow$ dens']}
if shot in [187076]:
    times=[1950,2450,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=np.searchsorted(data[shot]['time'],2500)
    actuator_choices={1950: [r'$\downarrow$ pinj, $\downarrow$ tinj',
                              r'$\downarrow$ pinj, $\uparrow$ tinj',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ tinj',
                              r'$\uparrow$ pinj, $\uparrow$ tinj'],
                      2450: [r'$\downarrow$ pinj, $\downarrow$ dens',
                              r'$\downarrow$ pinj, $\uparrow$ dens',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ dens',
                              r'$\uparrow$ pinj, $\uparrow$ dens'],
                      2950: [r'$\downarrow$ pinj, $\downarrow$ dens',
                             r'$\downarrow$ pinj, $\uparrow$ dens',
                             'const pinj',
                             r'$\uparrow$ pinj, $\downarrow$ dens',
                             r'$\uparrow$ pinj, $\uparrow$ dens']}
if shot in [187077]:
    times=[1950,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=len(data[shot]['time'])-1
    actuator_choices={1950: [r'$\downarrow$ pinj, $\downarrow$ dens',
                              r'$\downarrow$ pinj, $\uparrow$ dens',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ dens',
                              r'$\uparrow$ pinj, $\uparrow$ dens'],
                       2950: [r'$\downarrow$ pinj, $\downarrow$ dens',
                              r'$\downarrow$ pinj, $\uparrow$ dens',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ dens',
                              r'$\uparrow$ pinj, $\uparrow$ dens']}
if shot in [187078]:
    times=[1950,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=len(data[shot]['time'])-1
    actuator_choices={1950: [r'$\downarrow$ pinj, $\downarrow$ tinj',
                              r'$\downarrow$ pinj, $\uparrow$ tinj',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ tinj',
                              r'$\uparrow$ pinj, $\uparrow$ tinj'],
                       2950: [r'$\downarrow$ pinj, $\downarrow$ tinj',
                              r'$\downarrow$ pinj, $\uparrow$ tinj',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ tinj',
                              r'$\uparrow$ pinj, $\uparrow$ tinj']}
if shot in [187079]:
    times=[1950,2950]
    start_time_ind=np.searchsorted(data[shot]['time'],1000)
    switch_time_ind=np.searchsorted(data[shot]['time'],2000)
    switch_time_ind_2=len(data[shot]['time'])-1
    actuator_choices={1950: [r'$\downarrow$ pinj, $\downarrow$ ip',
                              r'$\downarrow$ pinj, THIS WAS AN INPUT ERROR: $\downarrow$ ip',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ ip',
                              r'$\uparrow$ pinj, $\uparrow$ ip'],
                       2950: [r'$\downarrow$ pinj, $\downarrow$ ip',
                              r'$\downarrow$ pinj, $\uparrow$ ip',
                              'const pinj',
                              r'$\uparrow$ pinj, $\downarrow$ ip',
                              r'$\uparrow$ pinj, $\uparrow$ ip']}

targets['etste'][:]=None
targets['etsinprs'][:]=None
targets['etsne'][:]=None

# dumped from GUI via "write data to file"
temp_2000=[2000.00, 1914.19, 1828.74, 1744.61, 1663.96, 1581.68, 1501.05, 1422.22, 1345.14, 
           1270.08, 1197.31, 1126.94, 1059.18,  994.49,  932.91,  874.66,  820.27,  769.87,  
           723.72,  682.04,  645.09,  612.63, 584.59, 560.00, 537.44, 514.51, 487.39, 451.87, 
           404.74, 349.17, 295.92, 153.85, 0.60]
temp_3000=[3000.00, 2871.28, 2743.11, 2616.92, 2495.94, 2372.52, 2251.58, 2133.33, 
           2017.71, 1905.11, 1795.97, 1690.41, 1588.76, 1491.74, 1399.36, 1312.00, 
           1230.40, 1154.81, 1085.57, 1023.05,  967.63,  918.94,  876.89,  840.00, 
           806.16,  771.77,  731.08,  677.80,  607.11,  523.76,  443.88,  230.78,    
           0.90]
temp_6000=[6000.00, 5742.57, 5486.22, 5233.84, 4991.88, 4745.03, 4503.15, 4266.66, 
           4035.43, 3810.23, 3591.93, 3380.81, 3177.53, 2983.48, 2798.72, 2623.99, 
           2460.80, 2309.61, 2171.15, 2046.11, 1935.26, 1837.88, 1753.77, 1680.00, 
           1612.32, 1543.54, 1462.16, 1355.61, 1214.23, 1047.52,  887.75,  461.55,    
           1.81]

temp_3000_experimental=[3338.94, 3206.46, 3073.97, 2942.79, 2817.49, 2689.43, 2562.45, 
                        2437.37, 2314.85, 2194.61, 2077.49, 1963.23, 1852.82, 1746.10, 
                        1643.61, 1546.06, 1453.69, 1366.88, 1286.31, 1212.42, 1145.36, 
                        1085.59, 1032.59,  985.72,  943.03,  901.24,  855.20,  798.03,  
                        723.08,  631.47,  539.04,  359.91,    1.00]
dens_10=[10.06, 8.85, 8.16, 7.67, 7.29, 6.97, 6.69, 6.44, 6.22, 6.02, 5.84, 5.68, 
         5.53, 5.39, 5.26, 5.14, 5.03, 4.92, 4.82, 4.73, 4.65, 4.57, 4.49, 4.42, 
         4.35, 4.28, 4.22, 4.16, 4.11, 4.05, 3.93, 2.91, 0.00]
dens_4=[4.18, 4.23, 4.28, 4.32, 4.36, 4.39, 4.43, 4.46, 4.48, 4.50, 4.52, 4.53, 4.53, 
        4.53, 4.52, 4.51, 4.48, 4.45, 4.41, 4.36, 4.29, 4.22, 4.14, 4.05, 3.96, 3.87, 
        3.80, 3.77, 3.77, 3.82, 3.78, 2.88, 0.00]

if shot in [187070,187071,187072]:
    targets['etste'][start_time_ind:switch_time_ind]=temp_2000
    targets['etste'][switch_time_ind:switch_time_ind_2]=temp_3000
    targets['etste'][switch_time_ind_2:]=temp_2000
if shot in [187073]:
    targets['etste'][start_time_ind:switch_time_ind]=temp_2000
    targets['etste'][switch_time_ind:switch_time_ind_2]=temp_3000
    targets['etste'][switch_time_ind_2:]=temp_6000    
if shot in [187074,187075,187076]:
    targets['etste'][start_time_ind:switch_time_ind,:3]=temp_2000[:3]
    targets['etsinprs'][start_time_ind:switch_time_ind,:3]=[47493,45691,43908.6]
    targets['etste'][switch_time_ind:switch_time_ind_2,:3]=temp_3000[:3]
    targets['etste'][switch_time_ind_2:,:3]=temp_6000[:3]
if shot in [187077,187078]:
    targets['etste'][start_time_ind:switch_time_ind]=temp_2000
    targets['etsne'][start_time_ind:switch_time_ind]=dens_10
    targets['etste'][switch_time_ind:switch_time_ind_2]=temp_3000_experimental
    targets['etsne'][switch_time_ind:switch_time_ind_2]=dens_4
if shot in [187079]:
    targets['etste'][start_time_ind:switch_time_ind]=temp_2000
    targets['etsne'][start_time_ind:switch_time_ind]=dens_10
    targets['etste'][switch_time_ind:switch_time_ind_2]=temp_3000_experimental

corresponding_sigs_ref={'etste': [], #['zipfit_etempfit','thomson_temp_linear_interp_1d','thomson_temp_mtanh_1d'],
                        'etsne': [], #['zipfit_edensfit','thomson_density_linear_interp_1d','thomson_density_mtanh_1d'],
                        'etscr': ['zipfit_trotfit','cer_rot_linear_interp_1d'],
                        'etsct': ['zipfit_itempfit'],
                        'etsinq': [],
                        'etsinprs': []} #['press_EFITRT2']}
corresponding_sigs={'etste': ['etstein'], #['etstein','etsteout'],
                    'etsne': ['etsnein'], #, 'etsneout'],
                    'etscr': ['ftscrot','ftssrot'], #['etscrin','ftscrot'],
                    'etsct': ['etsctin'],
                    'etsinq': ['etsinq'],
                    'etsinprs': ['etsinprs']}
means={}
stds={}
for sig in sigs:
    means[sig]=0
    stds[sig]=1
    for corresponding_sig in corresponding_sigs_ref[sig]+corresponding_sigs[sig]:
        means[corresponding_sig]=0
        stds[corresponding_sig]=1

stds['zipfit_etempfit']= 1000.
stds['thomson_temp_linear_interp_1d']= 1000.
stds['thomson_temp_mtanh_1d']= 1000.
stds['zipfit_edensfit']= 1.
stds['thomson_density_linear_interp_1d']= 1
stds['thomson_density_mtanh_1d']= 1
stds['etscr']= 1/1e3 / np.linspace(1.75,2.25,33) # ((2*3.14*1.67)*1e3)
stds['ftscrot']= 1/1e3 / np.linspace(1.75,2.25,15) #1.67
stds['ftssrot']= 1/1e3 / np.linspace(1.75,2.25,75) #1.67
stds['zipfit_itempfit']= 1000.

means['etstein']=1587
stds['etstein']=1560
means['etsteout']=1587
stds['etsteout']=1560
means['etsneout']=3.977
stds['etsneout']=2.764

means['etsnein']=3.977
stds['etsnein']=2.764

means['etsctin']=1235.26
stds['etsctin']=1385.5859

means['etscrin']=42.93
stds['etscrin']=58.47

means['etsinprs']=23303
stds['etsinprs']=35931
means['etsprsout']=23303
stds['etsprsout']=35931

means['etsinpwr6']=4072876.
stds['etsinpwr6']=3145593.
means['etsintor6']=3.38
stds['etsintor6']=2.7
means['etsincur6']=989467
stds['etsincur6']=389572.
means['etsinden6']=3.93
stds['etsinden6']=2.48

color_wheel=['r','g','b']
scalar_quantities=['etsintor6','etsinpwr6','etsincur6','etsinden6']

# for plot over time
if True:
    fig,axes=plt.subplots(len(sigs)+1+len(scalar_quantities),sharex=True)
    fig1,axes1=plt.subplots(len(sigs),sharex=True)
    axes=np.atleast_1d(axes)
    rho_ind=0
    time_ind=100
    for i in range(len(sigs)):
        sig=sigs[i]
        ax=axes[i]
        for corresponding_sig in corresponding_sigs_ref[sig]:
            ax.plot(data[shot]['time'],
                    data[shot][corresponding_sig][:,rho_ind]*stds[corresponding_sig]+means[corresponding_sig],
                    label=corresponding_sig)
        for corresponding_sig in corresponding_sigs[sig]:
            # make artificial shift for our shots that were messed up bc our cycle time was set too low
            fucked_time=data[shot]['time'] #(data[shot]['time']+time_offset)*time_crunch
            ax.plot(fucked_time,data[shot][corresponding_sig][rho_ind,:]*stds[corresponding_sig]+means[corresponding_sig],
                    label=corresponding_sig)
        # plot the targets (hand crafted above) 
        ax.plot(data[shot]['time'],targets[sig][:,rho_ind],label='target',c='k')
        ax.set_xlim((800,3000))
        ax.axvline(1000,c='k',linestyle='--')
        for j,time in enumerate(times):
            ax.axvline(time,c=color_wheel[j])
        if i==0:
            ax.set_title('shot {}, rho={}'.format(shot,np.linspace(0,1,33)[rho_ind]))
            #ax.legend(loc='upper right',bbox_to_anchor=(1,2))
        if sig in sig_names:
            ax.set_ylabel('{}'.format(sig_names[sig]))
        else:
            ax.set_ylabel('{}'.format(sig))

        ax=axes1[i]
        for corresponding_sig in corresponding_sigs[sig]:
            time_ind=np.searchsorted(data[shot]['time'],1000)
            ax.plot(np.linspace(0,1,33),
                    data[shot][corresponding_sig][:,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                    alpha=0.5,
                    c='k',
                    linestyle='--',
                    label='{}ms'.format(1000))            
            for j,time in enumerate(times): 
                time_ind=np.searchsorted(data[shot]['time'],time)
                ax.plot(np.linspace(0,1,33),
                        data[shot][corresponding_sig][:,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                        alpha=0.5,
                        label='{}ms'.format(time),
                        c=color_wheel[j])
            for j,time in enumerate(times): 
                time_ind=np.searchsorted(data[shot]['time'],time)
                ax.scatter(np.linspace(0,1,33)+1./200 *j,targets[sig][time_ind,:],
                           marker='x',s=70,linewidths=1.5,
                           c=color_wheel[j])
            ax.set_ylabel(sig_names[corresponding_sig])
            if i==0:
                ax.legend()
axes[len(sigs)].plot(data[shot]['time'],data[shot]['etswchprop'])
axes[len(sigs)].set_yticks([])
axes[len(sigs)].axvline(1000,c='k',linestyle='--')
for k in range(len(actuator_choices[times[0]])):
    axes[len(sigs)].text(1000,k-.3,actuator_choices[times[0]][k],fontsize=10)
for j in range(len(times)):
    axes[len(sigs)].axvline(times[j],c=color_wheel[j])
    if j<len(times)-1:
        for k in range(len(actuator_choices[time])):
            axes[len(sigs)].text(times[j],k-.3,actuator_choices[times[j+1]][k],fontsize=10)
axes[len(sigs)].set_ylabel('choice')
axes[len(sigs)].set_xlim((800,3000))

for i,corresponding_sig in enumerate(scalar_quantities):
    axes[len(sigs)+1+i].plot(data[shot]['time'],
                               data[shot][corresponding_sig]*stds[corresponding_sig]+means[corresponding_sig])
    axes[len(sigs)+1+i].set_xlim((800,3000))
    if corresponding_sig in sig_names:
        axes[len(sigs)+1+i].set_ylabel('{}'.format(sig_names[corresponding_sig]))
    else:
        axes[len(sigs)+1+i].set_ylabel('{}'.format(corresponding_sig))

for i in range(len(axes)):
    if i<len(axes)-1:
        plt.setp(axes[i].get_xticklabels(), visible=False)
axes[-1].set_xlabel('Time (ms)')

for i in range(len(axes1)):
    if i<len(axes1)-1:
        plt.setp(axes1[i].get_xticklabels(), visible=False) #axes1[i].set_xticks([])
axes1[-1].set_xlabel('rho')


plt.show()

if False:
    fig,axes=plt.subplots(len(sigs))
    axes=np.atleast_1d(axes)
    for i in range(len(sigs)):
        ax=axes[i]
        sig=sigs[i]
        if True:
            for corresponding_sig in corresponding_sigs_ref[sig]:
                ax.plot(np.linspace(0,1,65),
                        data[shot][corresponding_sig][time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                        label=corresponding_sig)
        for corresponding_sig in corresponding_sigs[sig]:
            if 'ftsc' in corresponding_sig:
                valid_inds=np.nonzero(data[shot]['ftsc1vld'][:,time_ind])
                ax.scatter(data[shot]['ftscpsin'][valid_inds,time_ind],
                           data[shot][corresponding_sig][valid_inds,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                           label=corresponding_sig)
            elif 'ftss' in corresponding_sig:
                print(corresponding_sig)
                valid_inds=np.nonzero(data[shot][corresponding_sig][:,time_ind])
                ax.scatter(data[shot]['ftsspsin'][valid_inds,time_ind],
                           data[shot][corresponding_sig][valid_inds,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                           label=corresponding_sig)
            else:
                fucked_time_ind=np.searchsorted(data[shot]['time'],data[shot]['time'][time_ind]) #/time_crunch-time_offset)
                ax.plot(np.linspace(0,1,33),
                        data[shot][corresponding_sig][:,fucked_time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                        label=corresponding_sig)
        if False:
            ax.plot(np.linspace(0,1,33),
                    data[shot][sig][:,time_ind]*stds[sig]+means[sig],
                    label='pcs')
        if sig in sig_names:
            ax.set_ylabel('{}'.format(sig_names[sig]))
        else:
            ax.set_ylabel('{}'.format(sig))
        ax.legend()
        if i==0:
            ax.set_title('{}ms, shot {}/{}'.format(data[shot]['time'][time_ind],shot,shot))

    plt.show()

