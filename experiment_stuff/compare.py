import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data.pkl','rb') as f:
    data=pickle.load(f)

ref_shot=182500
sim_shot=916609

time_ind=45
sigs=['etscr'] #['etste','etsne','etscr','etsct','etsinq']
corresponding_sigs_2={'etste': 'thomson_temp_linear_interp_1d',
                    'etsne': 'thomson_density_linear_interp_1d',
                    'etscr': 'cer_rot_linear_interp_1d',
                    'etsct': 'cer_temp_linear_interp_1d'}

corresponding_sigs_ref={'etste': ['zipfit_etempfit'],
                        'etsne': ['zipfit_edensfit'],
                        'etscr': ['zipfit_trotfit','cer_rot_linear_interp_1d'],
                        'etsct': ['zipfit_itempfit'],
                        'etsinq': []}
corresponding_sigs={'etste': ['etstein','etsneout'],
                    'etsne': ['etsnein', 'etsteout'],
                    'etscr': ['ftscrot','ftssrot'], #['etscrin','ftscrot'],
                    'etsct': ['etsctin'],
                    'etsinq': ['etsinq']}
means={}
stds={}
for sig in sigs:
    means[sig]=0
    stds[sig]=1
    for corresponding_sig in corresponding_sigs_ref[sig]+corresponding_sigs[sig]:
        means[corresponding_sig]=0
        stds[corresponding_sig]=1

if True:
    stds['zipfit_etempfit']= 1000.
    stds['etscr']= 1/1e3 / np.linspace(1.75,2.25,33) # ((2*3.14*1.67)*1e3)
    stds['ftscrot']= 1/1e3 / np.linspace(1.75,2.25,15) #1.67
    stds['ftssrot']= 1/1e3 / np.linspace(1.75,2.25,75) #1.67
    stds['zipfit_itempfit']= 1000.

    means['etstein']=1587
    stds['etstein']=1560
    means['etsneout']=1587
    stds['etsneout']=1560

    means['etsnein']=3.977
    stds['etsnein']=2.764
    means['etsctin']=1235.26
    stds['etsctin']=1385.5859
    means['etscrin']=42.93
    stds['etscrin']=58.47


fig,axes=plt.subplots(len(sigs))
axes=np.atleast_1d(axes)
for i in range(len(sigs)):
    ax=axes[i]
    sig=sigs[i]
    if True:
        for corresponding_sig in corresponding_sigs_ref[sig]:
            ax.plot(np.linspace(0,1,65),
                    data[ref_shot][corresponding_sig][time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                    label=corresponding_sig)
    for corresponding_sig in corresponding_sigs[sig]:
        if 'ftsc' in corresponding_sig:
            valid_inds=np.nonzero(data[sim_shot]['ftsc1vld'][:,time_ind])
            ax.scatter(data[sim_shot]['ftscpsin'][valid_inds,time_ind],
                       data[sim_shot][corresponding_sig][valid_inds,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                       label=corresponding_sig)
        elif 'ftss' in corresponding_sig:
            print(corresponding_sig)
            valid_inds=np.nonzero(data[sim_shot][corresponding_sig][:,time_ind])
            ax.scatter(data[sim_shot]['ftsspsin'][valid_inds,time_ind],
                       data[sim_shot][corresponding_sig][valid_inds,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                       label=corresponding_sig)
        else:
            ax.plot(np.linspace(0,1,33),
                    data[sim_shot][corresponding_sig][:,time_ind]*stds[corresponding_sig]+means[corresponding_sig],
                    label=corresponding_sig)
    if False:
        ax.plot(np.linspace(0,1,33),
                data[sim_shot][sig][:,time_ind]*stds[sig]+means[sig],
                label='pcs')
    ax.set_ylabel('rotation') #sig)
    ax.legend()
    if i==0:
        ax.set_title('{}ms, shot {}/{}'.format(data[ref_shot]['time'][time_ind],ref_shot,sim_shot))

plt.show()
