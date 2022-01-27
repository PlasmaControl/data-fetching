from toksearch import PtDataSignal, MdsSignal, Pipeline
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
#plt.rcParams.update({'text.usetex': True})
import numpy as np

shot=187076
req_sig_name='etsrpinj'
true_sig_name='pinj'
controlled_sig_name='etstein1'
offline_controlled_sig_name='etempfit'
te_std_dev=1.560
te_mean=1.587

req_sig=PtDataSignal(req_sig_name).fetch(shot)
true_sig=MdsSignal(true_sig_name,
                   'NB',
                   location='remote://atlas.gat.com').fetch(shot)
controlled_sig=PtDataSignal(controlled_sig_name).fetch(shot)
#offline_controlled_sig = MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(offline_controlled_sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times']).fetch(shot)
which_proposal=PtDataSignal('etswchprop').fetch(shot)
prop_colors=['r','r','y','g','g']
props=[]
for i in range(5):
    props.append(PtDataSignal(f'etsteprop{i}').fetch(shot))

req_scaling=1e-6
true_scaling=1e-3
controlled_scaling=1e-3
offline_controlled_scaling=1

fig,axes=plt.subplots(3,sharex=True,figsize=(8,16))
axes=np.atleast_1d(axes)
axes[0].plot(controlled_sig['times'],controlled_sig['data']*te_std_dev+te_mean,label='core Te (keV)',c='k')
if False: #for i,prop in enumerate(props):
    axes[0].plot(prop['times'],
                 prop['data']*te_std_dev+te_mean,
                 label=f'prediction, prop {i}',
                 c=prop_colors[i],
                 alpha=0.5)
axes[0].set_ylim((0,6.5))

# targets
xcoords=[1000,2000,2500,3000]
ycoords=[2,3,6]
for i in range(len(xcoords)-1):
    axes[0].plot([xcoords[i],xcoords[i+1]],[ycoords[i],ycoords[i]],c='k',linestyle='--')

proposals=np.zeros(np.shape(which_proposal['data']))
proposals[which_proposal['data']==2]=0
proposals[which_proposal['data']<2]=-.009
proposals[which_proposal['data']>2]=.009
axes[1].scatter(which_proposal['times'],proposals*1000,label=r'$\Delta$pinj chosen (MW/s)',c='k')
axes[1].set_ylim((-10,10))

axes[2].plot(req_sig['times'],req_sig['data']*req_scaling,label='pinj requested (MW)',c='k')
axes[2].plot(true_sig['times'],true_sig['data']*true_scaling,label='pinj actuated (MW)', alpha=.4,linewidth=.8,c='b')
axes[2].set_ylim((0,18))

for ax in axes:
    ax.legend()
    ax.set_xlim(900,3000)
    for time in [1000,2000,2500]:
        ax.axvline(time,c='r')

axes[-1].set_xlabel('Time (ms)')
axes[1].legend(loc=(.05,.6))
plt.savefig('187076_figure.png')
plt.show()

