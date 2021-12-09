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
controlled_sig_name='etste1'
offline_controlled_sig_name='etempfit'

req_sig=PtDataSignal(req_sig_name).fetch(shot)
true_sig=MdsSignal(true_sig_name,
                   'NB',
                   location='remote://atlas.gat.com').fetch(shot)
controlled_sig=PtDataSignal(controlled_sig_name).fetch(shot)
#offline_controlled_sig = MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(offline_controlled_sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times']).fetch(shot)
which_proposal=PtDataSignal('etswchprop').fetch(shot)

req_scaling=1e-6
true_scaling=1e-3
controlled_scaling=1e-3
offline_controlled_scaling=1

fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True)

axes[0,0].plot(controlled_sig['times'],controlled_sig['data']*controlled_scaling,label='core Te (keV)')
axes[0,0].set_ylim((0,6.5))
#axes[0,0].plot(offline_controlled_sig['times'],offline_controlled_sig['data'][:,0]*offline_controlled_scaling,label='Te zipfit (keV)', alpha=.4,linewidth=.8)

# targets
xcoords=[1000,2000,2500,3000]
ycoords=[2,3,6]
for i in range(len(xcoords)-1):
    axes[0,0].plot([xcoords[i],xcoords[i+1]],[ycoords[i],ycoords[i]],c='k')

axes[1,0].plot(req_sig['times'],req_sig['data']*req_scaling,label='pinj request (MW)')
axes[1,0].plot(true_sig['times'],true_sig['data']*true_scaling,label='pinj (MW)', alpha=.4,linewidth=.8)
axes[1,0].set_ylim((0,12))

errors=np.zeros(np.shape(controlled_sig['data']))
mask=controlled_sig['times']<1000
errors[mask]=None
mask=controlled_sig['times']>1000
errors[mask]=controlled_sig['data'][mask]-2000
mask=controlled_sig['times']>2000
errors[mask]=controlled_sig['data'][mask]-3000
mask=controlled_sig['times']>2500
errors[mask]=controlled_sig['data'][mask]-6000
axes[0,1].plot(controlled_sig['times'],errors*controlled_scaling,label='Te error (keV)')
axes[0,1].set_ylim((-4,2))
axes[0,1].axhline(0,c='k')

proposals=np.zeros(np.shape(which_proposal['data']))
proposals[which_proposal['data']==2]=0
proposals[which_proposal['data']<2]=-.009
proposals[which_proposal['data']>2]=.009
axes[1,1].plot(which_proposal['times'],proposals*1000,label=r'desired delta pinj (MW/s)')
axes[1,1].set_ylim((-10,10))
#dreq_dt=np.divide(np.diff(req_sig['data']),np.diff(req_sig['times']))
#axes[1,1].plot(req_sig['times'][:-1],dreq_dt*req_scaling*1000,label='requested delta pinj (MW/s)',alpha=0.5)
#axes[1,1].axhline(31.4,c='k',linestyle='--')
#axes[1,1].axhline(-31.4,c='k',linestyle='--')

for arr in axes:
    for ax in arr:
        ax.legend()
        ax.set_xlim(900,3000)
        for time in [1000,2000,2500]:
            ax.axvline(time,c='r')

axes[1,1].legend(loc=(0.05,0.6))
plt.show()

