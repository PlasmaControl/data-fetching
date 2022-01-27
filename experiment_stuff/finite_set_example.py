import matplotlib.pyplot as plt
import scipy.interpolate
from toksearch import MdsSignal
import numpy as np

shot=163303

fig,axes=plt.subplots(2,figsize=(10,18))
axes=np.atleast_1d(axes)
ind=160
axis_font_size=60
linewidth=2
big_linewidth=10
tick_font_size=40
label_font_size=30
x_axis=np.linspace(0,1,33)

sig_name='etempfit'
sig_info=MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times']).fetch(shot)
zipfit_axis=np.linspace(0,1,len(sig_info['data'][ind][:-20]))
true_temp=scipy.interpolate.interp1d(zipfit_axis,sig_info['data'][ind][:-20])(x_axis)

sig_name='edensfit'
sig_info=MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times']).fetch(shot)
zipfit_axis=np.linspace(0,1,len(sig_info['data'][ind][:-20]))
true_dens=scipy.interpolate.interp1d(zipfit_axis,sig_info['data'][ind][:-20])(x_axis)

axes[0].plot(x_axis,true_temp,c='k',label='INITIAL',linestyle='--',linewidth=linewidth)
axes[0].plot(x_axis[:6:2],pow(true_temp[:6:2],1.4),c='b',linestyle='--',
marker='x',markersize=tick_font_size,markeredgewidth=tick_font_size/10,
label='TARGET')
axes[0].plot(x_axis,true_temp*.95,c='r',label=r'$\downarrow$  pinj',linewidth=linewidth)
axes[0].plot(x_axis,true_temp*1.3,c='orange',label=r'$\uparrow$ pinj',linewidth=linewidth)
axes[0].plot(x_axis,true_temp*1.22,c='g',label=r'$\uparrow$ pinj, $\uparrow$ target density',linewidth=big_linewidth)

axes[1].plot(x_axis,true_dens,c='k',label='INITIAL',linestyle='--',linewidth=linewidth)
axes[1].plot(x_axis[-6::2],pow(true_dens[-6::2],1.4),c='b',linestyle='--',
marker='x',markersize=tick_font_size,markeredgewidth=tick_font_size/10,
label='TARGET')
axes[1].plot(x_axis,true_dens*1.05,c='r',label=r'$\downarrow$  pinj',linewidth=linewidth)
axes[1].plot(x_axis,true_dens*(1.1+x_axis*0.2),c='orange',label=r'$\uparrow$ pinj',linewidth=linewidth)
axes[1].plot(x_axis,true_dens*(1.15+x_axis*0.4),c='g',label=r'$\uparrow$ pinj, $\uparrow$ target density',linewidth=big_linewidth)

for ax in axes:
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        right=False,
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    ax.text(0.5, 0.5, 'Idealized Example', transform=ax.transAxes,
            fontsize=2*label_font_size, color='gray', alpha=0.5,
            ha='center', va='center', rotation='30')


handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=label_font_size, loc=(0.2,0.45))
axes[-1].set_xlabel('radius',size=axis_font_size)
axes[0].set_ylabel('temperature',size=axis_font_size)
axes[1].set_ylabel('density',size=axis_font_size)

plt.savefig('finite_set_example.png')
plt.show()
