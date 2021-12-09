from toksearch import PtDataSignal, MdsSignal
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

shot=936813
ref_shot=182500 #5769

mytime=2000

which_sig='temp'
zipfit_signame={'temp': 'etempfit', 'dens': 'edensfit', 'itemp': 'itempfit'}
ricardo_fit_signame={'temp': 'ftte', 'dens': 'ftne', 'itemp': 'ftti'}
ricardo_input_signame={'temp': 'temp', 'dens': 'dens', 'itemp': 'cti'}
ricardo_input_psi_correction={'temp': 1., 'dens': 1., 'itemp': 1}
ricardo_fit_psi_correction=1. #0.792
max_ind={'temp': 50, 'dens': 50, 'itemp': 16}

ricardo_input_max_ind=max_ind[which_sig]
ricardo_fit_max_ind=121

rho_or_psin='psin'
zipfit_norm={'temp': 1e3, 'dens': 1, 'itemp': 1}
ricardo_norm={'temp': 1e3, 'dens': 1, 'itemp': 1}


y=[]
x=[]
for i in range(1,ricardo_input_max_ind):
    sig_name='fts{}{}'.format(ricardo_input_signame[which_sig],i)
    print(sig_name)
    result=PtDataSignal(sig_name).fetch(shot)
    y.append(result['data'])
    if i==1:
        times=result['times']
    if which_sig in ['dens', 'temp']:
        sig_name='fts{}{}'.format(rho_or_psin,i)
    else:
        sig_name='fts{}{}'.format('cpsin',i)
    result=PtDataSignal(sig_name).fetch(shot)
    x.append(result['data'])

time_ind=np.searchsorted(times,mytime)

x=np.array(x).T
y=np.array(y).T

sig_name=zipfit_signame[which_sig]
zipfit_result=MdsSignal(r'\ZIPFIT01::TOP.PROFILES.{}'.format(sig_name),'ZIPFIT01',location='remote://atlas.gat.com',dims=['rhon','times']).fetch(ref_shot)
zipfit_time_ind=np.searchsorted(zipfit_result['times'],times[time_ind])
if True:
    rhon_sig = MdsSignal(r'\rhovn',
                          'EFIT01', 
                          location='remote://atlas.gat.com',
                          dims=['psi','times']).fetch(ref_shot)

    efit_time_ind=np.searchsorted(rhon_sig['times'],times[time_ind])
    rhon_to_psi=interpolate.interp1d(rhon_sig['data'][efit_time_ind],rhon_sig['psi'],bounds_error=False,fill_value=0)
    plt.plot(rhon_to_psi(zipfit_result['rhon']),zipfit_result['data'][zipfit_time_ind,:]*zipfit_norm[which_sig],
             label='zipfit, {:.2f} ms'.format(zipfit_result['times'][zipfit_time_ind]))
else:
    plt.plot(zipfit_result['rhon'],zipfit_result['data'][zipfit_time_ind,:]*zipfit_norm[which_sig],
             label='zipfit, {:.2f} ms'.format(zipfit_result['times'][zipfit_time_ind]))

plt.plot(x[time_ind]*ricardo_input_psi_correction[which_sig],y[time_ind], 
         label='ricardo input, {:.2f} ms'.format(times[time_ind]))

y_fit=[]
x_fit=[]
for i in np.arange(1,ricardo_fit_max_ind,4):
    sig_name='fts{}{}'.format(ricardo_fit_signame[which_sig],i)
    print(sig_name)
    result=PtDataSignal(sig_name).fetch(shot)
    y_fit.append(result['data'])
    if i==1:
        times_fit=result['times']
    sig_name='fts{}{}'.format('ftps',i)
    result=PtDataSignal(sig_name).fetch(shot)
    x_fit.append(result['data'])
x_fit=np.array(x_fit).T
y_fit=np.array(y_fit).T

is_all_zero=np.all(np.isclose(y_fit,0),axis=1)
x_fit=x_fit[np.logical_not(is_all_zero)]
y_fit=y_fit[np.logical_not(is_all_zero)]
times_fit=times_fit[np.logical_not(is_all_zero)]

fit_time_ind=min(np.searchsorted(times_fit,times[time_ind]),len(times_fit)-1)

plt.plot(x_fit[fit_time_ind]*ricardo_fit_psi_correction,y_fit[fit_time_ind]*ricardo_norm[which_sig],
         label="ricardo's fit, {} ms".format(times_fit[fit_time_ind]))

plt.ylabel(which_sig)
plt.xlabel(rho_or_psin)
plt.title('shot {}, {:.2f} ms'.format(shot,times[time_ind]))
plt.legend()
plt.show()
