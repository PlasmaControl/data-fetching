from toksearch import PtDataSignal, MdsSignal
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

shot=969000
shot2=936814 #936815
ref_shot=182500 #5769

mytime=2000

which_sig='itemp'
zipfit_signame={'temp': 'etempfit', 'dens': 'edensfit', 'itemp': 'itempfit'}
ricardo_fit_signame={'temp': 'ftte', 'dens': 'ftne', 'itemp': 'ftti'}
ricardo_input_signame={'temp': 'temp', 'dens': 'dens', 'itemp': 'cti'}
ricardo_input_psi_correction={'temp': 1., 'dens': 1., 'itemp': 1}
thomson_signame={'temp': 'te', 'dens': 'ne'}
joe_signame={'temp': 'temp', 'dens': 'density'}
ricardo_fit_psi_correction=1. #0.792
max_ind={'temp': 50, 'dens': 50, 'itemp': 16}

ricardo_input_max_ind=max_ind[which_sig]
ricardo_fit_max_ind=121

rho_or_psin='psin'
zipfit_norm={'temp': 1e3, 'dens': 1, 'itemp': 1}
ricardo_norm={'temp': 1e3, 'dens': 1, 'itemp': 1}
joe_norm={'temp': 1e3, 'dens': 1}

#see Thomson
if False:
    x=[]
    y=[]
    y_ref=[]
    for i in range(30):
        sig_name='tsscor{}{:02d}'.format(thomson_signame[which_sig],i)
        print(sig_name)
        result=PtDataSignal(sig_name).fetch(shot)
        y.append(result['data'])
        result_ref=PtDataSignal(sig_name).fetch(ref_shot)
        y_ref.append(result_ref['data'])
        sig_name='tsscorps{:02d}'.format(i)
        result=PtDataSignal(sig_name).fetch(shot)
        x.append(result['data'])
        if i==1:
            times=result['times']
            times_ref=result_ref['times']
    x=np.array(x).T
    y=np.array(y).T
    y_ref=np.array(y_ref).T

    time_ind=np.searchsorted(times,mytime)
    time_ind_ref=np.searchsorted(times_ref,mytime)

    plt.plot(x[time_ind],y[time_ind],
             label='thomson {}: {} ms'.format(shot,times[time_ind]))
    plt.plot(x[time_ind],y_ref[time_ind_ref],
             label='thomson {}: {} ms'.format(ref_shot, times_ref[time_ind_ref]))


if True:
    for this_shot in [shot]:
        y=[]
        x=[]
        for i in range(1,ricardo_input_max_ind,2):
            sig_name='fts{}{}'.format(ricardo_input_signame[which_sig],i)
            print(sig_name)
            result=PtDataSignal(sig_name).fetch(this_shot)
            y.append(result['data'])
            if i==1:
                times=result['times']
            if which_sig in ['dens', 'temp']:
                sig_name='fts{}{}'.format(rho_or_psin,i)
            else:
                sig_name='fts{}{}'.format('cpsin',i)
            result=PtDataSignal(sig_name).fetch(this_shot)
            x.append(result['data'])

        time_ind=np.searchsorted(times,mytime)

        x=np.array(x).T
        y=np.array(y).T

        plt.plot(x[time_ind]*ricardo_input_psi_correction[which_sig],y[time_ind], 
                 label='ricardo input {}, {:.2f} ms'.format(this_shot,times[time_ind]))


if True:
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

if True:
    y_fit=[]
    x_fit=[]
    for i in np.arange(1,ricardo_fit_max_ind,4):
        sig_name='fts{}{}'.format(ricardo_fit_signame[which_sig],i)
        print(sig_name)
        result=PtDataSignal(sig_name).fetch(ref_shot)
        y_fit.append(result['data'])
        if i==1:
            times_fit=result['times']
        sig_name='fts{}{}'.format('ftps',i)
        result=PtDataSignal(sig_name).fetch(ref_shot)
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

# include my offline code
if False:
    import pickle
    with open('new_data.pkl','rb') as f:
        data=pickle.load(f)
    thomson_time_ind=np.searchsorted(data[ref_shot]['time'],mytime)
    #thomson_temp_mtanh_1d
    plt.plot(np.linspace(0,1,len(data[ref_shot]['thomson_{}_linear_interp_1d'.format(joe_signame[which_sig])][thomson_time_ind])),
             data[ref_shot]['thomson_{}_linear_interp_1d'.format(joe_signame[which_sig])][thomson_time_ind]*joe_norm[which_sig],
             label='offline thomson {}: {} ms'.format(ref_shot,data[ref_shot]['time'][thomson_time_ind]))

plt.ylabel(which_sig)
plt.xlabel(rho_or_psin)
plt.legend()
plt.show()
