import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('new_data.pkl','rb') as f:
    data=pickle.load(f)

sig_name='rotation'
raw_name={'rotation': 'cer_rot', 'temp': 'thomson_temp'}

time_ind=60
shot=163303
psi_raw=data[shot]['{}_psi_raw_1d'.format(raw_name[sig_name])]
sig_raw=data[shot]['{}_raw_1d'.format(raw_name[sig_name])]
plt.scatter(psi_raw[time_ind],sig_raw[time_ind])

sig=data[shot]['{}_csaps_1d'.format(raw_name[sig_name])]
psi=np.linspace(0,1,sig.shape[1])
plt.plot(psi,sig[time_ind])
sig=data[shot][sig_name]
psi=np.linspace(0,1,sig.shape[1])
plt.plot(psi,sig[time_ind])

plt.title('shot {}, {}s'.format(shot,data[shot]['time'][time_ind]))
plt.show()
