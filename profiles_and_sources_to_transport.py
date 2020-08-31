import os
import pickle
import numpy as np

shot=163303
efit_type='EFIT01'

standard_psi=np.linspace(0,1,65)
transp_psi=np.linspace(0,1,20)

data_dir=os.path.join(os.path.dirname(__file__),'data')
with open(os.path.join(data_dir,'final_data_full_batch_0.pkl'),'rb') as f:
    full_data=pickle.load(f)
with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'rb') as f:
    data=pickle.load(f)

dn_dt = np.diff(data[shot]['thomson_dens_{}'.format(efit_type)],axis=-1)
ddv_dt = np.diff(data['dv'],axis=-1)
print(ddv_dt.shape)
# for time_ind in range(len(data[shot]['time'])):
#     data[shot]['thomson_dens_{}'.format(efit_type)][time_ind]
#     dGamma_dRho=dv*S-
