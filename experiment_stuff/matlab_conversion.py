from scipy.io import savemat
import numpy as np
import pickle

with open('raw_data_test.pkl','rb') as f:
    data=pickle.load(f)
    
shot=187076
y=data[shot]['temp'][:,0]
u=data[shot]['pinj']
# get mask for whether data is present 
mask=np.logical_not(np.logical_or(np.isnan(data[shot]['temp'][:,0]), np.isnan(data[shot]['pinj'])))
y=y[mask]
u=u[mask]

matlab_dict={}
matlab_dict['y_val']=y
matlab_dict['u_val']=u
savemat('te_pinj.mat',matlab_dict)
