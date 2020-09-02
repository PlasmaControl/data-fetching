import numpy as np
from scipy import interpolate
import pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'lib'))
from transport_helpers import my_interp
from plot_tools import plot_comparison_over_time

shot=163303
debug=True
debug_sig='G1'

data_dir=os.path.join(os.path.dirname(__file__),'data')
with open(os.path.join(data_dir,'final_data_full_batch_0.pkl'),'rb') as f:
    full_data=pickle.load(f)
with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'rb') as f:
    data=pickle.load(f)

r=full_data[shot]['EFIT01']['R']
z=full_data[shot]['EFIT01']['Z']
psi_grid=np.array(data[shot]['psirz'])
rho_grid=np.array(data[shot]['rho_grid'])
basis=np.linspace(.025,.975,20)
fit_psi=False

efit_psi=np.linspace(0,1,65)
num_psi=len(basis)
dr=np.diff(r)
dz=np.diff(z)
dv=np.zeros((psi_grid.shape[0],num_psi-1))
G1=np.zeros((psi_grid.shape[0],num_psi-1))

for time_ind in range(psi_grid.shape[0]):
    if fit_psi:
        basis_psi=basis
        psi_to_rho=my_interp(efit_psi,rho_grid[time_ind])
        basis_rho=psi_to_rho(basis)
    else:
        basis_rho=basis
        rho_to_psi = my_interp(rho_grid[time_ind],efit_psi)
        basis_psi=rho_to_psi(basis)
    dpsi=np.diff(basis_psi)
    drho=np.diff(basis_rho)
    dRho_dPsi=np.divide(dpsi,drho)

    for psi_ind in range(num_psi-1):
        gradient=np.gradient(psi_grid[time_ind])

        psi=basis_psi[psi_ind] #dpsi[psi_ind]*psi_ind
        hull=np.where(np.logical_and(psi_grid[time_ind]<psi+dpsi[psi_ind],
                      psi_grid[time_ind]>=psi))

        volume=0
        rho_gradient_squared=0
        for (z_ind,r_ind) in zip(*hull):
            # if z_ind==num_psi-1:
            #     z_ind=num_psi-2
            # if r_ind==num_psi-1:
            #     r_ind=num_psi-2
            volume+=r[r_ind]*dr[r_ind]*dz[z_ind]
            psi_gradient_squared=np.square(gradient[0][r_ind][z_ind])+np.square(gradient[1][r_ind][z_ind])
            rho_gradient_squared+=psi_gradient_squared*np.square(dRho_dPsi[psi_ind])

        volume*=2*np.pi
        dv[time_ind,psi_ind]=volume
        G1[time_ind,psi_ind]=rho_gradient_squared/len(hull[0])

# dv is differential so will be missing the last volume element
# this is a hack, but we add the last psi value for each time
# to the end to reshape it to match basis_psi
last_dv=np.atleast_2d(dv[:,-1])
last_G1=np.atleast_2d(G1[:,-1])
#transpose and untraspose to match what np.concatentate expects
dv=np.concatenate((dv.T,last_dv),axis=0).T
G1=np.concatenate((G1.T,last_G1),axis=0).T

data[shot]['dv']=dv
data[shot]['G1']=G1

with open(os.path.join(data_dir,'final_data_batch_0.pkl'),'wb') as f:
    pickle.dump(data,f)

if debug:
    print('Average volume: {:.2f}m^3'.format(np.mean(np.sum(data[shot]['dv'],axis=-1))))
    plot_comparison_over_time(xlist=[basis_rho],
                              ylist=[data[shot][debug_sig]],
                              time=data[shot]['time'],
                              ylabel=debug_sig,
                              xlabel='rho',
                              uncertaintylist=None,
                              labels=None)
