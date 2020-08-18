import numpy as np

def get_volume(r,z,psi_grid,basis_psi):

    num_psi=len(basis_psi)
    dpsi=np.diff(basis_psi) #1/num_psi
    dr=np.diff(r)
    dz=np.diff(z)

    dv=np.zeros((psi_grid.shape[0],num_psi-1))
    #G1=np.zeros(num_psi)
    for time_ind in range(psi_grid.shape[0]):
        for psi_ind in range(num_psi-1):
            gradient=np.gradient(psi_grid[time_ind])

            psi=basis_psi[psi_ind] #dpsi[psi_ind]*psi_ind
            hull=np.where(np.logical_and(psi_grid[time_ind]<psi+dpsi[psi_ind],
                          psi_grid[time_ind]>=psi))

            volume=0
            for (z_ind,r_ind) in zip(*hull):
                # if z_ind==num_psi-1:
                #     z_ind=num_psi-2
                # if r_ind==num_psi-1:
                #     r_ind=num_psi-2
                volume+=r[r_ind]*dr[r_ind]*dz[z_ind]
            volume*=2*np.pi
            dv[time_ind,psi_ind]=volume

    # dv is differential so will be missing the last volume element
    # this is a hack, but we add the last psi value for each time
    # to the end to reshape it to match basis_psi
    last_dv=np.atleast_2d(dv[:,-1])
    #transpose and untraspose to match what np.concatentate expects
    dv=np.concatenate((dv.T,last_dv),axis=0).T

    return dv
        # psi_gradient=0
        # for (z_ind,r_ind) in zip(*hull):
        #     psi_gradient+=np.square(gradient[0][r_ind][z_ind])+np.square(gradient[1][r_ind][z_ind])
        # G1[i]=psi_gradient/len(hull[0])

    

