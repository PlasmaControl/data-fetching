# before running compile the library via:
# gcc -lm -fPIC -shared mtanh_mpfit_lin_core.c mpfit.c -o libmtanh.so

import numpy as np
import ctypes
import os

lib_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      'libmtanh.so')

def array_to_ctypes_1d(A,desired_type=ctypes.c_double):
    # initialize properly sized array (to 0)
    arr=(desired_type*len(A))()
    for i in range(len(A)):
        arr[i]=A[i]
    return arr

def ctypes_to_array_1d(A,N):
    arr=np.zeros((N))
    for i in range(N):
        arr[i]=A[i]
    return arr

def padded_with_zeros(arr, new_length):
    tmp=np.zeros(new_length)
    tmp[:min(len(arr),new_length)]=arr[:new_length]
    return tmp

def mtanh_eval(psin, signal, error):
    clib=ctypes.cdll.LoadLibrary(lib_path)
    FITTING_NRINGTHOM=41
    FITTING_MAX_PAR=5
    NOUT=121

    num_points=len(signal)

    clib.mtanh_mpfit.restype=ctypes.c_int
    clib.mtanh_mpfit.argtypes=[ctypes.c_int, #nx
                               (ctypes.c_double*FITTING_NRINGTHOM), #x
                               (ctypes.c_double*FITTING_NRINGTHOM), #y
                               (ctypes.c_double*FITTING_NRINGTHOM), #ey
                               (ctypes.c_double*FITTING_MAX_PAR), #presult
                               (ctypes.c_double*NOUT), #prho
                               (ctypes.c_double*NOUT)] #pfit
    x=array_to_ctypes_1d(padded_with_zeros(psin, FITTING_NRINGTHOM))
    y=array_to_ctypes_1d(padded_with_zeros(signal, FITTING_NRINGTHOM))
    ey=array_to_ctypes_1d(padded_with_zeros(error, FITTING_NRINGTHOM))

    presult=(ctypes.c_double*FITTING_MAX_PAR)()
    prho=(ctypes.c_double*NOUT)()
    pfit=(ctypes.c_double*NOUT)()
    
    clib.mtanh_mpfit(num_points, x,y,ey,presult,prho,pfit)
    return (ctypes_to_array_1d(prho,NOUT), ctypes_to_array_1d(pfit,NOUT))
