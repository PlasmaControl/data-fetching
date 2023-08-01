import MDSplus
import gadata # download from https://diii-d.gat.com/diii-d/Gadata_py
import h5py
import pickle
import numpy as np
import time
# from tqdm import tqdm
import sys

def data2dict(shotn, signame, hf, atlconn) :
	data = gadata(signame, shotn, connection=atlconn)
	dict_group = hf.create_group(str(signame))
	dict_group['xdata'] = data.xdata
	dict_group['ydata'] = data.ydata
	dict_group['zdata'] = data.zdata
	dict_group['xunits'] = data.xunits
	dict_group['yunits'] = data.yunits
	dict_group['zunits'] = data.zunits
	return True

output_path = '/cscratch/jalalvanda/outputs/'
atlconn = MDSplus.Connection('atlas.gat.com')
ech_gytname = ['lei','luk','r2d']

shot_list = [np.int32(sys.argv[1])] # input shot number

signal_list= { # List of the profiles and actuators to be fetched
'profiles':['bcoil', 'betan', 'bmspinj', 'bmstinj', 'bt', 'dssdenest', 'edensfit', 'etempfit', 'fzns', 'ip', 'ipsip', 'iptipp', 'neutronsrate', 'pcbcoil', 'pinj', 'plasticfix', 'pnbi', 'pres', 'q', 'q95', 'tinj','n1rms','n2rms'],
'pinj':['pinjf_%dl' % k for k in [15,21,30,33]]+['pinjf_%dr' % k for k in [15,21,30,33]],
'tinj':['tinj_%dl' % k for k in [15,21,30,33]]+['tinj_%dr' % k for k in [15,21,30,33]],
'ech':['pech','echpwrc','echpwr']+['ec%sfpwrc' % (x) for x in ech_gytname]+['ec%sxmfrac' % (x) for x in ech_gytname]+['ec%spolang' % (x) for x in ech_gytname],
'shape':['kappa','triangularity_u','triangularity_l'],
'qmin':['qmin']
}

ece_pcece=True # if True fast ECE (tecefxx) and PCS available ECE (pcece) will be fetched.
for shotn in shot_list:
	t1=time.time()
	print('Shot #%d'%(shotn,))
	for grpname,signals in signal_list.items():
		hf = h5py.File(output_path+ str(shotn)+'_'+grpname+'.h5')
		for signame in signals:
                        #print(signame)
			_=data2dict(shotn,signame,hf,atlconn)
		hf.close()
	

	if ece_pcece:
		hf = h5py.File(output_path+ str(shotn)+'_ece_pcece_rtece.h5')
		pece_group = hf.create_group('pcece')	
		ece_group = hf.create_group('ece')
		#rtece_group = hf.create_group('rtece')
                
		for k in range(40):
			print('chn %i' % (k+1))
			pece_data = gadata('pcece%d' % (k+1), shotn, connection=atlconn)
			pece_group['pcece%02d' % (k+1)] = pece_data.zdata
			ece_data = gadata('tecef%02d' % (k+1), shotn, connection=atlconn)
			ece_group['tecef%02d' % (k+1)] = ece_data.zdata
                        #print('here')
			#rtece_data = gadata('ecsdata%d' % (k+97), shotn, connection=atlconn)
			#rtece_group['ecsdata%d' % (k+97)] = rtece_data.zdata
                        #print('ecsdata%d' % (k+97))
		pece_group['xdata'] = pece_data.xdata
		pece_group['ydata'] = pece_data.ydata
		pece_group['xunits'] = pece_data.xunits
		pece_group['yunits'] = pece_data.yunits
		pece_group['pceceunits'] = pece_data.zunits
                
		ece_group['xdata'] = ece_data.xdata
		ece_group['ydata'] = ece_data.ydata
		ece_group['xunits'] = ece_data.xunits
		ece_group['yunits'] = ece_data.yunits
		ece_group['eceunits'] = ece_data.zunits

		#rtece_group['xdata'] = rtece_data.xdata
		#rtece_group['ydata'] = rtece_data.ydata
		#rtece_group['xunits'] = rtece_data.xunits
		#rtece_group['yunits'] = rtece_data.yunits
		#rtece_group['rteceunits'] = rtece_data.zunits
		hf.close()
#	print('time per shot:%ds' % (time.time()-t1))
