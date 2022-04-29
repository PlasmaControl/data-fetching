import numpy as np

thomson_scale={'density': 1e19, 'temp': 1e3}
thomson_areas=['CORE','TANGENTIAL']

cer_scale={'temp': 1e3, 'rot': 1}
cer_areas=['TANGENTIAL', 'VERTICAL']
cer_channels={'TANGENTIAL': np.arange(5,25), #np.arange(1,33); np.arange(5,25) for realtime
              'VERTICAL': []} #np.arange(1,49); [] for realtime

zipfit_pairs={'cer_temp': 'itempfit',
              'cer_rot': 'trotfit',
              'thomson_temp': 'etempfit',
              'thomson_density': 'edensfit'}

# our PCS algo stuff
etemp_sigs=['etstein', 'etsnein','etscrin', 'etsctin','etsinq', 'etsinprs',
          'etsteout', 'etsneout', 'etsqout', 'etsprsout',
          'etste', 'etsne','etscr','etsct']
pcs_length={sig_name: np.arange(0,33) for sig_name in etemp_sigs}
pcs_length['ftscrot']=np.arange(1,16)
pcs_length['ftscpsin']=np.arange(1,16)
pcs_length['ftsc1vld']=np.arange(1,16)
pcs_length['ftsspsin']=np.arange(1,76)
pcs_length['ftssrot']=np.arange(1,76)

name_map={'zipfit_trotfit': 'rotation',
          'zipfit_itempfit': 'itemp',
          'zipfit_etempfit': 'temp',
          'zipfit_edensfit': 'dens',
          'vloop': 'vloop',
          'aminor': 'aminor',
          'li': 'li',
          'kappa': 'kappa',
          'volume': 'volume',
          'betan': 'betan',
          'betap': 'betap',
          'drsep': 'drsep',
          'rmaxis': 'rmaxis',
          'zmaxis': 'zmaxis',
          'zxpt1': 'zxpt1',
          'zxpt2': 'zxpt2',
          'rxpt1': 'rxpt1',
          'rxpt2': 'rxpt2',
          'seplim': 'seplim',
          'gapin': 'gapin',
          'gapout': 'gapout',
          'gapbot': 'gapbot',
          'gaptop': 'gaptop',
          'tritop': 'tritop',
          'tribot': 'tribot',
          'qpsi': 'q',
          'pres': 'press',
          'pinj': 'pinj',
          'tinj': 'tinj',
          'dstdenp': 'dstdenp',
          'dssdenest': 'dssdenest',
          'ipsiptargt': 'curr_target',
          'echpwr': 'echpwr',
          'dsifbonoff': 'dsifbonoff',
          'dsiwdens': 'dsiwdens',
          'dsitgtsrc': 'dsitgtsrc',
          'bmifbonoff': 'bmifbonoff',
          'bmipfbval': 'bmipfbval',
          'DUSTRIPPED': 'DUSTRIPPED',
          'bt': 'bt',
          'ip': 'curr',
          'gasA': 'gasA',
          'gasB': 'gasB',
          'gasC': 'gasC',
          'gasD': 'gasD',
          'gasE': 'gasE',
          'pfx1': 'pfx1',
          'pfx2': 'pfx2',
          'N1ICWMTH': 'N1ICWMTH',
          'N1IIWMTH': 'N1IIWMTH',
          'n1rms': 'n1rms',
          'n2rms': 'n2rms',
          'n3rms': 'n3rms',
          'wmhd': 'wmhd'
          }
for i in ['30','15','21','33']:
    for j in ['l','r']:
        name=f'bmspinj{i}{j}'
        name_map[name]=name
for i in ['30','150','210','330']:
    for j in ['l','r']:
        name=f'bmics{i}{j}'
        name_map[name]=name

for i in range(1,25):
    for position in ['L','U']:
        name='prad{}{}'.format(position,i)
        name_map[name]=name
for key in ['KAPPA','PRAD_DIVL','PRAD_DIVU','PRAD_TOT']:
    name=f'prad{key}'
    name_map[name]=name
for key in pcs_length.keys():
    name_map[key]=key
for i in range(1,9):
    for letter in ['a','b']:
        name=f'f{i}{letter}'
        name_map[name]=name
for i in [30,90,150,270,330]:
    for letter in ['U','L']:
        name=f'I{letter}{i}'
        name_map[name]=name
