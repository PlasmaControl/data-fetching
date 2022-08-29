import numpy as np

thomson_scale={'density': 1e19, 'temp': 1e3}
thomson_areas=['CORE','TANGENTIAL']

cer_scale={'temp': 1e3, 'rot': 1}
cer_areas=['TANGENTIAL', 'VERTICAL']
cer_channels_realtime={'TANGENTIAL': np.arange(5,25), #np.arange(1,33); np.arange(5,25) for realtime
                       'VERTICAL': []} #np.arange(1,49); [] for realtime
cer_channels_all={'TANGENTIAL': np.arange(1,33),
                  'VERTICAL': np.arange(1,49)}

zipfit_pairs={'cer_temp': 'itempfit',
              'cer_rot': 'trotfit',
              'thomson_temp': 'etempfit',
              'thomson_density': 'edensfit'}

# our PCS algo stuff
etemp_sigs=['etstein', 'etsnein','etscrin', 'etsctin','etsinq', 'etsinprs',
          'etsteout', 'etsneout', 'etsqout', 'etsprsout',
          'etste', 'etsne','etscr','etsct']
pcs_length={sig_name: np.arange(0,33) for sig_name in etemp_sigs}
for sig in ['ftscrot','ftscpsin','ftsc1v1d']:
    pcs_length[sig]=np.arange(1,16)
for sig in ['ftsspsin','pftssrot','ftsspsin','ftssmhat']:
    pcs_length[sig]=np.arange(1,76)
for sig in ['ftstemp','ftsterr','ftsdens','ftsnerr','ftspsin','ftsrho']:
    pcs_length[sig]=np.arange(1,50)
# {'zipfit_trotfit': 'rotation',
#           'zipfit_itempfit': 'itemp',
#           'zipfit_etempfit': 'temp',
#           'zipfit_edensfit': 'dens',
#           'qpsi': 'q',
#           'pres': 'press',
#           'ipsiptargt': 'curr_target',
#           'ip': 'curr',
#           }
