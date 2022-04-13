# run this script in OMFIT commandwindow (no other setup required)

shots=np.load('/home/abbatej/tokamak-transport/data/2021_shots.npy')
data_dic={}
OMFIT.loadModule('TRANSP',"OMFIT['TRANSP_for_RF']")
OMFIT['TRANSP_for_RF']['SETTINGS']['EXPERIMENT']['times']=np.arange(0,6000,10)
for shot in shots:
    OMFIT['TRANSP_for_RF']['SETTINGS']['EXPERIMENT']['shot']=shot
    OMFIT['TRANSP_for_RF']['SCRIPTS']['1_nml'].run()
    OMFIT['TRANSP_for_RF']['SCRIPTS']['DIII-D']['HEATING']['toray'].run()
    if 'ECP' in OMFIT['TRANSP_for_RF']['UFILES']:
        data_dic[shot]={}
        data_dic[shot]['ECP']=dict(OMFIT['TRANSP_for_RF']['UFILES']['ECP'])
        data_dic[shot]['ECP']['SCALARS']=dict(data_dic[shot]['ECP']['SCALARS'])
        for key in ['RFMODECH','XECECH','ZECECH','FREQECH']:
            data_dic[shot][key]=np.array(OMFIT['TRANSP_for_RF']['INPUTS']['TRANSP']['ELECTRON_CYCLOTRON_RESONANCE_HEATING_TORAY'][key])
        for key in ['ECB_WIDTH_HOR','ECB_WIDTH_VERT','ECB_CURV_HOR','ECB_CURV_VERT']:
            data_dic[shot][key]=np.array(OMFIT['TRANSP_for_RF']['INPUTS']['TRANSP']['ELECTRON_CYCLOTRON_RESONANCE_HEATING_TORBEAM'][key])
        if 'ECA' in OMFIT['TRANSP_for_RF']['UFILES'] and 'ECB' in OMFIT['TRANSP_for_RF']['UFILES']:
            data_dic[shot]['THE']=dict(OMFIT['TRANSP_for_RF']['UFILES']['ECA'])
            data_dic[shot]['THE']['SCALARS']=dict(data_dic[shot]['THE']['SCALARS'])
            data_dic[shot]['PHI']=dict(OMFIT['TRANSP_for_RF']['UFILES']['ECB'])
            data_dic[shot]['PHI']['SCALARS']=dict(data_dic[shot]['PHI']['SCALARS'])
        else:
            data_dic[shot]['THE']=np.array(OMFIT['TRANSP_for_RF']['INPUTS']['TRANSP']['ELECTRON_CYCLOTRON_RESONANCE_HEATING_TORAY']['THETECH'])
            data_dic[shot]['PHI']=np.array(OMFIT['TRANSP_for_RF']['INPUTS']['TRANSP']['ELECTRON_CYCLOTRON_RESONANCE_HEATING_TORAY']['PHAIECH'])

with open('/cscratch/abbatej/raw_read/ech_info.pkl','wb') as f:
    pickle.dump(data_dic,f)
