# this is an OMFIT script, so first do
# module load omfit
# then run with omfit nb_grab.py batch_index=_index_
# for _index_ the string to append to the filename
# see https://omfit.io/run.html for more omfit info

# batch_index is the string that gets appended to the filename
# shot_start_index is the index of the first shot we want included
#                                           within 2021_shots.npy,
# shot_end_index is the index of the last shot we want included
defaultVars(batch_index='', shot_start_index=0, shot_end_index=2)

# points are on a 0.1ms basis, so 10 downsamples to 1ms basis
downsample_rate=10
all_shots=np.load('/home/abbatej/tokamak-transport/data/2021_shots.npy')

shots=all_shots[shot_start_index:shot_end_index]
data_dic={}
failed_shots=[]
OMFIT.loadModule('TRANSP',"OMFIT['TRANSP_for_NB']")
OMFIT['TRANSP_for_NB']['SETTINGS']['EXPERIMENT']['times']=np.arange(0,6000,10)
important_keys=['NBEAM', 'DIVZA','DIVRA','RAPEDGA','REDGE','XZPEDGA','XZEDGE','XBZETA', 'RTCENA', 'XLBTNA', 'XYBSCA', 'XLBAPA', 'XYBAPA','ABEAMA','XZBEAMA','NLCO','NBAPSHA']
for shot in shots:
    try:
        data_dic[shot]={}
        OMFIT['TRANSP_for_NB']['SETTINGS']['EXPERIMENT']['shot']=shot
        OMFIT['TRANSP_for_NB']['SCRIPTS']['1_nml'].run()
        OMFIT['TRANSP_for_NB']['SCRIPTS']['DIII-D']['HEATING']['nubeam'].run()
        full_nml=OMFIT['TRANSP_for_NB']['INPUTS']['TRANSP']
        nubeam_nml = full_nml['NEUTRAL_BEAMS']
        NBEAM=nubeam_nml['NBEAM']
        data_dic[shot]['time'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['X0']['data'][::downsample_rate]
        data_dic[shot]['pinj'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['F']['data'][::downsample_rate, : NBEAM]
        data_dic[shot]['vinj'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['F']['data'][::downsample_rate, NBEAM : 2 * NBEAM]
        data_dic[shot]['frac1'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['F']['data'][::downsample_rate, 2 * NBEAM : 3 * NBEAM]
        data_dic[shot]['frac2'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['F']['data'][::downsample_rate, 3 * NBEAM : 4 * NBEAM]
        data_dic[shot]['BEAMS'] = OMFIT['TRANSP_for_NB']['UFILES']['NB2']['BEAMS'] #beam labels
        for key in important_keys:
            if key in nubeam_nml:
                data_dic[shot][key] = nubeam_nml[key]
    except:
        failed_shots.append(shot)
with open(f'/cscratch/abbatej/raw_read/beam_info{batch_index}.pkl','wb') as f:
    pickle.dump(data_dic,f)
print(f'failed shots: {failed_shots}')
