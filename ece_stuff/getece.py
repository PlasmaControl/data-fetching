#!/usr/bin/env python
'''
Simple script to utilize toksearch capabilities to build an ECE database
Saves data to pickle files with a single time array

Author: Oak Nelson (Nov 16 2020)

INPUTS: 
 - shots      (list)    | a list of shots to include
 - tmin       (float)   | minimum time to include in output
 - tmax       (float)   | maximum time to include in output
 - outdir     (string)  | output directory for pkl files

OUTPUTS:
 - <shot>.pkl (pkl)     | a pickle file with the ECE output
'''


from toksearch import PtDataSignal, MdsSignal, Pipeline
import numpy as np
import collections
import pprint
import pickle



###### USER INPUTS ######
#                       #

newshots = [164911,165249,166348,167303,167449,167605,169605,170090,170256,170338,170503,170783,171169]
doneshots = [171623,172453,173378,173857,173956,174767,174970,175288,175855,176189,176460,178609,179106,179372,179850,180280,180666,180721,164879,165248,166169,167297,167445,167572,169599,170089,170251,170337,170497,170781,171167,171569,172451,173267,173842,173955,174761,174968,175280,175854,176162,176457,177013,179101,179371,179828,180260,180661,180719,122117, 145387, 157102, 175711, 180619, 180625, 149388, 150610, 150616, 150792, 163117, 166578, 170881, 174082, 174084, 174819, 174823, 174864, 180908]#[149388, 174864]

shots = list(set(newshots) - set(doneshots))

tmin = 0
tmax = 5000

outdir = 'data'
#                       #
#### END USER INPUTS ####


#### START OF SCRIPT ####
#                       #

#### GATHER DATA ####

pipeline = Pipeline(shots) 

ptnms = []
ece_signals = {}
for i in range(1,41):
    ptnms += [r'\tecef' + str(i).zfill(2)]
    
for ptnm in ptnms:
    ece_signals[ptnm] = MdsSignal(ptnm, 'ECE', location='remote://atlas.gat.com')
    pipeline.fetch(ptnm, ece_signals[ptnm])

np.set_printoptions(threshold=3, precision=1)

records = pipeline.compute_serial() # spark() or ray() or serial()
print('Data collection complete!')
print('Number of records: {}. Should be {}.'.format(len(records),len(shots)))

#### CROP DATA TO TIME AXIS ####

results = {}
for record in records:
    if not record['errors']: # there is probably a better way to handle errors...
        results[record['shot']] = {}
        
        times = record[ptnms[0]]['times']
        tsel = (times > tmin) * (times < tmax)
        
        results[record['shot']]['time'] = record[ptnms[0]]['times'][tsel] # crop one time array
        for ptnm in ptnms:
            results[record['shot']][ptnm] = record[ptnm]['data'][tsel]
    
    else: 
        print('There were some errors for shot {}...'.format(record['shot']))

#### PRINT RESULTS ####

for shot in results:
    with open(outdir+'/ece_{}.pkl'.format(shot), 'wb') as f:
        pickle.dump(results[shot], f)
    print('Data saved for shot {}...'.format(shot))

print('Done!')
