from toksearch import Pipeline, PtDataSignal
import numpy as np
import pickle

signames=['ftssmhat','ftsspsin','ftssrot']
extra_sigs=['ftssn']
pipeline=Pipeline([936901])
for sig in signames:
    for i in range(1,76):
        signame='{}{}'.format(sig,i)
        signal=PtDataSignal(signame)
        pipeline.fetch(signame,signal)
for sig in extra_sigs:
    pipeline.fetch(sig,signal)

'''
@pipeline.map
def appending(record):
    record['time']=record['ftssn']['times']
    for sig in signames:
        record[sig]=[]
        for t_ind in range(len(record['time'])):
            tmp=[]
            for i in record['ftssn']['data'][t_ind]: #range(1,76):
                tmp.append(record['{}{}'.format(sig,i)]['data'][t_ind])
            record[sig].append(tmp)
        #record[sig]=np.array(record[sig])
'''
#pipeline.keep([sig for sig in signames]+[sig for sig in extra_sigs])
records = pipeline.compute_serial()
#print(records[0]['ftssmhat'])
#import pdb; pdb.set_trace()

#final_data={}
#for signame in signames: #+extra_sigs:
#    final_data[signame]=records[0][signame]

final_data={}
for sig in signames:
    for i in range(1,76):
        signame='{}{}'.format(sig,i)
        final_data[signame]=records[0][signame]['data']
with open('/cscratch/abbatej/for_rory/mhat_stuff.pkl','wb') as f:
    pickle.dump(final_data,f)
