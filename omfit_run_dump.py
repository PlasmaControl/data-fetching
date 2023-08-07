"""
module load omfit
omfit run_dump.py

(or copy/paste into an OMFIT command window and run from there)
"""

import h5py
import numpy as np
import time

filename='/cscratch/abbatej/run_dump.h5'
min_shot=140000
max_shot=196512
overwrite_runs=True

# splitting up into batches prob not necessary in retrospect...
# the thought was to not have too much stuff in RAM at a time
max_shots_per_run=10000

text_sigs=['text','topic','username']
run_sigs=['brief']
run_sig_string=','.join(run_sigs)
text_string=','.join(text_sigs)

shot_dividers=list(reversed(np.arange(min_shot,max_shot,max_shots_per_run)))

prev_time=time.time()
for i in range(len(shot_dividers)-1):
    prev_time=time.time()
    this_max_shot=shot_dividers[i]
    this_min_shot=shot_dividers[i+1]
    print(f"Starting {this_max_shot}-{this_min_shot}")
    sig_info=OMFITrdb(f"""SELECT DISTINCT UPPER(run) as run
    FROM shots
    WHERE shot>={this_min_shot} AND shot<{this_max_shot}""",db='d3drdb',server='d3drdb',by_column=True)
    runs=sig_info['run']
    run_string='({})'.format(','.join(["'"+run+"'" for run in runs]))
    tmp_dic={run: {sig: [] for sig in text_sigs} for run in runs}

    sig_info=OMFITrdb(f"""SELECT UPPER(run) as run,{run_sig_string}
    FROM runs
    WHERE run in {run_string}""",db='d3drdb',server='d3drdb',by_column=True)
    for i in range(len(sig_info['run'])):
        for sig in run_sigs:
            tmp_dic[str(sig_info['run'][i])][sig]=str(sig_info[sig][i])

    sig_info=OMFITrdb(f"""SELECT UPPER(run) as run,{text_string}
    FROM entries
    WHERE run in {run_string} AND shot IS NULL""",db='d3drdb',server='d3drdb',by_column=True)
    for i in range(len(sig_info['run'])):
        for sig in text_sigs:
            tmp_dic[str(sig_info['run'][i])][sig].append(str(sig_info[sig][i]))

    with h5py.File(filename, 'a') as f:
        for run in tmp_dic:
            if overwrite_runs and run in f:
                del f[run]
            else:
                f.require_group(run)
                for sig in tmp_dic[run]:
                    f[run][sig]=tmp_dic[run][sig]
    print(f'Took {time.time()-prev_time:.2f}s')
