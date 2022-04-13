import os
import math
import numpy as np

all_shots=np.load('/home/abbatej/tokamak-transport/data/2021_shots.npy')
# 15337 shots, about 2.85MB per shot; so 1750 gives 9 chunks of 5GB each
max_shots_per_file=1750

num_files=math.ceil(len(all_shots)/max_shots_per_file)
for i in [0]: #range(num_files):
    shot_start_index=i*max_shots_per_file
    shot_end_index=min(len(all_shots)-1,(i+1)*max_shots_per_file)
    print(f'Starting {all_shots[shot_start_index]}-{all_shots[shot_end_index]}')
    os.system('omfit nb_grab.py '\
                  f'batch_index={i} '\
                  f'shot_start_index={shot_start_index} '\
                  f'shot_end_index={shot_end_index}')
