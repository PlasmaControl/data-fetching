import os
import argparse
import yaml
import shutil
import numpy as np
from dump_shots import dump_shots

submit_runs=True
nprocesses=16
minimum_shot=189900
maximum_shot=190000
slurm_dir='slurm_test'
baseconfig_filename='configs/example.yaml'

shot_separators=np.linspace(minimum_shot, maximum_shot, nprocesses+1).astype(int)

root_dir=os.path.dirname(os.path.abspath(__file__))
with open(baseconfig_filename,"r") as f:
    cfg=yaml.safe_load(f)
output_filename_base=os.path.splitext(cfg['logistics']['output_file'])[0]
try:
    os.mkdir(slurm_dir)
except FileExistsError as e:
    raise FileExistsError(f'slurm_dir="{slurm_dir}" exists, either delete the directory or change slurm_dir to a different directory') from None

#keep copy of config file without ensemble_number for testing
shutil.copyfile(baseconfig_filename, os.path.join(slurm_dir,f'.yaml'))
for ensemble_number in range(nprocesses):
    appended_text=f"{shot_separators[ensemble_number]}_{shot_separators[ensemble_number+1]}"
    config_filename=os.path.join(slurm_dir,f'config{appended_text}.yaml')
    shots_filename=os.path.join(slurm_dir,f'shots{appended_text}.npy')
    dump_shots(shot_separators[ensemble_number], shot_separators[ensemble_number+1], shots_filename)
    cfg['data']['shots']=shots_filename
    cfg['logistics']['output_file']=output_filename_base+f'{appended_text}.h5'
    with open(config_filename,"w") as f:
        yaml.dump(cfg,f)
    log_filename=os.path.join(slurm_dir,f'log{appended_text}.out')
    slurm_text=f'''#!/bin/bash
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output {log_filename}
#SBATCH --time 24:00:00

root_dir={root_dir}
module load toksearch
cd $root_dir
python -u new_database_maker.py {config_filename}

exit'''
    slurm_filename=os.path.join(slurm_dir,f'job{appended_text}.slurm')
    with open(slurm_filename,'w') as f:
        f.write(slurm_text)
    if submit_runs:
        os.system(f'sbatch {slurm_filename}')
