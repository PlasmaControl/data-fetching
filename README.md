To run on iris:

`module load toksearch`

and pip install h5py==3.6.0 the first time you run.

For the basic case: 

In dump_shots.py edit the min_shots and max_shots to be whatever range you want, for testing use e.g. 163300 to 163310. Then run it to dump the shots you want to collect (but only those with plasma; and you can also edit to only take from certain run days in the file) via

`python dump_shots.py`

Then collect the signals from those shots. By default, path_to_config above will be configs/example.yaml which has all the signals, draws from data/shots.npy (which dump_shots.py dumps to), and dumps to output_file (MAKE SURE TO EDIT THIS). You can take out signals, e.g. for testing
- set sql_sig_names, scalar_sig_names, stability_sig_names, nb_sig_names, efit_profile_sig_names, efit_scalar_sig_names, thomson_sig_names, zipfit_sig_names to an empty array
- set include_radiation, include_full_ech_data, include_full_nb_data, include_gas_valve_info, include_log_info to False
Increase max_shots_per_run, this controls how often it checkpoints and spits out how long each batch takes; num_processes greater than 1 should automatically work on Saga and parallelizes.

When the config file is ready, run

`python new_database_maker.py path_to_config`

For large runs, use `launch_parallel_jobs.py` which manually dumps shots and splits into cases to run in parallel (toksearch theoretically can do this under the hood but in my experience it doesn't speed stuff up and is not robust). You can modify `combine_shots.py` to combine the h5 files it dumps into one. 
