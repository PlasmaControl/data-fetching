import h5py

filenames=['example_149057_140888.h5', 'example_157705_149058.h5',
           'example_165399_157706.h5', 'example_174042_165400.h5',
           'example_183223_174044.h5', 'example_191450_183224.h5']

with h5py.File('all_shots.h5', 'w') as combined_file:
    for filename in filenames:
        print(f'Starting {filename}')
        with h5py.File(filename, 'r') as individ_file:
            for key in individ_file:
                if key not in combined_file:
                    bytes_key=bytes(key, 'utf-8')
                    h5py.h5o.copy(individ_file.id, bytes_key,
                                  combined_file.id, bytes_key)

