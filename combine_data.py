import h5py

filenames=['hdf5s/example_small.h5', 'hdf5s/example1.h5']

with h5py.File('hdf5s/all_shots.h5', 'w') as combined_file:
    for filename in filenames:
        print(f'Starting {filename}')
        with h5py.File(filename, 'r') as individ_file:
            for key in individ_file:
                if key not in combined_file:
                    bytes_key=bytes(key, 'utf-8')
                    h5py.h5o.copy(individ_file.id, bytes_key,
                                  combined_file.id, bytes_key)
                else:
                    for sig in individ_file[key]:
                        if sig not in combined_file[key]:
                            combined_file[key][sig]=individ_file[key][sig][()]