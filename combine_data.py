# mostly used for folding in new signals to our big dataset used for training
# if overwriting signals, higher priority goes to the right in the list of filenames
# note that bigger files should go on left (and use overwrite if desired) so that
#    there are fewer individual signal copies (and more full-shot copies)
# you could do this in place by replacing 'combined_data.h5' with 'big.h5'
#    and only having 'small.h5' in the list, but this is more careful

import h5py

filenames=['big.h5', 'small.h5']
overwrite_signals=True

special_sigs=['times', 'spatial_coordinates']
with h5py.File('combined_data.h5', 'a') as combined_file:
    for filename in filenames:
        print(f'Starting {filename}')
        with h5py.File(filename, 'r') as individ_file:
            shots=list(individ_file.keys())
            for sig in special_sigs:
                shots.remove(sig)
                if sig not in combined_file:
                    combined_file[sig]=individ_file[sig][()]
            for shot in shots:
                if shot not in combined_file:
                    bytes_shot=bytes(shot, 'utf-8')
                    h5py.h5o.copy(individ_file.id, bytes_shot,
                                  combined_file.id, bytes_shot)
                else:
                    for sig in individ_file[shot]:
                        if sig not in combined_file[shot]:
                            combined_file[shot][sig]=individ_file[shot][sig][()]
                        elif overwrite_signals:
                            combined_file[shot][sig][()]=individ_file[shot][sig][()]
