import h5py
from pathlib import Path

data_path = Path("/cscratch/mehtav/data.h5")

def get_last_datapoint():
    f = h5py.File(data_path, 'r')
    all_dataset_shots = set(dataset.keys())
    all_dataset_shots.remove('times')
    shot_num_list = sorted([int(key) for key in all_dataset_shots])
    return shot_num_list[-1]


def main(most_recent_shot, dest):
    last_datapoint = get_last_datapoint()
    shots_to_get = np.array(list(range(last_datapoint + 1, most_recent_shot + 1))).astype(int)
    # write these to dest in a numpy file
    np.save(dest, shots_to_get)

if __name__ == '__main__':
    main(int(sys.argv[1]), Path(sys.argv[2]))
