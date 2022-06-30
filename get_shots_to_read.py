import h5py
from pathlib import Path
from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np

data_path = Path("/cscratch/mehtav/data.h5")

def get_last_datapoint():
    f = h5py.File(data_path, 'r')
    all_dataset_shots = set(dataset.keys())
    all_dataset_shots.remove('times')
    shot_num_list = sorted([int(key) for key in all_dataset_shots])
    return shot_num_list[-1]


def main(most_recent_shot, dest):
    last_datapoint = get_last_datapoint()
    # from https://diii-d.gat.com/DIII-D/software/ml/toksearch/latest/SQL.html
    conn = connect_d3drdb()
    parameterized_query = """
        select shot
        from shots_type
        where shot_type = 'plasma' and shot > %d
        order by shot desc
        """
    pipeline = Pipeline.from_sql(conn, parameterized_query, last_datapoint)
    results = pipeline.compute_serial()

    shots=np.array([elem['shot'] for elem in results])

    np.save('data/shots.npy',
            shots)

if __name__ == '__main__':
    main(int(sys.argv[1]), Path(sys.argv[2]))
