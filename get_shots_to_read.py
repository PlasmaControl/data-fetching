import sys
import h5py
from pathlib import Path
from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np

data_path = Path("/cscratch/mehtav/data.h5")

def get_last_datapoint():
    dataset = h5py.File(data_path, 'r')
    all_dataset_shots = set(dataset.keys())
    dataset.close()
    all_dataset_shots.remove('times')
    shot_num_list = sorted([int(key) for key in all_dataset_shots])
    return shot_num_list[-1]


def main(np_dest, txt_dest):
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

    shot_list = [elem['shot'] for elem in results]
    shots = np.array(shot_list)

    np.save(np_dest, shots)
    with txt_dest.open('w') as f:
        f.writelines([str(shot) + '\n' for shot in shot_list])
    print(f"wrote {len(shot_list)} new shots to {np_dest} and {txt_dest}")

if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
