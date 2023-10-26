from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np
# from https://diii-d.gat.com/DIII-D/software/ml/toksearch/latest/SQL.html

def dump_shots(minimum_shot, maximum_shot, save_filename):
    # for getting all plasma shots past a certain shot number
    print(f'Getting all plasma shots between {minimum_shot} and {maximum_shot}')
    if True:
        query = """
            select shot
            from shots_type
            where shot_type = 'plasma' and shot > {} and shot < {}
            order by shot desc
            """.format(minimum_shot-1,maximum_shot)
    # for pulling shots by experiment, e.g. these are high-qmin shots
    else:
        runs=['20191009A', '20190716', '20130510', '20120813', '20120720', '20120720A',
              '20111107', '20100322', '20100325', '20091214', '20090316', '20080509',
              '20020311', '20020227', '20020226A', '20020207', '20010515', '20010424A',
              '20010420', '20010418', '20010319']
        query = """
            select shot
            from shots
            where run in {}
            order by shot desc
            """.format(
            "({})".format(','.join([f"'{str(elem)}'" for elem in runs]))
            )

    conn = connect_d3drdb()
    pipeline = Pipeline.from_sql(conn, query)
    results = pipeline.compute_serial()

    shots=np.array([elem['shot'] for elem in results])

    if len(shots):
        print(f"    {len(shots)} shots, within {min(shots)}-{max(shots)}")
    else:
        print(f"    No shots in this interval")
    np.save(save_filename,
            shots)

if __name__ == "__main__":
    dump_shots(minimum_shot=140000,
               maximum_shot=194248,
               save_filename='shots.npy')
