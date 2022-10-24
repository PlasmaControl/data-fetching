import pickle
from toksearch.sql.mssql import connect_d3drdb
from toksearch import Pipeline

def main(test_shot_path, out_path):
    with open(test_shot_path, 'r') as f:
        test_shots = set([s.strip() for s in f.readlines()])

    # get all run numbers for test shots
    runs = set()
    conn = connect_d3drdb()
    query = f"""SELECT run
                FROM shots
                WHERE shot IN ({','.join(test_shots)})"""

    pipeline = Pipeline.from_sql(conn, query)

    records = pipeline.compute_serial()
    runs = set(records)

    query2 = f"""SELECT shot, run
                 FROM shots
                 WHERE run IN ({','.join(runs)})"""
    pipeline = Pipeline.from_sql(conn, query2)
    records = pipeline.compute_serial()
    control_shots = []
    for shot, run in records:
        #check if not test shot, add to list
        if shot not in test_shots:
            control_shots.append(shot)

    with out_path.open('w') as f:
        f.write('\n'.join(control_shots))
        f.write('\n')
