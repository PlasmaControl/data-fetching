import pickle
from toksearch.sql.mssql import connect_d3drdb

def main(test_shot_path, out_path):
    with open(test_shot_path, 'r') as f:
        test_shots = [s.strip() for s in f.readlines()]

    # get all run numbers for test shots
    runs = set()
    conn = connect_d3drdb()
    query = f"""SELECT run
                FROM shots
                WHERE shot IN ({','.join(test_shots)})"""





