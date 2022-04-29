from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb
import numpy as np

# from https://diii-d.gat.com/DIII-D/software/ml/toksearch/latest/SQL.html
conn = connect_d3drdb()
parameterized_query = """
    select shot
    from shots_type
    where shot_type = 'plasma' and shot > %d
    order by shot desc
    """
minimum_shot = 120000
pipeline = Pipeline.from_sql(conn, parameterized_query, minimum_shot)
results = pipeline.compute_serial()

shots=np.array([elem['shot'] for elem in results])

np.save('data/shots.npy',
        shots)
