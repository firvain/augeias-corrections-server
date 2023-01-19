import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import tensorflow as tf

POSTGRESQL_URL = 'postgresql://augeias:augeias@83.212.19.17:5432/augeias'


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def to_sql_date(start_date):
    return start_date.strftime('%Y-%m-%d %H:%M:%S')


def get_data(table, limit=None, start_date=None, end_date=None):
    if limit is None or start_date is None or end_date is None:
        sql = f"""select * from "{table}" order by timestamp """
    else:
        print(f"Getting data from {start_date} to {end_date}")
        start_date = to_sql_date(start_date)
        sql = f"""select * from "{table}" where timestamp between '{start_date}' and '{end_date}' order by timestamp desc limit {limit}"""

    engine = create_engine(POSTGRESQL_URL)
    try:
        data = pd.read_sql(sql, engine)
        data = data.set_index('timestamp')
        return data
    except Exception as e:
        print(e)
        return None
