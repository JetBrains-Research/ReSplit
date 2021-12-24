import json

import pandas as pd


def ipynb_to_df(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data['cells'])
    df.source = df.source.apply(lambda x: ''.join(x))
    df = df.reset_index()

    df = df.rename(columns={'index': 'cell_id'})

    return df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
