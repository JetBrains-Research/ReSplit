import pandas as pd
import os
import re


def merge_chunked_data(dataset, path='../dump2', save_path='../data', duplicates='../data/duplicates_id.csv'):
    chunks = os.listdir(f'{path}/{dataset}')

    df_list = [pd.read_csv(f'{path}/{dataset}/{chunk}') for chunk in chunks]
    df = pd.concat(df_list)

    cols_to_remove = [c for c in df.columns if re.search('Unnamed: ', c)]
    df = df.drop(columns=cols_to_remove)

    duplicates = pd.read_csv(duplicates)

    df = df[~df.notebook_id.isin(duplicates.dupl_id)]

    df.to_csv(f'{save_path}/{dataset}.csv')

    # return df
