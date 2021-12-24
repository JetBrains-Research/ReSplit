from Code2Nb import Code2Nb
import os
import numpy as np
import pandas as pd
import json
from pyminifier.minification import remove_comments_and_docstrings


def prepare_experiment(amount=5, data_path='../data', path='../../jupyter_experiment/test'):
    changes = pd.read_csv(f'{data_path}/changes.csv')

    changes = changes[(changes['orig_nb_len'] > 5) & (changes['orig_nb_len'] < 15)]
    changes = changes[(changes['percent'] > 0.3) & (changes['percent'] < 1.5)]

    ids_for_experiments = changes.sample(amount).loc[:, 'notebook_id'].tolist()

    random_mapping = np.random.choice(['A.B', 'B.A'], amount)
    random_mapping = [i.split('.') for i in random_mapping]

    orig = pd.read_csv(f'{data_path}/orig.csv')
    complete = pd.read_csv(f'{data_path}/complete.csv')

    nbconverter = Code2Nb()
    for num, nb_id in enumerate(ids_for_experiments):

        if not os.path.exists(f'{path}/Task{num}'):
            os.mkdir(f'{path}/Task{num}')

        script = orig.loc[orig.notebook_id == nb_id, 'source'].tolist()

        with open(f'{path}/Task{num}/script.py', 'w') as outfile:
            for ele in script:
                ele = remove_comments_and_docstrings(ele)
                outfile.write(ele + '\n')

        test = orig.loc[orig.notebook_id == nb_id, 'source'].tolist()
        test = [('code', i) for i in test]
        nbconverter.convert(test, path=f'{path}/Task{num}/notebook_{random_mapping[num][0]}.ipynb')

        test = complete.loc[complete.notebook_id == nb_id, 'source'].tolist()
        test = [('code', i) for i in test]
        nbconverter.convert(test, path=f'{path}/Task{num}/notebook_{random_mapping[num][1]}.ipynb')

        with open(f'{path[:-5]}/mapping.txt', 'w') as outfile:
            json.dump(random_mapping, outfile)

    return None  # random_mapping
