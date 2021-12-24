import pandas as pd
from tqdm import tqdm
import difflib
import os
from pandarallel import pandarallel

from .chain_extractor import ChainExtractor
from .merger import CellMerger
from .splitter import CellSplitter
from .utils import chunks


class NotebookProcessor:

    def __init__(self, config=None):
        self.extractor = ChainExtractor()
        self.merger = CellMerger()
        self.splitter = CellSplitter()

        self.config = config if config else {'merge': True, 'split': True}

    @staticmethod
    def _calc_changes(nb_df, merged_df, split_df):

        merge_diff = len(nb_df) - len(merged_df)
        split_diff = len(split_df) - len(merged_df)

        diff = {'additions': split_diff, 'deletions': merge_diff}

        return diff

    @staticmethod
    def _prepare_df(nb_df):

        if 'cell_lines' not in nb_df.columns:
            nb_df['cell_lines'] = nb_df.source.str.count('\n')
            nb_df.loc[nb_df.source.str.len() > 0, 'cell_lines'] = nb_df.loc[
                                                                      nb_df.source.str.len() > 0, 'cell_lines'] + 1

        # assign in order to avoid copy warning
        # nb_df = nb_df.assign(source=nb_df.loc[:, 'source'].str.rstrip("\n"))

        return nb_df

    def _compare_notebooks(self, original, merged_df, split_df):

        diff = self._calc_changes(original, merged_df, split_df)

        if (len(original) - diff['deletions'] + diff['additions']) != (len(split_df)):
            print('Incorrect amount of cell in final ')

        code_diff = list(difflib.ndiff("".join(original.source.tolist()), "".join(split_df.source.tolist())))

        for num, i in enumerate(code_diff):
            if (i[0] != ' ') & (i[2] != '\n'):
                print(f"Unexpected change in notebook {original.notebook_id.values[0]}:{i}, {num}")

    def process_notebook(self, nb_df, config=None):

        config = self.config if config is None else config
        try:
            nb_df = self._prepare_df(nb_df)

            merged_df = self.merger.merge_cells(nb_df) if config['merge'] else nb_df
            split_df = self.splitter.split_cells(merged_df) if config['split'] else merged_df

            self._compare_notebooks(nb_df, merged_df, split_df)

            if isinstance(split_df.index, pd.MultiIndex):
                split_df = split_df.reset_index(drop=True)
        except RecursionError as e:
            print(nb_df)

        return split_df

    def process_notebook_dataset(self, path='../../jupyter_data/dump/sklearn_full_cells.csv', save_path='../dump2/',
                                 nb_num=None, overwrite=False,
                                 chunk_size=1000):

        df = pd.read_csv(path, nrows=nb_num)
        df.source = df.source.astype(str)
        df = df.rename(columns={'index': 'cell_id', 'lines': 'cell_lines'}).sort_values(['notebook_id', 'cell_id'])
        notebook_list = df.notebook_id.unique().tolist()

        chunkify = list(enumerate(list(chunks(notebook_list, chunk_size))))

        pandarallel.initialize(nb_workers=8)
        calculated_chunks = os.listdir(save_path)

        for num, chunk in tqdm(chunkify):
            if ((f'chunk_{num}.fth' not in calculated_chunks) or overwrite) and nb_num is None:
                part = df[df.notebook_id.isin(chunk)]
                part = part.groupby('notebook_id').parallel_apply(self.process_notebook)
                part.to_csv(f'{save_path}/chunk_{num}.fth')
            else:
                print(f'Chunk {num} located, skipping')
                continue

        return df
