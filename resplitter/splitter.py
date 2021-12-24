import numpy as np
import pandas as pd

from .chain_extractor import ChainExtractor


class CellSplitter:

    def __init__(self):

        self.nb_df = None
        self.extractor = ChainExtractor()

    def __process_cells(self):

        cell_chains = {}

        parsed_cells = [self.extractor.parse_code(cell) for cell in self.nb_df.source.tolist()]
        parsed_cells = [(cell[0],) + cell[1] for cell in zip(self.nb_df.cell_id.tolist(), parsed_cells)]

        for cell_id, module, du, ancestors in parsed_cells:

            if du is None:
                continue

            anc_pair_list = self.extractor.extract_chains(du, ancestors)
            instruction_dict = {instruction: iid for (iid, instruction) in enumerate(module.body)}
            iid_line_mapping = {iid: instruction.lineno for (iid, instruction) in enumerate(module.body)}
            cell_mapping = self.__create_in_cell_chain_mapping(anc_pair_list, instruction_dict, iid_line_mapping)
            cell_chains[cell_id] = cell_mapping

        return cell_chains

    @staticmethod
    def __create_in_cell_chain_mapping(anc_pair_list, instruction_dict, iid_line_mapping):

        df = pd.DataFrame(anc_pair_list, columns=['source', 'target'])
        df = df.replace(instruction_dict).sort_values(['source', 'target'])
        df = df.join(df.replace(iid_line_mapping), rsuffix='_line')
        df = df.reset_index().rename(columns={'index': 'chain_id'})
        df['closenss'] = df.target - df.source

        return df

    @staticmethod
    def discover_chain_seq(cell_mapping):

        # TODO: find where cell_mapping become slice
        df = cell_mapping.copy()
        df = df.loc[df.closenss == 1, :].drop_duplicates(subset=['source', 'target'])
        # Increment if not in singular closeness
        df.loc[:, 'seq_group_id'] = df.loc[:, 'source'] - df.loc[:, 'target'].shift(1)
        df.seq_group_id = df.seq_group_id.bfill()
        # Get seq_id bu counting amount of changes
        df.seq_group_id = np.add.accumulate(df.seq_group_id)

        return df

    # ____SPLIT_MODULE____

    def split_cells(self, nb_df):

        self.nb_df = nb_df.copy()
        cell_mapping = self.__process_cells()
        # groupby here is a hack in order to simplify output of apply operation
        processed_cells = self.nb_df.groupby('cell_id').apply(lambda cell: self.lambda_split(cell, cell_mapping))

        return processed_cells

    def lambda_split(self, blank, cell_mapping):

        result = blank
        blank = blank.squeeze(axis=0)

        merged = ~blank['merged'] if 'merged' in blank else True

        if (blank['cell_type'] != 'markdown') & (blank.cell_id in cell_mapping.keys()) & merged:
            mapping = cell_mapping[blank.cell_id]
            # self.mapping = mapping
            chains = self.discover_chain_seq(mapping)

            chain_sizes = chains.groupby('seq_group_id').apply(len)
            chains = chains.loc[chains.seq_group_id.isin(chain_sizes[chain_sizes > 1].index)]
            # self.chains = chains
            if any(chains.groupby('seq_group_id').apply(len) > 1):
                result = self._split_singular_cell(chains, blank)
                result = pd.concat(result, axis=1).T

        return result

    def _split_singular_cell(self, chains, blank):

        cell_source = blank.source.split('\n')

        seq_lines = chains.groupby('seq_group_id') \
            .agg({'source_line': min, 'target_line': max}) \
            .sort_values(['source_line', 'target_line'])

        # Make sure that sequences are not overlapping
        seq_lines = seq_lines[(seq_lines.target_line - seq_lines.target_line.shift(-1)).fillna(0) <= 0]
        seq_lines.target_line = seq_lines.target_line + 1
        # TODO: This two can be merged

        new_cell_borders = self.create_new_cell_borders(seq_lines, cell_source)

        lines = self.__merge_small_cells(new_cell_borders)
        new_cells = self.__split_lines(lines, cell_source)
        blank = self.__covert_to_nb_format(new_cells, blank)

        return blank

    @staticmethod
    def __split_lines(lines, cell_source):
        # Here we convert from actual lines to lines ids in list
        lines.loc[lines.start_line > 0, 'start_line'] = lines.loc[lines.start_line > 0, 'start_line'] - 1
        lines.start_line = lines.start_line.astype(int)
        lines.loc[:, 'end_line'] = lines.loc[:, 'end_line'].astype(int) - 1

        # and split!
        borders = list(zip(lines.start_line, lines.end_line))
        # Temp fix: need to avoid creating zero splits at previous step
        new_cells = [cell_source[slice(*border)] for border in borders if cell_source[slice(*border)] != []]

        return new_cells

    @staticmethod
    def __covert_to_nb_format(new_cells, blank_base):
        new_cells_df = []
        for cell in new_cells:
            blank = blank_base.copy()
            new_source = "\n".join(cell)

            if cell != new_cells[-1]:
                new_source = new_source + "\n"

            blank.loc['source'] = new_source
            new_cells_df.append(blank)
        # new_cells_df = pd.concat(new_cells_df)
        return new_cells_df

    @staticmethod
    def __merge_small_cells(new_cell_borders):
        lines = pd.DataFrame(new_cell_borders, columns=['start_line', 'end_line'])
        lines['cell_len'] = lines.end_line - lines.start_line

        new_df = pd.DataFrame()
        tmp_df = pd.DataFrame()
        lines = lines.sort_values(['end_line', 'start_line'], ascending=False)
        for index, row in lines.iterrows():
            tmp_df = tmp_df.append(row, sort=True)
            if tmp_df.loc[:, ['cell_len']].sum().values[0] >= 3:
                x = tmp_df.agg({'start_line': min, 'end_line': max, 'cell_len': sum}).to_frame().T
                new_df = new_df.append(x, sort=True)
                tmp_df = pd.DataFrame()
        if len(tmp_df) > 0:
            x = tmp_df.agg({'start_line': min, 'end_line': max, 'cell_len': sum}).to_frame().T
            new_df = new_df.append(x, sort=True)

        lines = new_df.sort_values(['end_line', 'start_line'], ascending=True).copy()
        return lines

    @staticmethod
    def create_new_cell_borders(a, cell_source):
        cell_borders = [0]
        for seq_id in a.index:
            cell_borders.extend(a.loc[seq_id, :].tolist())

        cell_borders = list(dict.fromkeys(cell_borders))
        cell_length = len(cell_source) + 1
        if cell_borders[-1] != cell_length:
            cell_borders.append(cell_length)

        new_cell_borders = [cell_borders[i:i + 2] for i in range(len(cell_borders) - 1)]
        return new_cell_borders
