import pandas as pd

from .chain_extractor import ChainExtractor


class CellMerger:

    def __init__(self):

        self.nb_df = None
        self.nb_df_markdown = None
        self.extractor = ChainExtractor()

        self.inter_intra_change = 0.1
        self.length_cap = 10

    def merge_cells(self, nb_df):

        self.nb_df = nb_df[nb_df.cell_type == 'code']
        self.nb_df_markdown = nb_df[nb_df.cell_type != 'code']

        cell_list, line_list = self.__process_notebook()

        if cell_list:

            merge_stats = self._calculate_stats_for_merge(cell_list, line_list)
            merge_candidates = self._get_merge_candidates(merge_stats)
            merged_notebook = self._merge_cells(merge_candidates)

        else:
            # print('Merge failed: code is incorrect')
            merged_notebook = nb_df

        return merged_notebook

    def __process_notebook(self):

        strings_df = self.nb_df['source'].str.count('\n') + 1
        strings_df = strings_df.cumsum().to_frame('cell_ends').join(self.nb_df['cell_id'])

        code = '\n'.join(self.nb_df.source.tolist())
        module, du, ancestors = self.extractor.parse_code(code)

        if module is None:
            return None, None

        expression_line_mapping = {expression: expression.lineno for expression in module.body}
        expression_cell_mapping = {
            expression: strings_df.loc[strings_df.cell_ends >= expression.lineno, 'cell_id'].iloc[0] for expression in
            module.body}

        anc_pair_list = self.extractor.extract_chains(du, ancestors)

        cell_list = [self.extractor._node_to_cell(pair, expression_cell_mapping) for pair in anc_pair_list]
        line_list = [self.extractor._node_to_cell(pair, expression_line_mapping) for pair in anc_pair_list]

        return cell_list, line_list

    def _get_merge_candidates(self, cell_stats):

        cell_stats['no_output'] = True

        cell_stats['change_type'] = abs(
            cell_stats.intralink_ratio - cell_stats.merged_intralink_ratio) < self.inter_intra_change
        cell_stats['summed_length'] = cell_stats.cell_lines < self.length_cap / 2  # + cell_stats.cell_lines.shift(1)

        cell_stats[
            'merge_candidates'] = cell_stats.no_output & cell_stats.change_type & cell_stats.summed_length

        notebook_representation = self.nb_df.merge(cell_stats, left_on='cell_id',
                                                   right_on='cell_source', how='left').drop(columns=['cell_source'])
        notebook_representation.merge_candidates = notebook_representation.merge_candidates.fillna(False)

        # Add markdown and avoid merging with markdown cells
        # TODO: probably need to be included after merge in order to simplify.
        #  However in this way we could guaranty that we will not merge anything split by markdown
        notebook_representation = pd.concat([notebook_representation, self.nb_df_markdown]).sort_values('cell_id')
        next_cell_after_merge = notebook_representation.merge_candidates.shift(1).isna()
        notebook_representation.loc[next_cell_after_merge, "merge_candidates"] = False
        notebook_representation.merge_candidates = notebook_representation.merge_candidates.fillna(False)

        return notebook_representation

    def _calculate_stats_for_merge(self, cell_list, line_list):

        df = pd.concat([pd.DataFrame(cell_list), pd.DataFrame(line_list)], axis=1).reset_index()
        df.columns = ['expression_id', 'cell_source', 'cell_target', 'line_source', 'line_target']

        df['intracell_link'] = df.cell_source != df.cell_target
        df['intercell_link'] = df.cell_source == df.cell_target

        cell_stats = pd.concat([df.groupby('cell_source').intracell_link.apply(sum),
                                df.groupby('cell_source').intercell_link.apply(sum),
                                df.groupby('cell_source').expression_id.apply(len)], axis=1)

        cell_stats['intralink_ratio'] = cell_stats.intracell_link / cell_stats.expression_id
        cell_stats['interlink_ratio'] = cell_stats.intercell_link / cell_stats.expression_id
        cell_stats = cell_stats.reset_index()

        merged = cell_stats + cell_stats.shift(1)
        cell_stats['merged_intralink_ratio'] = merged.intracell_link / merged.expression_id
        cell_stats['merged_interlink_ratio'] = merged.intercell_link / merged.expression_id

        cell_stats = cell_stats.merge(self.nb_df.loc[:, ['cell_id', 'cell_lines']],
                                      left_on='cell_source', right_on='cell_id') \
            .drop(columns=['cell_id'])

        # cell_stats['cell_lines']

        return cell_stats

    @staticmethod
    # TODO: review this function - it is not so very ugly
    def _merge_cells(merger_df):

        merger_df['merged'] = False
        merger_df['merge_id'] = (~merger_df.loc[:, ['merge_candidates']]).cumsum()
        merger_df.loc[~(merger_df.merge_candidates | merger_df.merge_candidates.shift(-1)), 'merge_id'] = -1

        joined_source = merger_df[merger_df.merge_id > 0].groupby('merge_id').agg({"source": ''.join,
                                                                                   'notebook_id': min,
                                                                                   'repository_id': min,
                                                                                   'intercell_link': sum,
                                                                                   'intracell_link': sum,
                                                                                   'intralink_ratio': 'mean',
                                                                                   'interlink_ratio': 'mean', }).reset_index()

        joined_source[
            'resulted_intercell_ratio'] = joined_source.intercell_link / (
                joined_source.intercell_link + joined_source.intracell_link)
        joined_source = joined_source.merge(merger_df[['cell_id', 'merge_id']]) \
            .sort_values('cell_id', ascending=False) \
            .drop_duplicates(subset=['merge_id'])

        joined_source['merged'] = True
        joined_source['cell_type'] = 'code'
        merger_df = pd.concat([merger_df[~merger_df.merge_id.isin(joined_source.merge_id)], joined_source]).sort_values(
            'cell_id')

        return merger_df
