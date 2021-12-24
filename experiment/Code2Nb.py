import json
from copy import deepcopy
from pyminifier.minification import remove_comments_and_docstrings


class Code2Nb:

    def __init__(self, blank_notebook_path='blank.ipynb'):
        self.blank_cell = {'cell_type': 'code',
                           'execution_count': None,
                           'metadata': {'collapsed': True,
                                        "editable": False,
                                        "jupyter": {"outputs_hidden": False}},
                           'outputs': [],
                           'source': ['']}

        with open(blank_notebook_path) as json_file:
            self.blank = json.load(json_file)

    def convert(self, cells_data, path='test.ipynb'):
        notebook = self.blank
        notebook['cells'] = [self.__generate_cell(cell_data) for cell_data in cells_data]

        with open(path, 'w') as json_file:
            json.dump(notebook, json_file)

    def __generate_cell(self, cell_data):
        cell = deepcopy(self.blank_cell)
        cell['cell_type'] = cell_data[0]
        cell['source'][0] = remove_comments_and_docstrings(cell_data[1]) if cell_data[0] == 'code' else cell_data[1]

        return cell
