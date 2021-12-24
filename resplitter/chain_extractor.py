from contextlib import redirect_stdout

import beniget
import gast as ast


class ChainExtractor:

    def __init__(self):
        self.pair_list = []

        self.parsing_error_log = {
            'ValueError': [],
            'SyntaxError': [],
            'AssertionError': [],
            'IndexError': []
        }

    def parse_code(self, code):

        with redirect_stdout(None):
            try:

                module = ast.parse(code)

                du = beniget.DefUseChains()
                du.visit(module)

                ancestors = beniget.Ancestors()
                ancestors.visit(module)

            except ValueError as e:
                self.parsing_error_log['ValueError'].append(e)
                return None, None, None

            except SyntaxError as s:
                self.parsing_error_log['SyntaxError'].append(s)
                return None, None, None

            except AssertionError as a:
                self.parsing_error_log['AssertionError'].append(a)
                return None, None, None

            except IndexError as a:
                self.parsing_error_log['IndexError'].append(a)
                return None, None, None

        return module, du, ancestors

    def extract_chains(self, du, ancestors):
        self.pair_list = []
        for chain in du.chains:
            self.__traverse_chain(du.chains[chain], {}, chain)

        anc_pair_list = [self._node_to_first_ancestor(pair, ancestors) for pair in self.pair_list]

        return anc_pair_list

    def __traverse_chain(self, chain, visited, starting_node):

        if chain.node in visited:
            return None
        else:
            visited[chain.node] = len(visited)
            for node in chain.users():
                if len(node.users()) == 0:
                    self.pair_list.append((starting_node, node.node))
                else:
                    self.__traverse_chain(node, visited, starting_node)

    @staticmethod
    def _node_to_first_ancestor(node_list, ancestors):
        ancestor_list = [ancestors.parents(node)[1] if (len(ancestors.parents(node)) > 1) else node
                         for node in node_list]
        return ancestor_list

    @staticmethod
    def _node_to_cell(node_list, cell_mapping):
        cell_list = [cell_mapping[node] for node in node_list]
        return cell_list
