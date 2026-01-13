Repository archived by abarilo as no longer mantained

# ReSplit

An algorithm that re-splits Jupyter notebooks by merging and splitting their cells.

## Repository structure

The repository has the following structure:

`/resplitter/` — the implementation of **ReSplit**. 

`/data/`— the storage for the datasets.

`/analysis/` — the notebooks for running **ReSplit** and analysing the results.

`/experiment/` — the code for generating the sample for the user study that we used to evaluate the algorithm.

`/jupyter_experiment/` — the generated data and the code for hosting the Jupyter server for the experiment. 
Additionally, in `survey.pdf` you can find the form with the questions from the survey.   

## How to use

1. Install the requirements by running

    `pip install -r requirements.txt`

2. Download the data from [Zenodo](https://zenodo.org/record/5803345), unzip it, and place it in the `data` folder in the repository.

3. Currently, one can run **ReSplit** on the entire dataset of notebooks. To start, you need to create the `NotebokProcessor` object 
and specify whеther to preform splitting, merging, or both.

    `nbp = NotebookProcessor({'merge': True, 'split': True})`

4. Then, you can process a dataset using the `process_notebook_dataset` method or 
process a single notebook with the `process_notebook` method. For the first method, you need to provide a path for the dataset.
For the second method, you need to provide the notebook as DataFrame. For both formats, please look into the provided data files. 

## How this works

**ReSplit** uses the definition-usage chain analysis of the code to find candidates to merge or split,
as well as a number of heuristics to select the best among them. To see find out more about how the
algorithm works, please refer to our [SANER'22 paper](https://github.com/JetBrains-Research/ReSplit).

## Examples

**TBD: we will add a number of examples here to highlight the operation of ReSplit.**

## Contacts

If you have any questions or suggestions about the work, feel free to create an issue
or contacnt Sergey Titov at [sergey.titov@jetbrains.com](mailto:sergey.titov@jetbrains.com).
