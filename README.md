# ReSplit

## Requirements

All the requirements are in the `requirements.txt` file. In order to install, please use:

`pip install -r requirements.txt`

##  Replication:

Before replicating the results, you should download the data (`data.zip`), unzip it, and place it in the `data` folder in the repository.

The project has the following structure:

`/data`- data storage  
`/resplitter` - the implementation of the algorithm  
`/experiment` - the code for generating the sample for the user study  
`/jupyter_experiment` - the generated data and the code for hosting the Jupyter server for the experiment. 
Additionally, in `survey.pdf` you can find the form with the questions from the survey.   
`/analysis` - the notebooks for running ReSplit and analysing the results

Currently, one can run ReSplit on the entire dataset of notebooks. To start, you need to create the `NotebokProcessor` object 
and specify where to preform splitting, merging, or both.

    `nbp = NotebookProcessor({'merge': True, 'split': True})`

Then, you can process a dataset using the `process_notebook_dataset` method or 
process a single notebook with the `process_notebook` method. For the first method, you need to provide a path for the dataset.
For the second method, you need to provide the notebook as DataFrame. For both formats, please look into the provided data files. 

