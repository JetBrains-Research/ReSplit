import sklearn
import pandas as pd
import numpy as np

df = pd.read_csv('./sentiment/all_data.csv', header = 0)

df.columns

del df['Unnamed: 0']

df.head()

print(df.describe())

import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

df.boxplot()

df.hist()



