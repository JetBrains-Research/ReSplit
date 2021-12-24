import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats

from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data = pd.read_json("train.json")

data.features[10000]

data.shape

data.head(10)

import matplotlib.pyplot as plt
import pandas as pd


df = data[['bathrooms','price','interest_level']]

fig, ax = plt.subplots()

colors = {'low':'green', 'medium':'blue', 'high':'red'}

ax.scatter(df['bathrooms'], df['price'], c=df['interest_level'].apply(lambda x: colors[x]))
plt.ylim(ymax = 150000, ymin = -1000)

plt.show()


import matplotlib.pyplot as plt
import pandas as pd


df = data[['bedrooms','price','interest_level']]

fig, ax = plt.subplots()

colors = {'low':'green', 'medium':'blue', 'high':'red'}

ax.scatter(df['bedrooms'], df['price'], c=df['interest_level'].apply(lambda x: colors[x]))
plt.ylim(ymax = 150000, ymin = -1000)

plt.show()

import seaborn as sns
sns.boxplot(x="interest_level", y=np.log(data["price"]), data=data)
plt.ylim(ymax = 12, ymin = 3)

import seaborn as sns
sns.boxplot(x="interest_level", y="bedrooms", data=data)

df1 = data[['bedrooms','bathrooms']]
df1.hist()

df2 = np.log(data['price'])
df2.hist()

from sklearn import svm
from sklearn.model_selection import train_test_split
X = data[['price','bedrooms','bathrooms']]
y = data["interest_level"]
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train) 





