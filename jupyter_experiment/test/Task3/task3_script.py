from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.read_csv('wine_original.csv')
labels = data['class']
del data['class']


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

X_train


gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_train_pred = gnb.predict(X_train)


print ('Training accuracy = ' + str(np.sum(y_train_pred == y_train)/len(y_train)))
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))

X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2, random_state=5)

alphas = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 100]
best_alpha = 0.1
best_acc = 0.0

for alpha in alphas:
    
    clf = MultinomialNB(alpha=alpha)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_valid)
    accuracy = np.sum(y_pred == y_valid)/len(y_valid)
    print ('Validation accuracy = ' + str(accuracy) + ' at alpha = ' + str(alpha))
    if accuracy > best_acc:
        best_acc = accuracy
        best_alpha = alpha

print ('Best alpha = ' + str(best_alpha))        

X_train = np.concatenate((X_train, X_valid))
y_train = np.concatenate((y_train, y_valid))

clf = MultinomialNB(alpha=best_alpha)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)


print ('Training accuracy = ' + str(np.sum(y_train_pred == y_train)/len(y_train)))
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))





from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)


clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)


print ('Training accuracy = ' + str(np.sum(y_train_pred == y_train)/len(y_train)))
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))











