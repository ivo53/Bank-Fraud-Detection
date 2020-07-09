import numpy as np
import matplotlib.pyplot as plt

import csv

import pandas as pd

import seaborn as sb

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.over_sampling import ADASYN

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score

#

# reading and saving into variables 284 807 instances of credit card transactions in Europe in 2013

# database downloaded from https://www.kaggle.com/mlg-ulb/creditcardfraud

# by Ivo Mihov ( ID 9918179 )

# MPhys student at

# The University of Manchester

# ivo.mihov@protonmail.com

#

# step 1: extracting data

print('...........................\nstep 1: extracting data')

#

# Tried to construct a centering matrix with no success due to the size of the sample (shape (284000, 284000))
def H(n):
	return np.identity(n) - np.true_divide(1., n, dtype='f8') * np.ones((n, n))

def plot2d(x1, x2, a, colour):
	c = colour + 'x'
	a.plot(x1, x2, c)

# first create a dictionary attr object for our variables

attr = {}

file = open('creditcard.csv', 'r')

# counter i for the data extraction loop

i = 0

for line in file:
	split = line.split(",")
	# we get rid of symbols that make our labels look ugly and do not allow to cast data into float
	split = map(lambda each:each.strip("\n"), split)
	split = list(map(lambda each:each.strip('"'), split))
	# in the first iteration we get the names of the attributes to fill the names in the dictionary
	if i == 0:
		i += 1
		# initialise array of strings to save the attribute names
		names = np.empty(len(split), dtype='U25')
		for j in range(len(split)):
			# save the labels in names array
			names[j] = split[j]
			# label the attribute in dictionary and assign it an empty array
			attr[names[j]] = np.empty(284807, dtype='f8')
		# print(names)
		continue
	# for all but first row the below code is executed
	for k in range(len(names)):
		# save data in attr dictionary
		attr[names[k]][i-1] = split[k]
	i += 1

file.close()

# step 2: doing PCA to find variance and eventually exclude some attributes
print('...........................\nstep 2: doing PCA to find variance and eventually exclude some attributes')

data_matr = np.empty((len(attr.keys()) - 1, 284807))

i = 0

for key in attr.keys():
	if i == len(names) - 1:
		class_matr = attr[names[len(names) - 1]][:]
	else:
		data_matr[i][:] = attr[key][:]
	i += 1



# finding the covariance/correlation matrix

# will have to manually subtract the mean from each column so as not to construct a 280000 by 280000 matrix
#cntr = np.asarray(H(284807))
#data_cntrd = np.dot(cntr, data_matr)

# find average of each attribute and subtract from each row 

avg = np.average(data_matr, axis=1)

data_matr -= avg[:, None]

S_unb = np.true_divide(1., (284807. - 1.), dtype='f8') * np.dot(data_matr, data_matr.T)

# check if they are close
#print(np.testing.assert_allclose(S_unb, S1))

'''

tried another method to find correlation matrix, didn't work

'''

D = np.matrix(np.diag(np.power(np.diag(S_unb), -0.5)))

Z = np.matrix(data_matr.T * D)

R = np.matrix(1/float(284807 - 1) * Z.T * Z)

'''
check if correlation matrix is right


R_ = np.corrcoef(data_matr)
print(np.testing.assert_allclose(R, R_))
'''
# finding principal components with weights

values, vectors = np.linalg.eig(R)

sum_weights = np.sum(values)

weights = values / sum_weights



indices_pca = np.argsort(weights)[::-1]

values = values[indices_pca]

vectors = vectors[indices_pca]

weights = weights[indices_pca]

#print(vectors)

# save the correlation matrix into a csv file 

with open('Correlation.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(attr.keys())
	writer.writerows(np.array(R))

print('...........................\nCorrelation matrix saved in current directory.\n')

unitary = vectors[0]

for i in range(len(vectors)):
	if i != 0:
		unitary = np.hstack((unitary, vectors[i]))

unitary = np.array(unitary).reshape((30, 30))

pca_data = np.dot(data_matr.T, unitary)
'''

check where the counterfeit transactions are, so we include a substantial number in train sample

print('Indices of counterfeit transactions: ', np.nonzero(data_matr[len(data_matr)-1] == 1)[1])

print(np.mean(np.nonzero(data_matr[len(data_matr)-1] == 1)[1]))

'''

# divide data into train and test set to validate training

indices_test = np.arange(np.shape(data_matr)[1])
indices_test = indices_test[indices_test % 6 == 0]
indices_train = np.arange(np.shape(data_matr)[1])
indices_train = indices_train[indices_train % 6 != 0]

train_data = np.empty((30, len(indices_train)), dtype='f8')
test_data = np.empty((30, len(indices_test)), dtype='f8')

train_class = np.empty(len(indices_train))
test_class = np.empty(len(indices_test))

pca_train_data = np.empty((len(indices_train), 30), dtype='f8')
pca_test_data = np.empty((len(indices_test), 30), dtype='f8')

train_class[:] = class_matr[indices_train]
test_class[:] = class_matr[indices_test]

train_data[:] = data_matr[:, indices_train]
test_data[:] = data_matr[:, indices_test]

pca_train_data[:] = pca_data[indices_train, :]
pca_test_data[:] = pca_data[indices_test, :]

pca_train_class = np.empty(len(indices_train))
pca_test_class = np.empty(len(indices_test))

pca_train_class[:] = train_class[:]
pca_test_class[:] = test_class[:]


#print(train_data.shape, test_data.shape)

'''

Using the SMOTE class to do synthetic oversampling of the minority class
works by focusing on the minority class only, doing k-means on it to
group it into several minority classes, and then creating new minority classes 
by combining the features of the ones present. New ones lie on the line between their means.
This way all the information is kept and overfitting the minority class data is less of an issue. 

'''
# step 3: oversampling fraud data points to balance dataset
print('...........................\nstep 3: oversampling fraud data points to balance dataset')
sm = SMOTE(sampling_strategy=0.004)

tom = SMOTETomek(smote=sm)
#enn = SMOTEENN(smote=sm)
#train_data, train_class = tom.fit_resample(train_data.T, train_class)
#print(train_data.shape)

pca_train_data, pca_train_class = sm.fit_resample(pca_train_data, pca_train_class)
#train_data, train_class = sm.fit_resample(train_data, train_class)

#print(train_data.shape, train_class.shape)
#print(np.sum(train_class))
# As always, first try fitting a simple model to data and see what its accuracy is to compare

# Decided to fit a simple Linear Regression Model
# step 4: classification
print('...........................\nstep 4: classification')
print('Applying Logistic Regression Model...')

num_frauds_test = np.sum(test_class == 1)
print('number of frauds in test dataset is', num_frauds_test)
'''
#lr = LR(class_weight={0:1, 1:50}, solver='lbfgs', multi_class='ovr', penalty='l2', max_iter=10000, C=0.1).fit(train_data, train_class)

#print(cross_val_score(lr, train_data, train_class, scoring='precision'))

paramLR = lr.get_params()

#print(lr.predict(test_data.reshape(-1, 30)))

#predictionsLogR = lr.predict(test_data.reshape(-1, 30))





print('Accuracy:', accuracy_score(test_class, predictionsLogR))
print('Precision:', precision_score(test_class, predictionsLogR))
print('MCC:', matthews_corrcoef(test_class, predictionsLogR))
print('F1:', f1_score(test_class, predictionsLogR))
print('Confusion matrix (LogR):\n', confusion_matrix(test_class, predictionsLogR))

print('Logistic regression on PCA\'d data...........')
'''
lr2 = LR(class_weight={0:1, 1:1},
		 solver='lbfgs', 
		 multi_class='ovr', 
		 penalty='l2', 
		 max_iter=100000, 
		 C=10).fit(pca_train_data, pca_train_class)

predictionsLogR2 = lr2.predict(pca_test_data.reshape(-1, 30))

print('Accuracy:', accuracy_score(pca_test_class, predictionsLogR2))
print('Precision:', precision_score(pca_test_class, predictionsLogR2))
print('MCC:', matthews_corrcoef(pca_test_class, predictionsLogR2))
print('F1:', f1_score(pca_test_class, predictionsLogR2))
print('Confusion matrix on PCA data (LogR):\n', confusion_matrix(pca_test_class, predictionsLogR2))
#print('R^2 of the fit is ', lr.score(test_data.reshape(-1, 30), test_class))

# The specific problem with the credit card fraud data is that it is imbalanced (only 492 out of 284807 transactions are counterfeit) - 0.17%

# For most clustering algorithms (eg. support vector machines) this creates the problem of always having a bias towards the much more likely class

print('Applying Random Forest Model...')
'''
rf = RF(n_estimators=15, criterion='gini', class_weight={0:20, 1:1}).fit(train_data, train_class)

predictionsRF = rf.predict(test_data.reshape(-1, 30))

#print(predictionsRF)

print('Accuracy:', accuracy_score(test_class, predictionsRF))
print('Precision:', precision_score(test_class, predictionsRF))
print('MCC:', matthews_corrcoef(test_class, predictionsRF))
print('F1:', f1_score(test_class, predictionsRF))
print('Confusion matrix (RF):\n', confusion_matrix(test_class, predictionsRF))

rf2 = RF(n_estimators=20, criterion='gini', class_weight={0:3, 1:1}).fit(pca_train_data, pca_train_class)

predictionsRF2 = rf2.predict(pca_test_data.reshape(-1, 30))
print('Accuracy:', accuracy_score(pca_test_class, predictionsRF2))
print('Precision:', precision_score(pca_test_class, predictionsRF2))
print('MCC:', matthews_corrcoef(pca_test_class, predictionsRF2))
print('F1:', f1_score(pca_test_class, predictionsRF2))
print('Confusion matrix on PCA data (RF):\n', confusion_matrix(pca_test_class, predictionsRF2))
'''