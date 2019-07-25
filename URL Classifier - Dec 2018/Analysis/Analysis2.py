import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from random import sample
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Load required files

with open('./ObjectFiles/data_csrTRAIN.obj', 'rb') as pickle_file:
    data_CSR = pickle.load(pickle_file)

with open('./ObjectFiles/labelTRAIN.obj', 'rb') as pickle_file:
    label = pickle.load(pickle_file)

with open('./ObjectFiles/categorical_cols.obj', 'rb') as pickle_file:
    categorical_cols = pickle.load(pickle_file)

with open('./ObjectFiles/numerical_cols.obj', 'rb') as pickle_file:
    numerical_cols = pickle.load(pickle_file)

#Indices gathered using a sorted keepcols array
with open('./ObjectFiles/categorical_cols_index.obj', 'rb') as pickle_file:
    categorical_cols_index = pickle.load(pickle_file)

with open('./ObjectFiles/keepcols.obj', 'rb') as pickle_file:
    keepcols = pickle.load(pickle_file)



#Define Test set
def get_data(fname):
    data = load_svmlight_file(fname)
    print (fname + " loaded successfully.")
    return data[0], data[1]

#Needed so indices line up with training
def featureSelectandEncode(csr_input):
    # Remove noninformative columns using variance based feature selection (not size dependent)
    csr_input = csr_input[:,keepcols]

    #One hot encode the categorical values
    onehotencoder = OneHotEncoder(categorical_features=categorical_cols_index)
    csr_input = onehotencoder.fit_transform(csr_input.todense())

    onehotencoder = None

    return csr_matrix(csr_input)

#use the last day for testing purposes
data_CSR_TEST, label_TEST = get_data("url_svmlight/Day120.svm")
data_CSR_TEST = featureSelectandEncode(data_CSR_TEST)

data_CSR = None
label = None

X_train = data_CSR
Y_train = label
X_test = data_CSR_TEST
Y_test = label_TEST

'''

#PCA Visualization

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train.toarray())
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
label = pd.DataFrame({'Col 1': Y_train})
finalDf = pd.concat([principalDf, label], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,-1]
colors = ['r', 'b']
print(finalDf)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Col 1'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

'''

# Training models

#Can run different classifiers here where num_blocks = number of built classifiers in bagged method
#Note: if you don't want an ensemble, set this to 1
num_blocks = 1

#Can change the nnumber of bootstrapped samples in each training set
num_of_samples = 200

ran_num = [random.randint(0, num_of_samples) for x in range(len(Y_train))]

models = [RandomForestClassifier(n_estimators=200, max_depth=2,random_state=0) for i in range(num_blocks)]
for i in range(num_blocks):
    models[i].fit(X_train[[index for index, value in enumerate(ran_num) if value == i], :],
                  Y_train[[index for index, value in enumerate(ran_num) if value == i]])
    print("Model " + str(i + 1) + " done.")
ran_num = None

# Predicting
predicted_test = models[0].predict(X_test)
for i in range(1, num_blocks):
    predicted_test += models[i].predict(X_test)
    print("Model " + str(i + 1) + " done.")

y_pred_test = [1 if x >= 0 else -1 for x in predicted_test]

# Results

print("Testing set ...")
print(confusion_matrix(Y_test, y_pred_test))
print(classification_report(Y_test, y_pred_test))


