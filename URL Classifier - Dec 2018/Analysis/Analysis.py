# Where malicious is value of +1 and not malicious is value of -1

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

# Load Files from FeatureSelection.py output

with open('./ObjectFiles/categorical_cols.obj', 'rb') as pickle_file:
    categorical_cols = pickle.load(pickle_file)

with open('./ObjectFiles/numerical_cols.obj', 'rb') as pickle_file:
    numerical_cols = pickle.load(pickle_file)

#Indices gathered using a sorted keepcols array
with open('./ObjectFiles/categorical_cols_index.obj', 'rb') as pickle_file:
    categorical_cols_index = pickle.load(pickle_file)

with open('./ObjectFiles/keepcols.obj', 'rb') as pickle_file:
    keepcols = pickle.load(pickle_file)

#So the one hot encoding indices work
keepcols = sorted(keepcols)


# Read in sparse matrix and preprocess

def get_data(fname):
    data = load_svmlight_file(fname)
    print (fname + " loaded successfully.")
    return data[0], data[1]

def concatenate_csr_rows(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csr_matrix((new_data, new_indices, new_ind_ptr))

def featureSelectandEncode(csr_input):
    # Remove noninformative columns using variance based feature selection (not size dependent)
    csr_input = csr_input[:,keepcols]

    #One hot encode the categorical values
    onehotencoder = OneHotEncoder(categorical_features=categorical_cols_index)
    csr_input = onehotencoder.fit_transform(csr_input.todense())

    onehotencoder = None

    return csr_matrix(csr_input)

def sampleandCluster(csr_input, label_input):
    #Randomly choose 2000 points to represent subset of data
    if (csr_input.shape[0] <2000):
        SAMPLE_SIZE = csr_input.shape[0]
    else:
        SAMPLE_SIZE = 2000
    num_blocks = 200
    ran_num = [random.randint(0, num_blocks) for x in range(SAMPLE_SIZE)]
    csr_output = csr_input[[index for index, value in enumerate(ran_num)], :]

    #Cluster to 120 points with KMeans
    kmeans = KMeans(n_clusters=120, random_state=0).fit(csr_output)

    csr_output = csr_matrix(kmeans.cluster_centers_)

    #Classify clusters using KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(csr_input,label_input)
    label_output = knn.predict(csr_output)

    return csr_output, label_output


#Set reference CSR then iterate through all CSRs
data_CSR, label = get_data("url_svmlight/Day0.svm")
data_CSR = featureSelectandEncode(data_CSR)
data_CSR, label = sampleandCluster(data_CSR, label)

for i in range (1,120):
    data_CSR1, label1 = get_data("url_svmlight/Day" + str(i) +".svm")
    data_CSR1 = featureSelectandEncode(data_CSR1)
    data_CSR1, label1 = sampleandCluster(data_CSR1, label1)

    #onehot encoding files 49 and 114 causes issues due to size increase, so
    #these will be skipped until they can be parsed to find which categorical variables are
    #present in them that aren't present in any other file - this is a fixable issue but
    #will take a considerable amount of time and doesn't benefit the program much in terms
    #of accuracy, so for the sake of time these will be ignored
    if {i==49 or i==114}:
        continue

    data_CSR = concatenate_csr_rows(data_CSR, data_CSR1)
    label = np.concatenate((label,label1))
    data_CSR1 = None
    label1 = None


with open('./ObjectFiles/data_csr.obj', 'wb') as pickle_file:
    pickle.dump(data_CSR, pickle_file)

with open('./ObjectFiles/label.obj', 'wb') as pickle_file:
    pickle.dump(label, pickle_file)
