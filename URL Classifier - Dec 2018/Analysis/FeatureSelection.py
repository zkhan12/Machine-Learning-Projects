from sklearn.datasets import load_svmlight_file
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

with open('FeatureTypes') as file:
    file = file.read()
    file = file.split("\n")
    file.remove(file[len(file)-1])

num_col = list(map(int, file))

num_col = [x - 1 for x in num_col]

def get_data(fname):
    data = load_svmlight_file(fname)
    print (fname + " loaded successfully.")
    return data[0], data[1]


cutoff = .3
featureTrack = dict()

for i in range (0,121):
    data_CSR, label = get_data("url_svmlight/Day" + str(i) +".svm")

    nonzero = data_CSR.nonzero()
    freq = stats.itemfreq(nonzero[1])
    reduced = freq[freq[:, 1] > data_CSR.shape[0] * cutoff, :]
    for row in reduced:
        featureTrack[row[0]] = featureTrack.get(row[0], 0) + row[1]

    data_CSR = None
    label = None
    nonzero = None
    freq = None
    reduced = None

rows = []
cols = []
keepcols = []
totalcols = 2396130

for entry in featureTrack.keys():
    rows.append(entry)
    cols.append(featureTrack.get(entry)/totalcols)
    if (featureTrack.get(entry)/totalcols > 0.2):
        if (entry in num_col):
            print("Numerical column: ", entry)
        keepcols.append(entry)

with open('./ObjectFiles/keepcols.obj', 'wb') as pickle_file:
    pickle.dump(keepcols, pickle_file)

y_pos = np.arange(len(rows))
plt.bar(y_pos, cols, align = 'center', alpha = 0.5)
plt.ylabel('Frequency of nonzero value in column')
plt.xlabel('Number of columns passing filter')
plt.title('Feature selection data')

plt.show()