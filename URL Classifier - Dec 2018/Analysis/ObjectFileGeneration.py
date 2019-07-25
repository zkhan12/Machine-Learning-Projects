import pickle

with open('./ObjectFiles/keepcols.obj', 'rb') as pickle_file:
    keepcols = pickle.load(pickle_file)

with open('FeatureTypes') as file:
    file = file.read()
    file = file.split("\n")
    file.remove(file[len(file)-1])

num_col = list(map(int, file))

num_col = [x - 1 for x in num_col]

categorical_cols = []

numerical_cols = []

for i in range(len(keepcols)):
    if (keepcols[i] not in num_col):
        categorical_cols.append(keepcols[i])
        print(keepcols[i],' Categorical')
    else:
        numerical_cols.append(keepcols[i])
        print(keepcols[i],' Numerical')

print(sorted(keepcols))

categorical_cols_index = []

for i in range(len(keepcols)):
    if (keepcols[i] in categorical_cols):
        categorical_cols_index.append(i)

print(categorical_cols_index)

with open('./ObjectFiles/categorical_cols_index.obj', 'wb') as pickle_file:
    pickle.dump(categorical_cols_index, pickle_file)

with open('./ObjectFiles/categorical_cols.obj', 'wb') as pickle_file:
    pickle.dump(categorical_cols, pickle_file)

with open('./ObjectFiles/numerical_cols.obj', 'wb') as pickle_file:
    pickle.dump(numerical_cols, pickle_file)
