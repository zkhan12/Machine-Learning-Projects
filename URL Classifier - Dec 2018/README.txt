This analysis is done regarding the URL Reputation dataset from the UCI Machine Learning Repository.

Some basic techniques were used here to famialize myself with feature selection, data visualization, and classification modeling.
Due to an extremely large and extremely sparse dataset, some initial diagnostics were run to determine the number of malicious/nonmalicious samples. The figure distributionGraph.png demonstrates this distribution.
To get an idea of how sparse the dataset is, a sparsity graph determined which day's files contained the most useful data, where a lower sparsity score meant it was less sparse and more complete. This can be seen in sparsityGraph.png
Due to the limitation of techniques like SVD on large datasets, feature selection was used by only keeping the columns with a lot of nonzero values. This can be seen in the colFrequency.png graph.
Lastly, in the Analysis2.py file, multiple different techniques were used for classification including SVM's, Random Forest classifiers, ensembling (for SVMs), and KNN. The accuracy ranged between 80-90% for most classifiers. 

APPROACH:
The inital approach used was to train on 120 days of data and test on day 121. Due to the massive size of the dataset, dimensionality reduction was necessary which resulted in a train size of 14,400 and a test size of . This consisted of a twofold approch - feature selection and sample size reduction. For feature selection, columns with a nonzero frequency greater than 0.3 were kept. For sample size reduction, 2000 points were randomly sampled for each day (out of 120 days), and 120 clusters were made using the KMeans algorithm. Based on the nearest neighbor for each of these clusters, they were classified with the KNN algorithm and used for training once all 120 days were sampled and clustered.
A PCA graph was generated using these values to visualize high dimensional values in 2D. This can be seen in PCAoutput.png.