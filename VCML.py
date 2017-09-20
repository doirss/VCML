import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

with open('ml_training_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    # print(readCSV)
    locations = []
    location_IDs = []
    for row in readCSV:
        location = row[0]
        location_ID = row[1]

        locations.append(location)
        location_IDs.append(location_ID)

    dict_data = dict(zip(locations, location_IDs))
    # print (dict_data)
    X = dict_data.keys()
    y = dict_data.values()

    vect = CountVectorizer()
    # learn vocabulary of the trainig data
    vect.fit(X)
    # examine the fitted vocabulary,
    vect.get_feature_names()
    # transform training data inot document term matrix
    X_dtm = vect.transform(X)
    # print(X_dtm)
    # convert sparse matrix to a dense matrix
    X_dtm.toarray()
    # representing text as numerical data
    pd.DataFrame(X_dtm.toarray(), columns=vect.get_feature_names())

    simple_test = ["ZIP CODE SALESUT84087-2144"]
    simple_test_dtm = vect.transform(simple_test)
    simple_test_dtm.toarray()
    pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
    