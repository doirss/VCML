import csv
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics


# # #print(dataset.groupby('TARGET_LOC_ID').size())

#instantiate with default parameter
nb = MultinomialNB()

#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#instantiate with default parameter
nb = MultinomialNB()

with open('ml_training_data_copy.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")


    locations = []
    location_IDs = []
    for row in readCSV:
        location = row[0]
        location_ID = row[7]

        locations.append(location)
        location_IDs.append(location_ID)
    dict_data = dict(zip(locations, location_IDs))

    #X_train = dict_data.keys()
    X_train = locations
    #print(X_train)
    #y_train = dict_data.values()
    y_train = location_IDs
    #print(y_train)
    # split X and y into train and test data
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, random_state=42)

    # print(len(X_train1))
    # print(len(X_test1))
    # print(len(y_train1))
    # print(len(y_test1))

    vect = CountVectorizer()
    # learn vocabulary of the trainig data
    vect.fit(X_train1)
    # examine the fitted vocabulary,
    vf = vect.get_feature_names()
    # print(vect.get_feature_names())
    # transform training data inot document term matrix
    X_train_dtm = vect.transform(X_train1)
    # print(X_train_dtm)
    # convert sparse matrix to a dense matrix
    X_train_dtm.toarray()
    # representing text as numerical data examine the vocabulary and document-term matrix together
    pd.DataFrame(X_train_dtm.toarray(), columns=vf)

    #X_test = ["ZIP CODE SALESUT84087-2144"]
    X_test_dtm = vect.transform(X_test1)
    X_test_dtm.toarray()
    # print (X_test_dtm)
    pd.DataFrame(X_test_dtm.toarray(), columns = vf)

    # train the model using X_train_dtm
    #nb.fit(X_train_dtm, y_train1)  # y_train
    logreg = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_dtm, y_train1)
    #logreg.fit(X_train_dtm, y_train1)
    #text_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
    #text_clf = text_clf.fit(X_train, dict_data.target)


    # make class prediction for X_test_dtm
    y_predict_class = knn.predict(X_test_dtm)
    # tags = list(set(y_train.tolist()))
    # probs = y_predict_class.tolist()
    #print(y_predict_class)

    # calculate accuracy of class prediction
    # print(metrics.accuracy_score(y_test1, y_predict_class))
    accuracy = metrics.accuracy_score(y_test1, y_predict_class)
    print("Accuracy: %2f%%" %(accuracy *100.0))
