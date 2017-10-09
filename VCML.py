import csv , time
import numpy as np
# import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, model_selection
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from argparse import ArgumentParser
# # #print(dataset.groupby('TARGET_LOC_ID').size())
def introduction():
    print("""
        ValueCentric Machine Learning Project\n\n
        Developed by:\n
        Alexander Archer\n
        Samrita Malla\n
        Ming Hann Hsieh (Henry)\n
    """)

def parse():
    parser = ArgumentParser()
    parser.add_argument('-train', nargs=1, help="-train <training_data.csv>")
    parser.add_argument('-test', nargs=2, help="-test <test_data.csv> <answer_data.csv>")

    args = parser.parse_args()

    if args.train is not None:
        return args.train, 1
    elif args.test is not None:
        return args.test, 2

def train(filename):

    #instantiate with default parameter
    nb = MultinomialNB()
    #instantiate with default parameter
    nb = MultinomialNB()

    with open(filename) as csvfile:
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

        vect = CountVectorizer(stop_words='english')
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
        logreg.fit(X_train_dtm, y_train1)
        #text_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
        #text_clf = text_clf.fit(X_train, dict_data.target)


        # make class prediction for X_test_dtm
        y_predict_class = logreg.predict(X_test_dtm)
        # tags = list(set(y_train.tolist()))
        # probs = y_predict_class.tolist()
        #print(y_predict_class)

        # calculate accuracy of class prediction
        # print(metrics.accuracy_score(y_test1, y_predict_class))
        accuracy = metrics.accuracy_score(y_test1, y_predict_class)
        print("Accuracy: %2f%%" %(accuracy *100.0))
    print("Finished training the model, outputting to training.txt\n")
    open("training.txt", "w")


def test(test_data, answer_data):
    print("Finished testing the model, outputting to testing.txt")
    open("testing.txt", "w")


if __name__ == "__main__":

    introduction()

    filename, nargs = parse()

    if(nargs == 1):

        time.sleep(1)
        print("Taking in the training data...\n")

        train(filename[0])

    elif(nargs == 2):

        time.sleep(1)
        print("Testing the model...\n")

        test(filename[0], filename[1])
