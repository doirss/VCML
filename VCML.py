import sys, os, time, csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from argparse import ArgumentParser


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

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        next(csvfile)
        locations = []
        location_IDs = []

        for row in readCSV:
            location = row[0]
            location_ID = row[1]
            locations.append(location)
            location_IDs.append(location_ID)

        dict_data = dict(zip(locations, location_IDs))

        X = dict_data.keys()
        y = dict_data.values()

        vect = CountVectorizer()

        vect.fit(X)

        vect.get_feature_names()

        X_dtm = vect.transform(X)
        print(X_dtm)

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
