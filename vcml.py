import csv
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import sys
import argparse

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#data
test_data = "ml_test_data.csv"
training_data = "ml_training_data1.csv"
filename = sys.argv[0]

# def main():
#     filename = sys.argv[0]
#     data = np.loadtxt(filename, delimiter = ',')
#     for m in data.mean(axis = 1):
#         print (m)

# main()
# class MyParser(argparse.ArgumentParser):
#   def error(self, message):
#       sys.stderr.write('error: %s\n' % message)
#       self.print_help()
#       sys.exit(2)

# parser = MyParser()
# parser.add_argument('Error!', help='Please enter the .csv file you want to run as an argument after vcml.py!!!')
# # parser.add_argument('Error!', nargs = '+')
# args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('Error!', help='Please enter the .csv file you want to run as an argument after vcml.py!!!')
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()

# reader = csv.DictReader(open(filename, newline = ""), dialect = "excel")

# for row in reader:
#     next(reader)
#     print (row)

with open(training_data) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    next(csvfile)
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
    #learn vocabulary of the training data
    vect.fit(X)
    #examine the fitted vocabulary,
    vect.get_feature_names()
    #transform training data inot document term matrix
    X_dtm = vect.transform(X)
    print(X_dtm)
    # convert sparse matrix to a dense matrix
    # X_dtm.toarray()
    np.asarray(X_dtm)
    #representing text as numerical data
    # pd.DataFrame(np.asarray(X_dtm), columns=vect.get_feature_names())

    # simple_test = ["ZIP CODE SALESUT84087-2144"]
    # simple_test_dtm = vect.transform(simple_test)
    # # simple_test_dtm.asarray()
    # np.asarray(simple_test_dtm)
    # pd.DataFrame(np.asarray(simple_test_dtm), columns = vect.get_feature_names())