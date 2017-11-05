import csv , time
import numpy as np
from sklearn.model_selection import train_test_split
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
from sklearn.neighbors import KNeighborsClassifier

#Python GUI
from tkinter import*
from tkinter import ttk
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os

"""GUI"""

root = Tk()

def destroy():
    root.destroy()

myFormats = [('CSV files', '*.csv')]

# def choosefile():
#     file = tk.filedialog.askopenfile(filetypes=myFormats, mode='r')

tex = tk.Text(master=root)
tex.pack(side=tk.RIGHT)
bop = tk.Frame()
bop.pack(side=tk.LEFT)
"""Menu"""
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=0)
# filemenu.add_command(label="Choose File", command=choosefile)
filemenu.add_command(label="Close", command=destroy)

menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)
"""Menu"""

"""Window size"""
root.minsize(width=1000, height=400)
"""Window size"""

"""Frames and Labels"""
theLabel2 = Label(root, text="VCML Machine Learning!", bg="orange", fg="white")
theLabel2.pack(fill=X)

topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side = BOTTOM)
"""Frames and Labels"""

"""Buttons"""
button1 = Button(topFrame, text="Team Introduction!", bg="purple", fg="white")
button2 = Button(topFrame, text="Click here for help!",bg="blue", fg="white")
button3 = Button(topFrame, text="Choose CSV File!", bg="green", fg="white")
# button4 = Button(bottomFrame, text="Exit", bg="orange", fg="white")
button5 = Button(topFrame, text="Columns needed", bg="yellow",fg="white")
"""Buttons""" =

CHOSE_COLS = """Choose your columns!"""

def cbc(tex):
    return lambda : introduction(event)

def choosefile(event):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Enter Columns")
        self.l.pack()
        self.e=Entry(top)
        self.r=Entry(top)
        self.e.pack()
        self.r.pack()
        self.bu=Button(top,text='Ok',command=self.cleanup)
        self.bu.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

def introduction(event):
    s = 'ValueCentric Machine Learning Project\n\nDeveloped by:\nAlexander Archer\nSamrita Malla\nMing Hann Hsieh (Henry)\n\n'.format()
    tex.insert(tk.END, s)
    tex.see(tk.END)

def help(event):
    s = 'You have requested help!\n\nChoose a CSV by clicking on the top left!\nSupports up to 6000 lines of data for training!\n\n'.format()
    tex.insert(tk.END, s)
    tex.see(tk.END)

def choosefileUI(event):
    filename = tk.filedialog.askopenfilename()
    print(len(filename))
    if filename:

        #instantiate with default parameter
        nb = MultinomialNB()

        with open(filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")

            locations = []
            location_IDs = []
            for row in readCSV:
                location = row[6]
                location_ID = row[7]

                locations.append(location)
                location_IDs.append(location_ID)

            X_train = locations

            y_train = location_IDs

            X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, random_state=42)

            vect = CountVectorizer(stop_words='english')

            # learn vocabulary of the trainig data
            vect.fit(X_train1)

            # examine the fitted vocabulary,
            vf = vect.get_feature_names()

            # transform training data inot document term matrix
            X_train_dtm = vect.transform(X_train1)

            # convert sparse matrix to a dense matrix
            X_train_dtm.toarray()

            # representing text as numerical data examine the vocabulary and document-term matrix together
            pd.DataFrame(X_train_dtm.toarray(), columns=vf)

            X_test_dtm = vect.transform(X_test1)
            X_test_dtm.toarray()

            pd.DataFrame(X_test_dtm.toarray(), columns = vf)

            # train the model using X_train_dtm
            logreg = LogisticRegression()
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_dtm, y_train1)

            # make class prediction for X_test_dtm
            y_predict_class = knn.predict(X_test_dtm)

            # calculate accuracy of class prediction
            accuracy = metrics.accuracy_score(y_test1, y_predict_class)
            # print("Accuracy: %2f%%" %(accuracy *100.0))

        s = 'You have successfully chosen a file!\n\nYour Accuracy is %2f%%'%(accuracy *100.0)+'!'
        tex.insert(tk.END, s)
        tex.see(tk.END)


"""Buttons"""
button1.bind("<Button-1>",introduction)
button2.bind("<Button-1>",help)
button3.bind("<Button-1>",choosefileUI)
button5.bind("<Button-1>",choosecols)

button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side=LEFT)
button5.pack(side=LEFT)

tk.Button(bottomFrame, text="Exit", bg="red", fg="white", command=root.destroy).pack(side=BOTTOM)
"""Buttons"""

root.mainloop()
