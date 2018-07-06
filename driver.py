import numpy as np
import pandas as pd
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

training_path = "./aclImdb/"
testing_path = "./imdb_te.csv" 

def Unigram():

    inpath = "./" 
    name = "imdb_tr.csv"
    ngram = 1
    outfile="./unigram.output.txt"
    tfidf = False
    data = pd.read_csv(inpath+name)
    training_data, value_date = train_test_split(data, test_size=0.1, random_state=32)
    if not tfidf:
        vect = CountVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    else:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    Xtrain= vect.fit_transform(training_data['text'].tolist())
    Ytrain = training_data['polarity'].tolist()
    Xval = vect.transform(value_date['text'].tolist())
    Yval = value_date['polarity'].tolist()
    clf = SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(Xtrain, Ytrain)
    VALpredict = clf.predict(Xval)
    TRAINpredict = clf.predict(Xtrain)
    testing_data = pd.read_csv(testing_path, encoding='ISO-8859-1')
    Xtest = vect.transform(testing_data['text'].tolist())
    TESTpredict = clf.predict(Xtest)
    f = open(outfile, 'w')
    for item in TESTpredict:
        f.write("%d\n" % item)
    f.close()

def Bigram():
    inpath = "./"
    name = "imdb_tr.csv" 
    ngram = 2
    outfile="./bigram.output.txt"
    tfidf = False
    data = pd.read_csv(inpath+name)
    training_data, value_date = train_test_split(data, test_size=0.1, random_state=32)
    if not tfidf:
        vect = CountVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    else:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    Xtrain= vect.fit_transform(training_data['text'].tolist())
    Ytrain = training_data['polarity'].tolist()
    Xval = vect.transform(value_date['text'].tolist())
    Yval = value_date['polarity'].tolist()
    clf = SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(Xtrain, Ytrain)
    VALpredict = clf.predict(Xval)
    TRAINpredict = clf.predict(Xtrain)
    testing_data = pd.read_csv(testing_path, encoding='ISO-8859-1')
    Xtest = vect.transform(testing_data['text'].tolist())
    TESTpredict = clf.predict(Xtest)
    f = open(outfile, 'w')
    for item in TESTpredict:
        f.write("%d\n" % item)
    f.close()

def Unigramtfidf():
  
    inpath = "./"
    name = "imdb_tr.csv"
    ngram = 1
    outfile="./unigramtfidf.output.txt"
    tfidf = True
    data = pd.read_csv(inpath+name)
    training_data, value_date = train_test_split(data, test_size=0.1, random_state=32)
    if not tfidf:
        vect = CountVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    else:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    Xtrain= vect.fit_transform(training_data['text'].tolist())
    Ytrain = training_data['polarity'].tolist()
    Xval = vect.transform(value_date['text'].tolist())
    Yval = value_date['polarity'].tolist()
    clf = SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(Xtrain, Ytrain)
    VALpredict = clf.predict(Xval)
    TRAINpredict = clf.predict(Xtrain)
    testing_data = pd.read_csv(testing_path, encoding='ISO-8859-1')
    Xtest = vect.transform(testing_data['text'].tolist())
    TESTpredict = clf.predict(Xtest)
    f = open(outfile, 'w')
    for item in TESTpredict:
        f.write("%d\n" % item)
    f.close()

def Bigramtfidf():
    
    inpath = "./"
    name = "imdb_tr.csv"
    ngram = 2
    outfile="./bigramtfidf.output.txt"
    tfidf = True
    data = pd.read_csv(inpath+name)
    training_data, value_date = train_test_split(data, test_size=0.1, random_state=32)
    if not tfidf:
        vect = CountVectorizer(stop_words='english', ngram_range=(ngram,ngram))
    else:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(ngram,ngram)) 
    Xtrain= vect.fit_transform(training_data['text'].tolist())
    Ytrain = training_data['polarity'].tolist()
    Xval = vect.transform(value_date['text'].tolist())
    Yval = value_date['polarity'].tolist()
    clf = SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(Xtrain, Ytrain)
    VALpredict = clf.predict(Xval)
    TRAINpredict = clf.predict(Xtrain)
    testing_data = pd.read_csv(testing_path, encoding='ISO-8859-1')
    Xtest = vect.transform(testing_data['text'].tolist())
    TESTpredict = clf.predict(Xtest)
    f = open(outfile, 'w')
    for item in TESTpredict:
        f.write("%d\n" % item)
    f.close()
    
def imdb_data_preprocessing(inpath, outpath="./", name="imdb_tr.csv", mix=False):

    inpath += 'train/'
    f = open('./stopwords.en.txt', 'r')
    stopwords = list(f)
    stopwords = [w[:-1] for w in stopwords]
    f.close()
    comments=[]
    labels =[]
    for file in glob.glob(inpath+"pos/*.txt"):
        text_file = open(file, "r")
        line = text_file.read()
        comments.append(line)
        labels.append(1)
        text_file.close()
    for file in glob.glob(inpath+"neg/*.txt"):
        text_file = open(file, "r")
        line = text_file.read()
        comments.append(line)
        labels.append(0)
        text_file.close()
    df = pd.DataFrame.from_dict({'text':comments, 'polarity':labels})
    df = df[['text','polarity']]

    df.to_csv(outpath+name)

if __name__ == "__main__":

    print "writing to unigram.output.txt ..."
    Unigram()
    print "writing to bigram.output.txt ..."
    Bigram()
    print "writing to unigramtfidf.output.txt ..."
    Unigramtfidf()
    print "writing to bigramtfidf.output.txt ..."
    Bigramtfidf()
    print "done writing."
