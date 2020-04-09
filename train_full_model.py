import pickle
import pymysql
from sqlalchemy import create_engine
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.metrics import classification_report
import os
import re
from glob import glob
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

data_folder = 'satire/'
stop_words = set(stopwords.words('english'))

connection = pymysql.connect(host='csmysql.cs.cf.ac.uk',
                             user='c1979282',
                             password='Password2020.',
                             db='c1979282_coursework')

cursor = connection.cursor()

# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user='c1979282',
                               pw='Password2020.',
                               db='c1979282_coursework',
                               host='csmysql.cs.cf.ac.uk'))


def load_data(data_folder):
    files = os.listdir(data_folder)
    d = {}
    for infile in files:
        f = open(os.path.join(data_folder, infile), encoding='latin1')
        text = f.read()
        list_of_documents = text.split('\n')
        l = []
        for document in list_of_documents:
            tokens = word_tokenize(document)
            tokens_lower = [t.lower() for t in tokens]
            tokens_alpha = [t for t in tokens_lower if t.isalpha()]
            tokens_nostop = [t for t in tokens_alpha if t not in stop_words]
            wordnet_lemmatizer = WordNetLemmatizer()
            tokens_lemmatised = [wordnet_lemmatizer.lemmatize(t) for t in tokens_nostop]
            l.append(tokens_lemmatised)
        d[infile] = l
    return d


def process_input(sentence):
    tokens = word_tokenize(sentence)
    tokens_lower = [t.lower() for t in tokens]
    tokens_alpha = [t for t in tokens_lower if t.isalpha()]
    tokens_nostop = [t for t in tokens_alpha if t not in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens_lemmatised = [wordnet_lemmatizer.lemmatize(t) for t in tokens_nostop]

    return tokens_lemmatised


dict = load_data(data_folder)

df = pd.DataFrame(dict)
df_satire = pd.DataFrame({'text': df['satire'], 'label': '1'})
df_non_satre = pd.DataFrame({'text': df['non_satre'], 'label': '0'})
df_all = df_non_satre.append(df_satire, ignore_index=True)
df_all['document'] = df_all['text'].str.join(" ")

X_train, X_test, y_train, y_test = train_test_split(df_all['document'], df_all['label'], test_size=0.33)

# Create train and test dataframes
train_table = pd.concat([X_train, y_train], axis=1)
test_table = pd.concat([X_test, y_test], axis=1)

# Insert dataframes to the database
train_table.to_sql('Train_table', con=engine, if_exists='replace', chunksize=1000)
test_table.to_sql('Test_table', con=engine, if_exists='replace', chunksize=1000)


def train_classifiers():

    # versioning
    os.getcwd()
    files = glob("models/*.model")
    files
    a = files.sort()
    a

    try:
        cur_num = int(re.search(r"\d+(?=\.model)", files[-1]).group())
        cur_num += 1
    except IndexError:
        cur_num = 0

    cur_num


    df_train = pd.read_sql('SELECT * FROM Train_table', con=engine)
    df_test = pd.read_sql('SELECT * FROM Test_table', con=engine)

    count_vectorizer = CountVectorizer()
    count_train = count_vectorizer.fit_transform(df_train['document'])
    count_test = count_vectorizer.transform(df_test['document'])
    out_count_vectorizer = open('models/count_vectorizer.vec', 'wb')
    pickle.dump(count_vectorizer, out_count_vectorizer)
    out_count_vectorizer.close()

    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, df_train['label'])
    nb_pred = nb_classifier.predict(count_test)
    nb_score = metrics.accuracy_score(df_test['label'], nb_pred)
    print('Naive Bayes Score: ' + str(nb_score))
    nb_model_path = 'models/count_nb_' + str(cur_num) + '.model'
    out_nb_model = open(nb_model_path, 'wb')
    pickle.dump(nb_classifier, out_nb_model)
    out_nb_model.close()


    log_classifier = LogisticRegression(random_state=0, max_iter=1000)
    log_classifier.fit(count_train, df_train['label'])
    log_pred = log_classifier.predict(count_test)
    log_score = metrics.accuracy_score(df_test['label'], log_pred)
    print('Logistic Regression Score: ' + str(log_score))
    log_model_path = 'models/count_log_' + str(cur_num) + '.model'
    out_log_model = open(log_model_path, 'wb')
    pickle.dump(log_classifier, out_log_model)
    out_log_model.close()

    dictionary = {'model_name': ['naive_bayes', 'logistic_regression'], 'model_path': [nb_model_path, log_model_path],
                  'score': [nb_score, log_score]}
    df_classifier = pd.DataFrame(dictionary)
    df_classifier
    df_classifier.to_sql('Classifier_table', con=engine, if_exists='append', chunksize=1000)

    return nb_score, log_score

train_classifiers()
# classifier = pd.read_sql(
#     'SELECT * FROM Classifier_table WHERE score=(SELECT MAX(score) FROM Classifier_table)',
#     con=engine)
# classifier


# cm = metrics.confusion_matrix(y_test, pred)
# plot_confusion_matrix(nb_classifier, count_test, y_test, cmap=plt.cm.Blues, values_format='.0f')
# print(classification_report(y_test, pred)
