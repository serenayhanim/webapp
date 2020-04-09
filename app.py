import pickle

import pandas as pd
from sqlalchemy import create_engine

import train_full_model
from flask import Flask, render_template, request, url_for
import pymysql

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user='c1979282',
                               pw='Password2020.',
                               db='c1979282_coursework',
                               host='csmysql.cs.cf.ac.uk'))

labels = {'1': 'satire', '0': 'non_satre'}
app = Flask(__name__)


@app.route('/')
def form():
    return render_template('home.html')


@app.route('/submitted', methods=['POST'])
def submitted_form():

    vectorizer = pickle.load(open('models/count_vectorizer.vec', 'rb'))
    print('vectorizer loaded')
    model_path = pd.read_sql(
        'SELECT model_path FROM Classifier_table WHERE score=(SELECT MAX(score) FROM Classifier_table)',
        con=engine)
    model = pickle.load(open(model_path, 'rb'))
    print('model loaded')
    text = request.form['input_text']
    list_sentence = train_full_model.process_input(text)
    clean_sentence = ' '.join(map(str, list_sentence))
    vec = vectorizer.transform([clean_sentence])
    pred = model.predict(vec)[0]
    pred = labels[pred]

    print('prediction: ', pred)

    return render_template('Submission.html', sentence=clean_sentence, prediction=pred)


@app.route('/feedback_received', methods=['POST'])
def feedback_received():
    print('request: ', list(request.form.keys()))
    feedback = request.form['feedback']
    sentence = request.form['sentence']
    prediction = request.form['prediction']

    if feedback == 'incorrect':

        if prediction == 'satire':
            dict = {sentence: '0'}
        else:
            dict = {sentence: '1'}
        df_input = pd.DataFrame(dict.items(), columns=['document', 'label'])
        print(df_input)
        df_input.to_sql('Train_table', con=engine, if_exists='append', chunksize=1000)
        train_full_model.train_classifiers()

    print('feedback: ', feedback)
    print('sentence: ', sentence)
    print('prediction: ', prediction)

    return render_template('feedback.html', sentence=sentence, feedback=feedback)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
