from flask import Flask, render_template, request, redirect, url_for, Markup, send_from_directory
from wtforms import Form, TextAreaField, validators
import os
import socket
import csv

import pandas as pd
from pandas import Series,DataFrame
import numpy as np

from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)

df=None
tst_data=None

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Tokenizes and removes punctuation
    2. Removes  stopwords
    3. Stems
    4. Returns a list of the cleaned text
    """

    # tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed=tokenizer.tokenize(text)
    
    # removing any stopwords
    stoplist = stopwords.words('english')
    stoplist.append('twitter')
    stoplist.append('pic')
    stoplist.append('com')
    stoplist.append('net')
    stoplist.append('gov')
    stoplist.append('tv')
    stoplist.append('www')
    stoplist.append('twitter')
    stoplist.append('num')
    stoplist.append('date')
    stoplist.append('time')
    stoplist.append('url')
    stoplist.append('ref')

    text_processed = [word.lower() for word in text_processed if word.lower() not in stoplist]
    
    # steming
    porter_stemmer = PorterStemmer()
    
    text_processed = [porter_stemmer.stem(word) for word in text_processed]
    

    return text_processed



from sklearn.grid_search import GridSearchCV


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score



@app.route('/', methods = ['GET'])
#@app.route('/home', methods = ['GET'])
def home():
    return render_template('mhome.html')

@app.route('/aboutus',methods=['GET'])
def aboutus():
	return render_template('aboutus.html')

@app.route('/upload-file', methods=['POST'])
def upload():
    target = os.path.join(cur_dir,"upload/")

    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("author1-file"):
        filename = file.filename
	df1=file.filename
        dest = "/".join([target, filename])
        file.save(dest)

    print"author1 filename",df1

    df_data1=pd.read_csv('/home/lora/Desktop/be project/upload/'+df1)   
     
    print"len of author1 tweet",len(df_data1) 

    for file in request.files.getlist("author2-file"):
        filename = file.filename
	df2=file.filename
        dest = "/".join([target, filename])
        file.save(dest)

    print"suthor2 filename",df2

    df_data2=pd.read_csv('/home/lora/Desktop/be project/upload/'+df2)

    print"len of author2 tweet",len(df_data2)

    #file_names=[]
    #for file in os.listdir("/home/lora/Desktop/be project/upload"):
    	#if file.endswith(".csv"):
        	#print(os.path.join("/home/lora/Desktop/be project/upload", file))
		#file_names.append(file)


    for file in request.files.getlist("disputed-file"):
        filename = file.filename
	test=file.filename
        dest = "/".join([target, filename])
        file.save(dest)


    #print(file_names)
    #print(file_names[0])
    
    print"test tweet filename",test

    tst_data=pd.read_csv('/home/lora/Desktop/be project/upload/'+test)

    print"len of test tweet",len(tst_data)
   
    #with open('/home/lora/Desktop/be project/upload/'+test) as csvfile:
    	#reader = csv.DictReader(csvfile)
    	#for i,row in enumerate(reader):
        	#print row
        	#if(i >= 2):
            		#break
    #count=len(file_names)
    #print count

    #df=[]
    #for i in range(0,count):
	#df[i]=pd.read_csv('/home/lora/Desktop/be project/upload/'+file_names[i])
	#name='/home/lora/Desktop/be project/upload/'+file_names[i]
	#df.append(pd.read_csv(name))
	#print("i: ",i," len: ",len(df[i]))
	#print name

    import random
    #2000 random sample rows from files
    #for i in range(0,count):
    	#rows = random.sample(list(df[i].index), 2000)
    	#df[i] = df[i].ix[rows]
    	#print("i: ",i," len: ",len(df[i]))

    #joinin all data in one file
    #for i in range(1,count):
    	#df_new=df[0].append(df[i],ignore_index=True)
    #print"selected & joined tweets",len(df_new)


    #2000 random sample rows for KK
    rows = random.sample(list(df_data1.index), 2000)
    df_data1 = df_data1.ix[rows]
    #2000 random sample rows for HC
    rows = random.sample(list(df_data2.index), 2000)
    df_data2 = df_data2.ix[rows]
    #join back together
    df=df_data1.append(df_data2,ignore_index=True)
    print"selected & joined tweets",len(df)

    #data pre-processing
    df.drop(df[df.retweet==True].index, inplace=True)
    df['num_of_words'] = df["text"].str.split().apply(len)
    df.drop(df[df.num_of_words<4].index, inplace=True)
    df["text"].replace(r"http\S+", "URL", regex=True,inplace=True)
    df["text"].replace(r"@\S+", "REF", regex=True ,inplace=True)
    df["text"].replace(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+", "DATE", regex=True,inplace=True)
    df["text"].replace(r"(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    df["text"].replace(r"(\d{1,2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    df["text"].replace(r"\d+", "NUM", regex=True,inplace=True)
    print"data after preprocessing",len(df)


    #data pre-processing of test data
    tst_data.drop(tst_data[tst_data.retweet==True].index, inplace=True)
    tst_data['num_of_words'] = tst_data["text"].str.split().apply(len)
    tst_data.drop(tst_data[tst_data.num_of_words<4].index, inplace=True)
    tst_data["text"].replace(r"http\S+", "URL", regex=True,inplace=True)
    tst_data["text"].replace(r"@\S+", "REF", regex=True ,inplace=True)
    tst_data["text"].replace(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+", "DATE", regex=True,inplace=True)
    tst_data["text"].replace(r"(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    tst_data["text"].replace(r"(\d{1,2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    tst_data["text"].replace(r"\d+", "NUM", regex=True,inplace=True)
    print"test data after preprocessing",len(tst_data)

    twt_test=tst_data["text"]
    twt_train=df["text"]
    author_train=df["author"]
    author_test=tst_data["author"]

    print"twt test len",len(twt_test)
    print"twt train len",len(twt_train)
    print"author test len",len(author_test)
    print"author train len",len(author_train)
    print"author test",author_test

    ScoreSummaryByModelParams = list()

    def ModelParamsEvaluation (vectorizer,model,params,comment):
    	pipeline = Pipeline([
    	('vect', vectorizer),
    	('tfidf', TfidfTransformer()),
	('clf', model),
	])
	grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1)
	grid_search.fit(df['text'], df['author'])
	print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(params.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))
	        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])])


    #LinearSVC
    word_vector =  CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 5),max_df=0.5)
    char_vector = CountVectorizer(analyzer='char_wb',ngram_range=(3, 3),max_df=0.75)
    text_vector = CountVectorizer(analyzer='word',tokenizer=text_process,ngram_range=(3, 3),max_df=0.75)
    vectorizer = FeatureUnion([("chars", char_vector),("words", word_vector),("text", text_vector)])
    p = {'clf__C': (1,0.1,0.01,0.001,0.0001)}
    ModelParamsEvaluation(vectorizer,LinearSVC(),p,'LinearSVC vectorizer+text_vector')

    word_vector = CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 5),max_df=0.5)
    char_vector = CountVectorizer(analyzer='char_wb',ngram_range=(3, 3),max_df=0.75)
    text_vector = CountVectorizer(analyzer='word',tokenizer=text_process,ngram_range=(3, 3),max_df=0.75)
    vectorizer  = FeatureUnion([("chars", char_vector),("words", word_vector),("text", text_vector)])

    pipeline = Pipeline([
    	('vect', vectorizer),
    	('tfidf', TfidfTransformer()),
    	('clf', LinearSVC(C=1)),
    	])

    pipeline.fit(twt_train,author_train)
    author_predictions = pipeline.predict(twt_test)

    print"author predicted",author_predictions

    if author_predictions[0]==author_predictions[1]:
	author_predicted=author_predictions[0]
    else:
	author_predicted="cannot predict!"

    print"final prediction is: ",author_predicted


    return render_template('rerult.html',prediction = author_predicted )

if __name__ == '__main__':
    app.run(port=5008,debug = True)


