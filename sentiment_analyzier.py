import os 
curdir=os.getcwd()

import numpy as np
import string 
import re
from nltk.corpus import stopwords
from os.path import join,isdir
from os import listdir
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D,Dropout,Flatten,Dense

seed=1234
np.random.seed(seed)

def load_file(filename):
    
    file=open(filename,'r')
    text=file.read()
    file.close()
    
    return text

def clean_data(file,vocab):
    
    words=file.split()
    
    re_punc=re.compile('[%s]'% re.escape(string.punctuation))
    words_no_punc=[]
    for word in words:
        words_no_punc.append(re_punc.sub('',word))
        
    words=[]
    for word in words_no_punc:
        if word.isalpha():
            words.append(word)
            
    stop_words=stopwords.words('english')
    words_ex_stopwords=[]
    for word in words:
        if word not in stop_words:
            words_ex_stopwords.append(word)
            
    words_ex_small=[]
    for word in words_ex_stopwords:
        if len(word)>1:
            words_ex_small.append(word)
            
    words=[]
    for word in words_ex_small:
        if word in vocab:
            words.append(word)
    
    words=' '.join(words)
    
    return words
def find_maxlength(reviews):
    
    max_length=0
    for review in reviews:
        max_length=max(max_length,len(review.split()))
        
    return max_length
        
def data_loader(data_loc,purpose,vocab):
    
    data=[]
    labels=[]
    location=data_loc+purpose
    for folder in sorted(listdir(location),key=len):
        file_loc=join(location,folder)
        if isdir(file_loc):
            if folder=='neg':
                for file in sorted(listdir(file_loc),key=len):
                    filename=join(file_loc,file)
                    text=load_file(filename)
                    words=clean_data(text,vocab)
                    data.append(words)
                    labels.append(0)
            else:
                for file in sorted(listdir(file_loc),key=len):
                    filename=join(file_loc,file)
                    text=load_file(filename)
                    words=clean_data(text,vocab)
                    data.append(words)
                    labels.append(1)
                    
    return data,labels
    
def fit_tokenizer(words):
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(words)
    
    return tokenizer
    
def text_encoder(tokenizer,max_length,reviews):
    
    encoded=tokenizer.texts_to_sequences(reviews)
    padded=pad_sequences(encoded,maxlen=max_length,padding='post')
    
    return padded
    
def find_sentiment(filename,vocab,tokenizer,max_length,model):
    
    text=load_file(filename)
    words=clean_data(text,vocab)
    words=[words]
    data=text_encoder(tokenizer,max_length,words)
    
    prediction=model.predict(data)
    
    postivity=prediction[0,0]
    
    if round(postivity)==0 :
        confidence=1-postivity
        sentiment='Negative'
        
    else:
        confidence=postivity
        sentiment='Positive'
    
    return confidence,sentiment,text
    
def create_model(vocab_size,max_length):
    
    model=Sequential()
    model.add(Embedding(vocab_size,100,input_length=max_length))
    
    model.add(Conv1D(62,8,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    return model 
    
vocabFile=curdir+'/dictionary/'+'vocab.txt'
vocab=load_file(vocabFile)
vocab=vocab.split()

purpose='train'
data_loc=curdir+'/movie_reviews/'
Xtrain,Ytrain=data_loader(data_loc,purpose,vocab)
purpose='test'
Xtest,Ytest=data_loader(data_loc,purpose,vocab)


tokenizer=fit_tokenizer(Xtrain)
vocab_size=len(tokenizer.word_index)+1

max_length=find_maxlength(Xtrain)
Xtrain=text_encoder(tokenizer,max_length,Xtrain)
Xtest=text_encoder(tokenizer,max_length,Xtest)

retrain=False
epochs=10

if retrain==True :
    model=create_model(vocab_size,max_length)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    model.fit(Xtrain,Ytrain,epochs=epochs)
    model_name='sentNet1.hdf5'
    model_name=curdir+'/models/'+model_name
    model.save(model_name)
    
else :
    model_name='sentNet1.hdf5'
    model_name=curdir+'/models/'+model_name
    model=load_model(model_name)
    model.summary()
    

_,train_acc=model.evaluate(Xtrain,Ytrain)
_,test_acc=model.evaluate(Xtest,Ytest)

print('Trainning Accuracy: %.3f%%'% (train_acc*100))
print('Testing Accuracy: %.3f%%'% (test_acc*100))

review_file='review.txt'
confidence,sentiment,review=find_sentiment(review_file,vocab,tokenizer,max_length,model)


print('\n\n')
print('########Given review###########')
print('\n-------------------------------------------------------------------\n')
print(review)
print('\n-------------------------------------------------------------------\n')
print('\nSentiment: %s || Confidence: %.3f%%'%(sentiment,confidence*100))
print('\n\n')