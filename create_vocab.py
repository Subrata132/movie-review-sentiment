import os
curdir=os.getcwd()

import string
import re
from os import listdir
from os.path import join,isdir 
from collections import Counter
from nltk.corpus import stopwords

def open_text(filename):
    doc=open(filename,'r')
    text=doc.read()
    doc.close()
    
    return text 
    
def clean_text(text):

    # Split by whitespace
    words=text.split()
    
    # Remove punctuations
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    words_no_punc=[]
    for word in words:
        words_no_punc.append(re_punc.sub('',word))
        
    
    # Remove non-alphabetic words
    words=[]
    for word in words_no_punc:
        if word.isalpha() :
            words.append(word)
    
    # Remove stop words
    stop_words=stopwords.words('english')
    words_ex_stopwords=[]
    for word in words:
        if word not in stop_words :
            words_ex_stopwords.append(word)
            
    
    # Remove words of length less than '1'
    words=[]
    for word in words_ex_stopwords:
        if len(word)>1 :
            words.append(word)
    
    return words 

def save_vocab(name,dict):
    
    dict='\n'.join(dict)
    file=open(name,'w')
    file.write(dict)
    file.close()
    
    
data_loc=curdir+'/movie_reviews/train/'

vocab=Counter()
for folder in sorted(listdir(data_loc),key=len):
    file_loc=join(data_loc,folder)
    if isdir(file_loc):
        for file in sorted(listdir(file_loc),key=len):
            filename=join(file_loc,file)
            text=open_text(filename)
            words=clean_text(text)
            vocab.update(words)
            
            
min_occ=2
dictionary=[]
for k,c in vocab.items():
    if c>=min_occ:
        dictionary.append(k)
        
print(len(dictionary))    
dict_name=curdir+'/dictionary/'+'vocab1.txt'
save_vocab(dict_name,dictionary)    
    