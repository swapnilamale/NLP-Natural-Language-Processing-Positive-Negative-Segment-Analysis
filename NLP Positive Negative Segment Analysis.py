# -*- coding: utf-8 -*-

# Created on Thu Oct  7 14:29:29 2021

# NLTK - natural language toolkit
# text analytics - unstructured data

# regular expressions
import re

# split a sentence into words
s1 = 'I am learning NLTK using python'
re.split('\s', s1)

# \s --> identifier

''' identifiers in regular expressions
\s : white space
\S : anything but a white space
\d : number
\D : anything but a number
\w : any character
\W : anything but a character
'''

''' mofifiers in regular expressions
+ : matches 1 or more 
* : matches 0 or more
. : matches till the end of the line
[] : matches a range
() : matches a group of characters
{} : query
'''

# find all the words that have "ia"
s3 = "India is my Country. The capital is New Delhi. It is in Asia."
re.findall('[\w]+ia[\w]*', s3)

# get all the words that starts with a capital letter
re.findall('[A-Z]\w+',s3)

# get all words that are in lower case
re.findall('\s[a-z]+', s3)

# extract email IDs from a text
s4 = "primary email id is srirama@mymail.com, secondary mail id is sri@gmail.com, alternate mail id is sriramancr@hotemail.com"
print(s4)

emails = re.findall('[\w]+@[\w.]+', s4)
print(emails[2])

# extract values from an XML tag
s5 = "<firtname>sriraman</firstname>"
re.findall('>(\w+)<',s5) # group of characters

# extract numbers from a text
s6 = "the average experience of our employees is 5 years. There are people who have over 20 years of experience. The total employee count is 374. The revenue of the company last year was 1378 cr. Our office number is 45667896. We are expecting another 178 people to join next month"

numbers = re.findall('\d+', s6)
print(numbers)
int(numbers[0])

# get all the 8-digit numbers
re.findall('\d{8}',s6)

# get all 4-digit numbers
re.findall('\d{4}', s6) # wrong answer
re.findall('\d{3}', s6) # wrong answer

# to prevent this, use the lookahead and lookbehind technique
# all 3-digit numbers
re.findall('(?<=\s)\d{3}(?=\s)', s6)

# split a sentence based on characters 
# eg: split a sentence at [a,b,c]
s7 = 'the king of fruits is mango because of its unique taste and color'
re.split('[a-c]', s7)

# text cleansing
s8 = " ???? the sentence [[[ ))) ]]] has ** lots of special  + characters // & "

# text after cleansing
' '.join(re.sub('[\W]',' ',s8).split())


# read a text file in python
filename = 'services.xml'
f = open(filename,"r")
line = f.readline() # read a line from the file
print(line)

while line:
    line = f.readline()
    print(line)    

f.close()

# -------------------------------------------

# NLTK-2 --> NLTK

import nltk

# tokenization

# convert text into words 
t1 = "There are 7 days in a week. Sunday is the first day and Saturday is the last day. I hate Mondays"

words = nltk.word_tokenize(t1)

# sentence tokenization (convert text into sentences)
sentences = nltk.sent_tokenize(t1)
print(sentences)

# dealing with abbreviations
s1 = 'This is Dr. Singh. He is a renowned dentist with over 30 years of experience. Next to him is Ms. Swati, his close associate' 
nltk.word_tokenize(s1)
nltk.sent_tokenize(s1)

# dealing with apostrophes (can't, cdn't, shdn't etc...)
s2 = "I can't do this. It's too complicated. I don't want to spend time on this. It ain't worth it and doesn't interest me"

nltk.word_tokenize(s2) # tokenisation is not proper

# to get the tokenisation of apostrophe words properly
from nltk.tokenize import RegexpTokenizer
pattern = RegexpTokenizer("[\w']+")
pattern.tokenize(s2)

# sentence
nltk.sent_tokenize(s2) # this is ok

##
# stop words
# -----------

from nltk.corpus import stopwords

# get the default set of stop words
stop_words = set(stopwords.words('english'))
print(stop_words)

# update the set of stop_words with new words
stop_words.update({'jugaad','paisa','masala','rs', 'rs.'})
print(stop_words)

s9 = 'the movie was full of masala and bad acting. it was not worth the rs. stunts were jugaad. overall a bad movie'

words = nltk.word_tokenize(s9)
words

# remove stopwords from the sentence
new_words = [w for w in words if w not in stop_words]
print(new_words)
' '.join(new_words)

# POS tagging (Parts-of-Speech tagging)

# before POS tagging, tokenize the text
words = nltk.word_tokenize(s1)

pos_words = nltk.pos_tag(words)
print(pos_words)

# extract all the Nouns in the sentence
# nouns : NN/NNS/NNP/NNPS

nouns = [w for w,p in pos_words if p in ['NN','NNS','NNP','NNPS'] ]
print(nouns)

# n-grams
'''
from anaconda prompt, run the following command
pip install -U textblob
'''

from textblob import TextBlob

# remove stop words before forming n-grams

# convert the input text into blob format
s1_blob = TextBlob(s1)

# form the n-grams
n1 = s1_blob.ngrams(1); n1
n2 = s1_blob.ngrams(2); n2
n3 = s1_blob.ngrams(3); n3
n4 = s1_blob.ngrams(4); n4

# TF IDF (Term Frequency  Inverse Document Frequency)
# -----
# given a set of documents (text), what is the relevance of a given word ?

# before applying tf-idf on a document, do a stopwords on the doc

# term frequency (TF) : relevance of a word in a document
def tf(word,doc):
    token = nltk.word_tokenize(doc)
    len_token = len(token)
    word_count = doc.count(word)
    
    tf1 = word_count / len_token
    
    return(tf1)

doc = "python is a language. python is a snake. python is used in machine learning"

# remove stop words
words = nltk.word_tokenize(doc)
new_words = [w for w in words if w not in stop_words]
new_words.remove(".")
doc = ' '.join(new_words)

word = "python"
tf(word,doc)

# idf (Inverse document frequency) : relevance of word in a given set of documents

# idf = log(length_of_doc / word_count)

from math import log

def idf(word,list_of_doc):
    wordcount = 0
    
    for d in list_of_doc:
        if d.count(word) > 0:
            wordcount+=1
    
    # if word is not found in any of the documents
    if wordcount <= 0:
        wordcount = 1
     
    # document count
    doc_count = len(list_of_doc)
    
    # IDF
    idf1 = log(doc_count / wordcount)
    
    return(idf1)

d1 = "python is a language. python is a snake. python is used in machine learning"
d2 = "python is a scripting language"
d3 = "i love data analysis"

word = 'python'

list_of_doc = [d1,d2,d3]
idf(word,list_of_doc)

# combine tf and idf
# TFIDF = TF * IDF

def tfidf(word,doc,list_of_doc):
    return(tf(word,doc) * idf(word,list_of_doc))

# find the relevance of the word
tfidf(word,doc,list_of_doc)







#__________________________ 2nd Day Topics ___________________________
#_____________________________________________________________________
#_____________________________________________________________________
#_____________________________________________________________________
#_____________________________________________________________________

######################################################################
# using NaiveBayes classifier to do sentiment analysis

import pandas as pd
import textblob
from textblob.classifiers import NaiveBayesClassifier

# read the train and test data
train=pd.read_csv('F:/aegis/4 ml/dataset/supervised/nltk/nb/train.csv')
test=pd.read_csv('F:/aegis/4 ml/dataset/supervised/nltk/nb/test.csv')

train
test

# convert the train and test data into a tuple format
trainx = tuple(zip(train.data,train.sentiment))
testx = tuple(zip(test.data))


len(train)
len(test)

# build the NB classifier
model = NaiveBayesClassifier(trainx)

# predict the sentiment on the testx
# has to be done in a loop for each sentence

# store the predicted class for each sentence
preds = []

for t in testx:
    preds.append(model.classify(t))
    
# add the predictions to the test dataset
test['sentiment'] = preds

print(test)
######################################################################

######################################################################
# web crawling
import requests
from bs4 import BeautifulSoup as bs 

# read the url
url = "https://www.sify.com"

# get the request
req = requests.get(url)

# get the contents of the page
data = bs(req.text, "html.parser")
print(data)

# extract all the webpage links from the tag "li"
lst_links = []

for links in data.find_all("li"):
    lst_links.append(links)
    
# get a count of links
len(lst_links)

# extract the links under the tag "li" -> "a"
lst_links = []
for links in data.find_all("li"):
    for a in links.find_all("a"):
        lst_links.append(a)
        
print(lst_links)
lst_links[0]
lst_links[1]
lst_links[10]
lst_links[2]

# i) get all the values within the tag <a> </a>
# get only valid tag names. invalid tag names are those having blank spaces
tagnames = []
lst_links[29].text # to get a tag of a single element

for l in lst_links:
    tag = l.text.strip()
    if len(tag) > 0:
        tagnames.append(tag)
    
print(tagnames)


# class exercise

# get all the hyperlinks within the tags of lst_links

# method 1 -> dictionary method
taglinks=[]

for links in lst_links:
    lst = links.attrs
    for i,j in lst.items():
        taglinks.append(j)

taglinks

links = lst_links[1]
links.attrs

# method 2 -> regular expression method
lst_links[0:10]
for link in lst_links[0:10]:
    link = str(link)
    print(re.findall('(?<=a href=")(.*)(?=">)', link))
#######################################################################

