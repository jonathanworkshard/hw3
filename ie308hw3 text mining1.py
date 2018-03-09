#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:48:15 2018

@author: jsshenkman

text mining 1
"""
import base64
from pickle import dumps, loads
import unittest
import tempfile
import nltk
import pandas as pd
import numpy as np
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk,sent_tokenize
from nltk.stem import porter
import matplotlib
from collections import Counter
from nltk.corpus import names, stopwords
from nltk.util import ngrams
import random
import string
from nltk.data import load

nltk.download()

import os
import pickle
from sklearn import metrics
import sklearn.decomposition

import matplotlib.pyplot
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
import sklearn.preprocessing
from sklearn.model_selection import cross_val_score


#clean labelling documents
def clean_ceos():
    """
    function to clean the file with names of ceos
    
    for now, just doing last names bc it will be mentioned at some point in the
    article
    """
    raw_names = pd.read_csv('/Users/jsshenkman/Documents/python/all/ceo.csv',encoding='latin-1',header=None)
    # concat to one column
    combined_names = raw_names[0] + ' ' + raw_names[1]
    #drop na and duplicates
    combined_names = combined_names.unique()
    names = combined_names[~pd.isnull(combined_names)]
    return names



# get documents
documents_2014 = os.listdir('/Users/jsshenkman/Documents/python/2014')
documents_2014 = ['/Users/jsshenkman/Documents/python/2014/' + element for element in documents_2014]
documents_2013 = os.listdir('/Users/jsshenkman/Documents/python/2013')
documents_2013 = ['/Users/jsshenkman/Documents/python/2013/' + element for element in documents_2013]

documents = documents_2013+documents_2014

#### understand the data


# determine max number of 'words' for elements in labelled sets
ceo_data = clean_ceos()
percent_data = list(pd.read_csv('/Users/jsshenkman/Documents/python/all/percentage.csv',encoding='latin-1',header=None)[0])
company_data = list(pd.read_csv('/Users/jsshenkman/Documents/python/all/companies.csv',encoding='latin-1',header=None)[0])

#break up into words
ceo_words = [word_tokenize(element) for element in ceo_data]
percent_words = [word_tokenize(str(element)) for element in percent_data]
company_words = [word_tokenize(str(element)) for element in company_data]

# get number of words
ceo_num_words = [len(element) for element in ceo_words]
percent_num_words = [len(element) for element in percent_words]
company_num_words = [len(element) for element in company_words]

#plot
matplotlib.pyplot.hist(ceo_num_words,bins = np.arange(max(ceo_num_words)+2))
matplotlib.pyplot.title('Number of Words in Each CEO Name')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()
matplotlib.pyplot.hist(percent_num_words,bins = np.arange(max(percent_num_words)+2))
matplotlib.pyplot.title('Number of Words in Each Percent')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()
matplotlib.pyplot.hist(company_num_words,bins = np.arange(max(company_num_words)+2))
matplotlib.pyplot.title('Number of Words in Each Company Name')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()


# identify important words that might be at end
last_word_company = [element[-1] for element in company_words]
last_word_name = [element[-1] for element in ceo_words]
last_word_percent = [element[-1] for element in percent_words]

common_company = nltk.FreqDist(last_word_company).most_common(20)
common_name = nltk.FreqDist(last_word_name).most_common(20)
common_percent = nltk.FreqDist(last_word_percent).most_common(20)
print(common_company)
print(common_name)
print(common_percent)

# take just the words, not counts
common_company = [element[0] for element in common_company]
common_name = [element[0] for element in common_name]
common_percent = [element[0] for element in common_percent]
#get most common
# only company really worth adding features for


def get_data(filepath):
    ""
    ""
    with open (filepath, "r",errors='ignore') as myfile:
        data=myfile.read()
    # get rid of \ and \n and \r
    data = data.replace('\n',' ')
    data = data.replace('\r',' ')
    data = data.replace('\'',"")
    return data


# tokenize document to word so it's searchable
def get_words(filepath):
    """
    """
    data=get_data(filepath)
    # break into sentences
    sentences = sent_tokenize(data)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words
    

# make flat list of words

def word_list(tokenized_words):
    """
    """
    flat_list = [item for sublist in tokenized_words for item in sublist]
    return flat_list

# function to determine if string is all ascii
def is_ascii(string):
    return all(ord(c) < 128 for c in string)



def get_feature_word(words,word_index):
    """
    returns the features for an individual word
    Inputs:
        words: list of all the words in a document
        word_index: index of the word in question to get features from
    Output:
        list of values to be used as features
    """
    # inintialize stemmer
   # stemmer = porter.PorterStemmer()
    
    # return the word
    word = words[word_index]
    len_word = len(word)
    #pos = word_index
    #words_left = len(words[sentence_index]) - (word_index+1)
    #sentence_pos = sentence_index
    all_caps = int(word.upper() == word)
    lower_case = int(word.lower() == word)
    start_cap = int(word[0] in word.capitalize())
    contains_perc_symbol = int('%' in word)
    contains_percent = int('percent'in word)
    contains_point = int('point' in word)
    numeric = int(word.isnumeric())
    all_ascii = int(is_ascii(word))
    #lemma =stemmer.stem(word)
    contains_dash = int('-' in word)
    contains_dot = int('.' in word)
    contains_company_word = int(word in common_company)
    return [len_word,all_caps,lower_case,
    start_cap,contains_perc_symbol,contains_percent,contains_point,numeric,
    all_ascii,contains_dash,contains_dot,contains_company_word]



def get_pos_tags(data):
    """
    get a flat list of pos tags for the document
    """
    # break up into sentences
    data_sent = sent_tokenize(data)
    pos_tags = [pos_tag(word_tokenize(sentence)) for sentence in data_sent]
    # make into a flat list
    pos_tags_array = np.concatenate(pos_tags)
    pos_tags = pos_tags_array[:,1]
    return pos_tags


def get_full_features(data, data_words, pos_tags, word_index, num_words):
    """
    Get all the features necessary for a word/ngram
    Inputs:
        data: raw data file
        data_words: list of all the words in the data file
        pos_tags: flat list of pos tage retrieved from get_pos_tags
        word_index: index of the desired word in the words file
        num_words: number of words in ngram
    Outputs:
        
    """
    lookback = 4
    # get index for all data to pull
    index_desired = np.arange(word_index-lookback,word_index+lookback+num_words)
    # get features
    features = [get_feature_word(data_words,index) for index in index_desired]
    # combine features
    features = np.concatenate(features)
    # get pos tags
    pos_tags = pos_tags[index_desired]
    # add pos tags to features
    features = np.concatenate([features,pos_tags])
    return features
    

    
# return instances of where labelled word occurs
# intialize dict of indexes
ceo_index_dict = dict([(key,[]) for key in ceo_data])
ceo_features_dict = dict([(key,[]) for key in ceo_data])
percent_index_dict = dict([(key,[]) for key in percent_data])
percent_features_dict = dict([(key,[]) for key in percent_data])
company_index_dict = dict([(key,[]) for key in company_data])
company_features_dict = dict([(key,[]) for key in company_data])


for i in np.arange(len(documents)):
    print('on document', i)
    data = get_data(documents[i])
    data_words = word_tokenize(data)
    data_words_bigrams = list(nltk.bigrams(data_words))
    data_words_trigrams = list(nltk.trigrams(data_words))
    used_words = [element  for element in ceo_data if element in data]
    pos_tags = get_pos_tags(data)
    for word in used_words:
        # breakdown word into words
        word_tokens = word_tokenize(word)
        # get num components
        num_words = len(word_tokens)
        # get index in list of words
        if num_words == 1:
            try:
                word_index = data_words.index(word_tokens)
            except ValueError:
                word_index = 'none'
        elif num_words == 2:
            try:
                word_index = data_words_bigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 3:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 4:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        else:
            word_index = 'none'
        

        
        
        # continue if able to find the word
        if word_index != 'none':
            # get features
            try:
                ceo_features_dict[word].append(get_full_features(data, data_words, pos_tags,word_index, num_words))
            except IndexError:
                # if getting data calls beyond index, can't use
                word_index = 'none'
            
        
        if word_index != 'none':    
            ceo_index_dict[word].append([i,word_index])

            
    used_words = [element  for element in percent_data if element in data]
    for word in used_words:
        # breakdown word into words
        word_tokens = word_tokenize(word)
        # get num components
        num_words = len(word_tokens)
        # get index in list of words
        if num_words == 1:
            try:
                word_index = data_words.index(word_tokens)
            except ValueError:
                word_index = 'none'
        elif num_words == 2:
            try:
                word_index = data_words_bigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 3:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 4:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        else:
            word_index = 'none'
        
        # continue if able to find the word
        if word_index != 'none':
            # get features
            try:
                percent_features_dict[word].append(get_full_features(data, data_words, pos_tags,word_index, num_words))
            except IndexError:
                # if getting data calls beyond index, can't use
                word_index = 'none'
            
        
        if word_index != 'none':    
            percent_index_dict[word].append([i,word_index])
            
    used_words = [element  for element in company_data if element in data]
    for word in used_words:
        # breakdown word into words
        word_tokens = word_tokenize(word)
        # get num components
        num_words = len(word_tokens)
        # get index in list of words
        if num_words == 1:
            try:
                word_index = data_words.index(word_tokens)
            except ValueError:
                word_index = 'none'
        elif num_words == 2:
            try:
                word_index = data_words_bigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 3:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        elif num_words == 4:
            try:
                word_index = data_words_trigrams.index(tuple(word_tokens))
            except ValueError:
                word_index = 'none'
        else:
            word_index = 'none'
        
        # continue if able to find the word
        if word_index != 'none':
            # get features
            try:
                company_features_dict[word].append(get_full_features(data, data_words, pos_tags,word_index, num_words))
            except IndexError:
                # if getting data calls beyond index, can't use
                word_index = 'none'
            
        
        if word_index != 'none':    
            company_index_dict[word].append([i,word_index])

"""
# concatenate dicts into feature sets
def dict_to_features(dictionary):
    """
    """
    # get rid of words it couldnt find
    # WRONG   ONFOSNFOJKSDFLKSBFGLKSJBDFGLKBSDFLGKBSDLFKHGBSDLHJKFBGSEHBFGLJ
    list_of_arrays = [x for x in list(ceo_features_dict.values()) if x != []]
    # concatenate
    features = np.concatenate(list_of_arrays)
    return features
    
ceo_features_positive = np.concatenate(list(ceo_features_dict.values()).dropna())
"""    
    
# choose other random words to inlcude as negatives
# random words should match the same distribution of grams and number observations
def get_negative_sample_index(index_dict):
    """
    """
    # get info about positive samples
    positive_samples = list(index_dict.keys())
    positive_words = [word_tokenize(element) for element in positive_samples]
    # get lengths
    positive_length = [len(element) for element in positive_words]
    # get count for each length
    # in future, change to be count of number of observations, not just keys
    positive_counts = [np.sum(np.array(positive_length) == element) for element in list([1,2,3,4])]
    # generate random numbers for the document
    # sort so that
    random_index_1 = np.random.randint(0,len(documents),positive_counts[0])
    random_index_2 = np.random.randint(0,len(documents),positive_counts[1])
    random_index_3 = np.random.randint(0,len(documents),positive_counts[2])
    random_index_4 = np.random.randint(0,len(documents),positive_counts[3])
    # split up into parts for each document
    random_index_split_1 = np.array_split(random_index_1,len(documents))
    random_index_split_2 = np.array_split(random_index_2,len(documents))
    random_index_split_3 = np.array_split(random_index_3,len(documents))
    random_index_split_4 = np.array_split(random_index_4,len(documents))
    negative_sample_dict = {}
    for i in np.arange(len(documents)):
        print('on document', i)
        data = get_data(documents[i])
        data_words = word_tokenize(data)
        data_words_bigrams = list(nltk.bigrams(data_words))
        data_words_trigrams = list(nltk.trigrams(data_words))
        pos_tags = get_pos_tags(data)
        features_1 = [get_full_features(data, data_words, pos_tags,element, 1) for element in random_index_split_1[i]]
        features_2 = [get_full_features(data, data_words, pos_tags,element, 2) for element in random_index_split_2[i]]
        features_3 = [get_full_features(data, data_words, pos_tags,element, 3) for element in random_index_split_3[i]]
        features_4 = [get_full_features(data, data_words, pos_tags,element, 4) for element in random_index_split_4[i]]
        negative_sample_dict[i] = list([features_1,features_2,features_3,features_4])
    return negative_sample_dict
        

name_negative_sample_dict = get_negative_sample_index(ceo_features_dict)
percent_negative_sample_dict = get_negative_sample_index(percent_features_dict)
company_negative_sample_dict = get_negative_sample_index(company_features_dict)



# make function to store files

files = list([name_negative_sample_dict, percent_negative_sample_dict, 
              company_negative_sample_dict, ceo_features_dict, company_features_dict,
              ceo_index_dict, company_index_dict, percent_index_dict])


# get all the negative features
files_negative = list([name_negative_sample_dict, percent_negative_sample_dict, 
              company_negative_sample_dict])



# break up feature sets into smaller dicts the same size as random
def make_smaller_dicts(large_dict, negative_dict):
    """
    large_dict: large feature dict with keys as names of words and input are
    arrays with features for occurences
    negative_dict: smaller negative sample dictionary
    Output:
        dictionaries: a list of dictionaries of the correct size for neg features
    """
    # sample dict size was determined by the number of words in large dict
    negative_length = len(list(large_dict.keys()))
    # get number obs in larger
    positive_length = 0
    for name in list(large_dict.keys()):
        positive_length += len(large_dict[name])
    
    # determine number of splits
    # make half as much negative data for efficiency
    num_splits = np.floor(positive_length/(2*negative_length))
    # split values
    # list values
    values_list = np.array(list(large_dict.values()))
    values_split = np.array_split(values_list,num_splits)
    keys_list = np.array(list(large_dict.keys()))
    keys_split = np.array_split(keys_list,num_splits)
    dictionaries = [dict(zip(keys_split[i],values_split[i])) for i in np.arange(len(keys_split))]
    return dictionaries


names_dicts = make_smaller_dicts(ceo_features_dict,name_negative_sample_dict)
company_dicts = make_smaller_dicts(company_features_dict,company_negative_sample_dict)
percent_dicts = make_smaller_dicts(percent_features_dict,percent_negative_sample_dict)

"""
# get features for each of these
names_features = [dict_to_feature_positive(element) for element in names_dicts]
compan_features = [dict_to_feature_positive(element) for element in names_dicts]
"""

# save all the negative files
store_data(files_negative,'negative')

# save each of the other dicts
store_data(names_dicts,'names')
store_data(company_dicts,'company')
store_data(percent_dicts,'percent')




# load the files saved above
files_negative = load_data(np.arange(3),'negative')

names_dicts = load_data(np.arange(3),'names')
company_dicts = load_data(np.arange(10),'company')





"""
# split up the large file into 5 components
large_list = np.array(list(percent_features_dict.values()))
#split into parts
large_list_splits = np.array_split(large_list,5)
# do the same with keys
list_keys = list(percent_features_dict.keys())
keys_split = np.array_split(np.array(list_keys),5)


# recreate smaller dicts

dict_of_dicts = {}
for i in np.arange(len(keys_split)):
    small_dict = dict(zip(keys_split[i],large_list_splits[i]))
    dict_of_dicts[i] = small_dict
    


# store splits
store_data(large_list_splits,0)
store_data(list(percent_features_dict.keys()),5)

# store variables
store_data(files,6)


#load data
data = load_data(np.arange(9))   
        
"""




files_positive = list([ceo_features_dict, company_features_dict, percent_features_dict])
# order in this file is name, percent, company
negative_feature_dicts = [dict_to_feature_negative(element) for element in files_negative]
#positive_feature_dicts = [dict_to_feature_positive(element) for element in np.array(files_positive)[[0,1]]]

# get positive features






def store_data(files,type_file):
    """
    stores the files in list of files, files, in pickle.  just named with a number
    for the order in input list
    """
    for i in np.arange(len(files)):
        with open('/Users/jsshenkman/Documents/python/text_mining/' + type_file+str(i) + '.pickle', 'wb') as handle:
            pickle.dump(files[i], handle)
    return



#def store_data(files,number):
#    """
#    stores the files in list of files, files, in pickle.  just named with a number
#    for the order in input list
#    """
#    for i in np.arange(len(files)):
#        with open('/Users/jsshenkman/Documents/python/dict'+str(i+number) + '.pickle', 'wb') as handle:
#            s_dump(files[i], handle)
#    return
      

def load_data(files, type_file):
    """
    """
    b = list()
    for i in np.arange(len(files)):
        
        with open('/Users/jsshenkman/Documents/python/text_mining/' + type_file+str(i) + '.pickle', 'rb') as handle:
            b.append(pickle.load(handle))
    return b



def dict_to_feature_negative(data_dict):
    """
    turns data from a dictionary into a list dataframes.  Each element of
    list is a dataframe for observations with 1, 2, 3, and 4 words
    """
    possible_len = np.arange(1,5)
    
    # initialize dict for features
    features = {}
    # retrieve all words of each length and make it's own dataframe
    for length in possible_len:
        # get values
        list_values = list(data_dict.values())
        raw_features = [element[length-1] for element in list_values if element[length-1] != []]
        # concatenate and store in dict if values retrieved
        try:
            
            features[length] = np.concatenate(raw_features)
        except ValueError:
            # if no features made for that length, make empty set
            features[length] = []
    return features

 
def dict_to_feature_positive(data_dict):
    """
    """
    possible_len = np.arange(1,5)
    
    keys = list(data_dict.keys())
    key_words = [word_tokenize(element) for element in keys]
    len_keys = [len(element) for element in key_words]
    
    # initialize feature dict
    features = {}
    # make feature set for each of the lengths
    for length in possible_len:
        keys_of_length = np.array(keys)[np.where(np.array(len_keys) == length)]
        # get non-null feature sets using keys
        raw_features = [data_dict[element] for element in keys_of_length if data_dict[element] != []]
        # concatenate and store in dict if values retrieved
        try:
            
            features[length] = np.concatenate(raw_features)
        except ValueError:
            # if no features made for that length, make empty set
            features[length] = []
    return features

def get_numeric_data(data):
    """
    takes the dataframe of combined data and extracts a dataframe of just the
    numeric variables
    Input:
        data: combined data dataframe
    Output:
        numeric_data: exactly what is sounds like.  pd dataframe
    """
    # column name
    numeric_columns =list()
    for column in np.arange(data.shape[1]):
        try:
            float(data[0,column])
            numeric_columns.append(column)
        except: ValueError
    data = data[:,numeric_columns]
    return data
    

def fit_onehotencoder():
    """
    need all the differnt
    """
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    tags = list(tagdict.keys())
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(tags)
    return label_encoder
    
    


def get_onehotencoded(train,length_word):
    """
    returns the training data with pos tags as onehotencoded version of 
    string variables in data
    Inputs:
        train: np array of training data where last columns are pos_tag
        length_word: the number of words in the phrase
    Output:
        train:
        
    """
    
    # get string variables
    train_string = train[:,-(8+length_word):]
    # initialize label encoder
    label_encoder = sklearn.preprocessing.LabelEncoder()
    train_labels = pd.DataFrame(train_string).apply(label_encoder.fit_transform)
    # get onehotlabels
    onehotlabeler = sklearn.preprocessing.OneHotEncoder()
    onehotlabels = onehotlabeler.fit_transform(train_labels).toarray()
    # replace strings with the onehotlabels
    train = np.concatenate((train[:,:-(8+length_word)],onehotlabels),axis=1)
    return train




def build_model(raw_positive_dict,dict_of_features_negative):
    """
    build a single gradient boosted model for each length
    Inputs:
        raw_positive_dict: dict of values before transformed into
        keys by length.  smaller dict to match negative size
        dict_of_features_negative: dictionary with all negative features
        keys are the lengths of grams
    Outputs:
        models_list: a list of models created.  Can determine the number of
        elements by which lengths in negative samples have greater than min_obs obs
        ordered from lowest to highest length
        score: average accuracy of cross validation
        
    """
    #initialize outputs
    models_list = list()
    total_score = list()
    
    # convert to features dict
    dict_of_features_positive = dict_to_feature_positive(raw_positive_dict)
    # generate possible lengths
    possible_len = np.arange(1,5)
    # go through each value and train model for it
    min_obs = 100
    used = list()
    for length in possible_len:
        pos_length = len(dict_of_features_positive[length])
        neg_length = len(dict_of_features_negative[length])
        # only build model if more than min_obs item
        if (pos_length> min_obs) & (neg_length>min_obs):
            # repeat neg data so dataset balanced
            negatives = np.concatenate(np.repeat([dict_of_features_negative[length]],np.ceil(pos_length/neg_length),axis=0))
            train = np.concatenate((dict_of_features_positive[length],negatives))
            # turn pos labels into onehotencoded and append
            # train = get_onehotencoded(train,length)
            # only use numeric columns
            train = get_numeric_data(train)
            output = np.concatenate((np.ones(pos_length),np.zeros(negatives.shape[0])))
            # scale train data
            train = sklearn.preprocessing.scale(train)
            pca= sklearn.decomposition.PCA(10)
            train_pca = pca.fit_transform(train)
            # train model
            gbm = GradientBoostingClassifier()
            score = np.mean(cross_val_score(gbm,train_pca,output))
            gbm.fit(train_pca,output)
            models_list.append(gbm)
            total_score.append(score)
            used.append(length)
    total_score = np.mean(total_score)
    return [models_list,total_score, used]


def get_article_features(filepath):
    """
    gets the features from an article found at filepath
    
    Input:
        filepath: filepath of the article to get features from
    """
    data = get_data(filepath)
    # get tokenized words
    words = word_tokenize(data)
    #word_features = [get_feature_word(words,index) for index in np.arange(len(words))]
    pos_tags = get_pos_tags(data)
    unigram_features = get_numeric_data(np.array([get_full_features(data, words, pos_tags, index, 1) for index in np.arange(4,len(words) -1 -4)]))
    bigram_features = get_numeric_data(np.array([get_full_features(data, words, pos_tags, index, 2) for index in np.arange(4,len(words) -2 -4)]))
    trigram_features = get_numeric_data(np.array([get_full_features(data, words, pos_tags, index, 3) for index in np.arange(4,len(words) -3 -4)]))
    quadgram_features = get_numeric_data(np.array([get_full_features(data, words, pos_tags, index, 4) for index in np.arange(4,len(words) -4 -4)]))
    return words, unigram_features, bigram_features, trigram_features, quadgram_features


def make_prediction(filepath,models,length_used):
    """
    Inputs:
        filepath: filepath of the document to download
        models: list of lists of models to predict an entity
        length_used: list of the length words that were used

        
    """
    # determine which models will be running
    #above_thresh = [len(negative_feature_dict[i])>100 for i in np.arange(1,5)]
    # collect data
    words, unigram_features, bigram_features, trigram_features, quadgram_features = get_article_features(filepath)
    # initialize model counter
    counter = 0
    pca= sklearn.decomposition.PCA(10)
    unigrams_pred = list()
    bigrams_pred = list()
    trigrams_pred = list()
    quadgrams_pred = list()
    if length_used[min(counter,len(length_used)-1)] ==1:
        unigram_features = pca.fit_transform(unigram_features)
        unigram_predictions = np.array([element[counter].predict(unigram_features) for element in models])
        counter+=1
        consensus_predictions = np.where(unigram_predictions.mean(axis=0)>.5)[0]
        unigrams_pred = [words[prediction:prediction+1] for prediction in consensus_predictions]
    if length_used[min(counter,len(length_used)-1)] ==2:
        bigram_features = pca.fit_transform(bigram_features)
        bigram_predictions = np.array([element[counter].predict(bigram_features) for element in models])
        counter+=1
        # where mean prediction greater than .5, majority predict positive
        consensus_predictions = np.where(bigram_predictions.mean(axis=0)>.5)[0]
        bigrams_pred = [words[prediction:prediction+2] for prediction in consensus_predictions]
    if length_used[min(counter,len(length_used)-1)] ==3:
        trigram_features = pca.fit_transform(trigram_features)
        trigram_predictions = np.array([element[counter].predict(trigram_features) for element in models])
        counter+=1
        consensus_predictions = np.where(trigram_predictions.mean(axis=0)>.5)[0]
        trigrams_pred = [words[prediction:prediction+3] for prediction in consensus_predictions]
    if length_used[min(counter,len(length_used)-1)] ==4:
        quadgram_features = pca.fit_transform(quadgram_features)
        quadgram_predictions = np.array([element[counter].predict(quadgram_features) for element in models])
        counter+=1
        consensus_predictions = np.where(quadgram_predictions.mean(axis=0)>.5)[0]
        quadgrams_pred = [words[prediction:prediction+4] for prediction in consensus_predictions]
    
    predictions = unigrams_pred + bigrams_pred + trigrams_pred + quadgrams_pred
    return predictions

        

            
            
# get models
names_results = [build_model(element,negative_feature_dicts[0]) for element in names_dicts]
# separate the models from the score
# each element in models will be a list of the models for each ngram
names_models = [element[0] for element in names_results]
names_score = np.mean([element[1] for element in names_results])
names_used = names_results[0][2]
print('cross validation accuracy of ceo names trainer is', names_score)


company_results = [build_model(element,negative_feature_dicts[2]) for element in company_dicts]
# separate the models from the score
# each element in models will be a list of the models for each ngram
company_models = [element[0] for element in company_results]
company_score = np.mean([element[1] for element in company_results])
company_used = company_results[0][2]
print('cross validation accuracy of company trainer is', company_score)


percent_results = [build_model(element,negative_feature_dicts[1]) for element in percent_dicts]
# separate the models from the score
# each element in models will be a list of the models for each ngram
percent_models = [element[0] for element in percent_results]
percent_score = np.mean([element[1] for element in percent_results])
percent_used = percent_results[0][2]
print('cross validation accuracy of company trainer is', percent_score)


# save models
store_data(names_models,'names_models')
store_data(company_models,'company_models')

# load models
names_models = load_data(np.arange(3),'names_models')
company_models = load_data(np.arange(10),'company_models')



# make predictions
names_predictions = make_prediction(documents[1],names_models,names_used)
company_predictions = make_prediction(documents[1],company_models,company_used)

# export predictions to csv
pd.DataFrame(names_predictions).to_csv('/Users/jsshenkman/Documents/python/names_predictions.csv')
pd.DataFrame(company_predictions).to_csv('/Users/jsshenkman/Documents/python/company_predictions.csv')

def get_names():
    



# get features for negative words






def get_features(filepath):
    """
    returns features for all words in a document
    input:
        filepath: the filepath for the txt file to extract features for
    """
    
    
    # get features for classifiers
    
# make function to extract features from words in doc




