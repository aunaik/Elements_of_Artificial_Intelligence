#!/usr/bin/env python
# B551 Elements of AI, Prof. David Crandall

import re
import operator
import sys

# English Stopword list referred from "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
# and added more words.
stopword = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'off', 'yours', 'so', 'the', 'having', 'once', 'jobs','job','amp','im']

# Calculates the number of words in a city's tweets
def total_word_count(city):
    return sum(city[word] for word in city)

# Finds out the city a particular tweet belong to
# tweet: The tweet which we want to classify
# likelihood: A dictionary of probabilities of different words
# being in different cities (i.e, Multinomial distribution of words in each city)
# Prior: A Dictionary of probability of a tweet being in a particular city for all cities
def tweet_classification(tweet, likelihood, prior):
    temp={}
    for city in prior:
        p_value = 0
        for word in range(1,len(tweet)):
            if p_value == 0:
                p_value = likelihood[city].get(tweet[word],10**-6)
            else:
                p_value *= likelihood[city].get(tweet[word],10**-6)
        temp[city] = p_value * prior[city]
    return max(temp.iteritems(), key=operator.itemgetter(1))[0]

# Normalizes the probability value for each word occuring in tweets for all the cities.
# Which in a way creates the multinomial distribution of words belonging to tweets of a particular city.
def normalization(word_freq,prior,city_count):
    for city in word_freq:
        prior[city]=float(prior[city])/len(train)
        total=total_word_count(word_freq[city])
        for word in word_freq[city]:
            likelihood[city][word]= float(word_freq[city][word])/total

# Computes the frequency of a word being present in tweets of a particular city
# as well as calculates the number of tweets belonging to a particular city
def cal_freq(train, word_freq):
    for tweet in range(len(train)):
        if train[tweet][0] in word_freq:
            prior[train[tweet][0]]+=1
            for word in range(1,len(train[tweet])):
                word_freq[train[tweet][0]][train[tweet][word]] = word_freq[train[tweet][0]].get(train[tweet][word],0) + 1
        else:
            temp = {}
            for word in range(1,len(train[tweet])):
                temp[train[tweet][word]] = 1.0
            word_freq[train[tweet][0]] = temp
            prior[train[tweet][0]] = 1
    for city in prior:
        likelihood[city]={}
    city_count=len(prior)
    normalization(word_freq, prior, city_count)

train = []
test=[]
op=[]
city_count=0
prior = {}
word_freq = {}
likelihood = {}

# Reads the trainig data and clean it (removes the special characters as well as carriage return)
with open(sys.argv[1]) as inputfile:
    for line in inputfile:
        split_line = line.split(" ", 1)
        temp2=[word for word in [re.sub(r'[^a-zA-Z0-9]',r'',str(word)) for word in str(split_line[1]).lower().split() ] if word not in stopword]
        train.append(" ".join([str(split_line[0]), " ".join(re.sub(' +',' '," ".join(temp2)).split())]).split())

# Reads the testing data and clean it (removes the special characters as well as carriage return)
with open(sys.argv[2]) as inputfile:
    for line in inputfile:
        op.append(line)
        split_line = line.split(" ", 1)
        temp2=[word for word in [re.sub(r'[^a-zA-Z0-9]',r'',str(word)) for word in str(split_line[1]).lower().split() ] if word not in stopword]
        test.append(" ".join([str(split_line[0]), " ".join(re.sub(' +',' '," ".join(temp2)).split())]).split())

# Function call to calculate the word frequency matrx
cal_freq(train, word_freq)

# Open a file and write the output to the file in format('The predicted city' | 'with the actual test tweet')
file = open(sys.argv[3],"w")
[file.write(" ".join([tweet_classification(test[tweet],likelihood, prior),op[tweet]])) for tweet in range(len(test))]
file.close()

# Prints the top five words in each city
for city in word_freq:
    print city ,":", ", ".join(sorted(word_freq[city], key=lambda x:word_freq[city][x], reverse=True)[0:5])
