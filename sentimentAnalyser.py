# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:02:36 2020

@author: KRITANK SINGH
"""
#libraries needed
import os
import string
import numpy as np
import matplotlib.pyplot as plt

#defining stopwords
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
             'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
             'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
             'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
             'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
             'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
             'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone',
             'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for',
             'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
             'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less',
             'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine',
             'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once',
             'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
             'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 
             'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
             't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
             'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this',
             'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
             'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
             'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
             'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
             'yourselves']


#from nltk.stem import PorterStemmer
class Features:
    def __init__(self):
        self.vocab = {}
        self.features = []
        self.num_words =[] 
        self.cutoff_freq = 0
        
    def buildVocabulary(self,X_train):
        #ps = PorterStemmer()
        for i in range(len(X_train)):
            
            for word in X_train[i][1].split():
                word_new  = word.strip(string.punctuation).lower()
                if (len(word_new)>2)  and (word_new not in stopwords): 
                    #word_new=ps.stem(word)
                    if word_new in self.vocab:
                        self.vocab[word_new]+=1
                    else:
                        self.vocab[word_new]=1  
                        
    def plotVocabulary(self):
        self.num_words = [0 for i in range(max(self.vocab.values())+1)] 
        freq = [i for i in range(max(self.vocab.values())+1)] 
        for key in self.vocab:
            self.num_words[self.vocab[key]]+=1
        plt.plot(freq,self.num_words)
        plt.axis([1,10, 0, 20000])
        plt.xlabel("Frequency")
        plt.ylabel("No of words")
        plt.grid()
        plt.show()
        
    def setCutOff(self,num):
        self.cutoff_freq = num
        # For deciding cutoff frequency
        num_words_above_cutoff = len(self.vocab)-sum(self.num_words[0:self.cutoff_freq]) 
        print("Number of words with frequency higher than cutoff frequency({}) :".format(self.cutoff_freq),num_words_above_cutoff)

    def buildFeatures(self):
        for key in self.vocab:
            if self.vocab[key] >=self.cutoff_freq:
                self.features.append(key)
     
                
# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:
    
    def __init__(self,feature):
        # count is a dictionary which stores several dictionaries corresponding to each sentiment category
        # each value in the subdictionary represents the freq of the key corresponding to that setiment category 
        self.count = {}
        # classes represents the different sentiment categories
        self.classes = None
        self.feature=feature
    
    def fit(self,X_train,Y_train):
        # This can take some time to complete       
        self.classes = set(Y_train)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
            self.count[class_]['total_points'] = 0
        self.count['total_points'] = len(X_train)
        
        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.count[Y_train[i]][j]+=X_train[i][j]
                self.count[Y_train[i]]['total']+=X_train[i][j]
            self.count[Y_train[i]]['total_points']+=1
    
    def __probability(self,test_point,class_):
        
        log_prob = np.log(self.count[class_]['total_points']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i]*(np.log(self.count[class_][i]+1)-np.log(self.count[class_]['total']+total_words))
            log_prob += current_word_prob
        
        return log_prob
    
    
    def __predictSinglePoint(self,test_point):
        
        best_class = None
        best_prob = None
        first_run = True
        
        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_point,class_)
            if (first_run) or (log_probability_current_class > best_prob) :
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False
                
        return best_class
        
  
    def predict(self,X_test):
        # This can take some time to complete
        Y_pred = [] 
        for i in range(len(X_test)):
        # print(i) # Uncomment to see progress
            Y_pred.append( self.__predictSinglePoint(X_test[i]) )
        
        return Y_pred
    
    def accuracy(self,Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count):
        # returns the mean accuracy
        return (Pos_true_count+Neg_true_count)/(Pos_true_count+Neg_true_count+Pos_false_count+Neg_false_count)
        
    def recall(self,Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count):
        #returns recall
        return Pos_true_count/(Pos_true_count+Pos_false_count)
    
    def precision(self,Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count):
        #returns precision
        return Pos_true_count/(Pos_true_count+Neg_false_count)
    
    def fmeasure(self,recall,precision):
        #returns fmeasure
        return (2*recall*precision)/(recall+precision)
    
    def generateReport(self,Y_pred,Y_true):
        #generate report
        Pos_true_count = 0
        Neg_true_count = 0
        Pos_false_count = 0
        Neg_false_count = 0
        #count = 0
        for i in range(len(Y_pred)):
            #if Y_pred[i] == Y_true[i]:
            #    count+=1
            if Y_pred[i]==Y_true[i] and Y_true[i] =="Positive":
                Pos_true_count+=1
            elif Y_pred[i]==Y_true[i] and Y_true[i] =="Negative":
                Neg_true_count+=1
            elif Y_pred[i]!=Y_true[i] and Y_true[i] =="Positive":
                Pos_false_count+=1
            elif Y_pred[i]!=Y_true[i] and Y_true[i] =="Negative":
                Neg_false_count+=1;
        print("Report:")
        print("Accuracy:",round(self.accuracy(Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count),2))
        recall=self.recall(Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count)
        print("Recall:",round(recall,2))
        precision=self.precision(Pos_true_count,Neg_true_count,Pos_false_count,Neg_false_count)
        print("Precision:",round(precision,2))
        print("Fmeasure:",round(self.fmeasure(recall,precision),2))
        
    
    def getWordVectorCount(self,X):
        # To represent test data as word vector counts
        X_dataset = np.zeros((len(X),len(feature.features)))
        # This can take some time to complete
        for i in range(len(X)):
            # print(i) # Uncomment to see progress
            word_list = [ word.strip(string.punctuation).lower() for word in X[i][1].split()]
            for word in word_list:
                if word in feature.features:
                    X_dataset[i][feature.features.index(word)] += 1
        return X_dataset
    
    def getCleanedWordVectorCount(self,check):
        X_check = np.zeros((len(check),len(feature.features)))
        #ps = PorterStemmer()
        # This can take some time to complete
        for i in range(len(check)):
            # print(i) # Uncomment to see progress
            word_list = [ word.strip(string.punctuation).lower() for word in check[i].split()]
            word_list = [word for word in word_list if word not in stopwords]
            for word in word_list:
                #word=ps.stem(word)
                if word in feature.features:
                    X_check[i][feature.features.index(word)] += 1
        return X_check
        
    def predictClass(self,check):
        X_check=self.getCleanedWordVectorCount(check)
        Y_pred=self.predict(X_check)
        return Y_pred
        
        
# Helper Functions
def load(directory):
    X = [] # an element of X is represented as (filename,text)
    Y = [] # an element of Y represents the sentiment category of the corresponding X element
    for category in os.listdir(directory):
        for document in os.listdir(directory+'/'+category):
            with open(directory+'/'+category+'/'+document, "r", encoding="utf8") as f:
                X.append((document,f.read()))
                Y.append(category)
    return X,Y;




#Setting the data directory
#train_directory = 'Training_Data'
train_directory = 'train'
#test_directory = 'Test_Data'
test_directory = 'test'

#Loading the dataset
X_train,Y_train = load(train_directory)
X_test,Y_test = load(test_directory)

print("Data size:",len(X_train)+len(X_test))
print("Train Data size:",len(X_train))
print("Test Data size:",len(X_test))

#Annalysis of Data
#displaying postive and negative review count
pos_count=0
neg_count=0
for i in range(len(Y_train)):
    if(Y_train[i]== "Positive"):
        pos_count+=1
    else : neg_count+=1
for i in range(len(Y_test)):
    if(Y_test[i]== "Positive"):
        pos_count+=1
    else : neg_count+=1
           
print("Total Positive in dataset:",pos_count)
print("Total Negative in dataset:",neg_count)

#displaying average postive and negative review length
pos_len=0
neg_len=0

for i in range(len(Y_train)):
    if(Y_train[i]== "Positive"):
        pos_len+=len(X_train[i][1].split())
    else : neg_len+=len(X_train[i][1].split())
for i in range(len(Y_test)):
    if(Y_test[i]== "Positive"):
        pos_len+=len(X_test[i][1].split())
    else : neg_len+=len(X_test[i][1].split())

print("Average postive review Length:",round(pos_len/pos_count,2))
print("Average negative review Length:",round(neg_len/neg_count,2))





feature= Features()
feature.buildVocabulary(X_train) #building vocabulary
feature.plotVocabulary()        
feature.setCutOff(70)
feature.buildFeatures()          #building features

#building model
clf2 = MultinomialNaiveBayes(feature)
X_train_dataset=clf2.getWordVectorCount(X_train) 
X_test_dataset=clf2.getWordVectorCount(X_test)
clf2.fit(X_train_dataset,Y_train)     #training model
Y_test_pred = clf2.predict(X_test_dataset)
clf2.generateReport(Y_test_pred,Y_test)  
#print("Our score on testing data :",our_score_test)

while True:
    check = []
    replay = input("Want to test on custom reviews? ").lower()
    if replay in ("yes", "y"):
        review = input("Enter a movie Review : ")
        check.append(review)
        print()
        print(clf2.predictClass(check))
    elif replay in ("no", "n"):
        break
    else:
        print ("Sorry, I didn't understand that.")

print("done!")