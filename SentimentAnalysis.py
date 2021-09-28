"""
	This is a file for the Sentiment Analysis algorithm
"""

import nltk
from nltk.stem import WordNetLemmatizer


# returns a list of tuples, each of which is formated as (id, example_text) for test data and (id, label, example_text) for training data
def data_tuple_pairs(file_path, is_training):

	lines = open(file_path, "r").readlines()

	hold_returns = []

	for l in lines:
		separate_elements = l.split(',')
		i_d = separate_elements[0]
		if (is_training):
			label = separate_elements[1]
			text = separate_elements[2]
			hold_returns.append(tuple((i_d, label, text)))
		else:
			text = separate_elements[1]
			hold_returns.append(tuple((i_d, text)))

	# the first line is just the tag names, so remove it
	return (hold_returns[1:])


class SentimentAnalysis():

	def __init__(self):
		pass

	# preprocesses text data
	def preprocess_data(self):
		stop_words = open('./DataSets/stopwords.txt', "r")
		lemmatizer = WordNetLemmatizer() 

	# trains the model using the training data tupels
	def train_model(self, training_data):
		pass

	


if __name__ == '__main__':

	sa = SentimentAnalysis()

	# read data for training the model
	data_tuple_pairs("./DataSets/train.csv", True)

	#TODO: do stuff for training model here

	# read data for testing the model
	data_tuple_pairs("./DataSets/test.csv", False)

	#TODO: do stuff for testing accuracy of model here



	# run sa once with 1-grams
	# calculate accuracy

	# run sa once with 2-grams
	# calculate accuracy

	# run sa once with 3-grams
	# calculate accuracy

# ----------------------------------------------------------------------------------------------------------------------------------------



# import numpy as np
# import sys
# from collections import Counter
# import math
# import string
# import re


# # used to write classificaitons to file
# def write_to_file(filename, labels_list, class_list):
#     out_file = open(filename, "w")
    
#     return_list = []
#     if len(labels_list) == len(class_list):
    
#         idx = 0
#         while idx < len(labels_list):
#             return_list.append((labels_list[idx], class_list[idx]))
#             idx += 1

#         for i in enumerate(return_list):
#             output = i[1]
#             output = (str(output[0])+ " " + str(output[1]))        

#             out_file.write(output)
#             out_file.write("\n")
            
#     out_file.close()


    
# # for the following THREE functions...
# # classified labels is a list of strings of the labels assigned by the classifier
# # gold labels is a list of strings of the true labels
# # values is a return of a list in the order of (true positive, true negative, false negative, false positive)
# def precision(gold_labels, classified_labels): 

#     values = get_pos_neg(gold_labels, classified_labels) 
#     true_pos = values[0]
#     false_pos = values[3]
#     precision = true_pos / (true_pos + false_pos)
    
#     print("Precision is: ", precision)
#     print()
#     return precision # returns as a float

# def recall(gold_labels, classified_labels): 
    
#     values = get_pos_neg(gold_labels, classified_labels)
#     true_pos = values[0]
#     false_neg = values[2]
#     recall = true_pos / (true_pos + false_neg)
    
#     print("Recall is: ", recall)
#     return recall # returns as a float

# def f1(gold_labels, classified_labels): 
    
#     values = get_pos_neg(gold_labels, classified_labels)
#     recall_val = recall(gold_labels, classified_labels)
#     precision_val = precision(gold_labels, classified_labels)
#     f1 = (2*precision_val*recall_val) / (recall_val + precision_val)
    
#     return f1 # returns as a float


# # function to get the true pos/negs and false pos/negs to use in calculations above
# def get_pos_neg(gold_labels, predicted_labels):
#     true_pos = 0
#     true_neg = 0
#     false_neg = 0
#     false_pos = 0
    
#     idx = 0
#     while idx < len(predicted_labels):

#         if predicted_labels[idx] == 1 and gold_labels[idx] == 1:
#             true_pos += 1
#         elif predicted_labels[idx] == 0 and gold_labels[idx] == 0:
#             true_neg += 1
#         elif predicted_labels[idx] == 1 and gold_labels[idx] == 0:
#             false_pos += 1
#         elif predicted_labels[idx] == 0 and gold_labels[idx] == 1:
#              false_neg += 1

#         idx += 1

#     return (true_pos, true_neg, false_neg, false_pos)


# # F1 score should be higher
# class SentimentAnalysisImproved:

#     def __init__(self):
#         self.true_class_dict = {} # dictinary to hold true counts
#         self.false_class_dict = {} # dictionary to hold false counts
#         self.pos_count = {}
#         self.neg_count = {}
#         self.pos = 0 # class positive count
#         self.neg = 0 # class negative count
#         self.pos_word_counts = 0 # count of words of positive class
#         self.neg_word_counts = 0 # count of words of negative class
#         self.positive_probs = {} # positive probabilities
#         self.negative_probs = {} # negative probabilities
#         self.total_vocab = {} # dictionary to hold total vocab

#     def train(self, examples): # try normalization if score does not improve
        
#         features = []
#         # implement pre-processing
#         for e in examples:
            
#             # make sentence in the lower case
#             sentence = e[1]
#             sentence = sentence.lower()
            
#             # remove all non-alphanumeric values from string
#             sentence = ''.join([i for i in sentence if i.isalpha() or i.isspace()])
            
#             # remove all stop words and lemmatize words
#             hold_words = []
#             processed_sentence = []
#             hold_words.append(sentence.split())
#             for sentence in hold_words:
#                 for word in sentence:
#                     if word not in stop_words:
#                         word_new = lemmatizer.lemmatize(word)
#                         processed_sentence.append(word_new)
            
#             sentence = ' '.join(processed_sentence)            
#             features.append(self.featurize((e[0], sentence, e[2])))
        
#         features = [i for lst in features for i in lst]
        
#         for f in features:
#             # if positive
#             if f[1] == "1":
#                 self.pos += 1
#                 if f[0] not in self.pos_count:
#                     self.pos_count[f[0]] = 1
#                 else:
#                     self.pos_count[f[0]] += 1  
#             # if negative
#             elif f[1] == "0":
#                 self.neg += 1
#                 if f[0] not in self.neg_count:
#                     self.neg_count[f[0]] = 1
#                 else:
#                     self.neg_count[f[0]] += 1
            
#         # handle situation when word occurs in one class, but not in the other --> normalizing dictionary
#         for key, val in self.pos_count.items():
#             if key not in self.neg_count:
#                 self.neg_count[key] = 0

#         for key, val in self.neg_count.items():
#             if key not in self.pos_count:
#                 self.pos_count[key] = 0
                    
#         # get a total vocab
#         for word in self.pos_count:
#             if word not in self.total_vocab:
#                 self.total_vocab[word] = 1
#             else:
#                 self.total_vocab[word] += 1
#         for word in self.neg_count:
#             if word not in self.total_vocab:
#                 self.total_vocab[word] = 1
#             else:
#                 self.total_vocab[word] += 1
                
#         vals = self.total_vocab.values()
#         total_vocab_words = sum(vals)
        
#         #get prob of each word in pos class
#         for word in self.pos_count:
#             prob = self.pos_count[word] / self.pos
            
#             if prob == 0:
#                 # do laplace smoothing
#                 prob = (self.pos_count[word] + 1) / (self.pos + total_vocab_words)

#             self.positive_probs[word] = prob
        
#         #get prob of each word in neg class
#         for word in self.neg_count:
#             prob = self.neg_count[word] / self.neg
            
#             if prob == 0:
#                 # do laplace smoothing
#                 prob = (self.neg_count[word] + 1) / (self.neg + total_vocab_words)
                
#             self.negative_probs[word] = prob


#         self.pos_class_prob = self.pos / (self.pos + self.neg)
#         self.neg_class_prob = self.neg / (self.pos + self.neg)
        
#         print("positive class prob is: ", self.pos_class_prob)
#         print("negative class prob is: ", self.neg_class_prob)

            
#         print("Training finished.")

#     def score(self, data):
#         i_d = data[0]
#         sentence = data[1]
        
#         data_pos = (i_d, sentence, 1)
#         data_neg = (i_d, sentence, 0)
        
#         # get data in form of (word, label) for both negative and positive labels
#         feat_pos = self.featurize(data_pos)
#         feat_neg = self.featurize(data_neg)
        
#         prob_pos = self.pos_class_prob
#         for f in feat_pos:
#             if f[0] in self.positive_probs:
#                 prob_word = self.positive_probs[f[0]]
#                 prob_pos *= prob_word 
                            
#         prob_neg = self.neg_class_prob
#         for f in feat_neg:
#             if f[0] in self.negative_probs:
#                 prob_word = self.negative_probs[f[0]]
#                 prob_neg *= prob_word 

#         return(prob_pos, prob_neg)


#     def classify(self, data):
#         pos_prob, neg_prob = self.score(data)

#         if pos_prob > neg_prob: # is positive
#             return 1
#         else:
#             return 0 # is negative

#     def featurize(self, data):
#         sentence = data[1]
#         label = data[2]
        
#         sentence = sentence.split()
#         feat = []
#         for s in sentence:
#             feat.append((s, label))
            
#         return feat

#     def __str__(self):
#         return "NAME FOR YOUR CLASSIFIER HERE"


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
#         sys.exit(1)

#     training = sys.argv[1]
#     testing = sys.argv[2]

#     print("SentimentAnalysis Complete.")


     
#     # ================ running improved class ================
#     print()
#     print("======== SentimentAnalysisImproved Class ========")
#     sai = SentimentAnalysisImproved()
    
#     print()
#     print("Runing SentimentAnalysisImproved training...")
#     tuples_list = generate_tuples_from_file(training)
#     sai.train(tuples_list)
    
#     print()
#     print("Running SentimentAnalysisImproved classify...")
#     print("From dev_file.txt")
#     tuples_list = generate_tuples_from_file("dev_file.txt")
#     classification = []
#     files_labes = []
#     for i in tuples_list:
#         files_labes.append(int(i[2]))
#         classification.append(sai.classify(i))
        
#     print("classification from function: ", classification) # classified labels
#     print("classification from file:     ", files_labes) # gold labels
    
#     print()
#     print("Calculating F1 score...")
#     print("F1 score is: ", f1(files_labes, classification))
    
#     print()
#     print("Generating classifications from test data...")
#     tuples_list_test = generate_tuples_from_test(testing)
#     test_classificaitons = []
#     test_labels = []
#     for i in tuples_list_test:
#         test_labels.append(i[0])
#         test_classificaitons.append(sa.classify(i))
#     print("Classificaitons from test data finished.")
    
    
#     print()
#     print("SentimentAnalysisImproved Complete.")
    
