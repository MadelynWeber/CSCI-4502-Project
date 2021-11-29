"""
	This is a file for the Sentiment Analysis model
"""

import nltk
from nltk.stem import WordNetLemmatizer
import re
from contractions import contractions_dict # contains a mapping of frequent contractions in English and their expanded forms
import random
import math


# returns the calculated precision for the classification
def precision(gold_labels, classified_labels):

	values = calculate_pos_neg(gold_labels, classified_labels)
	true_pos = values[0]
	false_pos = values[3]
	precision = true_pos / (true_pos + false_pos)

	return precision

# returns the calculated recall for the classification
def recall(gold_labels, classified_labels):

	values = calculate_pos_neg(gold_labels, classified_labels)
	true_pos = values[0]
	false_neg = values[2]
	recall = true_pos / (true_pos + false_neg)

	return recall

# returns the true and false positive and negative values from the classificaiton
def calculate_pos_neg(gold_labels, predicted_labels):

	true_pos = 0
	true_neg = 0
	false_neg = 0
	false_pos = 0

	idx = 0
	while idx < len(predicted_labels):
		if predicted_labels[idx] == 1 and gold_labels[idx] == 1:
			true_pos += 1
		elif predicted_labels[idx] == 0 and gold_labels[idx] == 0:
			true_neg += 1
		elif predicted_labels[idx] == 1 and gold_labels[idx] == 0:
			false_pos += 1
		elif predicted_labels[idx] == 0 and gold_labels[idx] == 1:
			false_neg += 1
		idx += 1

	return(true_pos, true_neg, false_neg, false_pos)

# returns the calculated F1-score for the classification
def calculate_f1(gold_labels, classified_labels):

	values = calculate_pos_neg(gold_labels, classified_labels)
	recall_val = recall(gold_labels, classified_labels)
	precision_val = precision(gold_labels, classified_labels)
	f1 = (2*precision_val*recall_val) / (recall_val + precision_val)

	return f1

# returns a list of tuples, each of which is formated as (id, example_text) for test data and (id, label, example_text) for training data
def data_tuple_pairs(lines, is_training):

	hold_returns = []

	for l in lines:
		separate_elements = l.split('\t')
		if (is_training):
			label = separate_elements[1]
			if "\n" in label:
				label = label.strip()
			text = separate_elements[0]
			hold_returns.append(tuple((text, label)))
		else:
			text = separate_elements[0]
			hold_returns.append(text)

	return hold_returns

# returns list of uni-grams for each sentence 
def get_uni_grams(sentence, n):
	is_final = False # flag to mark final tuple for ngram
	hold_ngrams = []
	for t in range(len(sentence)):
		if is_final == False:
			ngram = sentence[t: t+n]
			hold_ngrams.append(tuple(ngram))
	return hold_ngrams


class SentimentAnalysis():

	def __init__(self):
		self.trained_prob_pos = {}		# probability of positive classificaiton from training for each word
		self.trained_prob_neg = {}		# probability of negative classification from training for each word
		self.class_positive_count = 0	# count of all positive instances from training
		self.class_negative_count = 0	# count of all negative instances from training
		self.positive_words_dict = {}		# holds total occurances of words in positive classification
		self.negative_words_dict = {}		# holds total occurances of words in negative classificaiton
		self.total_vocab_dict = {}		# holds total occurances of all words in vocabular (positive and negative classifications)
		self.calculated_class_positive_prob = 0		# probability of class being positive
		self.calculated_class_negative_prob = 0		# probability of class being negative


	# preprocesses text data - text is the sentences from the data
	def preprocess_data(self, text):

		stopwords_data = open('./DataSets/stopwords.txt', "r")
		lemmatizer = WordNetLemmatizer()

		# put all stopwords in lower case and remove new line characters
		stop_words = []
		for word in stopwords_data:
			word = word.lower()
			if "\n" in word:
				word = word.strip()
			stop_words.append(word)

		text = text.lower()

		text = ''.join([i for i in text if i.isalpha() or i.isspace()])
		word_list = text.split()

		# removes any residual non-English characters that weren't removed already
		cleaned_list = []
		for word in word_list:
			new_word = re.sub("[^a-zA-Z0-9]+", "",word)
			if new_word != '':
				cleaned_list.append(new_word)

		new_cleaned_list = []
		for word in cleaned_list:
			expansion = []
			if word in contractions_dict:
				word = contractions_dict[word]
				new_words = word.split()
				for idx, item in enumerate(new_words):
					new_cleaned_list.append(item)
			else:
				new_cleaned_list.append(word)

		for_return = []
		for word in new_cleaned_list:
			if word not in stop_words:
				word = lemmatizer.lemmatize(word)
				for_return.append(word)

		return for_return


	# trains the model using the training data tupels: (text_data, label) 
	def train_model(self, training_data):

		print("Conducting Training ...")

		new_training_data = []

		for data in training_data:
			preprocessed_sentence = self.preprocess_data(data[0])
			new_training_data.append((' '.join(preprocessed_sentence), data[1]))

		count = 0
		n = 1
		for item in new_training_data: # new_training_data is of format: (text, label)
			# handles positive features
			if item[1] == '1':
				self.class_positive_count += 1
				features = self.featurize(item, False)
				for f in features:
					word = f[0]
					if word not in self.positive_words_dict:
						self.positive_words_dict[f[0]] = 1
					else:
						self.positive_words_dict[f[0]] += 1

			# handles negative features
			if item[1] == '0':
				self.class_negative_count += 1
				features = self.featurize(item, False)

				for f in features:
					if f[0] not in self.negative_words_dict:
						self.negative_words_dict[f[0]] = 1
					else:
						self.negative_words_dict[f[0]] += 1

		# normalize dictinaory --> handles occurances of word appearing in one class, but not the other, which would give a zero-count for the word
		for key, val in self.positive_words_dict.items():
			if key not in self.negative_words_dict:
				self.negative_words_dict[key] = 0

		for key, val in self.negative_words_dict.items():
			if key not in self.positive_words_dict:
				self.positive_words_dict[key] = 0

		# collecting a dictionary for total vocab
		for word in self.positive_words_dict:
			if word not in self.total_vocab_dict:
				self.total_vocab_dict[word] = 1
			else:
				self.total_vocab_dict[word] += 1

		for word in self.negative_words_dict:
			if word not in self.total_vocab_dict:
				self.total_vocab_dict[word] = 1
			else:
				self.total_vocab_dict[word] += 1

		vals = self.total_vocab_dict.values()
		total_vocab_count = sum(vals) # vocabulary size

		# getting probability of each word in positive class
		for word in self.positive_words_dict:
			prob = self.positive_words_dict[word] / self.class_positive_count

			# conduct laplace smoothing
			if prob == 0:
				prob = (self.positive_words_dict[word] + 1) / (self.class_positive_count + total_vocab_count)
			self.trained_prob_pos[word] = prob

		# getting probability of each word in negative class
		for word in self.negative_words_dict:
			prob = self.negative_words_dict[word] / self.class_negative_count

			# conduct laplace smoothing
			if prob == 0:
				prob = (self.negative_words_dict[word] + 1) / (self.class_negative_count + total_vocab_count)
			self.trained_prob_neg[word] = prob

		self.calculated_class_positive_prob = self.class_positive_count / (self.class_positive_count + self.class_negative_count)
		self.calculated_class_negative_prob = self.class_negative_count / (self.class_positive_count + self.class_negative_count)

		print("POSITIVE CLASS PROBABILITY IS: ", self.calculated_class_positive_prob)
		print("NEGATIVE CLASS PROBABILITY IS: ", self.calculated_class_negative_prob)

		print("\nTraining Finished.\n")
		return

	# takes a given sentence with its label and splits it into individual words which hold an association to the sentence's given classificaiton label
	# data is of format: (text, label)
	def featurize(self, data, isTesting):

		sentence = data[0]
		label = data[1]

		sentence = sentence.split()

		features = []
		count = 0

		for word in sentence:
			features.append((word, label))

		return features

	# calculates the probability of a given piece of data to be classified as either positive or negative
	def score(self, text):

		prob_pos = self.calculated_class_positive_prob
		prob_neg = self.calculated_class_negative_prob
		classification_val = math.log(prob_pos/prob_neg)

		idx = 0
		sum_val = 0

		text = text.split()
		for word in text:
			# getting P(word | positive) and P(word | negative)
			if word in self.trained_prob_neg:
				if word in self.trained_prob_pos:
					sum_val += math.log(self.trained_prob_pos[word]/self.trained_prob_neg[word])

		liklihood = round(classification_val + sum_val, 2)
		return liklihood

	# classifies whether the input data is more likely to belong to the positive or negative class
	def classify(self, data):

		liklihood = self.score(data)

		if liklihood > 0.00:
			return 1 # positive classification
		if liklihood < 0.00:
			return 0 # negative classification
		if liklihood == 0.00:
			return -1 # neutral classification


# main function to test SentimentAnalysis model code
if __name__ == '__main__':

	print("================ running sentiment analysis ================\n")
	print()

	file = open("./DataSets/amazon_cells_labelled.txt", "r").readlines()
	file_2 = open("./DataSets/imdb_labelled.txt", "r").readlines()
	file_3 = open("./DataSets/yelp_labelled.txt", "r").readlines()

	file = file + file_2 + file_3

	data = []
	for line in file:
		data.append(line)

	random.shuffle(data)

	percent_80 = len(data)*.8
	percent_20 = len(data)*.2
	training_data = data[:int(percent_80)]
	testing_data = data[:int(percent_20)]

	# print("===> length of training file: ", len(training_data))
	# print("===> length of testing file: ", len(testing_data))

	# read data for training the model
	training_tuples = data_tuple_pairs(training_data, True)
	testing_tuples = data_tuple_pairs(testing_data, True)

	
	# running the model
	print("-------------- Running Sentiment Analysis --------------:\n")
	sa = SentimentAnalysis()

	# preprocessing testing data
	cleaned_testing_data = []
	for data in testing_tuples:
		preprocessed_sentence = sa.preprocess_data(data[0])
		cleaned_testing_data.append((' '.join(preprocessed_sentence), data[1]))

	print("Running training model...")
	sa.train_model(training_tuples)

	print("\nRunning classification...")
	classifications = [] 	# will hold classified labels (labels assigned by the classifier)
	gold_labels = [] 	# will hold gold labels (true labels)

	for i in cleaned_testing_data:
		gold_labels.append(int(i[1]))
		classifications.append(sa.classify(i[0]))

	recall_val = recall(gold_labels, classifications)
	precision_val = precision(gold_labels, classifications)
	f1_val = calculate_f1(gold_labels, classifications)

	print("\nRecall value: ", recall_val)
	print("Precision value: ", precision_val)
	print("F1-value: ", f1_val, "\n")

	print("Finsihed running model.\n")
