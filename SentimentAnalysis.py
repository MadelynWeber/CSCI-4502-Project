"""
	This is a file for the Sentiment Analysis algorithm
"""

import nltk
from nltk.stem import WordNetLemmatizer
import re
from contractions import contractions_dict # contains a mapping of frequent contractions in English and their expanded forms

# a helper function for DataAnalysis.py file --> runs through the sentiment analysis processes for each given text input
def sentiment_analysis_helper(text):
	# TODO
	pass

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
def data_tuple_pairs(file_path, is_training):

	lines = open(file_path, "r").readlines()

	hold_returns = []

	for l in lines:
		separate_elements = l.split('\t')
		#print(separate_elements)
		#i_d = separate_elements[0]
		if (is_training):
			label = separate_elements[1]
			if "\n" in label:
				label = label.strip()
			text = separate_elements[0]
			#hold_returns.append(tuple((i_d, label, text)))
			hold_returns.append(tuple((text, label)))
		else:
			text = separate_elements[0]
			hold_returns.append(text)
			#hold_returns.append(tuple((i_d, text)))

	# the first line is just the tag names, so remove it
	return (hold_returns[1:])

# returns list of ngrams for each sentence 
def get_ngrams(sentence, n):
	print("Sentence: ", sentence)
	is_final = False # flag to mark final tuple for ngram
	# tokens = sentence.split()
	hold_ngrams = []
	for t in range(len(sentence)):
		# print("t: ", t)
		# appending nothing onto last element on 2-grams
		if n == 2 and t == len(sentence)-1: 
			# print("TEST 1")
			ngram = sentence[t: t+n]
			ngram = [*ngram, ' ']
			hold_ngrams.append(tuple(ngram))
			is_final = True
		# appending empty character to last element on 3-gram
		if n == 3 and t == len(sentence)-2: 
			# print("TEST 2")
			ngram = sentence[t: t+n]
			ngram = [*ngram, ' ']
			hold_ngrams.append(tuple(ngram))
			is_final = True
		if is_final == False:
			# print("TEST 3")
			ngram = sentence[t: t+n]
			hold_ngrams.append(tuple(ngram))
	# print("====> ngrams: ", hold_ngrams)
	return hold_ngrams


class SentimentAnalysis():

	def __init__(self, n):
		self.trained_prob_pos = {}		# probability of positive classificaiton from training
		self.trained_prob_neg = {}		# probability of negative classification from training
		self.class_positive_count = 0	# count of all positive instances from training
		self.class_negative_count = 0	# count of all negative instances from training
		self.positive_words_dict = {}		# holds total occurances of words in positive classification
		self.negative_words_dict = {}		# holds total occurances of words in negative classificaiton
		self.total_vocab_dict = {}		# holds total occurances of all words in vocabular (positive and negative classifications)
		self.calculated_class_positive_prob = 0		# probability of class being positive
		self.calculated_class_negative_prob = 0		# probability of class being negative
		self.ngram = n 		# value to be used for ngram		


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
				features = self.featurize(item)

				for f in features:
					word = f[0]
					if word not in self.positive_words_dict:
						self.positive_words_dict[f[0]] = 1
					else:
						self.positive_words_dict[f[0]] += 1

			# handles negative features
			if item[1] == '0':
				self.class_negative_count += 1
				features = self.featurize(item)

				for f in features:
					if f[0] not in self.negative_words_dict:
						self.negative_words_dict[f[0]] = 1
					else:
						self.negative_words_dict[f[0]] += 1

					# # split into ngrams
					# if count != 15:
					# 	print("----> ", n, "gram: ", f)
					# 	count += 1


		# normalize dictinaory --> handles occurances of word appearing in one class, but not the other, which would give a zero-count for the word
		for key, val in self.positive_words_dict.items():
			if key not in self.negative_words_dict:
				self.negative_words_dict[key] = 0

		for key, val in self.negative_words_dict.items():
			if key not in self.positive_words_dict:
				self.positive_words_dict[key] = 0

		# print("POSITIVE WORDS DICT: ", self.positive_words_dict)
		# print()
		# print("NEGATIVE WORDS DICT: ", self.negative_words_dict)
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

		# print()
		# print("TOTAL VOCAB DICT: ", self.total_vocab_dict)
		# print()
		# print("TOTAL VOCAB SIZE: ", total_vocab_count)
		# print()
		# getting probability of each word in positive class
		for word in self.positive_words_dict:
			prob = self.positive_words_dict[word] / self.class_positive_count

			# conduct laplace smoothing
			if prob == 0:
				prob = (self.positive_words_dict[word] + 1) / (self.class_positive_count + total_vocab_count)
			self.trained_prob_pos[word] = prob
			# print("POS WORD: ", word, " POS PROB: ", prob)

		# getting probability of each word in negative class
		for word in self.negative_words_dict:
			prob = self.negative_words_dict[word] / self.class_negative_count

			# conduct laplace smoothing
			if prob == 0:
				prob = (self.negative_words_dict[word] + 1) / (self.class_negative_count + total_vocab_count)
			self.trained_prob_neg[word] = prob
			# print("NEG WORD: ", word, " NEG PROB: ", prob)

		self.calculated_class_positive_prob = self.class_positive_count / (self.class_positive_count + self.class_negative_count)
		self.calculated_class_negative_prob = self.class_negative_count / (self.class_positive_count + self.class_negative_count)

		print("POSITIVE CLASS PROBABILITY IS: ", self.calculated_class_positive_prob)
		print("NEGATIVE CLASS PROBABILITY IS: ", self.calculated_class_negative_prob)

		print("Training Finished.\n")
		return

	# takes a given sentence with its label and splits it into individual words which hold an association to the sentence's given classificaiton label
	# data is of format: (text, label)
	def featurize(self, data):

		sentence = data[0]
		label = data[1]

		sentence = sentence.split()

		# FOR TESTING BELOW
		count = 0
		if self.ngram != 1:
			ngram_sentence = get_ngrams(sentence, self.ngram)
			if count != 15:
				print("returned ngram: ", ngram_sentence)
				count += 1

		features = []
		for word in sentence:
			features.append((word, label))

		return features


	# calculates the probability of a given piece of data to be classified as either positive or negative
	def score(self, data):

		i_d = data[0]
		text = data[2]

		# create an instance of the text for both a negative and positive classification
		pos_classification = (i_d, 1, text)
		neg_classification = (i_d, 0, text)

		# for both classifications, collect features of words within text 
		pos_features = self.featurize(pos_classification)
		neg_features = self.featurize(neg_classification)

		# calclulate probability of each feature being positive or negative
		prob_positive = self.trained_prob_pos
		for feature in pos_features:
			if f[0] in self.pos_features:
				word_prob = self.pos_features[f[0]]
				prob_positive *= word_prob

		prob_negative = self.trained_prob_neg
		for feature in neg_features:
			if f[0] in self.neg_features:
				word_prob = self.neg_features[f[0]]
				prob_negative *= word_prob

		return (prob_positive, prob_negative)

	# classifies whether the input data is more likely to belong to the positive or negative class
	def classify(self, data):

		prob_pos, prob_neg = self.score(data)

		if prob_pos > prob_neg:
			return 1 # is a positive classification
		else:
			return 0 # is a negative classification



if __name__ == '__main__':

	# if len(sys.argv) != 3:
		# print("Usage: python SentimentAnalysis.py <training file> <testing file>")
		# sys.exit(1)

	# training = sys.argv[1]
	# testing_data = sys.argv[2]

	print("================ running sentiment analysis ================")
	print()

	sa = SentimentAnalysis(1)

	print("Running training model...")
	# read data for training the model
	training_tuples = data_tuple_pairs("./DataSets/amazon_cells_labelled.txt", True)  #("./DataSets/train.csv", True)
	#print(training_tuples)

	sa.train_model(training_tuples)

	print()
	print("Running classification...")
	# read data for testing the model
	data_tuples = data_tuple_pairs("./DataSets/test.csv", False)

	# classifications = [] # will hold classified labels (labels assigned by the classifier)
	# labels = [] # will hold gold labels (true labels)
	# for i in data_tuples:
	# 	labels.append(i[1])
	# 	classifications.append(sa.classify(i))

	# # calculate accuracies
	# print()
	# print("CLASSIFICAITON FROM FUNCTION: ", classifications) # classified labels
	# print("CLASSIFICATION FROM FILE: ", labels) # gold labels

	# print()
	# print("Calculating F1 score...")
	# print("F1 score: ", f1(labels, classifications))

	# print()
	# print("Generating classifications from test data...")
	# test_data_tuples = data_tuple_pairs(testing_data)
	# test_classificaitons = []
	# test_labels = []
	# for i in test_data_tuples:
	# 	labels.append(i[1])
	# 	test_classificaitons.append(sa.classifiy(i))
	# print("Classifications from test data finished.")


	# print("Sentiment analysis complete.")




# ------ TODO ------:
	# run sa once with 1-grams
	# calculate accuracy

	# run sa once with 2-grams
	# calculate accuracy

	# run sa once with 3-grams
	# calculate accuracy
