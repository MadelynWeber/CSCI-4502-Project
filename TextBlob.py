'''
This is a file for implementing the TextBlob sentiment analysis.
'''

from textblob import TextBlob
import nltk
import re
from contractions import contractions_dict # contains a mapping of frequent contractions in English and their expanded forms
from textblob import Word

nltk.download('punkt')

# a helper function for running TextBlob analysis from main DataAnalysis.py file
def textBlob_helper(text):
	
	tb = textblob_analysis()

	# preprocess test data
	processed_data = tb.preprocess(text)

	# create a blob for each sentence and gather a sentiment classification
	for item in processed_data:
		sentence = ' '.join(item)
		blob = TextBlob(sentence)

		result = blob.sentiment.polarity

		if result > 0.0:
			return 1 # the classification is positive
		if result < 0.0:
			return 0 # the classification is negative
		if result == 0.0:
			return -1 # this will be the neutual measure


	return result

class textblob_analysis():

	def __init__(self):
		pass

	# preprocess the text data for classificaiton
	def preprocess(self, text):
		
		return_list = []
		stopwords_data = open('./DataSets/stopwords.txt', "r")

		# removing stop words
		stop_words = []
		for word in stopwords_data:
			word = word.lower()
			if "\n" in word:
				word = word.strip()
			stop_words.append(word)

		sentence = text.lower()
		sentence = sentence.split()

		# removing non-alphabetical characters
		cleaned_list = []
		for word in sentence:
			new_word = re.sub("[^a-zA-Z0-9]+", "",word)
			if new_word != '':
				cleaned_list.append(new_word)

		# expanding contractions
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


		# lemmatize and remove stop words
		for_return = []
		for word in new_cleaned_list:
			if word not in stop_words:
				w = Word(word)
				lemmatized_word = w.lemmatize()
				for_return.append(lemmatized_word)

		return_list.append(for_return)

		return return_list
		

	# given some text data, creates a blob to be used for further analysis work
	def createBlob(self, text):
		return TextBlob(text)

	# gets the polarity calcualtion for the given piece of data
	def classify(self, text):
		
		result = text.sentiment.polarity

		if result > 0.0:
			return 1 # the classification is positive
		if result < 0.0:
			return 0 # the classification is negative
		if result == 0.0:
			return -1 # this will be the neutual measure


# main function to test the textblob analysis code
if __name__ == '__main__':

	print("================ running TextBlob sentiment analysis ================\n")
	print()

	# open this file for testing only
	file = open("./DataSets/amazon_cells_labelled.txt", "r").readlines()

	data = []
	for line in file:
		split_str = line.split('\t')
		data.append(split_str[0])


	# testing TextBlob model
	tb = textblob_analysis()

	# preprocess test data
	processed_data = tb.preprocess(data)

	for item in processed_data:
		sentence = ' '.join(item)
		blob = TextBlob(sentence)
		classification = tb.classify(blob)
		print(classification)
	
