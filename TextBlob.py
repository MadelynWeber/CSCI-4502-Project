'''
This is a file for implementing the TextBlob sentiment analysis.
'''

from textblob import TextBlob
import nltk
nltk.download('punkt')

class textblob_analysis():

	def __init__(self):
		pass

	# preprocess the text data for classificaiton
	def preprocess(self, text):
		# lemmatize
		# w = Word('running')
		# w.lemmatize("v") ## v here represents verb
		# >> run

		# remove stop words
		# conver to lowercase
		# remove non-alphabetical characters
		pass

	# given some text data, creates a blob to be used for further analysis work
	def createBlob(self, text):
		return TextBlob(text)

	# create list of n-grams for given data
	def ngrams(self, n):
		ngrams_list = []
		for ngram in data.ngrams(n):
			print(ngram)
			ngrams_list.append(ngram)

		return ngrams_list

	# gets the polarity calcualtion for the given piece of data
	def classify(self, text):

		return text.sentiment.polarity


if __name__ == '__main__':

	pass




