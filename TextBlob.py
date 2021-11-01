'''
This is a file for implementing the TextBlob sentiment analysis.
'''

from textblob import TextBlob

# given a sentence, creates a blob to be used for further analysis work
def createBlob(data):
	# blob = TextBlob(<sentence>)
	pass

# lemmatize each word
def lemmatize(data):
	# w = Word('running')
	# w.lemmatize("v") ## v here represents verb
	pass

# create n-grams
def ngrams(data, n):
	# for ngram in blob.ngrams(2):
	# print (ngram)
	# >> ['Analytics', 'Vidhya']
	# ['Vidhya', 'is']
	# ['is', 'a']
	# ['a', 'great']
	# ['great', 'platform']
	# ['platform', 'to']
	# ['to', 'learn']
	# ['learn', 'data']
	# ['data', 'science']
	pass

# collect sentiment classificaiton
def classify(data):
	# print (blob)
	# blob.sentiment
	# >> Analytics Vidhya is a great platform to learn data science.
	# Sentiment(polarity=0.8, subjectivity=0.75)
	pass