# This is where all of the data analysis will be done

import pandas as pd
from SentimentAnalysis import SentimentAnalysis
# from TextBlob import run_textBlob

if __name__ == '__main__':


	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	# removing all unnecessary columns for processes
	df = df[['id', 'user_location', 'date', 'text', 'hashtags']]

	# drop rows with NaN values
	df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	print("new length: ", len(df))

	# create new columns to hold information from classifications
	df["SentimentAnalysis Classificaiton"] = ""
	df["TextBlob SentimentAnalysis Classificaiton"] = ""

	df.info()

	df.head()

	# ========= TODO =========: 

		# get text data from dataset

		# run text data through SentimentAnalysis

		# run text data through TB Sentimetn analysis

		# run hashtag data through SentimentAnalysis

		# run hashtag data through TB Sentiment analysis

		# insert column for my SA results for text

		# insert column for TB SA results for text

		# examine hashtags and any linguistic data relating to any of the listed vaccine producers

		# insert columns for specified vaccine producers plus count of each time one is mentioned per row
				# column with highest count is most likely the producer the text data is about
				# if equal, then inconclusive or sentiment can count for both

		# graph my SA results against TB SA results

		# graph my SA restuls against vaccine producer 

		# graph my SA results against location

		# graph TB SA results against vaccine producer

		# graph TB SA resutls against location

		# graph my SA results, location, and vaccine producer for positive classificaiton

		# graph my SA results, location, and vaccine producer for negative classificaiton

		# graph TB SA results, location, and vaccine producer for positive classificaiton

		# graph TB SA results, location, and vaccine producer for negative classificaiton


