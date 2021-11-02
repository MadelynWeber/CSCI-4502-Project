# This is where all of the data analysis will be done

import pandas as pd
from SentimentAnalysis import SentimentAnalysis
# from TextBlob import run_textBlob

if __name__ == '__main__':


	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	# removing all unnecessary columns for processes
	df = df[['id', 'user_location', 'date', 'text', 'hashtags']]

	# ONLY DROP ROWS WITH NO TEXT DATA (IF ANY)


	# drop rows with NaN values
	# df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	# print("new length: ", len(df))

	# create new columns to hold information from classifications
	df["SentimentAnalysis Classificaiton"] = ""
	df["TextBlob SentimentAnalysis Classificaiton"] = ""

	df.info()

	df.head()

	# get text data from dataset

	# run text data through SentimentAnalysis

	# insert column for my SA results

	# insert column for TB SA results







	# ========= TODO =========: 
		# run dataset through my sentiment analysis model
		# run dataset through TextBlob's sentiment analysis model
		# compare results
		# vizualize:
			# sentiment of my results compared to vaccine producer, location -- on seperate graphs
			# sentiment of my results (positive, negative) and vaccine producer, location all on one graph
			# sentiment of TextBlob's sentiment analysis model compared to vaccine producer, location -- on seperate graphs
			# sentiment of TextBlob's results (positive, negative) and vaccine producer, location all on one graph

