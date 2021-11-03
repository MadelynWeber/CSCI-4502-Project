# This is where all of the data analysis will be done

import pandas as pd
from ast import literal_eval
from SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis import sentiment_analysis_helper
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

	count = 0
	for idx, row, in df.iterrows():
		if count != 20:
			print()
			print("==============================================================================================================================")
			# print("row: ", row)
			# print("idx: ", idx)
			text_data = df.loc[idx, "text"]
			hashtag_data = literal_eval(df.loc[idx, "hashtags"])

			text_with_hashtags = text_data + " " + ' '.join(hashtag_data)

			print("Text data: ", text_data)
			print("User location: ", df.loc[idx, "user_location"])
			print("Tweet date: ", df.loc[idx, "date"])
			print("Hashtags: ", df.loc[idx, "hashtags"])
			print("TEXT WITH HASHTAGS: ", text_with_hashtags)
			print("---------------------")
			print("Column BEFORE insertion: ", df.loc[idx, "SentimentAnalysis Classificaiton"])
			df.loc[idx, "SentimentAnalysis Classificaiton"] = "test " + str(count)
			print("Column AFTER insertion: ", df.loc[idx, "SentimentAnalysis Classificaiton"])
			count += 1

		text_data = df.loc[idx, "text"]
		# running text data through sentiment analysis model and adding classification to dataframe column
		# sa_classificaiton = sentiment_analysis_helper(text_with_hashtags)

		# df.loc[idx, "SentimentAnalysis Classificaiton"] = sa_classificaiton

		# running text data through TextBlob sentiment analysis model and adding classificaiton to dataframe column
		# tb_classificaiton = run_textBlob(text_with_hashtags)

		# df.loc[idx, "TextBlob SentimentAnalysis Classificaiton"] = tb_classificaiton

		# check for each vaccine mentioned (vaccines include: Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZaneca, Covaxin, Sputnik V.)
		if("pfizer" in text_with_hashtags.lower() or "biontech" in text_with_hashtags.lower()):
			pass
		if("sinopharm" in text_with_hashtags.lower()):
			pass
		if("sinovac" in text_with_hashtags.lower()):
			pass
		if("moderna" in text_with_hashtags.lower()):
			pass
		if("oxford" in text_with_hashtags.lower() or "astrazaneca" in text_with_hashtags.lower()):
			pass
		if("covaxin" in text_with_hashtags.lower()):
			pass
		if("sputnik" in text_with_hashtags.lower()):
			pass











	# ========= TODO =========: 

		# run text data through SentimentAnalysis

		# run text data through TB Sentimetn analysis

		# run hashtag data through SentimentAnalysis

		# run hashtag data through TB Sentiment analysis

		# insert column for my SA results for text

		# insert column for TB SA results for text

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


