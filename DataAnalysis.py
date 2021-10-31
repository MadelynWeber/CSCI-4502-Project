# This is where all of the data analysis will be done

import pandas as pd

if __name__ == '__main__':


	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	# removing all unnecessary columns for processes
	df = df[['id', 'user_location', 'date', 'text', 'hashtags']]

	df.info()

	# drop rows with NaN values
	df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	print("new length: ", len(df))





	# ========= TODO =========: 
		# run dataset through my sentiment analysis model
		# run dataset through TextBlob's sentiment analysis model
		# compare results
		# vizualize:
			# sentiment of my results compared to vaccine producer, location -- on seperate graphs
			# sentiment of my results (positive, negative) and vaccine producer, location all on one graph
			# sentiment of TextBlob's sentiment analysis model compared to vaccine producer, location -- on seperate graphs
			# sentiment of TextBlob's results (positive, negative) and vaccine producer, location all on one graph

