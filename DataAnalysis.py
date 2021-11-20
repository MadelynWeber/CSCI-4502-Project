# This is where all of the data analysis will be done

import pandas as pd
from ast import literal_eval
from SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis import sentiment_analysis_helper
from TextBlob import textBlob_helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':


	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	print("====> lenth: ", len(df))

	# removing all unnecessary columns for processes
	df = df[['id', 'date', 'text', 'hashtags']]

	# drop rows with NaN values
	df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	print("new length: ", len(df))

	# create new columns to hold information from classifications
	df["SentimentAnalysis Classificaiton"] = ""
	df["TextBlob SentimentAnalysis Classificaiton"] = ""

	# create new column to hold which vaccine(s) the tweet is mentioning
	df["Vaccine(s) mentioned"] = ""

	df.info()
	df.head()

	# =========================== TODO ===========================:
		# run training data through SentimentAnalysis model 
		# figure out a way to save the dictioary created from the training data 
	# ===============================================================

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

			# print("Text data: ", text_data)
			# print("User location: ", df.loc[idx, "user_location"])
			# print("Tweet date: ", df.loc[idx, "date"])
			# print("Hashtags: ", df.loc[idx, "hashtags"])
			print("TEXT WITH HASHTAGS: ", text_with_hashtags)
			print("---------------------")
			# print("Column BEFORE insertion: ", df.loc[idx, "SentimentAnalysis Classificaiton"])
			df.loc[idx, "SentimentAnalysis Classificaiton"] = "test " + str(count)
			# print("Column AFTER insertion: ", df.loc[idx, "SentimentAnalysis Classificaiton"])

			vaccines_mentioned = []
			# check for each vaccine mentioned (vaccines include: Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZaneca, Covaxin, Sputnik V.)
			if("pfizer" in text_with_hashtags.lower() or "biontech" in text_with_hashtags.lower()):
				vaccines_mentioned.append("pfizer/biotech")
			if("sinopharm" in text_with_hashtags.lower()):
				vaccines_mentioned.append("sinopharm")
			if("sinovac" in text_with_hashtags.lower()):
				vaccines_mentioned.append("sinovac")
			if("moderna" in text_with_hashtags.lower()):
				vaccines_mentioned.append("moderna")
			if("oxford" in text_with_hashtags.lower() or "astrazeneca" in text_with_hashtags.lower()):
				vaccines_mentioned.append("astrazeneca")
			if("covaxin" in text_with_hashtags.lower()):
				vaccines_mentioned.append("covaxin")
			if("sputnik" in text_with_hashtags.lower()):
				vaccines_mentioned.append("sputnik")
			if not vaccines_mentioned:
				vaccines_mentioned.append("N/A")

			# print("Vaccines: ", vaccines_mentioned)

			# --> FIRST START OFF WITH THE CASES IN WHICH THEY ONLY SPEAK ABOUT ONE VACCINE PRODUCER -- IF MORE OCCUR, HANDLE THAT LATER IF THERE IS TIME
			# insert columns for specified vaccine producers plus count of each time one is mentioned per row
				# column with highest count is most likely the producer the text data is about
				# if equal, then inconclusive or sentiment can count for both
			df.loc[idx, "Vaccine(s) mentioned"] = ','.join(vaccines_mentioned)
			print(df.loc[idx, "Vaccine(s) mentioned"])


			# split text_with_hashtags into the n-gram determined to be the most accurate

			# run training data into model so it has something to base itself off of

			# ==================================== TODO ====================================: 


				# run text and hashtag data through SentimentAnalysis
				# running text data through sentiment analysis model and adding classification to dataframe column
				# sa_classificaiton = sentiment_analysis_helper(text_with_hashtags)

				# insert column for my SA results for text
				# df.loc[idx, "SentimentAnalysis Classificaiton"] = sa_classificaiton

				# run text and hashtag data through TB Sentimetn analysis
				# running text data through TextBlob sentiment analysis model and adding classificaiton to dataframe column
				# tb_classificaiton = run_textBlob(text_with_hashtags)

				# insert column for TB SA results for text
				# df.loc[idx, "TextBlob SentimentAnalysis Classificaiton"] = tb_classificaiton
			# ========================================================================

			count += 1
		
	# FOR TESTING ONLY!
		else:
			break

	# plt.title("Test Plot")
	# plt.scatter(df['SentimentAnalysis Classificaiton'], df['Vaccine(s) mentioned'], color="pink")
	# plt.show()

	contract_df = pd.read_csv("./DataSets/vaccines_by_contract.csv", sep=",")
	country_df = pd.read_csv("./DataSets/vaccines_by_country.csv")

	print("=====================:")
	print(contract_df.head())
	print(country_df.head())

	x_1 = contract_df["vaccine"]
	y_1 = contract_df["number"]

	x_2 = country_df["vaccine"]
	y_2 = country_df["number"]



	# fig, ax = plt.subplots()


	# ax.bar(x_1, y_1, color='purple', width=0.2, align='edge')
	# ax.bar(x_2, y_2, color='pink', width=0.2, align='edge', alpha=0.2)
	# plt.title("Title")
	# plt.xlabel("X")
	# plt.ylabel("Y")
	# fig.tight_layout()

	# plt.show()




	x = range(7)
	plt.subplot(2,1,1)
	#This will create the bar graph for poulation
	plt.bar(x_1, y_1, color='pink', width=0.2, align='edge')
	plt.ylabel('Vaccines')
	# plt.xticks([],[])
	#The below code will create the second plot.
	plt.subplot(2,1,2)
	#This will create the bar graph for gdp i.e gdppercapita divided by population.
	plt.bar(x_2, y_2, color='pink', width=0.2, align='edge')
	plt.ylabel('Vaccines')
	# plt.xticks(x, datasort['country'], rotation='vertical')
	plt.xticks(rotation=45)
	plt.show()


	

	

	# ==================================== TODO: =============================================
	
	# graph my SA model's results against TB SA model's results (purly for checking accuracy)

	# plot density histogram for each count of each vaccine producer

	# plot each vaccine against positive classifcations with a density histogram

	# plot each vaccien against negative classifciations with a density historgram

	# (depending on how long the data spans for over time)
		# plot positive classificaitons against time
		# plot negative classifications against time

	# =================================================================================


	# n_groups = 7

	# contract_nums = contract_df["number"]
	# country_nums = country_df["number"]

	# fig, ax = plt.subplots()

	# index = np.arange(n_groups)
	# bar_width = 0.35

	# opacity = 0.8
	# # error_config = {'ecolor': '0.3'}

	# rects1 = ax.bar(index, contract_nums, bar_width,
 #                alpha=opacity, color='pink',
 #                label='Count by Contract')

	# rects2 = ax.bar(index + bar_width, country_nums, bar_width,
 #                alpha=opacity, color='purple',
 #                label='Count by Country')

	# ax.set_xlabel('Vaccine')
	# ax.set_ylabel('Doses (in hundred million)')
	# ax.set_title('Compairson Between Vaccine Contracts and Distribution by Country')
	# ax.set_xticks(index + bar_width / 2)
	# ax.set_xticklabels(('Pfizer', 'AstraZeneca', 'Moderna', 'Sinovac', 'Sinopharm', 'Covaxin', 'Sputnik'))
	# plt.xticks(rotation=45)

	# ax.legend()

	# fig.tight_layout()
	# plt.show()





