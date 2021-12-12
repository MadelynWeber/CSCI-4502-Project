# This is where all of the data analysis will be done

import pandas as pd
from ast import literal_eval
from SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis import data_tuple_pairs
from SentimentAnalysis import recall
from SentimentAnalysis import precision
from SentimentAnalysis import calculate_f1
from TextBlob import textBlob_helper
from TextBlob import textblob_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


if __name__ == '__main__':


	# reading in file to be classified
	# ========================================================================
	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	# removing all unnecessary columns for processes
	df = df[['id', 'date', 'text', 'hashtags']]

	# drop rows with NaN values
	df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

	# df.info()
	# df.head()

	# training model for classification
	# ========================================================================
	file = open("./DataSets/amazon_cells_labelled.txt", "r").readlines()

	data = []
	for line in file:
		data.append(line)

	df['classification'] = ''

	# read data for training the model
	training_tuples = data_tuple_pairs(data, True)

	# Sentiment Analysis model
	# ========================================================================
	sa_3 = SentimentAnalysis()

	print("Conducting Sentiment Analysis...\n")
	sa_3.train_model(training_tuples)

	neg_class = {}	# dictionary to hold counts of negative classifications for each vaccine
	pos_class = {}	# dictionary to hold counts of positive classificaitons for each vaccine
	net_class = {} 	# dictionary to hold counts of neutral classificaitons for each vaccine

	count = 0
	for idx, row, in df.iterrows():
			
		text_data = df.loc[idx, "text"]
		hashtag_data = literal_eval(df.loc[idx, "hashtags"])

		text_with_hashtags = text_data + " " + ' '.join(hashtag_data)

		# running text data through sentiment analysis model and adding classification to dataframe column
		sa_classificaiton = sa_3.classify(text_with_hashtags)

		df.loc[idx, 'classification'] = sa_classificaiton

		# assigning classifications to dictionaries
		if("pfizer" in text_with_hashtags.lower() or "biontech" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "pfizer/biontech" not in neg_class:
					neg_class["pfizer/biontech"] = 1
				else:
					neg_class["pfizer/biontech"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "pfizer/biontech" not in pos_class:
					pos_class["pfizer/biontech"] = 1
				else:
					pos_class["pfizer/biontech"] += 1
			else:
				# add to net dict
				if "pfizer/biontech" not in net_class:
					net_class["pfizer/biontech"] = 1
				else:
					net_class["pfizer/biontech"] += 1

		if("sinopharm" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "sinopharm" not in neg_class:
					neg_class["sinopharm"] = 1
				else:
					neg_class["sinopharm"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "sinopharm" not in pos_class:
					pos_class["sinopharm"] = 1
				else:
					pos_class["sinopharm"] += 1
			else:
				# add to net dict
				if "sinopharm" not in net_class:
					net_class["sinopharm"] = 1
				else:
					net_class["sinopharm"] += 1

		if("sinovac" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "sinovac" not in neg_class:
					neg_class["sinovac"] = 1
				else:
					neg_class["sinovac"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "sinovac" not in pos_class:
					pos_class["sinovac"] = 1
				else:
					pos_class["sinovac"] += 1
			else:
				# add to net dict
				if "sinovac" not in net_class:
					net_class["sinovac"] = 1
				else:
					net_class["sinovac"] += 1

		if("moderna" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "moderna" not in neg_class:
					neg_class["moderna"] = 1
				else:
					neg_class["moderna"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "moderna" not in pos_class:
					pos_class["moderna"] = 1
				else:
					pos_class["moderna"] += 1
			else:
				# add to net dict
				if "moderna" not in net_class:
					net_class["moderna"] = 1
				else:
					net_class["moderna"] += 1

		if("oxford" in text_with_hashtags.lower() or "astrazeneca" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "astrazeneca" not in neg_class:
					neg_class["astrazeneca"] = 1
				else:
					neg_class["astrazeneca"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "astrazeneca" not in pos_class:
					pos_class["astrazeneca"] = 1
				else:
					pos_class["astrazeneca"] += 1
			else:
				# add to net dict
				if "astrazeneca" not in net_class:
					net_class["astrazeneca"] = 1
				else:
					net_class["astrazeneca"] += 1

		if("covaxin" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "covaxin" not in neg_class:
					neg_class["covaxin"] = 1
				else:
					neg_class["covaxin"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "covaxin" not in pos_class:
					pos_class["covaxin"] = 1
				else:
					pos_class["covaxin"] += 1
			else:
				# add to net dict
				if "covaxin" not in net_class:
					net_class["covaxin"] = 1
				else:
					net_class["covaxin"] += 1

		if("sputnik" in text_with_hashtags.lower()):
			if sa_classificaiton == 0:
				# add to neg dict
				if "sputnik" not in neg_class:
					neg_class["sputnik"] = 1
				else:
					neg_class["sputnik"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "sputnik" not in pos_class:
					pos_class["sputnik"] = 1
				else:
					pos_class["sputnik"] += 1
			else:
				# add to net dict
				if "sputnik" not in net_class:
					net_class["sputnik"] = 1
				else:
					net_class["sputnik"] += 1

		else:
			if sa_classificaiton == 0:
				# add to neg dict
				if "N/A" not in neg_class:
					neg_class["N/A"] = 1
				else:
					neg_class["N/A"] += 1
			elif sa_classificaiton == 1:
				# add to pos dict
				if "N/A" not in pos_class:
					pos_class["N/A"] = 1
				else:
					pos_class["N/A"] += 1
			else:
				# add to net dict
				if "N/A" not in net_class:
					net_class["N/A"] = 1
				else:
					net_class["N/A"] += 1

		count += 1
		if count%10000 == 0:
			print("Calculating... {0:.2f}%".format(count/len(df)))

	print("Calculating... {0:.2f}%\n".format(100.00))

	print("Sentiment Analysis Complete.\n")

	print("\nClassified Values:")
	print("==========================================")
	print("Pos dict: ", pos_class)
	print("Neg dict: ", neg_class)
	print("Net dict: ", net_class)
	print("==========================================")

	# Graphing results from Sentiment Analysis Classifications:
	# ========================================================================
	neg_list = neg_class.items()
	neg_list = sorted(neg_list)
	x_1, y_1 = zip(*neg_list)

	pos_list = pos_class.items()
	pos_list = sorted(pos_list)
	x_2, y_2 = zip(*pos_list)

	net_list = net_class.items()
	net_list = sorted(net_list)
	x_3, y_3 = zip(*net_list)

	x = range(8)
	plt.subplot(3,1,1)
	plt.bar(x_1, y_1, color='red', width=0.5)
	plt.ylabel("Negative Classifications")
	plt.title("Classifications from Sentiment Analysis")

	plt.subplot(3,1,2)
	plt.bar(x_2, y_2, color='green', width=0.5)
	plt.ylabel("Positive Classifications")

	plt.subplot(3,1,3)
	plt.bar(x_3, y_3, color='orange', width=0.5)
	plt.ylabel("Neutral Classifications")

	plt.show()


	print("\nConducting TextBlob Sentiment Analysis...\n")

	# TextBlob Sentiment Analysis model
	# ========================================================================
	tb = textblob_analysis()

	neg_class_tb = {}	# dictionary to hold counts of negative classifications for each vaccine
	pos_class_tb = {}	# dictionary to hold counts of positive classificaitons for each vaccine
	net_class_tb = {}	# dictionary to hold counts of neutral classifications for each vaccine

	count = 0
	for idx, row, in df.iterrows():
			
		text_data = df.loc[idx, "text"]
		hashtag_data = literal_eval(df.loc[idx, "hashtags"])

		text_with_hashtags = text_data + " " + ' '.join(hashtag_data)

		# running text data through sentiment analysis model and adding classification to dataframe column
		tb_classificaiton = textBlob_helper(text_with_hashtags)

		# assigning classifications to dictionaries for plotting:
		if("pfizer" in text_with_hashtags.lower() or "biontech" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "pfizer/biontech" not in neg_class_tb:
					neg_class_tb["pfizer/biontech"] = 1
				else:
					neg_class_tb["pfizer/biontech"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "pfizer/biontech" not in pos_class_tb:
					pos_class_tb["pfizer/biontech"] = 1
				else:
					pos_class_tb["pfizer/biontech"] += 1
			else:
				# add to net dict
				if "pfizer/biontech" not in net_class_tb:
					net_class_tb["pfizer/biontech"] = 1
				else:
					net_class_tb["pfizer/biontech"] += 1

		if("sinopharm" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "sinopharm" not in neg_class_tb:
					neg_class_tb["sinopharm"] = 1
				else:
					neg_class_tb["sinopharm"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "sinopharm" not in pos_class_tb:
					pos_class_tb["sinopharm"] = 1
				else:
					pos_class_tb["sinopharm"] += 1
			else:
				# add to net dict
				if "sinopharm" not in net_class_tb:
					net_class_tb["sinopharm"] = 1
				else:
					net_class_tb["sinopharm"] += 1

		if("sinovac" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "sinovac" not in neg_class_tb:
					neg_class_tb["sinovac"] = 1
				else:
					neg_class_tb["sinovac"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "sinovac" not in pos_class_tb:
					pos_class_tb["sinovac"] = 1
				else:
					pos_class_tb["sinovac"] += 1
			else:
				# add to net dict
				if "sinovac" not in net_class_tb:
					net_class_tb["sinovac"] = 1
				else:
					net_class_tb["sinovac"] += 1

		if("moderna" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "moderna" not in neg_class_tb:
					neg_class_tb["moderna"] = 1
				else:
					neg_class_tb["moderna"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "moderna" not in pos_class_tb:
					pos_class_tb["moderna"] = 1
				else:
					pos_class_tb["moderna"] += 1
			else:
				# add to net dict
				if "moderna" not in net_class_tb:
					net_class_tb["moderna"] = 1
				else:
					net_class_tb["moderna"] += 1

		if("oxford" in text_with_hashtags.lower() or "astrazeneca" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "astrazeneca" not in neg_class_tb:
					neg_class_tb["astrazeneca"] = 1
				else:
					neg_class_tb["astrazeneca"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "astrazeneca" not in pos_class_tb:
					pos_class_tb["astrazeneca"] = 1
				else:
					pos_class_tb["astrazeneca"] += 1
			else:
				# add to net dict
				if "astrazeneca" not in net_class_tb:
					net_class_tb["astrazeneca"] = 1
				else:
					net_class_tb["astrazeneca"] += 1

		if("covaxin" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "covaxin" not in neg_class_tb:
					neg_class_tb["covaxin"] = 1
				else:
					neg_class_tb["covaxin"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "covaxin" not in pos_class_tb:
					pos_class_tb["covaxin"] = 1
				else:
					pos_class_tb["covaxin"] += 1
			else:
				# add to net dict
				if "covaxin" not in net_class_tb:
					net_class_tb["covaxin"] = 1
				else:
					net_class_tb["covaxin"] += 1

		if("sputnik" in text_with_hashtags.lower()):
			if tb_classificaiton == 0:
				# add to neg dict
				if "sputnik" not in neg_class_tb:
					neg_class_tb["sputnik"] = 1
				else:
					neg_class_tb["sputnik"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "sputnik" not in pos_class_tb:
					pos_class_tb["sputnik"] = 1
				else:
					pos_class_tb["sputnik"] += 1
			else:
				# add to net dict
				if "sputnik" not in net_class_tb:
					net_class_tb["sputnik"] = 1
				else:
					net_class_tb["sputnik"] += 1

		else:
			if tb_classificaiton == 0:
				# add to neg dict
				if "N/A" not in neg_class_tb:
					neg_class_tb["N/A"] = 1
				else:
					neg_class_tb["N/A"] += 1
			elif tb_classificaiton == 1:
				# add to pos dict
				if "N/A" not in pos_class_tb:
					pos_class_tb["N/A"] = 1
				else:
					pos_class_tb["N/A"] += 1
			else:
				# add to net dict
				if "N/A" not in net_class_tb:
					net_class_tb["N/A"] = 1
				else:
					net_class_tb["N/A"] += 1

		count += 1
		if count%10000 == 0:
			print("Calculating... {0:.2f}%".format(count/len(df)))

	print("Calculating... {0:.2f}%\n".format(100.00))


	print("TextBlob Sentiment Analysis Complete.\n")

	print("\nClassified Values:")
	print("==========================================")
	print("Pos dict TextBlob: ", pos_class_tb)
	print("Neg dict TextBlob: ", neg_class_tb)
	print("Net dict TextBlob: ", net_class_tb)
	print("==========================================")

	# Graphing results from TextBlob Sentiment Analysis Classifications:
	# ========================================================================
	neg_list_tb = neg_class_tb.items()
	neg_list_tb = sorted(neg_list_tb)
	x_1, y_1 = zip(*neg_list_tb)

	pos_list_tb = pos_class_tb.items()
	pos_list_tb = sorted(pos_list_tb)
	x_2, y_2 = zip(*pos_list_tb)

	net_list_tb = net_class_tb.items()
	net_list_tb = sorted(net_list_tb)
	x_3, y_3 = zip(*net_list_tb)

	x = range(8)
	plt.subplot(3,1,1)
	plt.bar(x_1, y_1, color='red', width=0.5)
	plt.ylabel("Negative Classifications")
	plt.title("Classifications from TextBlob Sentiment Analysis")

	plt.subplot(3,1,2)
	plt.bar(x_2, y_2, color='green', width=0.5)
	plt.ylabel("Positive Classifications")

	plt.subplot(3,1,3)
	plt.bar(x_3, y_3, color='orange', width=0.5)
	plt.ylabel("Neutral Classifications")

	plt.show()



	# for graphing both vaccine data files -- only needed to be run once 
	# ========================================================================

	# contract_df = pd.read_csv("./DataSets/vaccines_by_contract.csv", sep=",")
	# country_df = pd.read_csv("./DataSets/vaccines_by_country.csv")

	# print("=====================:")
	# print(contract_df.head())
	# print(country_df.head())

	# x_1 = contract_df["vaccine"]
	# y_1 = contract_df["number"]

	# x_2 = country_df["vaccine"]
	# y_2 = country_df["number"]

	# for graphing both vaccine data files -- only needed to be run once 
	# x = range(7)
	# plt.subplot(2,1,1)
	# plt.bar(x_1, y_1, color='pink', width=0.2, align='edge')
	# plt.ylabel('Number of Doses (in hundred million)')
	# plt.title("Vaccine Doses by Contract")
	# plt.xlabel("Vaccines")

	# plt.subplot(2,1,2)
	# plt.bar(x_2, y_2, color='pink', width=0.2, align='edge')
	# plt.ylabel('Number of Countries/Territories')
	# plt.title("Number of Countries and Territories Using Each Vaccine")
	# plt.xlabel("Vaccines")
	# plt.show()





