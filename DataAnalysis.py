# This is where all of the data analysis will be done

import pandas as pd
from ast import literal_eval
from SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis import data_tuple_pairs
from SentimentAnalysis import run_trigram
from SentimentAnalysis import recall
from SentimentAnalysis import precision
from SentimentAnalysis import calculate_f1
from SentimentAnalysis import get_ngrams
from TextBlob import textBlob_helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


if __name__ == '__main__':


	df = pd.read_csv("./DataSets/vaccination_all_tweets.csv")

	# removing all unnecessary columns for processes
	df = df[['id', 'date', 'text', 'hashtags']]

	# drop rows with NaN values
	df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	print("new length: ", len(df))

	df.info()
	df.head()

	file = open("./DataSets/amazon_cells_labelled.txt", "r").readlines()

	data = []
	for line in file:
		data.append(line)

	random.shuffle(data)

	percent_80 = len(data)*.8
	percent_20 = len(data)*.2
	training_data = data[:int(percent_80)]
	testing_data = data[:int(percent_20)]

	# read data for training the model
	training_tuples = data_tuple_pairs(training_data, True)
	testing_tuples = data_tuple_pairs(testing_data, True)

	# run the tri-gram model
	sa_3 = SentimentAnalysis(3)

	# preprocessing testing data
	# cleaned_testing_data = []
	# for data in testing_tuples:
	# 	preprocessed_sentence = sa_3.preprocess_data(data[0])
	# 	cleaned_testing_data.append((' '.join(preprocessed_sentence), data[1]))


	sa_3.train_model(training_tuples)

	# print("\nRunning classification...")
	# classifications = [] 	# will hold classified labels (labels assigned by the classifier)
	# gold_labels = [] 	# will hold gold labels (true labels)

	# for i in cleaned_testing_data:
	# 	gold_labels.append(int(i[1]))
	# 	classifications.append(sa_3.classify(i[0]))

	# recall_val = recall(gold_labels, classifications)
	# precision_val = precision(gold_labels, classifications)
	# f1_val = calculate_f1(gold_labels, classifications)

	neg_class = {}	# dictionary to hold counts of negative classifications for each vaccine
	pos_class = {}	# dictinoary to hold counts of positive classificaitons for each vaccine

	count = 0
	for idx, row, in df.iterrows():
		if count != 2000: # ---> for testing only
			
			text_data = df.loc[idx, "text"]
			hashtag_data = literal_eval(df.loc[idx, "hashtags"])

			text_with_hashtags = text_data + " " + ' '.join(hashtag_data)

			# running text data through sentiment analysis model and adding classification to dataframe column
			sa_classificaiton = sa_3.classify(text_with_hashtags)
			

			# ==================================== TODO ====================================: 
				# data put into model is not being split into n-grams before running through! 

				# run text and hashtag data through TB Sentimetn analysis
				# running text data through TextBlob sentiment analysis model and adding classificaiton to dataframe column
				# tb_classificaiton = run_textBlob(text_with_hashtags)

				# insert column for TB SA results for text
				# df.loc[idx, "TextBlob SentimentAnalysis Classificaiton"] = tb_classificaiton
			# ========================================================================

			# try making two dictionaries to hold each count and graph those:
			if("pfizer" in text_with_hashtags.lower() or "biontech" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "pfizer/biontech" not in neg_class:
						neg_class["pfizer/biontech"] = 1
					else:
						neg_class["pfizer/biontech"] += 1
				else:
					# add to pos dict
					if "pfizer/biontech" not in pos_class:
						pos_class["pfizer/biontech"] = 1
					else:
						pos_class["pfizer/biontech"] += 1

			if("sinopharm" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "sinopharm" not in neg_class:
						neg_class["sinopharm"] = 1
					else:
						neg_class["sinopharm"] += 1
				else:
					# add to pos dict
					if "sinopharm" not in pos_class:
						pos_class["sinopharm"] = 1
					else:
						pos_class["sinopharm"] += 1

			if("sinovac" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "sinovac" not in neg_class:
						neg_class["sinovac"] = 1
					else:
						neg_class["sinovac"] += 1
				else:
					# add to pos dict
					if "sinovac" not in pos_class:
						pos_class["sinovac"] = 1
					else:
						pos_class["sinovac"] += 1

			if("moderna" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "moderna" not in neg_class:
						neg_class["moderna"] = 1
					else:
						neg_class["moderna"] += 1
				else:
					# add to pos dict
					if "moderna" not in pos_class:
						pos_class["moderna"] = 1
					else:
						pos_class["moderna"] += 1

			if("oxford" in text_with_hashtags.lower() or "astrazeneca" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "astrazeneca" not in neg_class:
						neg_class["astrazeneca"] = 1
					else:
						neg_class["astrazeneca"] += 1
				else:
					# add to pos dict
					if "astrazeneca" not in pos_class:
						pos_class["astrazeneca"] = 1
					else:
						pos_class["astrazeneca"] += 1

			if("covaxin" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "covaxin" not in neg_class:
						neg_class["covaxin"] = 1
					else:
						neg_class["covaxin"] += 1
				else:
					# add to pos dict
					if "covaxin" not in pos_class:
						pos_class["covaxin"] = 1
					else:
						pos_class["covaxin"] += 1

			if("sputnik" in text_with_hashtags.lower()):
				if sa_classificaiton == 0:
					# add to neg dict
					if "sputnik" not in neg_class:
						neg_class["sputnik"] = 1
					else:
						neg_class["sputnik"] += 1
				else:
					# add to pos dict
					if "sputnik" not in pos_class:
						pos_class["sputnik"] = 1
					else:
						pos_class["sputnik"] += 1

			else:
				if sa_classificaiton == 0:
					# add to neg dict
					if "N/A" not in neg_class:
						neg_class["N/A"] = 1
					else:
						neg_class["N/A"] += 1
				else:
					# add to pos dict
					if "N/A" not in pos_class:
						pos_class["N/A"] = 1
					else:
						pos_class["N/A"] += 1

			count += 1

		
	# FOR TESTING ONLY!
		else:
			break

	df.head()
	df.info()

	print("==========================================")
	print("Pos dict: ", pos_class)
	print("Neg dict: ", neg_class)
	print("==========================================")

	# plot results for both positive and negative classifications
	# plt.bar(neg_class.keys(), neg_class.values(), width=1.0, color='r')
	# plt.bar(pos_class.keys(), pos_class.values(), width=1.0, color='g')

	# plt.show()



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


	

	

	# ==================================== TODO: =============================================
	
	# graph my SA model's results against TB SA model's results (purly for checking accuracy)


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





