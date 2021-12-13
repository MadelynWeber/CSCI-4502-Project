# Using Sentiment Analysis to Predict General Feelings Toward COVID-19 Vaccines 
An Analysis of the General Opinions Toward COVID-19 Vaccines and the Relationship Between Vaccine Producers, Region, and Temporal Data as Found From COVID-19-Related Tweets.

## Project Summary:
This is a project which makes use of a Naïve Bayesian Sentiment Analysis model to conclude the general sentiment associated with some of the COVID-19 vaccines currently available for distribution. Through collecting sentiment classifications assigned to text data collected from Tweets relating to COVID-19 vaccines, we are able to predict the general feelings that English-speaking Twitter users feel towards the vaccines examined. Each Tweet was screened for mentions of the following vaccines: Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZeneca, Covaxin, and Sputnik V. Through visualizations of the data collected, we are able to generate an approximate distribution of where each vaccine falls within the sentiment classifications of either positive, negative, or neutral. From these results, we can then begin to think of potential improvements to the current vaccine distribution plans, by taking in both the vaccines examined and the overall sentiment assigned to each, in an attempt to increase the numbers of the current vaccinated public. 

## Key Results:
It was found from the results of this analysis that the general overall sentiment toward COVID-19 vaccines is largely non-negative. It was also found that the frequency in which certain vaccines came up within the text data did not match the frequency of real-world distribution data. That is to say, the vaccines which are in high distribution currently are not necessarily those which were mentioned the most often within the dataset examined here. It was also found that some of the vaccines which received the highest ratio of non-negative overall sentiment were not among the top vaccines seen in real-world distribution data. From this, it can be concluded that a potential improvement to the current vaccine distribution plans may include increasing the vaccines available for individuals to select from within certain regions. All visual results can be found in the folder named "Results". The final plot for sentiment distribution amongst the vaccines examined as produced by the model is as shown below.
![image of plot](https://github.com/MadelynWeber/CSCI-4502-Project/blob/main/Results/results%20from%20sentiment%20analysis%20model.png)

## Data Files:

### StopWords.txt
- This text file contains a condensed list of English stop words used for the text pre-processing phase of the model. It includes both English stop words, as well as a few added stop words specifically used on Twitter/online text, such as "user" or "lol".

### amazon_cells_labelled.txt, imbd_labelled.txt, yelp_labelled.txt
- These three files are used as a means of training the Sentiment Analysis model. They consist of a piece of text and a pre-labeled classifications, which were combined together into one large set to be read by the model. The original datasets can be found from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences).

### vaccination_all_tweets.csv
- This is the dataset run through the Sentiment Analysis model to collect results. After cleaning the data and removing unnecessary columns, the model had 152,229 text instances to collect data from. Each text instance fed into the model consisted of a concatenated string of the Tweet text itself and the hashtag text associated with it. The full dataset can be found from [Kaggle](https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets).

### vaccines_by_contract.csv
- This is a dataset created from the information related to the number of reported dosages of different COVID-19 vaccines. The original information can be found from [Nikkei Asia](https://vdata.nikkei.com/en/newsgraphics/coronavirus-vaccine-status/). This datafile was used for an easier visualization of the current number of vaccine doses distributed amongst the vaccines examined within this analysis. 

### vaccines_by_country.csv
- This is a dataset created from the information related to the number of countries/territories officially using each of the vaccines examined within this analysis. The original information can be found from [BBC News](https://www.bbc.com/news/world-56237778). Similar to the dataset above, this datafile was used for an easier visualization of the current number of vaccine doses distributed amongst the vaccines examined within this analysis, to be used as a comparison tool of the results.

## Project Files:

### DataAnalysis.py 
- This is the file where all the work is done. It makes calls to the file holding the Sentiment Analysis class as well as the file holding the TextBlob Sentiment Analysis class. All classifications for the datafile examined are done within this file, and the histograms used for plotting the results are also created when running this file.

### SentimentAnalysis.py 
- This is the Sentiment Analysis model class. By running this model, the training dataset will be run through the model and an approximate estimate for the model's F1-score is calculated, as a means of measuring how accurate the model is. This file can be run on its own to test the model itself with the testing dataset.

### TextBlob.py
- This is the TextBlob implementation of Sentiment Analysis. This file is very similar to how the SentimentAnalysis.py file runs to test the model, except it is using TextBlob's implementation of sentiment classification. This file was created as a means of checking the output of my model's classifications against an already existing and accurate Sentiment Analysis model. This file will not work if run on its own, it is written to work when called from the DataAnalysis.py file.

### Contractions.py
- This is a file holding the class for all English contractions, which is used in the text pre-processing phase within the model.

## Running the Project:
To run the project to get the results of the dataset used for analysis, the DataAnalysis.py file must be run using Python3. There is no need to run any other file within this repository, as DataAnalysis.py makes calls to the other classes used for this method. Both the SentimentAnalysis.py and TextBlob.py files were created as a means of testing the individual models before implementing them in the larger project file with the Twitter dataset. 

## Future Work:
The data examined for this analysis comes from English-speaking Twitter users, and the results were not sorted by region or location of each user. Due to the fact that some of the vaccines examined here are only available in certain regions of the world, there is the potential for bias, which may throw off the accuracy of the predictions made from the results of this analysis. Thus, incorporating a means of locational analysis, such that each item of text, along with its corresponding classification from the model, can be sorted by geographical location. This would allow us to examine discrepancies between user location and the overall sentiment assigned to certain vaccines, leading to further potential improvements to vaccine distribution plans.

