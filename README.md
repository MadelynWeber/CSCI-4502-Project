# Using Sentiment Analysis to Predict General Feelings Toward COVID-19 Vaccines 
An Analysis of the General Opinions Toward COVID-19 Vaccines and the Relationship Between Vaccine Producers, Region, and Temporal Data as Found From COVID-19-Related Tweets.

## Project Summary:
This is a project which makes use of a Naïve Bayesian Sentiment Analysis model to conclude the general sentiment associated with some of the COVID-19 vaccines currently avaialble for distribution. Through collecting sentiment classificaitons assinged to text data collected from Tweets relating to COVID-19 vaccines, we are able to predict the general feelings that English-speaking Twitter users feel towards the vaccines examined. Each Tweet was screened for mentions of the following vaccines: Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZeneca, Covaxin, and Sputnik V. Through vizualizations of the data collected, we are able to generate an approximate distribution of where each vaccine falls within the sentiment classifications of either positive, negative, or neutral. From these results, we can then begin to think of potential improvements to the current vaccine distribution plans, by taking in both the vaccines examined and the overall sentiment assigned to each, in an attempt to increase the numbers of the current vaccinated public. 

## Key Results:
It was found from the results of this analysis that the general overall sentiment toward COVID-19 vaccines is largely non-negative. It was also found that the frequency in which certain vaccines came up within the text data did not match the frequency of real-world distribution data. That is to say, the vaccines which are in high distiribuiton currenlty are not necessarily those which were mentioned the most often within the dataset examined here. It was also found that some of the vaccines which received the highest ratio of non-negative overall sentiment were not among the top vaccines seen in real-world distribution data. From this, it can be concluded that a potential improvement to the current vaccine distribution plans may include increasing the vaccines avaialble for individuals to select from within certain regions. 

## Data Files:

### StopWords.txt
- This text file contains a condensed list of English stop words used for the text pre-processing phase of the model. It includes both English stop words, as well as a few added stop words specifically used on Twitter/online text, such as "user" or "lol".

### amazon_cells_labelled.txt, imbd_labelled.txt, yelp_labelled.txt
- These three files are used as a means of training the Sentiment Analysis model. They consist of a piece of text and a pre-labeled classificaion, which were combined together into one large set to be read by the model. The original datasets can be found from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences).

## Project Files:

### DataAnalysis.py 
- This is the file where all the work is done. It makes calls to the file holding the Sentiment Analysis class as well as the file holding the TextBlob Sentiment Analysis class. All classificaitons for the datafile examined are done within this file, and the histograms used for plotting the results are also created when running this file.

### SentimentAnalysis.py 
- This is the Sentiment Analysis model class. By running this model, the training dataset will be run through the model and an approximate estimate for the model's F1-score is calculated, as a means of measuring how accurate the model is.

### TextBlob.py
- This is the TextBlob implementation of Sentiment Analysis. This file is very simiilar to how the SentimentAnalysis.py file runs to test the model, except it is using TextBlob's implementation of sentiment classification. This file was created as a means of checking the output of my model's classificaiotns against an already existing and accurate Sentiment Analysis model. 

## Running the Project:
To run the project to get the results of the dataset used for analysis, the DataAnalysis.py file must be run. There is no need to run any other file wihtin this repository, as DataAnalysis.py makes calls to the other classes used for this method. Both the SentimentAnalysis.py and TextBlob.py files were created as a means of testing the individual models before implementing them in the larger project file with the Twitter dataset. 

## Future Work:
The data examined for this analysis comes from English-speaking Twitter users, and the results were not sorted by region or location of each user. Due to the fact that some of the vaccines examined here are only avaialble in certain regions of the world, there is the potential for bias, which may throw off the accuract of the predictions made from the results of this analysis. Thus, incorporating a means of locational analysys, such that each item of text, along with its corresponding classificaiton from the model, can be sorted by geographical location. This would allow us to examine discrepancies between user location and the overall sentiment assigned to certain vaccines, leading to further potential improvments to vaccine distribution plans.

