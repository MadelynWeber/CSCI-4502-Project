# Using Sentiment Analysis to Predict General Feelings Toward COVID-19 Vaccines 
An Analysis of the General Opinions Toward COVID-19 Vaccines and the Relationship Between Vaccine Producers, Region, and Temporal Data as Found From COVID-19-Related Tweets.

## Project Summary:
This is a project which makes use of a Naïve Bayesian Sentiment Analysis model to conclude the general sentiment associated with some of the COVID-19 vaccines currently avaialble for distribution. Through collecting sentiment classificaitons assinged to text data collected from Tweets relating to COVID-19 vaccines, we are able to predict the general feelings that English-speaking Twitter users feel towards the vaccines examined. Each Tweet was screened for mentions of the following vaccines: Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZeneca, Covaxin, and Sputnik V. Through vizualizations of the data collected, we are able to generate an approximate distribution of where each vaccine falls within the sentiment classifications of either positive, negative, or neutral. From these results, we can then begin to think of potential improvements to the current vaccine distribution plans, by taking in both the vaccines examined and the overall sentiment assigned to each, in an attempt to increase the numbers of the current vaccinated public. 

## Key Results:
It was found from the results of this analysis that the general overall sentiment toward COVID-19 vaccines is largely non-negative. It was also found that the frequency in which certain vaccines came up within the text data did not match the frequency of real-world distribution data. That is to say, the vaccines which are in high distiribuiton currenlty are not necessarily those which were mentioned the most often within the dataset examined here. It was also found that some of the vaccines which received the highest ratio of non-negative overall sentiment were not among the top vaccines seen in real-world distribution data. From this, it can be concluded that a potential improvement to the current vaccine distribution plans may include increasing the vaccines avaialble for individuals to select from within certain regions. 

## Data Files:

## Project Files:

### DataAnalysis.py 
- This is the Analysis file

### SentimentAnalysis.py 
- This is the Sentiment Analysis model 

### TextBlob.py
- This is the TextBlob implementation of Sentiment Analysis

## Running the Project:

## Future Work:
The data examined for this analysis comes from English-speaking Twitter users, and the results were not sorted by region or location of each user. Due to the fact that some of the vaccines examined here are only avaialble in certain regions of the world, there is the potential for bias, which may throw off the accuract of the predictions made from the results of this analysis. Thus, incorporating a means of locational analysys, such that each item of text, along with its corresponding classificaiton from the model, can be sorted by geographical location. This would allow us to examine discrepancies between user location and the overall sentiment assigned to certain vaccines, leading to further potential improvments to vaccine distribution plans.

