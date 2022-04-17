# Credit_Risk_Analysis

Machine Learning (ML) offers great promise of improving the value of analysis. To see ML in action, I applied two key analytical techniques to a collection of loan data. I wanted to see if ML might give me a way to predict, given a broad number of data points, which loans would be high-risk.

## Overview of the analysis: Explain the purpose of this analysis.
I focused on two ML techniques in this challenge: **Oversampling** (amplifying the values of the smaller dataset to improve analysis) and **Undersampling** (diminishing the values of the larger dataset). 

I used the dataset column *loan_status* as the target value. the factors of the data included a wide number of factors, from *loan_amount* to annual income, to other factors from the applicants' credit histories. 

When I ran a *value_counts()* command, I found a huge gap between the *low_risk* and the *high_risk* loans. The total of 347 bad loans was 0.5% of the total 68,817 loans in the given period. Looking for evidence to predict bad loans, based on evidence that consisted of 99.5% low-risk loans, would be like looking for a needle in a haystack.

Perhaps ML would help me to make the high-risk needles bigger (via oversampling), or it would make my low-risk haystack small enough to find the high-risk needles with ease.

Using **sklearn**, I set up my training and testing data along the traditional 75/25 split of data. Then I dove into my ML analysis.

### First: a bigger needle. 
I began with **Native Random Oversampling** using *RandomOversampler*. I predicted the scaled data by fitting the resampled **X** and **y** data.

The *confusion_matrix* showed an ominous results. There were only 60 true positive result as opposed to 10,763 false positives. The algorithm was 179 times as likely to miss a high-risk loan as to identify one. The accuracy score on the oversampled set was 53%, not much better than a coin flip. And when I ran a classification report, I found results that were highly accurate for low-risk loans (1.00) but highly inaccurate for low-risk scores (0.01). The one area where oversampling scored high was in its *recall/sensitivity*. With a score of 0.69, the test had identified a high percentage of the high-risk loans, but it came from such a large sample as to hold out little hope.



## Results: 
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

## Summary: 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
