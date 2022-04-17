# Credit_Risk_Analysis

Machine Learning (ML) offers great promise of improving the value of analysis. To see ML in action, I applied two key analytical techniques to a collection of loan data. I wanted to see if ML might give me a way to predict, given a broad number of data points, which loans would be high-risk.

## Overview of the analysis: Explain the purpose of this analysis.
I focused on two ML techniques in this challenge: **Oversampling** (amplifying the values of the smaller dataset to improve analysis) and **Undersampling** (diminishing the values of the larger dataset). 

I used the dataset column *loan_status* as the target value. the factors of the data included a wide number of factors, from *loan_amount* to annual income, to other factors from the applicants' credit histories. 

When I ran a *value_counts()* command, I found a huge gap between the *low_risk* and the *high_risk* loans. The total of 347 bad loans was 0.5% of the total 68,817 loans in the given period. Looking for evidence to predict bad loans, based on evidence that consisted of 99.5% low-risk loans, would be like looking for a needle in a haystack.

Perhaps ML would help me to make the high-risk needles bigger (via oversampling), or it would make my low-risk haystack small enough to find the high-risk needles with ease.

Using **sklearn**, I set up my training and testing data along the traditional 75/25 split of data. Then I dove into my ML analysis.

### First: a bigger needle. 
I began with **Native Random Oversampling** (NRO) using *RandomOversampler*. I predicted the scaled data by fitting the resampled **X** and **y** data.

The *confusion_matrix* showed an ominous results. There were only 60 true positive result as opposed to 10,763 false positives. The algorithm was 179 times as likely to miss a high-risk loan as to identify one.
![Oversampling](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/native_random_sampl.png)

Next I used the *SMOTE* algorithm to oversample the data. SMOTE offers the promise of more precision by adding support data using the concept of *closest neighbors*, i.e. oversampling with data close to the values in the original set. The *confusion_matrix* was similar to NRO, if slightly worse, yielding 56 true positive high-risk loans to 10,137 false-positive results.
![smote](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/smote_oversampl.png)

### Second: a smaller haystack.
I used sklearn's *LogisticRegression* to undersample the dataset, hoping to find a more successful way to find the "needles" of high-risk loans. This decreased the values of the predominant dataset. My first look at the precision, *Confusion_matrix* held little promise. There were 59 true positive test results, along with 10,599 false positives.
![LogRegression](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/logistic_regression_undersampl.png)

My final test was a combination of oversampling and undersampling known as **SMOTEENN**, which combines the SMOTE and Edited Nearest Naighbors (ENN) algorithms. I re-created samples of the data for training and testing and made predictions based on the sampled data. The results remained persistenly similar to the data from all previous tests (see below). 
![smoteenn](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/smotteen.png)

### Third: a forest of questions.
The final set of analyses used **forests** to improved accuracy. First, I downloaded *BalancedRandomForestClassifier*. This breaks down the data into "forests" where loan criteria are branched into smaller and smaller formulae in order to predict and classify. Perhaps this would be the method that would give ma a high degree of accuracy for predicting high-risk loans and eliminating them from future loan portfolios.
![randomForest](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/brfclassifier.png)

Finally, I used **adaptive boosting, a.k.a "AdaBoost"** to train a model using the training data. This boost gives extra weight to errors from previous models of testing, repeating the error-rate until the errors are minimized. The directory I downloaded from sklearn for this step was *EasyEnsembleClassifier*. Unfortunately, the results held the 0.01 rate of precision that I found in previous tests, and the other reports were similar
![ada boost](https://github.com/JDittes/Credit_Risk_Analysis/blob/main/images/AdaBoost.png)

## Results: 
- **NRO**: The accuracy score on the oversampled set was 53.04%, not much better than a coin flip. And when I ran a classification report, I found results that were highly accurate for low-risk loans (1.00) but highly inaccurate for low-risk scores (0.01). The one area where oversampling scored high was in its *recall/sensitivity*. With a score of 0.69, the test had identified a high percentage of the high-risk loans, but it came from such a large sample as to hold out little hope.
- **SMOTE**: The results of the SMOTE analysis tracked closely to the data from Native Random Oversampling. The accuracy score of 52.57% was just less than that of the NRO. The data for the high-risk loans was equally imprecise to NRO and had a slightly lower recall score (0.64) than NRO.
- **LogisticRegression**: I found minimal difference between the LogisticRegression results and those of SMOTE or NRO. The *F1 score* for high-risk loans was 0.01, the same as the two oversampled tests. 
- **SMOTEENN**: The accuracy score of 52.95% was the 2nd-highest of the samples I tested, second only to NRO. The recall score of 0.70 was the highest among the sets, but the precision lingered at 0.01, as did the F1 score for the high-risk loans. The imbalanced score of 0.27 found in the classification report was unchanged from previous tests.
- **Balanced Random Forests**: this test yielded the highest accracy score yet: 58.12%. The precision of the results (52 true positives, opposed to 7,170 false positives) was still low at 0.01. F1 and recall scores were similar. The only other significant difference I noticed was in the score of Index of Balanced Accuracy (IBA), which was 7% higher than in the previous over- and under-sampled tests.
- **AdaBoost Classifier**: as with the Balanced Random Forest approach, I had a higher level of accuracy with this test (58.02%) than I had in the over- and under-sampled cases. However the precision of the data was very, very low, meaning that a minimal percentage of "high-risk" classifications were actually high-risk loans. The recall on this data was also low. Only half (0.50) of all the high-risk loans had been found in the tested data.

## Summary: 
The most basic element one learns from the data I evaluated is this: it takes quality data to get quality analysis. With one subset (low-risk loans) making up 99.5% of the provided data, the process of making accurate predictions based on a miniscule subset of high-risk loans was like finding the proverbial needle in a haystack--even with the assistance of some powerful Machine Learning algorithms.

After testing algorithms for both oversampling and undersampling, I found very little accuracy. The values were close to 50%, which is the same rate as flipping a coin after reading a loan application to decide if it's high-risk or not.

The more advanced algorithms, which used learning trees and ensemble learning proved more accurate, but they still lacked precision in identifying high-risk loans.

Because of the disparity in data between subsets, I don't think Machine Learning was useful in bridging the gap between 99.5% and 0.5%. More data on high-risk loans would need to be added to the dataset to improve the ability of Machine Learning to identify factors.

Another option would be reducing the number of factors to look for more precise factors in risk, say, by looking only at interest rate or annual income. This more-granual look would take more time and be more expensive, but one should recall that it compete with a coin-flip for the aptitude in predicting high-risk among loan applications.
