#### SERX94: Machine Learning Evaluation
#### Chatbot Implementation using NLP (Question & Answering)
#### Vamsi Krishna Yadav Loya
#### 11/20/2022

## Evaluation Metrics
### Metric 1
**Name:** Accuracy

**Choice Justification:** I have choosen Accuracy as my first metric. Accuracy measures the overall correctness of the model by calculating the ratio of correctly predicted instances to the total instances. 
                          In the context of my project question-answering classification problem, accuracy represents the proportion of correctly predicted instances
                          (correctly identified the top sentences from a context related to the given context) out of the total number of instances in the test set. 
                          If the predicted top sentences contains the actual answer given in the test set then I am considering the prediction as positive (which means correct prediction).
                          It provides a general assessment of how well the model is performing across all classes.

Interpretation:** Precision

### Metric 2
**Name:** Precision

**Choice Justification:** Second metric I have choosen is precision. Precision measures the accuracy of the positive predictions made by the KNN model. 
                          In the context of my project question-answering classification problem. Precision indicates how many of the predicted context were actually relevant to the given question 
                          and after that predicting the correct answers using the context. A high precision score means that when the model predicts a certain question context and answer, which is likely to be correct.

### Metric 3
**Name:** Recall

**Choice Justification:** Recall which is also known as sensitivity or true positive rate, measures the model's ability to capture all the relevant positive instances. 
                          In the context of my project question-answering classification problem. Recall indicates how many of the actual question contexts were successfully predicted by the KNN model.
                          A high recall score means that the model is sensitive to identifying a large portion of the relevant question contexts.



## Alternative Models
### Alternative model 1: KNN Model with "TF-IDF as vectorizer" with a choosen k value of "5".
**Construction:** This model uses the KNN algorithm with a TF-IDF vectorizer.
                  The TF-IDF vectorizer converts context and questions into TF-IDF vectors during training. While performing testing I am using the same
                  TF-IDF vectorizer to convert questions and the top "5" sentences since the KNN model is constructed with a chosen k value of 5.

**Evaluation:** I have evaluated my model using the test data based on accuracy, precision and recall. Since it is a kind of multi-label classification
                problem, I have treated each answer as a individual label and predicted accuracy, precision and recall. I have used macro averaging to calculate precision and recall,
                where Macro-averaging calculates the precision,recall for each class independently and then takes the average. It treats all classes equally, regardless of their frequency in the dataset.
                These metrics can give you an overall sense of how well your model is performing in terms of precision across all the classes.
For this model the values are as below:  
Testing Accuracy: 0.74
Precision: 0.73
Recall: 0.72

### Alternative model 2: KNN Model with "Bag of Words (BoW)" with a choosen k value of "5".
**Construction:** TODO

**Evaluation:**I have evaluated my model using the test data based on accuracy, precision and recall. Since it is a kind of multi-label classification
                problem, I have treated each answer as a individual label and predicted accuracy, precision and recall. I have used macro averaging to calculate precision and recall,
                where Macro-averaging calculates the precision,recall for each class independently and then takes the average. It treats all classes equally, regardless of their frequency in the dataset.
                These metrics can give you an overall sense of how well your model is performing in terms of precision across all the classes.
For this model the values are as below: 
Testing Accuracy: 0.80
Precision: 0.80
Recall: 0.79

### Alternative model 3: KNN Model with "Bag of Words" with a choosen k value of "1".
**Construction:** TODO

**Evaluation:** I have evaluated my model using the test data based on accuracy, precision and recall. Since it is a kind of multi-label classification
                problem, I have treated each answer as a individual label and predicted accuracy, precision and recall. I have used macro averaging to calculate precision and recall,
                where Macro-averaging calculates the precision,recall for each class independently and then takes the average. It treats all classes equally, regardless of their frequency in the dataset.
                These metrics can give you an overall sense of how well your model is performing in terms of precision across all the classes.
For this model the values are as below: 
Testing Accuracy: 0.63
Precision: 0.64
Recall: 0.62

## Best Model

**Model:** Among the above three alternate models, The Knn model with Bag of words Vectorizer and K value of 5 is best performing among the rest of the models 
           because it has high accuracy,precision and recall compared to the other models after than Alternative model1 performs well but slightly 
           less accurate than Alternative Model.

## Visualization
### Visual 1
**Analysis:** 
Box plot: This plot gives insights about outliers, how the data is being spread across and its central tendencies. This plot shows the questions which are lengthy than the other questions.

### Visual 2
**Analysis:** 
Scatter Plot 1: This plot is drwan for word count vs uniques word count. This plot clearly shows that as the word count increases tye unique word also tends to increase. This clearly shows the positive linear relationship or we can say +ve corelation.

### Visual 3
**Analysis:** 
Scatter Plot 2: This plot is drawn for word count vs character count. Even the word count and character counts are also showing a clear positive corelation or simply +ve linear realtionship.

### Visual 4
**Analysis:** 
Scatter Plot 3: This plot is drawn for unique word count vs character count. Even the unique word count and character counts are also showing a clear positive corelation or simply +ve linear realtionship.

### Visual 5
**Analysis:** 
Histogram plot 1: This histogram provides insights into the frequency of different questions in the dataset. It also helps to understand the distribution of question data.

### Visual 6
**Analysis:** 
Histogram plot 2: This histogram provides insights into the frequency of different answers in the dataset. It also helps to understand the distribution of answer data.
