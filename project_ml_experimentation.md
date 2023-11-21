#### SERX94: Experimentation
#### Chatbot Implementation using NLP (Question & Answering)
#### Vamsi Krishna Yadav Loya
#### 11/20/2023


## Explainable Records
### Record 1
**Raw Data:** 
Context: The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). 
         Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, 
         the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers 
         reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau
         Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a 
         Buechner Prize for Preaching.

Question:Where is the headquarters of the Congregation of the Holy Cross?

Answer:Rome

Prediction Explanation:** After the model is trained on the Context, question data then the model prediction is performed based on the question. From the above the question the model predicts the answer by first mapping which context is relevant to the question so first it outputs the Context related to the given question because our data has different context, and each context has multiple questions. In order to predict the context K value is taken as 1 so that it will only identify the top relevant context, once that is done the same model is used to identify the top sentences that are relevant to the given question in the already predicted context. In this case the model predicts the output as “The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome)” As the sentence is taking about the Congregation of Holy cross and its headquarters location which is basically what the question is all about in making the prediction positive (“Answer found”)
### Record 2
**Raw Data:** 

Context: In 2014 the Notre Dame student body consisted of 12,179 students, with 8,448 undergraduates, 2,138 graduate and professional and 1,593 professional (Law, M.Div., Business, M.Ed.) students. Around 21â€“24% of students are children of alumni, and although 37% of students come from the Midwestern United States, the student body represents all 50 states and 100 countries. As of March 2007 update. The Princeton Review ranked the school as the fifth highest 'dream school' for parents to send their children. As of March 2015 update The Princeton Review ranked Notre Dame as the ninth highest. The school has been previously criticized for its lack of diversity, and The Princeton Review ranks the university highly among schools at which "Alternative Lifestyles are Not an Alternative." It has also been commended by some diversity oriented publications; Hispanic Magazine in 2004 ranked the university ninth on its list of the topâ€“25 colleges for Latinos, and The Journal of Blacks in Higher Education recognized the university in 2006 for raising enrollment of African-American students. With 6,000 participants, the university's intramural sports program was named in 2004 by Sports Illustrated as the best program in the country, while in 2007 The Princeton Review named it as the top school where "Everyone Plays Intramural Sports." The annual Bookstore Basketball tournament is the largest outdoor five-on-five tournament in the world with over 700 teams participating each year, while the Notre Dame Men's Boxing Club hosts the annual Bengal Bouts tournament that raises money for the Holy Cross Missions in Bangladesh.

Question: How many students in total were at Notre Dame in 2014?

Answer: 12,719

Prediction Explanation:** The question is provided to the model to predict the top answer that is included in exact sentence. From the above question the model correctly identifies context related to the given question because our data has different context, and each context has multiple questions. After that the same model is used to identify the top sentences that are relevant to the given question in the already predicted context. In this case the model predicts the output as “In 2014 the Notre Dame student body consisted of 12,179 students, with 8,448 undergraduates, 2,138 graduate and professional and 1,593 professional (Law, M.Div., Business, M.Ed.) students”. The model's output reflects an understanding of the institutional structures and entities within the domain of the  Notre Dame university along with the additional details were provided the distribution of students as well.


## Interesting Features
### Feature A
**Feature:** Question

**Justification:** The "question" feature is crucial because it serves as the input that prompts the model to generate a relevant answer. 
                   In case of question-answering context, the specificity, clarity, and contextual nuances present in the question directly influence the model's ability 
                   to comprehend and generate accurate responses. A well-constructed question provides essential cues and context for the model to understand the user's 
                   intent and retrieve relevant information. Therefore, variations in the structure, wording, and complexity of questions can significantly impact the model's performance.
                   By focusing on the "question" feature, the model can better capture the subtleties of user queries and improve its overall predictive accuracy.


### Feature B
**Feature:** Context

**Justification:** The context is crucial in understanding the meaning and intent behind a question. Different contexts might lead to different interpretations and answers. Some questions might be ambiguous and can only be resolved by considering the context in which they are asked. For instance, the same question may have different answers depending on whether it is related to technology, science, or literature.The meaning and relevance of a question can be better understood and addressed when considering the context in which it is asked. This, in turn, should impact the model's ability to predict the correct context for a given question.

## Experiments

### Varying A
**Prediction Trend Seen:** In this scenario, varying Feature A(Question) impacts the prediction a lot. For a fixed context, different question are asked. In one question I used only stop words like "Is it and or the?". Even how well the model is trained the model struggles to give the answer. It just gives the common text as output, since the input question lacks informative content.
The predictions completely different compared to situation where meaningful words and context are present in the question like "Is it a university or the church In Notre Dame?".

### Varying B
**Prediction Trend Seen:**  The question  feature is Fixed insted (Varying Feature B)context feature is varied. So the question is no where related to the information present in the context. The context talks about University of Notre Dame where as the question is Which two intelligence groups merged to form Joint Intelligence Service? this question is basically related to a movie context which results in the  answer predicted by model is completely irrelavant and the predictions change as I present the model with different varying contexts
 
### Varying A and B together
**Prediction Trend Seen:** Varying both questions and contexts simultaneously, by ensuring they are correlated. I have seen interactions where certain combinations of questions and contexts lead to more accurate predictions and the model learnt to recognize patterns that involve both features and using those patterns while predicting the answers. The accuracy of the model in this case is around 70 percent so as recall and precision are also displaying the similar percentage.

### Varying A and B inversely
**Prediction Trend Seen:** Varying both questions and contexts simultaneously, ensuring they are inversely correlated.It results in some situations where the question becomes more complex paricularly question while the context has more generic terms. So the question and context provide conflicting information which results in the accuracy of the model to decrease as the answers provided in most of the cases are generic. I have performed this experimentation on around 10 questions out of which not even one answer is correctly predicted.

