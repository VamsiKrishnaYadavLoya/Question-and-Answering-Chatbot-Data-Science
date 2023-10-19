#### SER594: Exploratory Data Munging and Visualization
#### Data Munging & Visualization for Chatbot Implementation     Project with NLP.
#### Vamsi Krishna Yadav Loya
#### 10/18/2023

## Basic Questions
**Dataset Author(s):** Rajpurkar,Pranav,Zhang,Jian,Lopyrev,
                       Konstantin,Liang,and Percy.

**Dataset Construction Date:** 2016. This dataset is last updated 6                               months ago.
                               Dataset URL: https://huggingface.co/datasets/squad

**Dataset Record Count:** 20,000. The original datset contains 84k                          rows. But we have partial dataset for time                          being with 20,000 rows.

**Dataset Field Meanings:** My data set is .csv file which contains                            data in the form of table with fields                            named as "id", "title", "context",                            "question" (These fields contain data in                            form of "Strings"), these fields have                            information on "id" with appropriate                            title/heading for a specific piece of                            content, and context field typically                            contains content from which question and                            answers are derived."answers" field                            is sequence type which contains answers                            to the questions. 

**Dataset File Hash(es):**  1b6e732552c531d87f6ad3953ace5819

## Interpretable Records
### Record 1
**Raw Data:** id: 17	
              title: 5733a6424776f41900660f4f	
			  content: University_of_Notre_Dame	The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study â€“ aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering â€“ with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.	
			  Question: Before the creation of the College of Engineering similar studies were carried out at which Notre Dame college?	
			  Answer: the College of Science



**Interpretation:** 
In the above record given, the raw data provides information about a specific entity with "id" of 17. The tilte for that entity is given above. The content for the record contains some historical information about University of Notre Dame, and it's College of Engineering which is established in 1920. It clearly mentions early courses related to civil and mechanical engineering were part of College of Science.
The College of Engineering, currently situated in Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, encompasses five departments of study: aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering. The college offers eight B.S. degrees, and it also provides five-year dual degree programs in collaboration with the Colleges of Arts and Letters and of Business, awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.
The question associated with this record is: "Before the creation of the College of Engineering similar studies were carried out at which Notre Dame college?" 
The answer to this question is "the College of Science," indicating that similar engineering studies were conducted at the College of Science prior to the establishment of the College of Engineering at the University of Notre Dame

### Record 2
**Raw Data:** id: 42	
              title: 5733ae924776f41900661016	
			  content: University_of_Notre_Dame	Notre Dame is known for its competitive admissions, with the incoming class enrolling in fall 2015 admitting 3,577 from a pool of 18,156 (19.7%). The academic profile of the enrolled class continues to rate among the top 10 to 15 in the nation for national research universities. The university practices a non-restrictive early action policy that allows admitted students to consider admission to Notre Dame as well as any other colleges to which they were accepted. 1,400 of the 3,577 (39.1%) were admitted under the early action plan. Admitted students came from 1,311 high schools and the average student traveled more than 750 miles to Notre Dame, making it arguably the most representative university in the United States. While all entering students begin in the College of the First Year of Studies, 25% have indicated they plan to study in the liberal arts or social sciences, 24% in engineering, 24% in business, 24% in science, and 3% in architecture.	
			  Question: What percentage of students at Notre Dame participated in the Early Action program?	
			  Answer: 39.10%


**Interpretation:** 
The above given record has "id" of 42 with title 5733ae924776f41900661016. Content is same like the past record University_of_Notre_Dame which contains information about the Universoty of Notre Dame, including details abouts its admissions, academic profile, early action, and the breakdown of students intended fields of study.
Question: What percentage of students at Notre Dame participated in the Early Action program? Answer: 39.10% In this record, the question is asking for the percentage of students who participated in the Early Action program at the University of Notre Dame, and the answer is provided as 
Answer: 39.10%. 

## Background Domain Knowledge

Introduction: The project "Chatbot Implementation Project with NLP" is useful to address the communication needs of any company efficiently. Mainly it leverages the power of Chatbots immediately instead of depending on human customer support. This project explores the utilization of NLP-Natural Language Processing techniques in Python to create a Chatbot system. I will be trying to program the Chatbot to autonomously interpret and respond to user queries based on the dataset I train Chatbot.

Ket concepts to have knowledge on:

1. NLP(Natural Language Processing): It is a core technology which enables chatbots to understand and generate human language. It can perform a range of tasks including text parsing, sentiment analysis, and language recognition.
2. Text Preprocessing: First the intial step is text processing in NLP which involves various techniques to clean the text to be suitable for analysis.
3. Tokenization: This is a process of breaking down a text into smaller units called tokens. Tokenization in NLP allows chatbot to work with individual words.
4. Machine Learning: Machine Learning techniques such as deep learning enables chatbots to learn from user interactions and improve their response over time which is cruicial for Chatbot.

Reference:
1. G. (2023, June 9). Conversational AI: A Comprehensive Guide (2023). Gupshup. https://www.gupshup.io/resources/guide/conversational-ai-comprehensive-guide
2. Schlicht, M. (2023, April 16). The Complete Beginner’s Guide To Chatbots - Chatbots Magazine. Medium. https://chatbotsmagazine.com/the-complete-beginner-s-guide-to-chatbots-8280b7b906ca
3. Chatbots Development Using Natural Language Processing: A Review. (2022, July 1). IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/document/10017592

## Data Transformation
### Transformation 1
**Description:** 
Bi-Grams Transformation: It involves extracting sequences of n words from both question and answer. I have used ntlk package to get the bi-grams.

**Soundness Justification:** 
N-grams transfornation is widely used technique in natural language processing for preserving the context of words in text. It helps in capturing phrases that convey specific meanings. This operation does not change the semantics of the data, and it preserves meaningful information within the text.

### Transformation 2
**Description:** 
Part-of-Speech Tags: I used pos_tag from NLTK to extract part-of-speech tags for words in questions and answers. This can help the model understand the grammatical structure.Each word is tagged as NN(Noun),ADJ(adjective) etc.

**Soundness Justification:** 
This operation is crucial for understanding the grammatical structure of text. It does not alter the core meaning of the text but aids in the analysis of language structure, helping the model understand the roles that words play in the sentence.

### Transformation 3
**Description:** 
Named Entity Recognition (NER): Applied NER transformation to identify and label entities (e.g., names, dates, locations) in the text. I have used "ne_chunk" from NLTK. It is useful for handling specific information.

**Soundness Justification:** 
NER is important for recognizing specific information within text, which is valuable for chatbot applications. It helps the model provide contextually relevant responses. The transformation does not change the essential meaning of the data but enhances the ability to identify and handle specific entities.

### Transformation 4
**Description:** 
Sentiment analysis is employed to determine the sentiment (positive, negative, or neutral) of both questions and answers using sentiment analysis tools. I have imported "SentimentIntensityAnalyzer" from NLTK. This information helps the model respond with an appropriate tone or emotion.

Sentiment Analysis: To Determine the sentiment of the questions and answers using sentiment analysis tools.
1. if neg(negative) > 0.5 the senence is in a negative tone
2. if pos(positive) > 0.5 the senence is in a positive tone
3. if neu(neutral) > 0.5 the senence is in a neutral tone

**Soundness Justification:** 
Sentiment analysis is applied to understand the emotional tone of text. It doesn't change the primary information but adds an emotional context, allowing the chatbot to respond with empathy and relevance based on user sentiment. The transformation does not discard data, introduce errors, or introduce outliers; instead, it enriches the data with emotional context.


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





