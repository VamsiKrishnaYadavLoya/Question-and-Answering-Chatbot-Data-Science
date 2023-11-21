import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

def perform_prediction(vectorizer,label_encoder_context,test_data,top_n):
    
    # Load the trained model using pickle
    with open('models/knn_model.pickle', 'rb') as model_file:
        knn_model = pickle.load(model_file)
    
    # Combine 'question' and 'context' for testing
    test_combined_text = test_data['question'] + ' ' + test_data['context']

    # Convert test data to TF-IDF vectors using the same vectorizer
    X_test = vectorizer.transform(test_combined_text)

    # Make predictions based on the trained model
    predictions_encoded_context = knn_model.predict(X_test)

    # Map predicted labels back to original labels using inverse_transform
    predicted_contexts = label_encoder_context.inverse_transform(predictions_encoded_context)

    predicted_answer = []
    # Display the actual questions, original contexts, and predicted contexts
    for i, (question, answer, predicted_context) in enumerate(zip(test_data['question'], test_data['answers'], predicted_contexts)):

        # Tokenize the context into sentences
        sentences = [sentence.strip() for sentence in predicted_context.split('.') if sentence.strip()]


        if len(question) > 3:
            # Create vectors for the question and sentences in the context
            question_vector = vectorizer.transform([question])
            context_vectors = vectorizer.transform(sentences)

            # Calculate cosine similarity between the question and each sentence
            similarities = cosine_similarity(question_vector, context_vectors).flatten()
            #top_n = 5

             # Get the indices of the top sentences
            top_indices = similarities.argsort()[-top_n:][::-1]
            # Check if the answer is found in any of the top sentences
            answer_found = any(answer.lower() in sentences[index].lower() for index in top_indices)
            result = answer if answer_found else "answer not found"
            predicted_answer.append(result)

        else:
            predicted_answer.append("not found")
            continue

    test_data['predicted_answers'] = predicted_answer
    test_accuracy = accuracy_score(predicted_answer, test_data['answers'].values)
    
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_data['answers'].values, predicted_answer)

    # Calculate precision and recall
    precision = precision_score(test_data['answers'].values, predicted_answer, average='macro')
    recall = recall_score(test_data['answers'].values, predicted_answer, average='macro')
    
    #test_data.to_csv("evaluation/predicted_data.csv",index=False)
    

    return test_accuracy,precision,recall