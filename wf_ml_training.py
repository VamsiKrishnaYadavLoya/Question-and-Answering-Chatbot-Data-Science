from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd


def train_knn_model(choose_vectorizer):
    
    train_data = pd.read_csv('data_processed/train_data.csv')
    test_data = pd.read_csv('data_processed/test_data.csv')
    
    # Encode labels using LabelEncoder for context
    label_encoder_context = LabelEncoder()
    train_labels_encoded_context = label_encoder_context.fit_transform(train_data['context'])

    # Combine 'question' and 'context' for training
    train_combined_text = train_data['question'] + ' ' + train_data['context']

    
    if choose_vectorizer == "TF-IDF":
        # Convert text data to TF-IDF vectors for training
        print("Going into tfidf")
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_combined_text)
    
    elif choose_vectorizer == "Count":
        # Convert text data to Count vectors for training
        print("Going into count")
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_combined_text)
        
    else:
        # Convert text data to binary Bag of Words vectors for training
        print("Going into BOW")
        vectorizer = CountVectorizer(binary=True)
        X_train = vectorizer.fit_transform(train_combined_text)

    # Train a KNN model on clustered data
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, train_labels_encoded_context)
    
    # Save the trained model
    model_filename = 'models/knn_model.pickle'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(knn_model, model_file)
    
    return knn_model,vectorizer,label_encoder_context
    
    
    