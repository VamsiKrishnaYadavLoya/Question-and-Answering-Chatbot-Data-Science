import pandas as pd
import wf_ml_training
import wf_ml_prediction
from sklearn.model_selection import train_test_split

def perform_data_split(df):
    df = df[['context','question','answers']]
    train_data, test_data = train_test_split(df[['context', 'question', 'answers']], test_size=0.2, random_state=42)
    train_data.to_csv('data_processed/train_data.csv',index=False)
    test_data.to_csv('data_processed/test_data.csv',index=False)
    return train_data,test_data

def save_model(choose_vectorizer):
    knn_model,vectorizer,label_encoder_context = wf_ml_training.train_knn_model(choose_vectorizer) 
    return knn_model,vectorizer,label_encoder_context

def get_predictions(vectorizer,label_encoder_context,test_data,k,choose_vectorizer):   
    test_accuracy,precision,recall = wf_ml_prediction.perform_prediction(vectorizer,label_encoder_context,test_data,k)
    
    if k == 1:
        model_name = "KNN_"+ choose_vectorizer + "_K_equals_to_one"  
    else:      
        model_name = "KNN_"+ choose_vectorizer + "_K_equals_to_five"
         
    try:
        summary_df = pd.read_csv('evaluation/summary.txt', sep='\t')
    except FileNotFoundError:
        summary_df = pd.DataFrame(columns=['model_name', 'accuracy', 'precision', 'recall'])
           
    # Check if the summary_df is empty and initialize it
    if summary_df is not None:
        
        # Add the results to the DataFrame
        new_row = pd.DataFrame([[model_name, round(test_accuracy, 2), round(precision, 2), round(recall,2)]], columns=['model_name', 'accuracy', 'precision', 'recall'])

        # Append the new row to the summary_df
        summary_df = pd.concat([summary_df, new_row], ignore_index=True)

        # Save the updated summary_df to the summary.txt file
        summary_df.to_csv('evaluation/summary.txt', sep='\t', index=False)
    print(f"{model_name} - Testing Accuracy: {test_accuracy:.2f}\n")
    print(f"{model_name} - Precision: {precision:.2f}\n")
    print(f"{model_name} - Recall: {recall:.2f}\n\n")
    
# The main block
if __name__ == "__main__":
    data = pd.read_csv('data_processed/squad_qa_processed.csv')
    train_data,test_data = perform_data_split(data)

    #First model
    K = 5
    choose_vectorizer = "TF-IDF"
    knn_model,vectorizer,label_encoder_context = save_model(choose_vectorizer)
    get_predictions(vectorizer,label_encoder_context,test_data,K,choose_vectorizer)

    #Second Model
    choose_vectorizer = "BOW"
    knn_model,vectorizer,label_encoder_context = save_model(choose_vectorizer)
    get_predictions(vectorizer,label_encoder_context,test_data,K,choose_vectorizer)
    
    #Third Model
    K = 1
    choose_vectorizer = "BOW"
    knn_model,vectorizer,label_encoder_context = save_model(choose_vectorizer)
    get_predictions(vectorizer,label_encoder_context,test_data,K,choose_vectorizer)
    
    #Fourth model
    choose_vectorizer = "Count"
    knn_model,vectorizer,label_encoder_context = save_model(choose_vectorizer)
    get_predictions(vectorizer,label_encoder_context,test_data,K,choose_vectorizer)
    