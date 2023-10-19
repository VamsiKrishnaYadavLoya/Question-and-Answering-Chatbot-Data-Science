import wf_dataprocessing
import wf_visualization
import pandas as pd


# Step 1: Process data using wf_processing.py.
def preprocessing(input_data):
    processed_data = wf_dataprocessing.process_data(input_data)
    processed_data.to_csv('data_processed/squad_qa_processed.csv',index=False)
    return processed_data

# Step 2: Generate statistics and visuals using wf_visualization.py.
def visualizations(df):
    wf_visualization.compute_summary_statistics(df)
    wf_visualization.compute_pairwise_correlations(df)
    wf_visualization.create_visualization(processed_data)

    
# The main block
if __name__ == "__main__":
    input_data = pd.read_csv('data_original/squad_qa_data.csv')
    processed_data = preprocessing(input_data)
    visualizations(processed_data)