import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def compute_summary_statistics(df):
    # Qualitative features
    qual_features = ['question', 'answers']

    summary_statistics = pd.DataFrame(columns=['Feature', 'Number of Categories', 'Most Frequent Category', 'Least Frequent Category'])

    # Calculate summary statistics for qualitative features
    for feature in qual_features:
        num_categories = df[feature].nunique()
        most_frequent_category = df[feature].mode().to_list()
        least_frequent_category = df[feature].value_counts(ascending=True).index[0]

        summary_statistics = summary_statistics.append({
            'Feature': feature,
            'Number of Categories': num_categories,
            'Most Frequent Category': most_frequent_category,
            'Least Frequent Category': least_frequent_category
        }, ignore_index=True)

    # Calculate describe statistics for the selected features
    selected_features = ['char_count', 'word_count', 'unique_word_count']
    statistics = df[selected_features].describe()

    statistics = statistics.transpose()

    # Save both qualitative and quantitative statistics to a single text file
    with open('data_processed/summary.txt', 'w') as file:
        file.write("Summary Statistics for Qualitative Features:\n")
        file.write(summary_statistics.to_string(index=False))
        file.write("\n\nSummary Statistics for Quantitative Features:\n")
        file.write(statistics.to_string())

def compute_pairwise_correlations(df):
    
    quantitative_features = df[['char_count', 'word_count', 'unique_word_count']]  

    # Compute the correlation matrix
    correlation_matrix = quantitative_features.corr()
    
    # Extract the lower triangular part of the matrix (including the diagonal)
    correlation_lower = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape)).astype(bool))
    
    correlation_lower = correlation_lower.replace(np.nan, '')
    
    # Save the correlation coefficients to a text file
    with open('data_processed/correlations.txt', 'w') as file:
         file.write(correlation_lower.to_string())


def create_visualization(df):
    
    #create a histogram for question data
    plt.figure(figsize=(8, 6))
    sns.histplot(df['question tokens'], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of question tokens')
    plt.xlabel('question tokens') #x-axis represents the question number
    plt.ylabel('Frequency') #the number of times the question was taken.
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/question_histogram.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file

    #create a histogram for answer data
    plt.figure(figsize=(8, 6))
    sns.histplot(df['answers'], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of answer tokens')
    plt.xlabel('answer tokens') #x-axis represents the answer number
    plt.ylabel('Frequency') #the number of times the answer was taken.
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/answer_histogram.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file
    
    
    # Create box plots
    #Box plots reveals outliers, the spread of the data, and central tendencies.
    #char_count plot shows that there are 2 questions which are very length compared to remining ones.
    plt.figure(figsize=(10, 6))
    df[['char_count','word_count', 'unique_word_count']].boxplot()
    plt.title('Box Plots for Numerical Columns')
    plt.xticks(rotation=45)
    plt.ylabel('Counts')
    plt.savefig('visuals/box_plot.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file
    
    
    # Create a scatter plot
    #As word_count increases, the unique tends to increase.This shows a positive linear relationship or positive correlation
    plt.figure(figsize=(8, 6))
    plt.scatter(df['word_count'], df['unique_word_count'], alpha=0.5)
    plt.title(f'Scatter Plot: word_count vs unique_word_count')
    plt.xlabel('word_count')
    plt.ylabel('unique_word_count')
    plt.grid(True)
    plt.savefig('visuals/scatter_plot_word_vs_unique.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file
    
    # Create a scatter plot
    #Even the wordcount and character count  also shows a positive linear relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(df['word_count'], df['char_count'], alpha=0.5)
    plt.title(f'Scatter Plot: word_count vs char_count')
    plt.xlabel('word_count')
    plt.ylabel('char_count')
    plt.grid(True)
    plt.savefig('visuals/scatter_plot_word_vs_char.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file
    
    
    # Create a scatter plot
    #Even the unique word count and character count  also shows a positive linear relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(df['unique_word_count'], df['char_count'], alpha=0.5)
    plt.title(f'Scatter Plot: unique_word_count vs char_count')
    plt.xlabel('unique_word_count')
    plt.ylabel('char_count')
    plt.grid(True)
    plt.savefig('visuals/scatter_plot_unique_vs_char.png',dpi=300, bbox_inches='tight') # Save the plot as a .png file
    
