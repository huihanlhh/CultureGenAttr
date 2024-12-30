# Import required libraries
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataframe by removing specific cultures, handling NaNs, and filtering rows based on conditions."""
    # Remove rows with "Culture" as Russian and re-index
    df = df[df['Culture'] != 'Russian']
    df = df.reset_index(drop=True)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Remove rows if the sum of Yes, No, and Not know is not 3
    df = df[(df['Yes'] + df['No'] + df['Not know']) == 3]
    df = df.reset_index(drop=True)
    
    return df

def save_data(df, file_path):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def extract_unique_values(df):
    """Extract unique cultures and topics from the DataFrame."""
    return df['Culture'].unique(), df['Topic'].unique()

def create_evaluation_dict(df, cultures, topics):
    """Convert the dataframe into a dictionary structured by topic and culture."""
    eval_dict = {}
    for topic in topics:
        eval_dict[topic] = {}
        for culture in cultures:
            eval_dict[topic][culture] = {}
            rows = df[(df['Topic'] == topic) & (df['Culture'] == culture)]
            for _, row in rows.iterrows():
                entity = row['Entity']
                eval_dict[topic][culture][entity] = 1 if row['Yes'] == 2 else 0
    return eval_dict

def save_dict_to_json(dictionary, file_path):
    """Save a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(dictionary, f, indent=4)

def calculate_statistics(eval_dict, cultures, topics):
    """Calculate and print detailed statistics including averages for each topic."""
    results_dict = {}
    for topic in topics:
        topic_results = {}
        print(f"Evaluating results for topic: {topic}")
        for culture in cultures:
            symbols = eval_dict[topic][culture].keys()
            lst_responses_eval = [eval_dict[topic][culture][sym] for sym in symbols]
            
            accuracy = accuracy_score(lst_responses_eval, lst_responses_eval)
            precision = precision_score(lst_responses_eval, lst_responses_eval, zero_division=0)
            recall = recall_score(lst_responses_eval, lst_responses_eval)
            f1 = f1_score(lst_responses_eval, lst_responses_eval, average='macro')
            
            topic_results[culture] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        results_dict[topic] = topic_results
        
        # Calculate and print averages for the topic
        avg_accuracy = sum(d['accuracy'] for d in topic_results.values()) / len(topic_results)
        avg_precision = sum(d['precision'] for d in topic_results.values()) / len(topic_results)
        avg_recall = sum(d['recall'] for d in topic_results.values()) / len(topic_results)
        avg_f1 = sum(d['f1_score'] for d in topic_results.values()) / len(topic_results)
        
        print(f"Average Results for {topic}:")
        print(f"Accuracy: {avg_accuracy}")
        print(f"Precision: {avg_precision}")
        print(f"Recall: {avg_recall}")
        print(f"F1 Score: {avg_f1}")

    return results_dict

if __name__ == "__main__":
    # Define file paths
    input_file = "_Human_Evaluation.csv"
    output_file = "_human_eval_processed.csv"
    json_path = "_human_eval_results.json"
    
    # Load, preprocess, and save data
    df = load_data(input_file)
    df_processed = preprocess_data(df)
    save_data(df_processed, output_file)
    
    # Extract information and create evaluation dictionary
    cultures, topics = extract_unique_values(df_processed)
    eval_dict = create_evaluation_dict(df_processed, cultures, topics)
    save_dict_to_json(eval_dict, json_path)
    
    # Calculate and print results and averages
    results = calculate_statistics(eval_dict, cultures, topics)
    results_json_path = "_evaluation_results.json"
    save_dict_to_json(results, results_json_path)