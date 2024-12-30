import logging
import argparse
import math
from collections import Counter
import csv
from elasticsearch import Elasticsearch
from es import count_documents_containing_phrases, get_document_ids_containing_phrases, get_documents_containing_phrases
import string
import re
import numpy as np
import json
import math
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from transformers import AutoTokenizer

# Load olmo-7b tokeniser
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

MAX_SEQ_LEN = 2048

# Elasticsearch connection setup
es = Elasticsearch(
    cloud_id="<inset cloud ID>",
    api_key="<insert API key>",
    retry_on_timeout=True,
    http_compress=True,
    request_timeout=180,
    max_retries=10
)
index = "docs_v1.5_2023-11-02"

# Custom serializer for numpy types
def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# -------------------***********************************************------------------- #
"""
Functions to calculate the d-SNR metric of a document given a set of query terms which were used to retrieve the document
"""
# Compute the term frequency for all cultures in the document
def compute_relevant_tf(document, relevant_terms):
    """
    Computes term frequency for relevant terms in a document using an optimized approach.
    
    :param document: A list of words representing the document.
    :param relevant_terms: A set of terms for which to compute the frequency.
    :return: A dictionary with relevant terms as keys and their frequencies as values.
    """
    # Filter the document to include only relevant terms using a set intersection
    filtered_terms = (term for term in document if term in relevant_terms)
    
    # Use Counter to efficiently count occurrences of each relevant term
    term_count = Counter(filtered_terms)

    return term_count

# Compute the d-SNR score of a document given a set of query terms
def score_query(document, query_culture, nationalities_list):
    """
    Scores a document based on a list of query terms using normalized TF for specific terms.
    
    :param document: A list of words representing the document.
    :param query_terms: A list of query terms.
    :param query_culture: The culture of interest.
    :param nationalities_list: A list of tuples (nation, nationality).
    :return: Total score for the document based on the query terms.
    """
    query_culture = query_culture.lower()
    relevant_terms = {nation.lower() for nation, _ in nationalities_list}
    relevant_terms.update({nationality.lower() for _, nationality in nationalities_list})
    
    term_count = compute_relevant_tf(document, relevant_terms)

    frequency_denominator = 0
    query_nation = ""
    for nation, nationality in nationalities_list:
        lower_nation = nation.lower()
        lower_nationality = nationality.lower()

        if lower_nationality != query_culture:
            frequency_denominator += term_count.get(lower_nationality, 0)
            frequency_denominator += term_count.get(lower_nation, 0)
        elif query_nation == "":
            query_nation = lower_nation

    frequency_num = term_count.get(query_culture, 0) + term_count.get(query_nation, 0)

    # Avoid division by zero and negative infinity scenario
    if frequency_num > 0:
        score = math.log2(frequency_num / (1 + frequency_denominator))
    else:
        score = -np.inf

    return score
# -------------------***********************************************------------------- #
    
"""
Function to calculate the minimum sentence distance between the query terms in a document (d_SENT)
"""
def calc_min_sentence_distance(text, word1, word2):
    text = text.lower()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentences = [re.sub(r'(?<!\s)-', ' ', s).translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    
    word1_indices = [i for i, s in enumerate(sentences) if word1 in s]
    word2_indices = [i for i, s in enumerate(sentences) if word2 in s]
    
    min_distance = min((abs(i - j) for i in word1_indices for j in word2_indices), default=1e+10)
    return int(min_distance)

# -------------------***********************************************------------------- #
"""
Function to calculate the minimum token distance between the query terms in a document (d_TOK)
"""
def mark_tokens(text, word, symbol):
    #logging.info(f"Text: {text}")
    # Tokenize the input text with offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = encoding.tokens()  # Use .tokens() to get the token strings directly
    token_offsets = encoding['offset_mapping']

    # Initialize marks array
    marks = [0] * len(tokens)

    # Helper function to mark tokens within a specific range
    def mark_range(start_idx, end_idx, mark):
        for i, (start, end) in enumerate(token_offsets):
            if start != None and end != None:  # Avoid special tokens like [CLS], [SEP]
                if start < end_idx and end > start_idx:
                    marks[i] = max(marks[i], mark)  # Use max to respect symbol over word if overlapping

    # Find and mark all instances of the symbol and word
    def find_and_mark(phrase, mark):
        start = 0
        while True:
            start = text.find(phrase, start)
            if start == -1:  # No more occurrences found
                break
            end = start + len(phrase)
            mark_range(start, end, mark)
            start += len(phrase)  # Move start index past the current occurrence to find next ones

    # First mark symbols, then words (symbol takes precedence with max in mark_range)
    find_and_mark(symbol, 2)
    find_and_mark(word, 1)

    return tokens, marks

# Function to calculate shortest distance between two marked tokens
def calc_shortest_inter_token_distance(marks):
    # Case - I : When w1 is before w2
    marks_case_one = []
    for i in range(len(marks)):
        if marks[i] == 1:
            if i != 0 and i != len(marks)-1:
                if (marks[i-1] != 1 and marks[i+1] != 1) or (marks[i-1] != 1 and marks[i+1] == 1):
                    marks_case_one.append(1)
                else:
                    marks_case_one.append(0)
            elif i == 0:
                marks_case_one.append(1)
            else:
                marks_case_one.append(0)
        elif marks[i] == 2:
            if i != 0 and i != len(marks)-1:
                if (marks[i-1] != 2 and marks[i+1] != 2) or (marks[i-1] == 2 and marks[i+1] != 2):
                    marks_case_one.append(2)
                else:
                    marks_case_one.append(0)
            elif i == len(marks)-1:
                marks_case_one.append(2)
            else:
                marks_case_one.append(0)
        else:
            marks_case_one.append(0)

    # Case - II : When w2 is before w1
    marks_case_two = []
    for i in range(len(marks)):
        if marks[i] == 1:
            if i != 0 and i != len(marks)-1:
                if (marks[i-1] != 1 and marks[i+1] != 1) or (marks[i-1] == 1 and marks[i+1] != 1):
                    marks_case_two.append(1)
                else:
                    marks_case_two.append(0)
            elif i == len(marks)-1:
                marks_case_two.append(1)
            else:
                marks_case_two.append(0)
        elif marks[i] == 2:
            if i != 0 and i != len(marks)-1:
                if (marks[i-1] != 2 and marks[i+1] != 2) or (marks[i-1] != 2 and marks[i+1] == 2):
                    marks_case_two.append(2)
                else:
                    marks_case_two.append(0)
            elif i == 0:
                marks_case_two.append(2)
            else:
                marks_case_two.append(0)
        else:
            marks_case_two.append(0)

    #logging.info(f"Marks case one: {marks_case_one}")
    #logging.info(f"Marks case two: {marks_case_two}")
    # Shortest distance -  case I
    distance_case_one = math.inf
    local_count = 0
    flag = False
    for i in range(len(marks_case_one)):
        if marks_case_one[i] == 1:
            local_count = 1
            flag = True
        elif marks_case_one[i] == 2:
            local_count += 1
            if flag:
                distance_case_one = min(distance_case_one, local_count)
            local_count = 0
        else:
            local_count += 1
            
    # Shortest distance -  case II
    distance_case_two = math.inf
    local_count = 0
    flag = False
    for i in range(len(marks_case_two)):
        if marks_case_two[i] == 2:
            local_count = 1
            flag = True
        elif marks_case_two[i] == 1:
            local_count += 1
            if flag:
                distance_case_two = min(distance_case_two, local_count)
            local_count = 0
        else:
            local_count += 1
    #print(distance_case_one, distance_case_two)
    #logging.info(f"token distance : {min(distance_case_one, distance_case_two)}")
    return min(distance_case_one, distance_case_two)

# -------------------***********************************************------------------- #
"""
Define the function to be executed in parallel
"""
def process_document(doc, query_terms, query_culture, countries_nationalities_list):
    doc_reqd_1, doc_id = doc['_source']['text'], doc['_id']
    sential_dist = calc_min_sentence_distance(doc_reqd_1, query_terms[0], query_terms[1])

    # Replace "-" not followed by a space with a space in the document
    doc_reqd_1 = re.sub(r'(?<!\s)-', ' ', doc_reqd_1)
    doc_reqd_1 = doc_reqd_1.translate(str.maketrans('', '', string.punctuation))
    # Lowercase the document
    doc_reqd_1 = doc_reqd_1.lower()
    
    # Calculate the ranking score
    doc_1_lst = [word.lower() for word in doc_reqd_1.split()]
    ranking_score = score_query(doc_1_lst, query_culture, countries_nationalities_list)

    # Initialize token distance
    token_distance = 1e+10
    # If ranking score is positive, calculate the token distance
    if sential_dist != 1e+10:
        if ranking_score > 0 or sential_dist < 2:
            # Calculate the shortest distance between the two words
            _, marks = mark_tokens(doc_reqd_1, query_terms[0], query_terms[1])
            token_distance = calc_shortest_inter_token_distance(marks)

    result = {
        "doc_id": doc_id,
        "sentence_distance": sential_dist,
        "token_distance": token_distance,
        "ranking_score": ranking_score
    }
    return result

# -------------------***********************************************------------------- #
"""
Multi-proc functions
"""
def process_single_document(doc, query_terms, query_culture, countries_nationalities_list):
    result = process_document(doc, query_terms, query_culture, countries_nationalities_list)
    if (result['ranking_score'] > 0 and result['token_distance'] < MAX_SEQ_LEN) or (result['sentence_distance'] < 2 and result['ranking_score'] > -1):
        return {
            'doc_id': result['doc_id'],
            'sentence_distance': result['sentence_distance'],
            'ranking_score': result['ranking_score'],
            'token_distance': result['token_distance']
        }
    return None

def process_documents(docs, query_terms, query_culture, countries_nationalities_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_single_document, [(doc, query_terms, query_culture, countries_nationalities_list) for doc in docs])
    
    # Filter out None results and separate data
    filtered_results = [result for result in results if result is not None]
    doc_ids = [res['doc_id'] for res in filtered_results]
    sentence_distances = [res['sentence_distance'] for res in filtered_results]
    ranking_scores = [res['ranking_score'] for res in filtered_results]
    token_distances = [res['token_distance'] for res in filtered_results]

    return doc_ids, sentence_distances, ranking_scores, token_distances

# -------------------***********************************************------------------- #
"""
Function to find cultures which generated a specific symbol
"""
def find_countries_with_symbol(responses, symbol):
    """
    Find countries with responses containing the specified symbol.
    
    :param responses: Dictionary of responses categorized by nationality and gender.
    :param symbol: Symbol to search for in responses.
    :return: List of countries where the symbol appears in any response.
    """
    symbol_lower = symbol.lower()  # Lowercase the symbol once for efficiency
    lst_of_countries = []

    for nationality, responses_ in responses.items():
        if nationality:  # Ensure nationality is not an empty string
            responses_new = responses_['male'] + responses_['female'] + responses_['']
            # Use any to check if the symbol is in any response to avoid counting
            temp_symbol = symbol + ';' # Add a semicolon to the symbol for better matching
            temp_symbol_two = symbol + '.' # Add a period to the symbol for better matching
            if any(symbol_lower in response.lower() or temp_symbol.lower() in response.lower() or temp_symbol_two.lower() in response.lower() for response in responses_new):
                lst_of_countries.append(nationality)

    return lst_of_countries


# Main function
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name to use for classification")
    parser.add_argument("--topic", type=str, default="clothing", help="Topic of interest")
    parser.add_argument("--multiproc", action="store_true", help="Use multiproc for processing documents")     # argument for if you want multiproc

    args = parser.parse_args()
    # Loading nationalities from CSV
    with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    # Topic of interest
    topic = args.topic
    print(f"Topic: {topic}")

    # Setup logging
    logging.basicConfig(filename=f"output_{topic}.log", filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Path to non_independent_symbols 
    path_to_non_independent_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_non_independent_symbols.json"

    # JSON path for storing results
    path_to_doc_ranking = f"{args.home_dir}/memoed_scripts/training_document_classification_{args.model_name}_{topic}_left.json"

    # Path to shortened responses
    path_to_shortened_responses = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_new_shortened.json"

    # Get the list of non-independent symbols
    with open(path_to_non_independent_symbols, "r") as r:
        non_independent_symbols = json.load(r)

    # Get the list of non-independent symbols for the topic
    non_independent_symbols_topic = non_independent_symbols[topic]

    # Get the shortened responses
    with open(path_to_shortened_responses, "r") as read_file:
        shortened_responses = json.load(read_file)
    responses = shortened_responses[topic]["neighbor"]

    # If path exists 
    try:
        with open(path_to_doc_ranking, "r") as r:
            ranking_score = json.load(r)
    except FileNotFoundError:
        ranking_score = {}

    # Loop through the non-independent symbols
    for cnt, non_independent_symbol in tqdm(enumerate(non_independent_symbols_topic), desc="Looping through non-independent symbols"):
        print(f"Non-independent symbol: {non_independent_symbol}")
        #logging.info(f"Non-independent symbol: {non_independent_symbol}")

        # If the non-independent symbol is already processed
        if non_independent_symbol in ranking_score:
            if len(ranking_score[non_independent_symbol]) == len(countries_nationalities_list):
                print(f"Non-independent symbol {non_independent_symbol} already processed")
                #logging.info(f"Non-independent symbol {non_independent_symbol} already processed")
                continue
            else:
                print(f"Non-independent symbol {non_independent_symbol} partially processed")
                #logging.info(f"Non-independent symbol {non_independent_symbol} partially processed")
        else:
            ranking_score[non_independent_symbol] = {}

        # Get the list of nationalities for the non-independent symbol
        lst_of_countries = find_countries_with_symbol(responses, non_independent_symbol)
        print(f"List of countries: {lst_of_countries}")
        logging.info(f"List of countries: {lst_of_countries}")

        # Loop through the nationalities
        for cnt_, tuple in tqdm(enumerate(lst_of_countries), desc="Looping through nationalities"):
            nationality = tuple.lower()
            print(f"Nationality: {nationality}")
            #logging.info(f"Nationality: {nationality}")

            if nationality in ranking_score[non_independent_symbol]:
                print(f"Nationality {nationality} already processed")
                continue

            # Query terms
            query_terms = [nationality, non_independent_symbol]
            query_culture = nationality
            print(f"Query terms: {query_terms}")
            logging.info(f"Query terms: {query_terms}")

            # lst_docs = get_documents_containing_phrases(index=index, es=es, phrases=["kimono", "albanian", "indian"], all_phrases=True, return_all_hits=True, sort_field="created")
            """
            GET ALL THE DOCUMENT IDs
            """
            count_docs = count_documents_containing_phrases(index, query_terms, es=es, all_phrases=True)
            logging.info(f"Count of documents: {count_docs}")
            print(f"Count of documents: {count_docs}")
            docs = get_documents_containing_phrases(index=index, es=es, phrases=query_terms, all_phrases=True, return_all_hits=True, sort_field="created")

            # Usage of the multiproc function
            if args.multiproc:
                doc_ids, sentence_distances, ranking_scores, token_distances = process_documents(docs, query_terms, query_culture, countries_nationalities_list)
            else:
                # Sequential processing of the documents
                sentence_distances = []
                token_distances = []
                ranking_scores = []
                doc_ids = []

                i = 0
                for doc in docs:
                    i += 1
                    print(f"{i}/{count_docs}")
                    #logging.info(i)
                    result = process_document(doc, query_terms, query_culture, countries_nationalities_list)
                    # Process results
                    if (result['ranking_score'] > 0 and result['token_distance'] < MAX_SEQ_LEN) or (result['sentence_distance'] < 2 and result['ranking_score'] > -1):  # KEPT IT 30 AS PREVIOUS BOUND WAS A SENTENCE DISTANCE OF 2
                        #print(f"Document ID: {result['doc_id']}, Sentence distance: {result['sentence_distance']}, Ranking Score: {result['ranking_score']}, Token distance: {result['token_distance']}")
                        #logging.info(f"Document ID: {result['doc_id']}, Sentence distance: {result['sentence_distance']}, Ranking Score: {result['ranking_score']}, Token distance: {result['token_distance']}")
                        doc_ids.append(result['doc_id'])
                        sentence_distances.append(result['sentence_distance'])
                        ranking_scores.append(result['ranking_score'])
                        token_distances.append(result['token_distance'])

            # Check if there are no co-occuring docs
            if len(doc_ids) != 0:
                print(f"Processed {count_docs} documents")    
                print(f"Number of relevant documents: {len(doc_ids)}, Ratio of relevant documents: {len(doc_ids) / count_docs}")

                # Calculate average, max and min of the sentence distances 
                sentence_distances = np.array(sentence_distances)
                ranking_scores = np.array(ranking_scores)

                if len(sentence_distances) != 0 and len(ranking_scores != 0) and len(token_distances) != 0:
                    avg_sentence_distance = np.mean(sentence_distances)
                    max_sentence_distance = np.max(sentence_distances)
                    max_sentence_distance_index = np.argmax(sentence_distances)
                    min_sentence_distance = np.min(sentence_distances)
                    min_sentence_distance_index = np.argmin(sentence_distances)

                    # Calculate average, max and min of the ranking scores without infinities
                    ranking_scores_2 = ranking_scores[~np.isinf(ranking_scores)]
                    avg_ranking_score = np.mean(ranking_scores_2)
                    max_ranking_score = np.max(ranking_scores_2)
                    max_ranking_score_index = np.argmax(ranking_scores_2)
                    min_ranking_score = np.min(ranking_scores_2)
                    min_ranking_score_index = np.argmin(ranking_scores_2)

                    # Calculate average, max and min of the token distances
                    token_distances = np.array(token_distances)
                    avg_token_distance = np.mean(token_distances)
                    max_token_distance = np.max(token_distances)
                    max_token_distance_index = np.argmax(token_distances)
                    min_token_distance = np.min(token_distances)
                    min_token_distance_index = np.argmin(token_distances)

                # Write all the stats to a txt file
                # Check if directory exists else create it
                if not os.path.exists(f"symbol_log_txts"):
                    os.makedirs(f"symbol_log_txts")
                with open(f"symbol_log_txts/{args.model_name}_{query_terms[0]}_{topic}_statistics.txt", "a") as f:
                    f.write(f"Symbol: {query_terms[1]}\n")
                    f.write(f"Processed {count_docs} documents\n")
                    f.write(f"Number of relevant documents: {len(doc_ids)}, Ratio of relevant documents: {len(doc_ids) / count_docs}\n")
                    
                    if len(sentence_distances) != 0 and len(ranking_scores) != 0 and len(token_distances) != 0:
                        f.write(f"Average sentence distance: {avg_sentence_distance}\n")
                        f.write(f"Max sentence distance: {max_sentence_distance}, Corresponding Rank: {ranking_scores[max_sentence_distance_index]}\n")
                        f.write(f"Min sentence distance: {min_sentence_distance}, Corresponding Rank: {ranking_scores[min_sentence_distance_index]}\n")
                        f.write(f"Average ranking score: {avg_ranking_score}\n")    
                        f.write(f"Max ranking score: {max_ranking_score}, Corresponding Sentence Distance: {sentence_distances[max_ranking_score_index]}\n")
                        f.write(f"Min ranking score: {min_ranking_score}, Corresponding Sentence Distance: {sentence_distances[min_ranking_score_index]}\n")
                        f.write(f"Average token distance: {avg_token_distance}\n")
                        f.write(f"Max token distance: {max_token_distance}, Corresponding Rank: {ranking_scores[max_token_distance_index]}\n")
                        f.write(f"Min token distance: {min_token_distance}, Corresponding Rank: {ranking_scores[min_token_distance_index]}\n")
                        f.write("\n---------------*********************************************---------------\n")
                    else:
                        f.write("No documents found\n")
                        f.write("\n---------------*********************************************---------------\n")

                # sort by ranking score and store in a json file
                doc_id_score = list(zip(doc_ids, ranking_scores, sentence_distances, token_distances))
                doc_id_score.sort(key=lambda x: x[1], reverse=True)

            else:
                print(f"No documents found")
                logging.info(f"No documents found")
                doc_id_score = []

            # Storing the relevant documents
            ranking_score[query_terms[1]][query_terms[0]] = doc_id_score

        with open(path_to_doc_ranking, "w") as w:
            json.dump(ranking_score, w, default=default_converter, indent=4)