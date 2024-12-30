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

# Setup logging
logging.basicConfig(filename=f"output_overshadow.log", filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Elasticsearch connection setup
es = Elasticsearch(
    cloud_id="<insert cloud ID>",
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

"""
Functions to calculate the ranking score for a document given query terms used to retrieve the document
"""
# Computing term frequency of cultures in the given pre-training document
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

# Function to compute the d-SNR for the given query
def score_query(document, query_terms, nationalities_list):
    """
    Scores a document based on a list of query terms using normalized TF for specific terms.
    
    :param document: A list of words representing the document.
    :param query_terms: A list of query terms.
    :param nationalities_list: A list of tuples (nation, nationality).
    :return: Total score for the document based on the query terms.
    """
    query_culture_one = query_terms[0].lower()
    query_culture_two = query_terms[1].lower()
    query_nation_one = [x[0] for x in nationalities_list if x[1].lower() == query_culture_one][0].lower()
    query_nation_two = [x[0] for x in nationalities_list if x[1].lower() == query_culture_two][0].lower()

    # Add all nationalities to the set of relevant terms
    relevant_terms = []
    for (nation, nationality) in nationalities_list:
        lower_nation = nation.lower()
        lower_nationality = nationality.lower()
        relevant_terms.append(lower_nation)
        relevant_terms.append(lower_nationality)
    
    term_count = compute_relevant_tf(document, relevant_terms)

    frequency_denominator = 0
    for nation, nationality in nationalities_list:
        lower_nation = nation.lower()
        lower_nationality = nationality.lower()

        if lower_nationality != query_culture_one and lower_nationality != query_culture_two:
            frequency_denominator += term_count.get(lower_nationality, 0)
            frequency_denominator += term_count.get(lower_nation, 0)

    frequency_num = term_count.get(query_culture_one, 0) + term_count.get(query_culture_two, 0) + term_count.get(query_nation_one, 0) + term_count.get(query_nation_two, 0)

    # Avoid division by zero and negative infinity scenario
    if frequency_num > 0:
        score = math.log2(frequency_num / (1 + frequency_denominator))
    else:
        score = -np.inf

    return score
    
"""
Function to calculate the minimum sentence distance between two words in a document
"""
def calc_min_sentence_distance(text, word1, word2):
    text = text.lower()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentences = [re.sub(r'(?<!\s)-', ' ', s).translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    
    word1_indices = [i for i, s in enumerate(sentences) if word1 in s]
    word2_indices = [i for i, s in enumerate(sentences) if word2 in s]
    
    min_distance = min((abs(i - j) for i in word1_indices for j in word2_indices), default=1e+10)
    return int(min_distance)

"""
Function to calculate the minimum word distance between two words in a document (2048 is the required upper bound)
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


# Define the function to be executed in parallel
def process_document(doc, query_terms, countries_nationalities_list):
    doc_reqd_1, doc_id = doc['_source']['text'], doc['_id']
    sential_dist = calc_min_sentence_distance(doc_reqd_1, query_terms[0], query_terms[1])

    # Replace "-" not followed by a space with a space in the document
    doc_reqd_1 = re.sub(r'(?<!\s)-', ' ', doc_reqd_1)
    doc_reqd_1 = doc_reqd_1.translate(str.maketrans('', '', string.punctuation))
    # Lowercase the document
    doc_reqd_1 = doc_reqd_1.lower()
    
    # Calculate the ranking score
    doc_1_lst = [word.lower() for word in doc_reqd_1.split()]
    ranking_score = score_query(doc_1_lst, query_terms, countries_nationalities_list)

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

"""
Multi-proc functions
"""
def process_single_document(doc, query_terms, countries_nationalities_list):
    result = process_document(doc, query_terms, countries_nationalities_list)
    if (result['ranking_score'] > 0 and result['token_distance'] < MAX_SEQ_LEN) or (result['sentence_distance'] < 2 and result['ranking_score'] > -1):
        return {
            'doc_id': result['doc_id'],
            'sentence_distance': result['sentence_distance'],
            'ranking_score': result['ranking_score'],
            'token_distance': result['token_distance']
        }
    return None

def process_documents(docs, query_terms, countries_nationalities_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_single_document, [(doc, query_terms, countries_nationalities_list) for doc in docs])
    
    # Filter out None results and separate data
    filtered_results = [result for result in results if result is not None]
    doc_ids = [res['doc_id'] for res in filtered_results]
    sentence_distances = [res['sentence_distance'] for res in filtered_results]
    ranking_scores = [res['ranking_score'] for res in filtered_results]
    token_distances = [res['token_distance'] for res in filtered_results]

    return doc_ids, sentence_distances, ranking_scores, token_distances

# Function to find countries with a specific symbol
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


# Main function call
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name to use for classification")
    parser.add_argument("--topic", type=str, default=None, help="Topic of interest")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol of interest")
    parser.add_argument("--culture_one", type=str, default=None, help="Culture 1 of interest")
    parser.add_argument("--culture_two", type=str, default=None, help="Culture 2 of interest")
    parser.add_argument("--multiproc", action="store_true", help="Use multiprocessing for processing documents")

    args = parser.parse_args()
    # Loading nationalities from CSV
    with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    # Check for valid cultures
    if args.culture_one == None or args.culture_two == None:
        print("Please provide the cultures of interest")
        exit()

    # Check if cultures exist in the list of nationalities
    if args.culture_one.lower() not in [x[1].lower() for x in countries_nationalities_list] or args.culture_two.lower() not in [x[1].lower() for x in countries_nationalities_list]:
        print("Please provide valid cultures")
        exit()
    else:
        query_culture_one = args.culture_one.lower()
        query_culture_two = args.culture_two.lower()
        query_nation_one = [x[0] for x in countries_nationalities_list if x[1].lower() == query_culture_one][0]
        query_nation_two = [x[0] for x in countries_nationalities_list if x[1].lower() == query_culture_two][0]

    # JSON path for storing results
    path_to_doc_ranking = f"{args.home_dir}/memoed_scripts/overshadowing_generalisation_docs_{args.model_name}_{args.topic}.json"

    # If path exists 
    try:
        with open(path_to_doc_ranking, "r") as r:
            ranking_score = json.load(r)
    except FileNotFoundError:
        ranking_score = {}

    # Query terms
    if args.symbol == None:
        query_terms = [args.culture_one, args.culture_two]
        symbol = None
    else:
        query_terms = [args.culture_one, args.culture_two, args.symbol]

    print(f"Query terms: {query_terms}")
    logging.info(f"Query terms: {query_terms}")

    # Check if culture one is in the ranking score
    if query_terms[0] in ranking_score:
        if query_terms[1] in ranking_score[query_terms[0]]:
            if args.symbol == None:
                print(f"This combination has already been processed as {query_terms[0]} and {query_terms[1]}")
                exit()
            else:
                if query_terms[2] in ranking_score[query_terms[0]][query_terms[1]]:
                    print(f"This combination has already been processed as {query_terms[0]} and {query_terms[1]} with symbol {query_terms[2]}")
                    exit()
    
    # Check for reverse combination
    if query_terms[1] in ranking_score:
        if query_terms[0] in ranking_score[query_terms[1]]:
            if args.symbol == None:
                print(f"This combination has already been processed as {query_terms[1]} and {query_terms[0]}")
                exit()
            else:
                if query_terms[2] in ranking_score[query_terms[1]][query_terms[0]]:
                    print(f"This combination has already been processed as {query_terms[1]} and {query_terms[0]} with symbol {query_terms[2]}")
                    exit()
                  
    # lst_docs = get_documents_containing_phrases(index=index, es=es, phrases=["kimono", "albanian", "indian"], all_phrases=True, return_all_hits=True, sort_field="created")
    """
    GET ALL THE DOCUMENT IDs
    """
    count_docs = count_documents_containing_phrases(index, query_terms, es=es, all_phrases=True)
    logging.info(f"Count of documents: {count_docs}")
    print(f"Count of documents: {count_docs}")
    #docs = get_documents_containing_phrases(index=index, es=es, phrases=query_terms, all_phrases=True, num_documents=10)
    docs = get_documents_containing_phrases(index=index, es=es, phrases=query_terms, all_phrases=True, return_all_hits=True, sort_field="created")

    if args.multiproc:
        # Usage of the multiproc function
        doc_ids, sentence_distances, ranking_scores, token_distances = process_documents(docs, query_terms, countries_nationalities_list)
    else:
        doc_ids = []
        sentence_distances = []
        ranking_scores = []
        token_distances = []
        for doc in tqdm(docs):
            doc_id, sential_dist, ranking_score, token_distance = process_document(doc, query_terms, countries_nationalities_list)
            if (ranking_score > 0 and token_distance < MAX_SEQ_LEN) or (sential_dist < 2 and ranking_score > -1):
                doc_ids.append(doc_id)
                sentence_distances.append(sential_dist)
                ranking_scores.append(ranking_score)
                token_distances.append(token_distance)


    # Check if there are no co-occuring docs
    if len(doc_ids) != 0:
        print(f"Processed {count_docs} documents")    
        print(f"Number of relevant documents: {len(doc_ids)}, Ratio of relevant documents: {len(doc_ids) / count_docs}")
        #logging.info(f"Processed {i} documents")
        #logging.info(f"Number of relevant documents: {len(doc_ids)}, Ratio of relevant documents: {len(doc_ids) / i}")

        # sort by ranking score and store in a json file
        doc_id_score = list(zip(doc_ids, ranking_scores, sentence_distances, token_distances))
        doc_id_score.sort(key=lambda x: x[1], reverse=True)

    else:
        print(f"No documents found")
        logging.info(f"No documents found")
        doc_id_score = []


    # Storing the relevant documents
    if query_terms[0] not in ranking_score:
            ranking_score[query_terms[0]] = {}
    if query_terms[1] not in ranking_score[query_terms[0]]:
        ranking_score[query_terms[0]][query_terms[1]] = {}
    if args.symbol == None:
        ranking_score[query_terms[0]][query_terms[1]][""] = doc_id_score
    else:
        ranking_score[query_terms[0]][query_terms[1]][query_terms[2]] = doc_id_score

    with open(path_to_doc_ranking, "w") as w:
        json.dump(ranking_score, w, default=default_converter, indent=4)