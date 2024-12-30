import logging
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import math
import time
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
from matplotlib.colors import LogNorm
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# For Infigram queries
import requests
INFIGRAM_URL = 'https://api.infini-gram.io/'
INDEX = 'v4_dolma-v1_7_llama'
QUERY_TYPE = 'count'

# # Setup logging
logging.basicConfig(filename=f"output_memo_assignment.log", 
                    filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# memorization THRESHOLD
memorization_RATIO = 1/110
memorization_Z_SCORE = 2.6

# Elasticsearch connection setup
es = Elasticsearch(
    cloud_id="<INSERT CLOUD ID>",
    api_key="<INSERT API KEY>",
    retry_on_timeout=True,
    http_compress=True,
    request_timeout=180,
    max_retries=10
)
index = "docs_v1.5_2023-11-02"

# Function definitions for normalizing and computing statistics
def normalise(x):
    total = sum(x)
    if total == 0:
        # If sum is zero, return a list of zeros to prevent division by zero
        return [0 for _ in x]
    return [float(i)/sum(x) for i in x]

# Function to calculate the entropy of a distribution
def entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities+1e-10))

# Function to calculate the Gini coefficient
def gini_coefficient(values):
    sorted_values = np.sort(values)
    cumulative_sum = np.cumsum(sorted_values)
    cumulative_sum_total = cumulative_sum[-1]
    n = len(values)
    return (n + 1 - 2 * np.sum(cumulative_sum) / cumulative_sum_total) / n

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

# Plot world map
def plot_world_map(home_dir, path_to_memorization_stats, topic):
    # Open the memorization stats JSON
    with open(path_to_memorization_stats, "r") as f:
        memorization_stats = json.load(f)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_country_names = world['name'].values

    name_to_gpd_mapping ={
        "Bosnia and Herzegovina":"Bosnia and Herz.",
        "Dominican Republic":"Dominican Rep.",
        "United States":"United States of America",
    }

    world['memorization_counts'] = None

    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    for country, nationality in countries_nationalities_list:
        if country not in world_country_names:
            if country in name_to_gpd_mapping:
                new_country = name_to_gpd_mapping[country]
            else:
                continue
        else:
            new_country = country
        # lowercase the nationality
        nationality_lower = nationality.lower()
        world.loc[world['name'] == new_country, 'memorization_counts'] = len(memorization_stats[nationality_lower]["symbols"])

    world['memorization_counts'] = world['memorization_counts'].fillna(0)
    world['memorization_counts'] = world['memorization_counts'].astype(int)

    # Plot the world map with heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot the countries
    world.boundary.plot(ax=ax, linewidth=1, color='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Plot the heatmap
    world.plot(column='memorization_counts', ax=ax, cmap='OrRd', legend=True, missing_kwds={'color': 'gray'}, cax=cax)

    plt.savefig(f"memorization_world_map_{topic}_Z=[{memorization_Z_SCORE}].png")
    # for country, nationality in countries_nationalities_list:
    #     mapped_country = name_to_gpd_mapping.get(country, country)
    #     if mapped_country in world_country_names:
    #         nationality_lower = nationality.lower()
    #         world.loc[world['name'] == mapped_country, 'memorization_counts'] = len(memorization_stats.get(nationality_lower, {}).get("symbols", []))

    # world['memorization_counts'] = world['memorization_counts'].fillna(0)
    # world['memorization_counts'] = world['memorization_counts'].astype(int)

    # fig, ax = plt.subplots(figsize=(12, 8))
    # world.boundary.plot(ax=ax, linewidth=1, color='black')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)

    # # Color map and normalization adjustments
    # cmap = plt.cm.Blues
    # norm = plt.Normalize(vmin=0, vmax=world['memorization_counts'].max())

    # # Set color for zero counts
    # cmap.set_under('gray')  # Any color that stands out, such as 'gray'

    # world.plot(column='memorization_counts', ax=ax, cmap=cmap, legend=True, norm=norm, missing_kwds={'color': 'lightgray'}, cax=cax)
    # plt.savefig(f"memorization_world_map_{topic}_Z=[{memorization_Z_SCORE}].png")

# Calculate occurence of nationalities
def calculate_occurence_nationalities(nationalities_):
    # Get a dict for the number of occurrences of the country
    occurrences_dict = {country: 0 for country in nationalities_}

    for nation in tqdm(nationalities_):
        """
        If using Infinigram
        """
        # payload1 = {
        #     'index': INDEX,
        #     'query_type': QUERY_TYPE,
        #     'query': f"{nation}",
        #     'max_diff_tokens': 1000,
        #     'max_clause_freq': 500000
        # }
        # while True:
        #     try:
        #         response1 = requests.post(INFIGRAM_URL, json=payload1)
        #         response1.raise_for_status()  # Raises an HTTPError for bad responses
                
        #         count1 = response1.json()["count"]
        #         occurrences_dict[nation] = count1
        #         break
        #     except requests.exceptions.RequestException as e:
        #         print(f"An error occurred: {e}. Retrying...")
        #         time.sleep(2)  # Wait for 2 seconds before retrying
        """
        If using Elasticsearch
        """
        cnt = count_documents_containing_phrases(index, [nation], es=es)
        occurrences_dict[nation] = cnt

    return occurrences_dict

"""
Function to assess cultural overmemorization
"""
def cultural_overmemorization(memo_data, responses, topic, model_name):
    cultural_overmemo_dict = {}
    for culture in memo_data.keys():
        memo_symbols = memo_data[culture]["symbols"]
        for memo_symbol in memo_symbols:
            if memo_symbol in cultural_overmemo_dict:
                continue
            cultural_overmemo_dict[memo_symbol] = 0
            countries = find_countries_with_symbol(responses[topic]["neighbor"], memo_symbol)
            countries = [country.lower() for country in countries]
            for country in countries:
                if country != culture.lower() and memo_symbol not in memo_data[country]["symbols"]:
                    cultural_overmemo_dict[memo_symbol] += 1
    cultural_overmemo_dict = {k: v for k, v in sorted(cultural_overmemo_dict.items(), key=lambda item: item[1], reverse=True)}

    # For dictionary get the average, max, min
    avg = sum(cultural_overmemo_dict.values())/len(cultural_overmemo_dict)
    max_val = max(cultural_overmemo_dict.values())
    min_val = min(cultural_overmemo_dict.values())

    # Get the symbols at the max and min
    max_symbol = [k for k, v in cultural_overmemo_dict.items() if v == max_val]
    min_symbol = [k for k, v in cultural_overmemo_dict.items() if v == min_val]

    # Print the results
    print(f"Topic: {topic} - Cultural Over-memorization Stats")
    print(f"Average no. of countries for which a memorized symbol is generated: {avg}")
    print(f"Max no. of countries for which a memorized symbol is generated: {max_val}")
    print(f"Min no. of countries for which a memorized symbol is generated: {min_val}")
    print(f"Symbols with max no. of countries: {max_symbol}")
    print(f"Symbols with min no. of countries: {min_symbol}")

    # Save the dictionary to a JSON file
    with open(f"cultural_overmemo_{model_name}_{topic}.json", "w") as f:
        json.dump(cultural_overmemo_dict, f, indent=4)

    non_zero = len([k for k, v in cultural_overmemo_dict.items() if v > 0])
    total = len(cultural_overmemo_dict)
    print("Symbols which are generated for atleast for one other culture:")
    print(f"{topic.capitalize()}: Non-zero: {non_zero}, Total: {total}, Percentage: {non_zero/total if total else 0}")

    # Save unique memorized symbols
    unique_memo_symbols = list(cultural_overmemo_dict.keys())
    with open(f"memorized_symbols_{model_name}_{topic}.json", "w") as f:
        json.dump(unique_memo_symbols, f, indent=4)
        
# Main function call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name to use for classification")
    parser.add_argument("--topic", type=str, default="clothing", help="Topic of interest")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol of interest")
    parser.add_argument("--calc_memorization", action="store_true", help="Calculate memorization statistics")
    parser.add_argument("--plot_world_map", action="store_true", help="Plot the world map with memorization counts")
    parser.add_argument("--calc_corr", action="store_true", help="Calculate correlation between number of memorized symbols and the number of occurrences of the country")
    parser.add_argument("--assess_cultural_overmemo", action="store_true", help="Assess cultural over-memorization")

    args = parser.parse_args()

    # Load JSON containing relevant documents for each culture-symbol pairing
    path_to_json = f"{args.home_dir}/memoed_scripts/training_document_classification_{args.model_name}_{args.topic}.json"
    with open(path_to_json, "r") as f:
        doc_ranking = json.load(f)

    # Load shortened responses JSON
    path_to_shortened_responses = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_new_shortened.json"
    with open(path_to_shortened_responses, "r") as f:
        shortened_responses = json.load(f)

    # Path for storing memorization statistics
    path_to_memorization_stats = f"{args.home_dir}/memoed_scripts/memorization_stats_{args.model_name}_{args.topic}_Z=[{memorization_Z_SCORE}].json"

    # Path to store generalised symbols
    path_to_generalised_symbols = f"{args.home_dir}/memoed_scripts/generalised_symbols_{args.model_name}_{args.topic}_Z=[{memorization_Z_SCORE}].json"

    # Load the list of all nationalities from CSV
    with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [row[1] for row in reader]

    # Get list of nationalities where the symbol appears
    # Define the symbol for processing
    symbol = args.symbol

    # If symbol is not None, calculate memorization statistics for the symbol
    if args.calc_memorization:
        print("Calculating memorization statistics...")
        if symbol is not None:
            print(f"Symbol: {symbol}")
            responses = shortened_responses[args.topic]["neighbor"]
            countries_with_symbol = find_countries_with_symbol(responses, symbol)
            countries_with_symbol = [country.lower() for country in countries_with_symbol]
            print(f"Countries with symbol: {countries_with_symbol}")

            # Calculate document counts and ratios
            total_documents = count_documents_containing_phrases(index, [symbol], es=es)
            relevant_documents = {country: len(doc_ranking[symbol][country]) for country in countries_with_symbol}
            ratios = {country: relevant_documents[country]/total_documents for country in countries_with_symbol}
            normalized_ratios_lst = normalise(list(ratios.values()))
            normalized_ratios = {country: normalized_ratios_lst[i] for i, country in enumerate(countries_with_symbol)}

            # Sort the countries by the ratio of relevant documents to get two sorted separate lists
            sorted_normalized_ratios = {k: v for k, v in sorted(normalized_ratios.items(), key=lambda item: item[1], reverse=True)}
            sorted_countries = list(sorted_normalized_ratios.keys())
            # Capitalize the countries
            sorted_countries = [country.capitalize() for country in sorted_countries]
            sorted_normalized_ratios = list(sorted_normalized_ratios.values())

            # Calculate Mean, Standard Deviation, and Z-scores
            mean = np.mean(normalized_ratios_lst)
            std_dev = np.std(normalized_ratios_lst)
            z_scores = [(x - mean) / std_dev for x in normalized_ratios_lst]

            # Calculate statistics
            gini_index = gini_coefficient(normalized_ratios_lst)
            entropy_value = entropy(np.array(normalized_ratios_lst))
            print(f"Gini coefficient: {gini_index}")
            print(f"Entropy of the distribution: {entropy_value}")
            print(f"Mean: {mean}")
            print(f"Standard Deviation: {std_dev}")
            print(f"Z-scores: {z_scores}")

            # Plotting the probability distribution
            plt.figure(figsize=(10, 10))
            plt.barh(sorted_countries, sorted_normalized_ratios, color='skyblue')
            plt.xlabel('Probability')
            plt.title(f'Probability Distribution over Cs for Symbol: {symbol}')
            plt.tight_layout()
            plt.savefig(f"prob_dist_{symbol}.png")
            plt.show()
        else:
            # Path to non_independent_symbols 
            path_to_non_independent_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_non_independent_symbols.json"
            with open(path_to_non_independent_symbols, "r") as f:
                non_independent_symbols = json.load(f)

            # Lowercase the countries
            countries_nationalities_list = [country.lower() for country in countries_nationalities_list]

            # Set up a dictionary for storing nationalities and their memorized symbols
            memorized_symbols = {country: {"count" : 0, "symbols" : []} for country in countries_nationalities_list}

            # generalised symbols list
            generalised_symbols = []

            # Get all symbols
            symbols = non_independent_symbols[args.topic]
            symbol_stats_dictionary = {}
            for symbol in tqdm(symbols):
                if symbol not in list(doc_ranking.keys()):
                    print(f"Skipping symbol due to missing symbol: {symbol}")
                    logging.info(f"Skipping symbol due to missing symbol: {symbol}")
                    continue 
                print(f"Symbol: {symbol}")
                symbol_stats_dictionary[symbol] = {}
                responses = shortened_responses[args.topic]["neighbor"]
                countries_with_symbol = find_countries_with_symbol(responses, symbol)

                # Lowercase the countries
                countries_with_symbol = [country.lower() for country in countries_with_symbol]

                print(f"Countries with symbol: {countries_with_symbol}")

                # Check if the countries are in doc_ranking, else skip the symbol
                if not all([country in doc_ranking[symbol].keys() for country in countries_with_symbol]):
                    # Get list of countries which are not in doc_ranking
                    countries_not_in_doc_ranking = [country for country in countries_with_symbol if country not in doc_ranking[symbol].keys()]
                    print(symbol)
                    print(countries_not_in_doc_ranking)
                    logging.info(f"symbol: {symbol}")
                    logging.info(f"countries_not_in_doc_ranking: {countries_not_in_doc_ranking}")
                    # Set them with empty lists
                    for country in countries_not_in_doc_ranking:
                        doc_ranking[symbol][country] = []

                # Calculate document counts and ratios
                total_documents = count_documents_containing_phrases(index, [symbol], es=es)
                relevant_documents = {country: len(doc_ranking[symbol][country]) for country in countries_with_symbol}

                # If there are no relevant documents or no countries with symbol, skip the symbol
                if sum(relevant_documents.values()) == 0 or len(countries_with_symbol) == 0:
                    print(f"Skipping symbol due to no relevant docs or no countries: {symbol}")
                    symbol_stats_dictionary[symbol]["mean"] = 0
                    symbol_stats_dictionary[symbol]["median"] = 0
                    symbol_stats_dictionary[symbol]["std_dev"] = 0
                    symbol_stats_dictionary[symbol]["iqr"] = 0
                    symbol_stats_dictionary[symbol]["median_plus_iqr"] = 0
                    symbol_stats_dictionary[symbol]["gini_index"] = 0
                    symbol_stats_dictionary[symbol]["entropy_value"] = 0
                    symbol_stats_dictionary[symbol]["z_scores"] = 0

                    # Add to generalised symbols
                    generalised_symbols.append(symbol)
                    continue
                
                # Calculate the ratio for each country
                ratios = {country: relevant_documents[country]/total_documents for country in countries_with_symbol}
                ratios_lst = list(ratios.values())
                normalized_ratios_lst = normalise(list(ratios.values()))
                normalized_ratios = {country: normalized_ratios_lst[i] for i, country in enumerate(countries_with_symbol)}

                # Get countries which have a memorization ratio greater than the threshold
                countries_passing_memorization_ratio_threshold = [country for country in countries_with_symbol if ratios[country] >= memorization_RATIO]

                # Sort the countries by the ratio of relevant documents to get two sorted separate lists
                sorted_normalized_ratios = {k: v for k, v in sorted(normalized_ratios.items(), key=lambda item: item[1], reverse=True)}
                sorted_countries = list(sorted_normalized_ratios.keys())
                sorted_countries = [country.capitalize() for country in sorted_countries]
                sorted_normalized_ratios = list(sorted_normalized_ratios.values())

                # Calculate Mean, Standard Deviation, and Z-scores
                mean = np.mean(ratios_lst)
                std_dev = np.std(ratios_lst)
                if std_dev > 1e-20:  # small threshold to avoid division by very small numbers
                    z_scores = [(x - mean) / std_dev for x in ratios_lst]
                else:
                    z_scores = [0 for _ in ratios] 

                # Calculate IQR
                q1 = np.percentile(ratios_lst, 25)
                q3 = np.percentile(ratios_lst, 75)
                iqr = q3 - q1

                # Get median
                median = np.median(ratios_lst)

                # Store z-scores in a dictionary
                z_scores_dict = {country: z_scores[i] for i, country in enumerate(countries_with_symbol)}

                # If z-scores are nan, convert them to 0
                z_scores_dict = {country: 0 if math.isnan(z_scores_dict[country]) else z_scores_dict[country] for country in z_scores_dict}

                # Store countries which passed the first threshold and then the z-score threshold
                countries_passing_z_score = [country for country in countries_passing_memorization_ratio_threshold if z_scores_dict[country] >= memorization_Z_SCORE]

                # Add to the dictionary depending on number of countries symbol is in
                flag = 0
                if len(countries_with_symbol) < 6:
                    for country in countries_passing_memorization_ratio_threshold:
                        flag = 1
                        memorized_symbols[country]["count"] += 1
                        memorized_symbols[country]["symbols"].append(symbol)
                else:
                    for country in countries_passing_z_score:
                        flag = 1
                        memorized_symbols[country]["count"] += 1
                        memorized_symbols[country]["symbols"].append(symbol)

                # If flag is 0, the symbol is a generalised symbol
                if flag == 0:
                    generalised_symbols.append(symbol)

                # Calculate statistics
                gini_index = gini_coefficient(normalized_ratios_lst)
                entropy_value = entropy(np.array(normalized_ratios_lst))

                # Add statistics to dictionary
                symbol_stats_dictionary[symbol]["mean"] = mean
                symbol_stats_dictionary[symbol]["median"] = median
                symbol_stats_dictionary[symbol]["std_dev"] = std_dev
                symbol_stats_dictionary[symbol]["iqr"] = iqr
                symbol_stats_dictionary[symbol]["median_plus_iqr"] = median + iqr
                symbol_stats_dictionary[symbol]["gini_index"] = gini_index
                symbol_stats_dictionary[symbol]["entropy_value"] = entropy_value
                symbol_stats_dictionary[symbol]["z_scores"] = z_scores_dict

                # Dump dictionary to JSON
                with open(f"{args.home_dir}/memoed_scripts/prob_dist_plots_{args.topic}_{args.model_name}/dict_prob_dist_{args.topic}.json", "w") as f:
                    json.dump(symbol_stats_dictionary, f, indent=4)

                # Plotting the probability distribution
                plt.figure(figsize=(10, 10))
                plt.barh(sorted_countries, sorted_normalized_ratios, color='skyblue')
                plt.xlabel('Probability')
                plt.title(f'Probability Distribution over Cs for Symbol: {symbol}')
                plt.tight_layout()
                plt.savefig(f"{args.home_dir}/memoed_scripts/prob_dist_plots_{args.topic}/prob_dist_{symbol}.png")
                plt.show()

            # Sort the memorized symbols dictionary by count
            memorized_symbols = {k: v for k, v in sorted(memorized_symbols.items(), key=lambda item: item[1]["count"], reverse=True)}

            # Dump memorized symbols to JSON
            with open(path_to_memorization_stats, "w") as f:
                json.dump(memorized_symbols, f, indent=4)

            # Dump generalised symbols to JSON
            generalised_symbols_dict = {"generalised_symbols": generalised_symbols}
            with open(path_to_generalised_symbols, "w") as f:
                json.dump(generalised_symbols_dict, f, indent=4)

    # Plotting the memorized symbols
    if args.plot_world_map:
        print(f"Topic: {args.topic}")
        print("Calculating memorization statistics...")
        # Load the memorization stats JSON
        with open(path_to_memorization_stats, "r") as f:
            memorization_stats = json.load(f)
        # Print how many countries have atleast one memorized symbol
        cnt = 0
        cnts = []
        for key in memorization_stats.keys():
            if memorization_stats[key]["count"] > 0:
                cnt += 1
            cnts.append(memorization_stats[key]["count"])
        print(f"Number of countries with atleast one memorized symbol: {cnt}")
        print(f"Average number of memorized symbols: {sum(cnts)/len(cnts)}")
        print("\n\n")
        print("Plotting world map...")
        # Plot world map
        plot_world_map(args.home_dir, path_to_memorization_stats, args.topic)

    # Calculate correlation between number of memorized symbols and the number of occurrences of the country
    if args.calc_corr:
        print("Calculating correlation between number of memorized symbols and the number of occurrences of the country...")
        # Load the memorization stats JSON
        with open(path_to_memorization_stats, "r") as f:
            memorization_stats = json.load(f)

        # Make a dict with the nationality and the number of memorized symbols
        memorized_symbols_count_dict = {country: memorization_stats[country.lower()]["count"] for country in list(memorization_stats.keys())}

        # Get the list of nationalities
        nationalities_correlation = list(memorized_symbols_count_dict.keys())

        # Calculate the occurrences of the nationalities
        occurrences_dict = calculate_occurence_nationalities(nationalities_correlation)

        # Get the two values in two separate lists
        memorized_symbols_count = []
        occurrences = []
        for nationality in nationalities_correlation:
            memorized_symbols_count.append(memorized_symbols_count_dict[nationality])
            occurrences.append(occurrences_dict[nationality])

        # Calculate the spearman and kendall correlation
        spearman_corr, _ = spearmanr(memorized_symbols_count, occurrences)
        kendall_corr, _ = kendalltau(memorized_symbols_count, occurrences)

        print(f"Spearman correlation: {spearman_corr}")
        print(f"Kendall correlation: {kendall_corr}")

    # Assess cultural over-memorization
    if args.assess_cultural_overmemo:
        print("Assessing cultural over-memorization...")
        # Load the memorization stats JSON
        with open(path_to_memorization_stats, "r") as f:
            memorization_stats = json.load(f)