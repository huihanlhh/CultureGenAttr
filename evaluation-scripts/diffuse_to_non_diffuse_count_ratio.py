import pandas as pd
import csv
from tqdm import tqdm
import logging
import json
import argparse
import requests

# For Infigram queries
INFIGRAM_URL = 'https://api.infini-gram.io/'
INDEX = 'v4_dolma-v1_7_llama'
QUERY_TYPE = 'count'

# Main function call
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name")
    parser.add_argument("--topic_list", nargs="+", default=None, help="List of topics to prompt")

    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(filename=f"{args.home_dir}/memoed_scripts/output_ratio_diffuse_to_non-diffuse.log", filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Path to symbols
    path_non_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_non_diffuse_symbols.json"
    path_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_diffuse_symbols.json"

    # Load the data from these paths
    with open(path_non_diffuse_symbols, "r") as file:
        non_diffuse_symbols = json.load(file)

    with open(path_diffuse_symbols, "r") as file:
        diffuse_symbols = json.load(file)

    print("Data loaded successfully")

    # Check topic list
    if args.topic_list != None:
        list_of_topics = args.topic_list
    else:
        # List of topics
        list_of_topics = ["food", "clothing"]

    # Iterate through the topics
    for topic in list_of_topics:
        diffuse_symbols_topic = diffuse_symbols[topic]
        non_diffuse_symbols_topic = non_diffuse_symbols[topic]

        # Iterate through the non-diffuse symbols and get the count
        non_diffuse_symbols_dict = {}
        for symbol in tqdm(non_diffuse_symbols_topic, desc=f"Looping through non-diffuse symbols for {topic}"):
            payload = {
                'index': INDEX,
                'query_type': QUERY_TYPE,
                'query': symbol
            }
            result = requests.post(INFIGRAM_URL, json=payload).json()
            cnt = result["count"]
            logging.info(f"Symbol: {symbol}, Count: {cnt}")
            non_diffuse_symbols_dict[symbol] = cnt

        # Iterate through the diffuse symbols and get the count
        diffuse_symbols_dict = {}
        for symbol in tqdm(diffuse_symbols_topic, desc=f"Looping through diffuse symbols for {topic}"):
            payload = {
                'index': INDEX,
                'query_type': QUERY_TYPE,
                'query': symbol
            }
            result = requests.post(INFIGRAM_URL, json=payload).json()
            cnt = result["count"]
            logging.info(f"Symbol: {symbol}, Count: {cnt}")
            diffuse_symbols_dict[symbol] = cnt

        # Calculate the ratio of diffuse to non-diffuse symbols and store in a dictionary
        ratio_dict = {}
        for diffuse_symbol in diffuse_symbols_topic:
            cnt_diffuse = diffuse_symbols_dict[diffuse_symbol]
            ratio_dict[diffuse_symbol] = {}

            for non_diffuse_symbol in non_diffuse_symbols_topic:
                cnt_non_diffuse = non_diffuse_symbols_dict[non_diffuse_symbol]
                ratio = (cnt_diffuse + 1) / (cnt_non_diffuse + 1)
                ratio_dict[diffuse_symbol][non_diffuse_symbol] = ratio

            # Sort the dictionary by value
            ratio_dict[diffuse_symbol] = dict(sorted(ratio_dict[diffuse_symbol].items(), key=lambda item: item[1], reverse=True))

        # Save the ratio dictionary to a file
        path_to_save = f"{args.home_dir}/probable_data/ratio_diffuse_to_non_diffuse_{args.model_name}_{topic}.json"
        with open(path_to_save, "w") as file:
            json.dump(ratio_dict, file, indent=4)