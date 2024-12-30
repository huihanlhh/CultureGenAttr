import os
import json
import argparse
import csv
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, "r") as read_file:
        return json.load(read_file)

def load_nationalities(path):
    with open(path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        return [row[1] for row in reader]

def count_cultures_per_symbol(symbols, responses):
    symbol_culture_count = {sym: 0 for sym in symbols}
    responses_group = responses["neighbor"]
    for symbol in tqdm(symbols, desc="Counting cultures per symbol"):
        for nationality, resp in responses_group.items():
            if nationality == '':
                continue
            combined_responses = resp["male"] + resp["female"] + resp[""]
            if any(symbol.lower() in response.lower() for response in combined_responses):
                symbol_culture_count[symbol] += 1
    return {k: v for k, v in sorted(symbol_culture_count.items(), key=lambda item: item[1], reverse=True)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--topic_list", nargs="+", default=None)
    parser.add_argument("--probably",action="store_true", help="Whether 'probably' is added in the prompt")
    parser.add_argument("--overwrite",action="store_true", help="Whether you want to overwrite")

    args = parser.parse_args()
    logger.info(args)

    if args.model_name =="gpt-4":
        model_path = "gpt-4"
    elif args.model_name == "llama2-13b":
        model_path = "meta-llama/Llama-2-13b-hf"
    elif args.model_name == "mistral-7b":
        model_path = "mistralai/Mistral-7B-v0.1"
    elif args.model_name == "olmo-7b":
        model_path = "allenai/OLMo-7B-hf"
    elif args.model_name == "olmo7b-instruct":    
        model_path = "ssec-uw/OLMo-7B-Instruct-hf"

    if args.topic_list is None:
        args.topic_list = [
                        "food",
                        "clothing",
                    ]

    # Load shortened responses
    home_dir = args.home_dir
    responses_path = f"{home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_new_shortened.json"
    shortened_responses = load_json(responses_path)

    # Load nationalities
    nationalities_path = f"{home_dir}/data/nationalities.csv"
    nationalities = load_nationalities(nationalities_path)

    # Path with the value-mapping
    path_to_value_mapping = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_value_to_culture_mapping.json"

    # Path to store the symbols
    path_to_non_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_non_diffuse_symbols.json"

    # Path to diffuse symbols
    path_to_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_diffuse_symbols.json"

    # Check if it exists, load the dict else create a new one
    if os.path.exists(path_to_non_diffuse_symbols):
        with open(path_to_non_diffuse_symbols, "r") as f:
            non_diffuse_symbol_dict = json.load(f)
    else:
        non_diffuse_symbol_dict = {}

    # Load the dictionalry of diffuse symbols
    if os.path.exists(path_to_diffuse_symbols):
        with open(path_to_diffuse_symbols, "r") as f:
            diffuse_symbols_dict = json.load(f)
    else:
        diffuse_symbols_dict = {}

    # Iterate over the topics
    for topic in args.topic_list:
        logger.info(f"Topic: {topic}")
        path_to_value_mapping_ = path_to_value_mapping.replace(".json", f"_{topic}.json")
        # Open the value mapping
        with open(path_to_value_mapping_, "r") as f:
            value_culture_mapping = json.load(f)

        # Check if the topic is already processed
        if topic in non_diffuse_symbol_dict and not args.overwrite:
            logger.info(f"Topic {topic} already processed")
            continue

        # Get all unique symbols for the topic
        symbols = list(value_culture_mapping.keys())

        # Get diffuse symbols of the topic
        if topic in diffuse_symbols_dict:
            diffuse_symbols_topic = diffuse_symbols_dict[topic]
        else:
            # Count the number of cultures per symbol
            symbol_culture_count = count_cultures_per_symbol(symbols, shortened_responses)
            diffuse_symbols_topic = [symbol for symbol, count in symbol_culture_count.items() if count >= 55] 
            diffuse_symbols_dict[topic] = diffuse_symbols_topic
            
        # Remove the diffuse symbols
        non_diffuse_symbols = [symbol for symbol in symbols if symbol not in diffuse_symbols_topic]

        # Add to topic_symbol_dict
        non_diffuse_symbol_dict[topic] = non_diffuse_symbols

        print(f"Number of symbols for {topic}: {len(symbols)}")
        print(f"Number of diffuse symbols for {topic}: {len(diffuse_symbols_topic)}")
        print(f"Number of non-diffuse symbols for {topic}: {len(non_diffuse_symbols)}")

    # Save the non-diffuse symbols to the path
    with open(path_to_non_diffuse_symbols, "w") as f:
        json.dump(non_diffuse_symbol_dict, f, indent=4)

    # Save the diffuse symbols to the path
    with open(path_to_diffuse_symbols, "w") as f:
        json.dump(diffuse_symbols_dict, f, indent=4)