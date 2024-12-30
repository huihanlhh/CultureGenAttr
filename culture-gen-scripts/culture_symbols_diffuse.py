import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    elif args.model_name == "olmo7b-instruct":    #### ADDED PATH FOR ALIGNED OLMO MODEL
        model_path = "ssec-uw/OLMo-7B-Instruct-hf"

    if args.topic_list is None:
        args.topic_list = [
                        "food",
                        "clothing",
                    ]

    # Path with the value-mapping
    path_to_value_mapping = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_value_to_culture_mapping.json"

    # Path to store the symbols
    path_to_store_symbols = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_non_diffuse_symbols.json"

    # Path to diffuse symbols
    path_to_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_diffuse_symbols.json"

    # Check if it exists, load the dict else create a new one
    if os.path.exists(path_to_store_symbols):
        with open(path_to_store_symbols, "r") as f:
            topic_symbol_dict = json.load(f)
    else:
        topic_symbol_dict = {}

    # Load the dictionalry of diffuse symbols
    with open(path_to_diffuse_symbols, "r") as f:
        diffuse_symbols = json.load(f)

    # Iterate over the topics
    for topic in args.topic_list:
        logger.info(f"Topic: {topic}")
        path_to_value_mapping_ = path_to_value_mapping.replace(".json", f"_{topic}.json")

        # Get diffuse symbols of the topic
        if topic in diffuse_symbols:
            diffuse_symbols_topic = diffuse_symbols[topic]
        else:
            raise ValueError(f"General symbols for {topic} not found")    

        # Open the value mapping
        with open(path_to_value_mapping_, "r") as f:
            value_culture_mapping = json.load(f)

        # Check if the topic is already processed
        if topic in topic_symbol_dict and not args.overwrite:
            logger.info(f"Topic {topic} already processed")
            continue

        # Get the keys as a list
        symbols = list(value_culture_mapping.keys())

        # Remove the diffuse symbols
        symbols = [symbol for symbol in symbols if symbol not in diffuse_symbols_topic]

        # Add to topic_symbol_dict
        topic_symbol_dict[topic] = symbols

        print(f"Number of symbols for {topic}: {len(symbols)}")

    # Save the topic_symbol_dict
    with open(path_to_store_symbols, "w") as f:
        json.dump(topic_symbol_dict, f, indent=4)