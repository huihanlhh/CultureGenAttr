import json
import csv
import argparse
import logging
from tqdm import tqdm

def load_json(path):
    with open(path, "r") as read_file:
        return json.load(read_file)

def load_nationalities(path):
    with open(path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        return [row[1] for row in reader]

def count_symbols_in_culture(nationalities, symbols, responses):
    nationality_symbol_count = {nat: 0 for nat in nationalities}
    for symbol in tqdm(symbols, desc="Counting symbols in each culture"):
        responses_group = responses["neighbor"]
        for nationality, resp in responses_group.items():
            if nationality == '':  # Skip if nationality is missing
                continue
            combined_responses = resp["male"] + resp["female"] + resp[""]
            if any(symbol.lower() in response.lower() for response in combined_responses):
                nationality_symbol_count[nationality] += 1
    return {k: v for k, v in sorted(nationality_symbol_count.items(), key=lambda item: item[1], reverse=True)}

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

def save_json(data, path):
    with open(path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def main(home_dir, model_name, topic):
    logging.basicConfig(filename=f"{home_dir}/culture_gen_analysis.log", filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    json_path = f"{home_dir}/probable_data/categories_nationality_100_{model_name}_prob=True_non_independent_symbols.json"
    symbols = load_json(json_path)[topic]

    responses_path = f"{home_dir}/probable_data/categories_nationality_100_{model_name}_prob=True_new_shortened.json"
    shortened_responses = load_json(responses_path)

    nationalities_path = f"{home_dir}/data/nationalities.csv"
    nationalities = load_nationalities(nationalities_path)
    logger.info("Nationalities loaded successfully")

    # Analysis for the number of symbols in each culture
    nationality_symbol_count = count_symbols_in_culture(nationalities, symbols, shortened_responses)
    save_json(nationality_symbol_count, f"{home_dir}/probable_data/{topic}_symbols_in_culture.json")

    # Analysis for the number of cultures per symbol
    symbol_culture_count = count_cultures_per_symbol(symbols, shortened_responses)
    save_json(symbol_culture_count, f"{home_dir}/probable_data/{topic}_cultures_per_symbol.json")

    logger.info("Analysis completed and results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, required=True, help="Home directory for storing and reading results")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name")
    parser.add_argument("--topic", type=str, default="food", help="Topic to analyze")

    args = parser.parse_args()
    main(args.home_dir, args.model_name, args.topic)