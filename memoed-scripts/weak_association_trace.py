import os
import json
import argparse
import csv
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# Z-score for memorization (can be changed if measured memorization with a different z-score)
MEMORIZATION_Z_SCORE = 2.6

# Function to prompt olmo-7b-instruct for generating the definition of a memorised symbol and trace it to a generalization
def prompt_olmo7b_instruct(model, tokenizer, symbol, culture, topic):
    topic_wise_prompt = { 
        "clothing": "clothing item which looks like a type of",
        "food": "food item which is a type of"
    }

    # Capitalise the culture
    culture = culture.capitalize()

    # Choose the article according to the culture
    if culture.startswith("A") or culture.startswith("E") or culture.startswith("I") or culture.startswith("O") or culture.startswith("U"):
        article = "an"
    else:
        article = "a"

    """
    For Using OLMO-7b-instruct
    """
    chat = [
        {"role": "user", 
            "content": f"""
        <<SYS>>
            You are a helpful agent who is knewledgable about celtures from all places. You are talking to a persen who is curious about different cultures and their {topic}.            
        <</SYS>> 
        [INST]
        COMPLETE THE SENTENCE BELOW WITH A FEW WORDS OR A PHRASE. IF YOU DO NOT KNOW THE ANSWER, PRINT DONT KNOW. 
        
        {symbol} is probably {article} {culture} {topic_wise_prompt[topic]}
        
        [/INST]\n"""}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, 
                            add_special_tokens=False, 
                            return_tensors="pt")

    texts = model.generate(input_ids=inputs.to(model.device),
                                max_new_tokens=75, 
                                do_sample=True,
                                top_k=50, 
                                top_p=1, 
                                num_return_sequences=10, 
                                pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens
    responses = tokenizer.batch_decode(texts, skip_special_tokens=True)

    # post-proc the responses
    post_proc_responses = []
    for response in responses:
        response = response.split("<|assistant|>")[1].strip()
        # Only take portion till the first full stop, if no full stop, take the whole response
        if "." in response:
            response = response.split(".")[0]
        post_proc_responses.append(response)

    return post_proc_responses

# Function to calculate overlap coefficient between two strings and use it to trace generalised symbols to memorized and diffuse symbols
def overlap_coefficient(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name")
    parser.add_argument("--topic", type=str, default=None, help="Topic of interest")
    parser.add_argument("--overlap_coefficient", type=float, default=0.75, help="Overlap coefficient threshold")

    args = parser.parse_args()
    topic = args.topic
    model_name = args.model_name

    # Check if the given overlap coefficient is valid
    try:
        float(args.overlap_coefficient)
    except ValueError:
        raise ValueError("Overlap coefficient should be a float")
    if args.overlap_coefficient <= 0 or args.overlap_coefficient > 1:
        raise ValueError("Overlap coefficient should be between 0 and 1")

    # Path to the instruct model for generating definitions of memorized symbols
    model_path = "ssec-uw/OLMo-7B-Instruct-hf"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", do_sample=True)

    # Path to memorized symbol stats
    memorized_stats_path = f"{args.home_dir}/memoed_scripts/memorization_stats_{model_name}_{topic}_Z=[{MEMORIZATION_Z_SCORE}].json"

    # Path to weak association symbols
    generalized_symbols_path = f"{args.home_dir}/memoed_scripts/weak_association_symbols_{model_name}_{topic}_Z=[{MEMORIZATION_Z_SCORE}].json"

    # Path to diffuse association symbols
    diffuse_symbols_path = f"{args.home_dir}/probable_data/categories_nationality_100_{model_name}_prob=True_diffuse_symbols.json"

    # Load memorized stats
    with open(memorized_stats_path, "r") as f:
        memorized_stats = json.load(f)

    # Load generalized symbols
    with open(generalized_symbols_path, "r") as f:
        generalized_symbols = json.load(f)
    generalized_symbols = generalized_symbols["generalized_symbols"]

    # Load diffuse symbols
    with open(diffuse_symbols_path, "r") as f:
        diffuse_symbols = json.load(f)
    diffuse_symbols_topic = diffuse_symbols[topic]

    # path to store weak association generalizations
    weak_association_generalizations_path = f"{args.home_dir}/memoed_scripts/weak_association_generalizations_{model_name}_{topic}_Z=[{MEMORIZATION_Z_SCORE}].json"

    # Check if it exists
    if os.path.exists(weak_association_generalizations_path):
        with open(weak_association_generalizations_path, "r") as f:
            weak_association_generalizations = json.load(f)
    else:
        weak_association_generalizations = {}

    """
    TRACING GENERALIZATIONS TO MEMORIZED SYMBOLS
    """
    # Iterate through the memorized symbols
    for culture in tqdm(memorized_stats.keys()):
        memorized_symbol_lst = memorized_stats[culture]["symbols"]
        print(f"Culture: {culture}")
        for memorized_symbol in tqdm(memorized_symbol_lst):
            if memorized_symbol in list(weak_association_generalizations.keys()):
                continue
            weak_association_generalizations[memorized_symbol] = []
            print(f"memorized Symbol: {memorized_symbol}")

            # Prompt the model and get responses
            responses = prompt_olmo7b_instruct(model, tokenizer, memorized_symbol, culture, topic)

            # Iterate through the generalized symbols and see if they are in the responses
            for generalized_symbol in generalized_symbols:
                for response in responses:
                    if generalized_symbol in response:
                        weak_association_generalizations[memorized_symbol].append(generalized_symbol)
                        break
                    else:
                        # compute overlap coefficient
                        overlap_coeff = overlap_coefficient(generalized_symbol, response)
                        if overlap_coeff >= args.overlap_coefficient:
                            weak_association_generalizations[memorized_symbol].append(generalized_symbol)
                            break

        # Save the weak association generalizations
        with open(weak_association_generalizations_path, "w") as f:
            json.dump(weak_association_generalizations, f, indent=4)

    print("weak association generalizations saved successfully!")

    # Print the statistics 
    weak_asso_gens = []
    for symbol in weak_association_generalizations.keys():
        weak_asso_gens.extend(weak_association_generalizations[symbol])
    print(f"Topic: {topic}")
    print(f"Number of unique weak association generalizations: {len(set(weak_asso_gens))}")
    print(f"Number of unique memorized symbols: {len(weak_association_generalizations.keys())}")

    """
    TRACING GENERALIZATIONS TO DIFFUSE SYMBOLS
    """
    # Iterate through the diffuse symbols and check for overlap with generalised symbols clothing and create a mapping
    traceable_to_diffuse_mapping = {}
    for symbol in diffuse_symbols_topic:
        traceable_to_diffuse_mapping[symbol] = []
        for generalized_symbol in generalized_symbols:
            if generalized_symbol in symbol:
                traceable_to_diffuse_mapping[symbol].append(generalized_symbol)
            else:
                # compute overlap coefficient
                overlap_coeff = overlap_coefficient(generalized_symbol, symbol)
                if overlap_coeff >= args.overlap_coefficient:
                    traceable_to_diffuse_mapping[symbol].append(generalized_symbol)

    # Get unique number of generalizations traceable to diffuse symbols
    traceable_gens_diffuse = []
    for symbol in traceable_to_diffuse_mapping.keys():
        traceable_gens_diffuse.extend(traceable_to_diffuse_mapping[symbol])
    print(f"Number of unique weak association generalizations to diffuse symbols: {len(set(traceable_gens_diffuse))}")

    # Save the mapping
    traceable_to_diffuse_mapping_path = f"{args.home_dir}/memoed_scripts/traced_to_diffuse_{topic}.json"
    with open(traceable_to_diffuse_mapping_path, "w") as f:
        json.dump(traceable_to_diffuse_mapping, f, indent=4)