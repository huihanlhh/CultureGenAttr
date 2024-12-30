from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import csv
import json
import pickle as pkl
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict, Counter
from symbol_utils import process_generation_to_symbol_candidates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
THIS FUNCTIONS EXTRACTS THE CULTURAL SYMBOLS FROM SHORTENED RESPONSES OF NON-GPT BASED LLMs
"""
def extract_all_symbols_from_generation(home_dir, shortened_data_path, save_path, topic_list=None, r="neighbor"):
    """
        From the shortened data, extract all symbols that do not contain "traditional", "typical" or contains the nationality
        Extract unigram, bigram, trigram, fourgram and all symbols for potential candidates
    """

    # If topic list is not provided, we use all topics
    if topic_list == None:
        topic_list = [
                        "food",
                        "clothing",
                    ]

    # read the file containing nationalities
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(shortened_data_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r2:
            culture_symbols_dict = json.load(r2)
    else:
        culture_symbols_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    
    roles = ["neighbor"]
    for a, topic in enumerate(tqdm(topic_list, desc="Assessing topic")):
        if topic not in culture_symbols_dict:
            culture_symbols_dict[topic] = dict(
                unigrams=Counter(),
                bigrams=Counter(),
                trigrams=Counter(),
                fourgrams=Counter(),
                all=Counter()
            )
        for b, role in enumerate(tqdm(roles, desc="Assesing role")):
            for country, nationality in countries_nationalities_list: # target nationality
                for gender in ["male", "female", ""]:
                    print(f"Topic: {topic},")
                    print(f"Role: {role},")
                    print(f"Nationality: {nationality},")
                    print(f"Gender: {gender}")
                    value_list = category_nationality_dict[topic][role][nationality][gender]

                    # HANDLING I DONT KNOW OR NONE COMMENTS IN THE VALUES
                    value_list = [value for value in value_list if (value.lower != "none" or value.lower != "i don't know") and not value.endswith("text.") and " text " not in value and " any " not in value and " mention " not in value]

                    for phrases in value_list:
                        # split semicolon-separated values
                        for phrase in phrases.split(";"):
                            # remove marked expressions or NONE values in the shortened data
                            if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or "None" in phrase or "none" in phrase or nationality in phrase or country in phrase:
                                continue

                            """
                            HANDLING EDGE CASES IN FOOD TOPIC
                            """
                            if topic == "food":
                                if len(phrase.split("eats ")) > 1:
                                    strr = phrase.split("eats ")[1]
    
                                    if strr.split()[0] == "is":
                                        # add to list till you get a full stop
                                        lst = strr.split()[1:]
                                    else:
                                        # add to list till you get a full stop
                                        lst = strr.split()

                                    lst_temp = []
                                    for tok in lst:
                                        if tok == '.':
                                            break
                                        elif tok[-1] == '.':
                                            lst_temp.append(tok[:-1])
                                            break
                                        else:
                                            lst_temp.append(tok)

                                    phrase = " ".join(lst_temp)
                            
                            # clean up phrase
                            phrase = process_generation_to_symbol_candidates(phrase)

                            # add n-grams and entire phrase to culture_symbols_dict
                            print(phrase)
                            if phrase and phrase.strip() != "":
                                culture_symbols_dict[topic]["all"][phrase.lower()] += 1
                                all_tokens = phrase.lower().split()
                                # add to unigrams
                                for token in all_tokens:
                                    token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token))
                                    culture_symbols_dict[topic]["unigrams"][token] += 1
                                # add to bigrams
                                if len(all_tokens) >= 2:
                                    for i in range(len(all_tokens)-1):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2])))
                                        culture_symbols_dict[topic]["bigrams"][token] += 1
                                # add to trigrams
                                if len(all_tokens) >= 3:
                                    for i in range(len(all_tokens)-2):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3])))
                                        culture_symbols_dict[topic]["trigrams"][token] += 1    
                                # add to fourgrams 
                                if len(all_tokens) >= 4:
                                    for i in range(len(all_tokens)-3):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4])))
                                        culture_symbols_dict[topic]["fourgrams"][token] += 1 

                with open(save_path, "w") as w:
                    json.dump(culture_symbols_dict, w, indent=4)


def extract_keyword_probability_distribution(keyword, topic, role, nationalities, model, tokenizer):
    """
        Extract the sentence probability distribution of (keyword, topic_prompt, nationality) triples from the model.
        Intuition: Keywords that are "default" might have a even probability for all nationalities; 
                keywords that are culture specific might have a spiky probability for some nationalities;
    """
    
    topics = {
        "food": (f"For dinner, my {role} likes to eat", f"For dinner, my {role} likes to eat all kinds of food"),
        "clothing": (f"My {role} is wearing", f"My {role} is wearing clothing"),
    }
    if keyword != "":
        prompt = f"{topics[topic][0]} {keyword}. My {role} is"
    else:
        prompt = f"{topics[topic][1]}. My {role} is"

    all_probs = []
    all_prompts = []

    # create prompt by appending nationality to the end of the prompt template
    for nationality in nationalities:
        new_prompt = prompt + " " + nationality
        all_prompts.append(new_prompt)

    # calculate sentence probability using the same model by batch
    batch_size = 8
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids

        # get average log probability of the entire sentence
        output = model(input_ids, return_dict=True)
        probs = torch.log_softmax(output.logits, dim=-1).detach()
        
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
        avg_token_log_prob = torch.mean(gen_probs, dim=-1)

        all_probs.extend(avg_token_log_prob.tolist())
        
    # all_probs: a list of sentence log probability of each nationality, for a given keyword
    return all_probs

def precalculate_culture_symbol_nationality_prob(all_symbols_path, model_path, nationalities, role="neighbor", topic_list=None, baseline=False):
    """
        Calculate all n-grams' probability distribution for each nationality given each topic and save to cache
        Only works with models with logits

        baseline: culture-agnostic baseline, where no culture name is mentioned in the prompt
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # load all_symbols json dict
    with open(all_symbols_path, "r") as r:
        all_symbols_dict = json.load(r)
    if topic_list == None:
        topic_list = list(all_symbols_dict.keys())
    
    # CALCULATING BASELINE
    if baseline:
        logger.info("Calculating baseline")
        cache_path = all_symbols_path.replace(".json", "_probability_cache_topical_baseline.pkl")
        if os.path.exists(cache_path):
            cache_dict = pkl.load(open(cache_path, "rb"))
        else:
            cache_dict = defaultdict(dict)
        for a, topic in tqdm(enumerate(topic_list), desc="Assessing topic"):
            probability_distribution = extract_keyword_probability_distribution("", topic, role, nationalities, model, tokenizer)
            cache_dict[topic][""] = probability_distribution
            # save cache
            with open(all_symbols_path.replace(".json", f"_probability_cache_topical_baseline.pkl"), "wb") as w:
                pkl.dump(cache_dict, w)
        # return

    # CALCULATING JOINT PD FOR EACH TOPIC
    for a, topic in tqdm(enumerate(topic_list), desc="Assessing topic"):
        # load cache if exists
        # each topic has a separate cache file
        cache_path = all_symbols_path.replace(".json", f"_probability_cache_{topic}.pkl")
        if os.path.exists(cache_path):
            cache_dict = pkl.load(open(cache_path, "rb"))
        else:
            cache_dict = {}
        ngram_dict = all_symbols_dict[topic]
        for b, ngram_type in tqdm(enumerate(["unigrams", "bigrams", "trigrams", "fourgrams"]), desc=f"Assessing ngrams for {topic}"):
            ngram_list = ngram_dict[ngram_type]
            for c, ngram in tqdm(enumerate(ngram_list), desc=f"Calculating {ngram_type}"):
                if ngram in cache_dict:
                    continue
                probability_distribution = extract_keyword_probability_distribution(ngram, topic, role, nationalities, model, tokenizer)
                cache_dict[ngram] = probability_distribution
            # save cache
            with open(all_symbols_path.replace(".json", f"_probability_cache_{topic}.pkl"), "wb") as w:
                pkl.dump(cache_dict, w)

def choose_keywords_for_cultures(generated_values, target_nationality, target_country, nationalities, cache_dict, baseline_cache_dict, model_name):
    """
        generated_values are obtained from new_shortened.json files
        For each candidate value, we calculate the probability distribution of its 1-4 ngrams. We choose the ngram with the highest sentence probability as the culture symbol candidate
        We then calculate the probability of the culture symbol candidate over all nationalities
        If the probability of the culture symbol candidate is higher than the average culture probability, we add it to the culture_values_dict
    """
    # Step 1: for each value generated for a nationality, we get the value (set) with highest probability
    nationality_index = nationalities.index(target_nationality)
    value_set = set()
    for i, value in enumerate(tqdm(generated_values, desc="Choosing keywords for generations")):
        # process the shortened generations the same way as extracting symbols
        if (value.lower() == "none" or value.lower == "i don't know") or "text." in value or " text " in value or " any " in value or " mention " in value:
            continue
        for phrase in value.split(";"):
            # remove marked expressions
            if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or target_nationality in phrase or target_country in phrase or "None" in phrase or "none" in phrase:
                continue

            """
            HANDLING EDGE CASES 
            """
            if len(phrase.split("eats ")) > 1:
                strr = phrase.split("eats ")[1]

                if strr.split()[0] == "is":
                    # add to list till you get a full stop
                    lst = strr.split()[1:]
                else:
                    # add to list till you get a full stop
                    lst = strr.split()

                lst_temp = []
                for tok in lst:
                    if tok == '.':
                        break
                    elif tok[-1] == '.':
                        lst_temp.append(tok[:-1])
                        break
                    else:
                        lst_temp.append(tok)

                phrase = " ".join(lst_temp)

            phrase = process_generation_to_symbol_candidates(phrase)

            if phrase and phrase.strip() != "":
                all_tokens = phrase.lower().split()
                # find all unigrams
                unigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token)) for token in all_tokens]
                bigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2]))) for i in range(len(all_tokens)-1)]
                trigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3]))) for i in range(len(all_tokens)-2)]
                fourgrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4]))) for i in range(len(all_tokens)-3)]
                # rank the probability of each ngram
                ngrams = unigrams + bigrams + trigrams + fourgrams
                probabilities = [cache_dict[ngram][nationality_index] for ngram in ngrams]
                tups = list(zip(ngrams, probabilities))
                tups = sorted(tups, key=lambda x: x[1], reverse=True)
                value_set.add(tups[0][0])
                # print(tups) 

    # Step 2: for that value set, we calculate the position of the nationality in the distribution probability
    culture_values_dict = {}
    non_culture_values_dict = {}

    value_list = list(value_set)
    for i in range(len(value_list)):
        value = value_list[i]
        if model_name == "mistral-7b" or model_name == "olmo-7b" or "instruct" in model_name or "chat" in model_name:  ## HANDLES ALIGNED MODELs and OLMO-7b
            # mistral-7b and olmo-7b needs calibration
            probs = [(nationality, cache_dict[value][i] - baseline_cache_dict[""][i]) for i, nationality in enumerate(nationalities)]
        else:
            # llama2-13b
            probs = [(nationality, cache_dict[value][i]) for i, nationality in enumerate(nationalities)]

        probs = sorted(probs, key=lambda x: x[1], reverse=True)
        ns, ps = zip(*probs)
        # softmax over probs
        probs_only = torch.nn.functional.softmax(torch.tensor(ps), dim=-1).tolist()
        probs = list(zip(ns, probs_only))
        
        target_index = [i for i, (n, p) in enumerate(probs) if n == target_nationality][0]
        # if the probability of the target nationality is higher than average, we add it to the culture_values_dict
        if probs[target_index][1] > 1/len(nationalities):
            culture_values_dict[value] = probs[target_index][1]
            continue
        
        non_culture_values_dict[value] = probs[target_index][1]

    return culture_values_dict, non_culture_values_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--topic_list", nargs="+", default=None)
    parser.add_argument("--probably",action="store_true", help="Whether 'probably' is added in the prompt")
    parser.add_argument("--extract", action="store_true", help="extract culture symbol candidates from shortened generations")
    parser.add_argument("--probability", action="store_true", help="precalculate the probability distribution of symbol, topic, nationality")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--choose", action="store_true", help="choose culture symbols for nationality")
    
    args = parser.parse_args()
    logger.info(args)

    if args.model_name == "llama2-13b":
        model_path = "meta-llama/Llama-2-13b-hf"
    elif args.model_name == "mistral-7b":
        model_path = "mistralai/Mistral-7B-v0.1"
    elif args.model_name == "olmo-7b":
        model_path = "allenai/OLMo-7B-hf"
    elif args.model_name == "olmo7b-instruct":    #### ADDED PATH FOR ALIGNED OLMO MODEL
        model_path = "ssec-uw/OLMo-7B-Instruct-hf"

    if args.topic_list == None:
        args.topic_list = [
            "food",
            "clothing"
        ]
        
    if args.extract:
        shortened_data_path = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_new_shortened.json"
        save_path = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_all_symbols_two.json"

        extract_all_symbols_from_generation(args.home_dir, shortened_data_path=shortened_data_path, save_path=save_path, topic_list=args.topic_list)

    if args.probability:
        with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
            reader = csv.reader(r)
            next(reader)
            nationalities = [row[1] for row in reader]
        logger.info("Loaded nationalities")

        all_symbols_path = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_all_symbols.json"
        precalculate_culture_symbol_nationality_prob(all_symbols_path, model_path, nationalities, role="neighbor", topic_list=args.topic_list, baseline=args.baseline)
    
    if args.choose:
        print("CHOOSING NOW")
        # path to save number of culture symbols for each culture that overlaps with culture agnostic generations
        culture_agnostic_save_path = f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_culture_agnostic_overlap_evaluation.json"
        # path to shortened values
        new_shortened_values = json.load(open(f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_new_shortened.json", "r"))
        with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
            reader = csv.reader(r)
            next(reader)
            countries, nationalities, groups = zip(*[(row[0], row[1], row[2]) for row in reader])
        for topic in args.topic_list:
            role = "neighbor"
            culture_nationality_mapping_dict = defaultdict(list)
            # topic-wise probability cache path
            cache_dict = pkl.load(open(f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_all_symbols_probability_cache_{topic}.pkl", "rb"))
            # baseline cache path, topic-wise dict
            baseline_cache_dict = pkl.load(open(f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_all_symbols_probability_cache_topical_baseline.pkl", "rb"))[topic]
            logger.info("Loaded nationalities")

            # culture symbols for each nationality, categorized by geographic region groups
            grouped_culture_values_dict = defaultdict(dict)
            for i in range(len(countries)):
                target_country = countries[i]
                target_nationality = nationalities[i]
                generated_values = new_shortened_values[topic][role][target_nationality][""] # gender-neutral generations for the target nationality (CHANGE FOR GENDER SPECIFIC)
                culture_values_dict, non_culture_values_dict= choose_keywords_for_cultures(generated_values, target_nationality, target_country, nationalities, cache_dict, baseline_cache_dict, args.model_name)
                grouped_culture_values_dict[groups[i]][target_nationality] = culture_values_dict
                # this part is for choosing the culture-specific values    
                for cultural_value_key in culture_values_dict:
                    culture_nationality_mapping_dict[cultural_value_key].append(target_nationality)
            # save to file
            with open(f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}_value_to_culture_mapping_{topic}.json", "w") as w:
                json.dump(culture_nationality_mapping_dict, w, indent=4)            