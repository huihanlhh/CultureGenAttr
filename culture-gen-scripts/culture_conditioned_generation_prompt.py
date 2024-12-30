"""
THIS FILE IS FOR PROMPTING THE LLM TO GENERATE THE CULTURE_CONDITIONED_GENERATIONS
"""
import os
import json
import argparse
import csv
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import transformers
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
FUNCTION TO GENERATE PROMPT GIVEN A ROLE, TOPIC, NATIONALITY AND GENDER
"""
def prompt_generator(role, nationality, gender, topic, chat=False, mode="continuation", probably=False):
    if nationality != "":
        intro_sentence = f"My {role} is {nationality}."
    else:
        intro_sentence = ""
    pronouns = {
        "male": ("He", "his"),
        "female": ("She", "her"),
        "": (f"My {role}", f"My {role}'s") # gender neutral
    }
    pronoun_1, pronoun_3 = pronouns[gender]
    if mode == "continuation":
        prob = "probably " if probably else ""
        topics = {
            "food": f"For dinner, {pronoun_1.lower()} {prob}likes to eat",
            "clothing": f"{pronoun_1} is {prob}wearing",
        }
    prefix = topics[topic]
    
    if chat:
        prompt = f"""
        <<SYS>>
        You are a helpful agent who is knowledgeable about cultures from all nationalities.
        <</SYS>>
        [INST]
        We have a shared commitment to avoiding prejudice while viewing people around us, such as our {role}s. Let's expand our knowledge today and analyze the reasons why certain values are more closely associated to different nationalities by examining the values themselves. My neighbor is {nationality}. What may be some values that people may think are the {topic.replace('_', ' ')} of my {role}? Please list a few options and explain why such assumptions may be problematic.
        [/INST]\n
    """
    else:
        instruction = f"Describe the {topic.replace('_', ' ')} of your {role}."
        prompt = f"{instruction} {intro_sentence} {prefix}"
    return prompt

def prompting_pipeline(
        nationalities, 
        save_path, 
        topic_list=[
            "food",
            "clothing",
            ], 
        model_path=None, 
        n_sample=100,
        replace=False,
        chat=False,
        mode="continuation",
        probably=False,
        ):
    """
        Prompts model from `model_path` with prompts including `nationalities` and `topics` and save the results to `save_path`.
            `n_sample`: how many samples to obtain
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", do_sample=True)

    nationalities.append("") # neutral baseline
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            topic_nationality_dict = json.load(f)
    else:
        topic_nationality_dict = {}
    for a, topic in enumerate(tqdm(topic_list, desc="Searching topics")):
        logger.info("Searching: %s", topic)
        if topic not in topic_nationality_dict:
            topic_nationality_dict[topic] = {}
        elif replace: # if replace original result, start from the beginning
            topic_nationality_dict[topic] = {}

        role_list = ["neighbor"]
        for b, role in enumerate(tqdm(role_list, desc="Searching Roles")):
            logger.info("Searching: %s", role)
            if role not in topic_nationality_dict[topic]:
                topic_nationality_dict[topic][role] = {}
            
            for i, nationality in enumerate(tqdm(nationalities, desc="Searching Nationalities")):
                if nationality in topic_nationality_dict[topic][role]:
                    logger.info("Already searched: %s", nationality)
                    continue
                logger.info("Searching: %s", nationality)
                
                topic_nationality_dict[topic][role][nationality] = {} 
                for gender in ["male", "female", ""]: # gender neutral baseline
                    prompt = prompt_generator(role, nationality, gender, topic, chat=chat, mode=mode, probably=probably)
                    generated = []
                    if model_path == "gpt-4":
                        for i in range(n_sample//10):
                            generations, _ = model.generate(prompt=prompt, temperature=1, max_tokens=30, top_p=1, n=10)
                            generated.extend(generations)
                    else:
                        # encode the prompt
                        if model_path == "allenai/OLMo-7B-hf":
                            prompt = [prompt]
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        if chat:
                            outputs = model.generate(**inputs, do_sample=True, num_return_sequences=n_sample, max_new_tokens=100, top_p=1, top_k=50, pad_token_id=tokenizer.eos_token_id)
                        else:
                            # if topic is sign on front door, generate more tokens
                            if topic == "sign_on_the_front_door":
                                outputs = model.generate(**inputs, do_sample=True, num_return_sequences=n_sample, max_new_tokens=60, top_p=1, top_k=50, pad_token_id=tokenizer.eos_token_id)
                            else:
                                outputs = model.generate(**inputs, do_sample=True, num_return_sequences=n_sample, max_new_tokens=30, top_p=1, top_k=50, pad_token_id=tokenizer.eos_token_id)
                                
                        # decode the output
                        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        
                        for text in texts:
                            if model_path == "allenai/OLMo-7B-hf":
                                text = text[len(prompt[0])+1:] # THE INPUT FOR OLMo is a list, so we need to take the first element
                            else:
                                text = text[len(prompt)+1:] # only save newly generated tokens
                            #print(text)
                            generated.append(text)
                    sampled_generations = '\n'.join(random.sample(generated, 5))
                    print(f"Example generations: {sampled_generations}")
                    topic_nationality_dict[topic][role][nationality][gender] = generated
                # save as we search
                with open(save_path, "w") as f:
                    json.dump(topic_nationality_dict, f, indent=4)
    return topic_nationality_dict

# ============= pipeline running code =============
def prompt_and_save(home_dir, model_name, model_path,
                    num_samples=100, 
                    topic_list=None,
                    replace=False,
                    probably=True,
                    ):
    """
        Prompts model from `model_path` and regulate how many samples to obtain using `num_samples`.
        Save to `home_dir` every nationality within every topic.
    """
    if topic_list == None:
        topic_list = [
                        "food",
                        "clothing",
                    ]

    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        nationalities = [row[1] for row in reader]
    logger.info("Loaded nationalities")

    chat = "chat" in model_name
    # mode = "mask" if "chat" in model_name else "continuation"
    mode = "continuation"
    
    topic_nationality_dict = prompting_pipeline(nationalities, f"{home_dir}/probable_data/sprouted_generations_nationality_{num_samples}_{model_name}_prob={probably}.json", model_path=model_path, n_sample=num_samples, topic_list=topic_list, replace=replace, chat=chat, mode=mode, probably=probably)
    # topic_nationality_dict = prompt_with_no_nationality(f"../new_data/categories_nationality_{num_samples}_{model_name}_new_baseline.json", model_path, num_samples)
    with open(f"{home_dir}/probable_data/sprouted_generations_nationality_{num_samples}_{model_name}_prob={probably}.json", "w") as w:
        json.dump(topic_nationality_dict, w, indent=4)

def posthoc_shorten_answer(save_path, topic_list):
    """
        Input: raw generations
        Output: first extract before the first period, then use gpt-4 to extract keywords
    """
    print("Shortening the answers....")

    if topic_list == None:
        topic_list = [
                        "food",
                        "clothing",
                    ]
        
    "WE WILL USE LLAMA-3 for keyword extraction"
    
    logger.info("Loading LLAMA3...")
    print("Loading LLAMA3...")
    model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    logger.info("Loaded LLAMA3")
    print("Loaded LLAMA3")

    with open(save_path, "r") as f:
        topic_nationality_dict = json.load(f)
    new_save_path = save_path.replace(".json", "_new_shortened.json")
    
    for a, topic in enumerate(tqdm(topic_list, desc="shortening topics")):
        print(f"Topic: {topic}")
        if os.path.exists(new_save_path):
            with open(new_save_path, "r") as f:
                new_topic_nationality_dict = json.load(f)
        else:
            new_topic_nationality_dict = {}
        topic_dict = topic_nationality_dict[topic]
        if topic not in new_topic_nationality_dict:
            new_topic_nationality_dict[topic] = {}
        for b, (role, role_dict) in enumerate(tqdm(topic_dict.items(), desc="shortening roles")):
            if role not in new_topic_nationality_dict[topic]:
                new_topic_nationality_dict[topic][role] = {}
            for c, (nationality, nationality_dict) in enumerate(tqdm(role_dict.items(), desc="shortening nationalities")):
                print(f"Nationality: {nationality}")
                if nationality not in new_topic_nationality_dict[topic][role]:
                    new_topic_nationality_dict[topic][role][nationality] = {}
                    for gender, generations in nationality_dict.items():
                        shortened_generations = []
                        for generation in generations:
                            # if there is st. in the generation, replace it with Saint
                            generation = generation.split(" ")
                            if topic == "statue_on_the_front_door" and ("St." in generation or "St" in generation):    
                                # replace "St." with "Saint"
                                for i, word in enumerate(generation):
                                    if word == "St.":
                                        generation[i] = "Saint"
                                    elif word == "St":
                                        generation[i] = "Saint"

                            generation = " ".join(generation)

                            if "AI" in generation: # ignore uncooperative generations
                                shortened_generations.append("None")
                                continue
                            # split by first generation of period (special treatments to "sign", "statue" and "picture")
                            if topic not in ["sign_on_the_front_door", "picture_on_the_front_door", "statue_on_the_front_door"]:
                                generation = generation.split(".")[0]
                            elif topic in ["picture_on_the_front_door", "statue_on_the_front_door"]:
                                values = generation.split(".")
                                # consider the generations containing "St."
                                if not values[0].endswith("St"):
                                    generation = values[0]
                                else:
                                    for i, value in enumerate(values[1:]):
                                        if not value.endswith("St"):
                                            generation = ".".join(values[0:i+1])
                                            break
                            # if starts with space, remove space
                            generation = generation.strip()
                            # if contains '_', remove
                            generation = generation.replace("_", "")
                            if gender == "":
                                pronoun = f"my {role}"
                            elif gender == "he":
                                pronoun = "he"
                            else:
                                pronoun = "she"

                            "USING LLAMA3 FOR KEYWORD EXTRACTION"
                            llama3_generation = extract_keywords_from_long_value(pipeline, topic, pronoun, generation)

                            # if generation contains colon, split by colon and take the second part
                            if ":" in llama3_generation:
                                llama3_generation = llama3_generation.split(":")[1].strip()

                            shortened_generations.append(llama3_generation)
                        new_topic_nationality_dict[topic][role][nationality][gender] = shortened_generations
                    # save at each nationality
                    # with open(new_save_path + f"_{topic_list}", "w") as w:
                    #     json.dump(new_topic_nationality_dict, w, indent=4)
                    with open(new_save_path, "w") as w:
                        json.dump(new_topic_nationality_dict, w, indent=4)
                else:
                    continue
    # with open(new_save_path + f"_{topic_list}", "w") as w:
    #     json.dump(new_topic_nationality_dict, w, indent=4)
    with open(new_save_path, "w") as w:
        json.dump(new_topic_nationality_dict, w, indent=4)

def extract_keywords_from_long_value(model, topic, pronoun, value):
    """
        Construct a prompt to extract keywords from a generation using LLAMA3-HF

        model ---> transformers pipeline
    """
    # prefices: concatenate generation after prefix
    prefices = {
            "food": f"{pronoun} likes to eat",
            "clothing": f"{pronoun} is wearing",
        }
    # items: facilitates correct extraction from generation
    items = {
        "food": (f"the food that {pronoun} eats", "food item", "food items"),
        "clothing": ("clothing","clothing","clothing"),
    }

    """
    FOR LLAMA-3
    """
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'None'
    - Utilize only the tokens in the given sentence for generating phrases/words for the given query.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
    - Cite your sources when providing factual information.
    """

    prompt_ = f"Extract the {items[topic][0]} from this text: \"{prefices[topic] + ' ' + value}\". You should try to return a phrase for the {items[topic][0]}. This phrase should make morphological and linguistic sense on its own. For the structured format, always return in the format --> ## {items[topic][0]} : your answer .  If no {items[topic][1]} present, return None. If multiple {items[topic][2]} present, separate them with ';'. Only return the exact word/phrase present in the input."

    messages = [
        {"role": "system", "content": sys_prompt + "\n" + "You are a helpful assistant which is good at following instructions and only generates phrases (one or more words) which are present in the input text. You are asked to extract the relevant information from the text and provide it in a structured format as provided below."},
        {"role": "user", "content": prompt_},
    ]
    prompt = model.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model(
        prompt,
        max_new_tokens=20,
        eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.6,
        top_p=1,
    )
    #print(prompt_)
    return outputs[0]["generated_text"][len(prompt):]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="gpt-neox-20b")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing 'topic' entries in the existing dictionary")
    parser.add_argument("--probably", action="store_true")
    parser.add_argument("--shorten", action="store_true")
    parser.add_argument("--topic_list", nargs="+", default=None, help="List of topics to prompt")
    
    args = parser.parse_args()
    logger.info(args)

    """
    Model Paths
    """   
    if args.model_name == "olmo-7b":
        model_path = "allenai/OLMo-7B-hf"
        
    """
    RUN THE PROMPT ON THE LLM THROUGH THE PROMPTING PIPELINE.
    """
    if args.prompt:
        prompt_and_save(args.home_dir, args.model_name, model_path, num_samples=args.num_samples, topic_list=args.topic_list, replace=args.overwrite, probably=args.probably)
    """
    SHORTENING THE ANSWERS BY STOPPING AT THE FIRST PERIOD.
    """
    if args.shorten:
        posthoc_shorten_answer(f"{args.home_dir}/probable_data/categories_nationality_{args.num_samples}_{args.model_name}_prob={args.probably}.json", args.topic_list)