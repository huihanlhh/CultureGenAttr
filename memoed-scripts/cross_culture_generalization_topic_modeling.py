import json
import logging
import argparse
import string
import nltk
import csv
import re
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, XLMRobertaModel, XLMRobertaTokenizer
import transformers
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
lemmatizer = WordNetLemmatizer()
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  
import gensim.corpora as corpora
from gensim.models import LdaModel
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

# Set cuda visible devices
# visible devices are 0 and 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Setup logging
logging.basicConfig(filename=f"output_topic_modelling.log", filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# dolma server, index is "docs_v1.5_2023-11-02"
es = Elasticsearch(
        cloud_id="<cloud_id>",
        api_key="<api_key>",
        retry_on_timeout=True,
        http_compress=True,
        request_timeout=180,
        max_retries=10,
    )
index = "docs_v1.5_2023-11-02"  ## DOLMA indexing

"""
Functions for performing Topic Modelling on the splits derived from a training document to delineate the topic being covered in the pre-training document
"""
# Function to run a sliding window over the document text and create splits of the text
def sliding_window(text, window_size=1024, step_size=512):
    print(len(text))
    windows = []
    for i in range(0, len(text), step_size):
        window = text[i:i+window_size]
        windows.append(window)
    return windows

# Function to preprocess text per split
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and lemmatize words
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    return text

# Function to extract topics from the LLAMA-3.1 output which was fed with the LDA input
def post_process_response(response):
    final_lst_items = []

    # Get the listed items
    items = re.findall(r'\d+\.\s*(.*)', response)
    
    # Post-process the items
    items = [item.strip() for item in items]
    items = [item for item in items if item != "DONE"]
    for j, item in enumerate(items):
        if '(' in item:
            item = item.split('(')[0].strip()
        if ':' in item:
            item = item.split(':')[0].strip()
        if ' - ' in item or '--' in item:
            item = item.split('-')[0].strip()
        if '.' in item:
            item = item.split('.')[0].strip()
        if ',' in item or ', ' in item:
            item = item.split(',')[0].strip()
        if item == ' ' or item == '':
            continue
        if len(item.split()) > 4:
            continue
        final_lst_items.append(item)

    return final_lst_items

# Function to do LDA on the splits derived from a pre-training document
def latent_dirichlet_allocation_doc(text, 
                                    num_topics=5, 
                                    random_state=100, 
                                    update_every=1, 
                                    chunksize=100, 
                                    passes=15, 
                                    alpha='auto', 
                                    per_word_topics=True):
    # Preprocess the document
    windows = sliding_window(text)
    windows_new = []
    for window in windows:
        window = preprocess_text(window)
        windows_new.append(window)

    # Create Dictionary
    id2word = corpora.Dictionary([doc.split() for doc in windows_new])

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text.split()) for text in windows_new]

    # Set up the LDA model
    lda_model = LdaModel(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics,
                        random_state=random_state,
                        update_every=update_every,
                        chunksize=chunksize,
                        passes=passes,
                        alpha=alpha,
                        per_word_topics=per_word_topics)
    # Get the LDA topics
    lda_output = ""
    for idx, topic in lda_model.print_topics(-1):
        lda_output += f"Topic: {idx} \nWords: {topic} \n"

    return lda_output

# Prompt for LLAMA-3.1 to input the LDA output and generate discernible topics from the document
def generate_llama3_topic_modelling_prompt(lda_output):
    sys_prompt = f"""
        Instructions:
        - Be helpful and answer questions concisely. If you don't know the answer, say 'None'
        - Utilize only the tokens in the given sentence for generating phrases/words for the given query.
        - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
        - Cite your sources when providing factual information.
        """

    prompt_ = f"""
    I had a document on which I ran Latent Dirichlet Allocation and I got the following outputs: \n
    {lda_output} \n
    From this LDA output, list almost 5 possible topics which can be modelled from the document. A topic is a phrase or a word denoting a theme and you can infer it from words or probabilities given above. Do not provide any explanation or description for the topics you generate. If you do not know more topics, stop and say DONE."""

    messages = [
        {"role": "system", "content": sys_prompt + "\n" + "You are an expert in converting an Latent Dirichlet Allocation topic-probability output to topics."},
        {"role": "user", "content": prompt_},
    ]
    return messages

# Function to prompt LLAMA-3.1 and generate topics from the document
def generate_model_responses_topic_modelling(pipeline, messages):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=100,
        eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.6,
        top_p=1
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return response

"""
Function to get sentence embeddings for removing similarities in topics by creating sentence embeddings and removing high cosine similarity topics
"""
def get_batched_embeddings(topics, model, tokenizer, batch_size=16):
    embeddings_dict = {}
    for i in tqdm(range(0, len(topics), batch_size)):
        batch = topics[i:i + batch_size]
        
        # Tokenize the batch of sentences
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Move inputs to the appropriate device
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        # Get the model's output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the last hidden state (sequence_output) from the model output
        last_hidden_state = outputs.last_hidden_state
        
        # Pool the output to get fixed-sized embedding vectors (e.g., by averaging)
        batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Map each original phrase (topic) to its corresponding embedding vector
        for phrase, embedding in zip(batch, batch_embeddings):
            embeddings_dict[phrase] = embedding
    
    return embeddings_dict

"""
Function to extract top_k keywords from the modelled topics (from LLAMA-3.1)
"""
def extract_keywords(args, top_k=10):
    nltk.download('stopwords')
    culture_one = args.culture_one
    culture_two = args.culture_two
    symbol = args.symbol

    # Load the extracted topics
    path_to_extracted_topics = f"{args.home_dir}/memoed_scripts/topic_modelling_responses_{args.model_name}_extracted_{args.topic}.json"
    with open(path_to_extracted_topics, "r") as f:
        extracted_topics = json.load(f)

    # Get the texts for the symbol
    texts = extracted_topics[culture_one][culture_two][symbol]
    combined_text = " ".join(texts)

    # Load the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Fit and transform the text to get the tfidf_matrix
    tfidf_matrix = vectorizer.fit_transform([combined_text])

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get the score for each feature
    dense = tfidf_matrix.todense()
    episode = dense[0].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)

    # Extract the top k keywords
    keywords = [feature_names[word_id] for word_id, score in sorted_phrase_scores[:top_k]]
    print("Keywords for the entire corpus:", keywords)

# Main function
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name to use for classification")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol of interest")
    parser.add_argument("--topic", type=str, default=None, help="Topic of interest")
    parser.add_argument("--culture_one", type=str, default=None, help="Culture 1 of interest")
    parser.add_argument("--culture_two", type=str, default=None, help="Culture 2 of interest")
    parser.add_argument("--model_topics", action="store_true", help="Model topics from the documents")
    parser.add_argument("--extract_topics", action="store_true", help="Extract topics from the documents")
    parser.add_argument("--generate_embeddings", action="store_true", help="Generate embeddings")
    parser.add_argument("--extract_keywords", action="store_true", help="Extract keywords from documents")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Loading nationalities from CSV
    with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    # Lowercase the nationalities
    countries_nationalities_list = [(country.lower(), nationality.lower()) for country, nationality in countries_nationalities_list]

    # Get path to list of docs
    path_to_list_of_docs = f"{args.home_dir}/memoed_scripts/overshadowing_generalisation_docs_{args.model_name}_{args.topic}.json"

    # Load the list of docs
    with open(path_to_list_of_docs, "r") as f:
        doc_ranking = json.load(f)

    if args.symbol is None:
        symbol = ""
    else:
        symbol = args.symbol

    # Path to store final topic_modelling responses
    path_to_topic_modelling_responses = f"{args.home_dir}/memoed_scripts/topic_modelling_responses_{args.model_name}_{args.topic}.json"

    # Path to store extracted topics
    path_to_extracted_topics = f"{args.home_dir}/memoed_scripts/topic_modelling_responses_{args.model_name}_extracted_{args.topic}.json"

    # Path to store embeddings
    path_to_embeddings = f"{args.home_dir}/memoed_scripts/topic_modelling_responses_{args.model_name}_embeddings_{args.topic}.pkl"

    """
    Model Topics from a Pre-Training Document
    """
    if args.model_topics:
        # Load the responses
        try:
            with open(path_to_topic_modelling_responses, "r") as f:
                responses = json.load(f)
        except FileNotFoundError:
            responses = {}

        # Check if args.culture_one and args.culture_two are in the responses
        if args.culture_one in responses:
            if args.culture_two in responses[args.culture_one]:
                if symbol in responses[args.culture_one][args.culture_two]:
                    print("Response already exists")
                    exit()
            else:
                responses[args.culture_one][args.culture_two] = {}
        elif args.culture_two in responses:
            if args.culture_one in responses[args.culture_two]:
                if symbol in responses[args.culture_two][args.culture_one]:
                    print("Response already exists")
                    exit()
            else:
                responses[args.culture_two][args.culture_one] = {}
        else:
            responses[args.culture_one] = {}
            responses[args.culture_one][args.culture_two] = {}

        # Check for combination of culture_one and culture_two in the doc_ranking
        if args.culture_one in doc_ranking:
            list_of_relevant_docs = doc_ranking[args.culture_one][args.culture_two][symbol]
        elif args.culture_two in doc_ranking:
            list_of_relevant_docs = doc_ranking[args.culture_two][args.culture_one][symbol]
        else:
            print("Combination of cultures not found in the doc_ranking")
            exit()

        # Load LLAMA-3.1-8B for post-proc on LDA o/p
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # Load the model
        print("Loading LLAMA3")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        # model.to("cuda")
        print(f"Is cuda available: {torch.cuda.is_available()}")
        print(print(next(model.parameters()).device))

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("Loaded LLAMA3")

        # Get the doc_ids in a list
        relevant_doc_ids = [doc[0] for doc in list_of_relevant_docs]

        # List for llama responses
        llama_responses = []

        # Iterate over the relevant documents
        for id in tqdm(relevant_doc_ids, desc="Looping through relevant documents"):
            print(f"Doc ID: {id}")
            logging.info(f"Doc ID: {id}")
            try:
                doc = es.get(index=index, id=id)
                doc_text = doc["_source"]["text"]
                #print(f"Doc Text: {doc_text}")
            except NotFoundError:
                print(f"Doc ID: {id}")
                print("Document not found in the index")
                continue
            
            # Get the LDA output
            lda_output = latent_dirichlet_allocation_doc(doc_text)

            #print("LDA done")

            """
            Send LDA output to LLAMA-3.1 for post-processing
            """
            # Generate prompt for LLAMA-3.1
            messages = generate_llama3_topic_modelling_prompt(lda_output)
            # Generate topics using LLAMA-3.1
            response = generate_model_responses_topic_modelling(pipeline, messages)
            # Add the response to the list
            llama_responses.append(response)
            # print(response)
            # print("\n")

        # Add the responses to the responses dictionary
        responses[args.culture_one][args.culture_two][symbol] = llama_responses

        # Save the responses
        with open(path_to_topic_modelling_responses, "w") as f:
            json.dump(responses, f, indent=4)

    """
    Extract Topics from the LLAMA-3.1 responses
    """
    if args.extract_topics:
        print("Extracting Topics")
        # Check if path_to_extracted_topics exists
        try:
            with open(path_to_extracted_topics, "r") as f:
                extracted_topics = json.load(f)
        except FileNotFoundError:
            extracted_topics = {}

        # Get the responses
        with open(path_to_topic_modelling_responses, "r") as f:
            responses = json.load(f)

        # Check if args.culture_one and args.culture_two are in the responses and get topic_modelled responses
        if args.culture_one in responses:
            if args.culture_two in responses[args.culture_one]:
                if symbol in responses[args.culture_one][args.culture_two]:
                    topics_modelled_before_post_proc = responses[args.culture_one][args.culture_two][symbol]
        elif args.culture_two in responses:
            if args.culture_one in responses[args.culture_two]:
                if symbol in responses[args.culture_two][args.culture_one]:
                    topics_modelled_before_post_proc = responses[args.culture_two][args.culture_one][symbol]
        else:
            print("Combination of cultures not found in the responses")
            exit()

        # Check if args.culture_one and args.culture_two are in the extracted_topics
        if args.culture_one in extracted_topics:
            if args.culture_two in extracted_topics[args.culture_one]:
                if symbol in extracted_topics[args.culture_one][args.culture_two]:
                    print("Topics already extracted")
                    exit()
            else:
                extracted_topics[args.culture_one][args.culture_two] = {}
        elif args.culture_two in extracted_topics:
            if args.culture_one in extracted_topics[args.culture_two]:
                if symbol in extracted_topics[args.culture_two][args.culture_one]:
                    print("Topics already extracted")
                    exit()
            else:
                extracted_topics[args.culture_two][args.culture_one] = {}
        else:
            extracted_topics[args.culture_one] = {}
            extracted_topics[args.culture_one][args.culture_two] = {}

        # List for extracted topics
        extracted_topics_list = []

        # Iterate over the topics_modelled_before_post_proc
        for response in tqdm(topics_modelled_before_post_proc):
            lst_of_topics = post_process_response(response)
            extracted_topics_list.extend(lst_of_topics)

        # Lowercase the extracted topics
        extracted_topics_list = [topic.lower() for topic in extracted_topics_list]

        # Remove repeated topics
        extracted_topics_list = list(set(extracted_topics_list))

        # Get the names of country and nationality of culture_one and culture_two
        country_one = [country for country in countries_nationalities_list if country[1] == args.culture_one][0][0]
        country_two = [country for country in countries_nationalities_list if country[1] == args.culture_two][0][0]

        # If the country names are in the extracted topics, remove the mention from the string
        for i, topic in enumerate(extracted_topics_list):
            if args.culture_one in topic:
                topic = topic.replace(args.culture_one, "").strip()
            if args.culture_two in topic:
                topic = topic.replace(args.culture_two, "").strip()
            if country_one in topic:
                topic = topic.replace(country_one, "").strip()
            if country_two in topic:
                topic = topic.replace(country_two, "").strip()
            extracted_topics_list[i] = topic
        
        # Remove empty strings
        extracted_topics_list = [topic for topic in extracted_topics_list if topic != ""]

        # Remove repeated topics
        extracted_topics_list = list(set(extracted_topics_list))

        # Print the number of extracted topics
        print(f"Number of extracted topics: {len(extracted_topics_list)}")

        # Add the extracted topics to the extracted_topics dictionary
        extracted_topics[args.culture_one][args.culture_two][symbol] = extracted_topics_list

        # Save the extracted topics
        with open(path_to_extracted_topics, "w") as f:
            json.dump(extracted_topics, f, indent=4)

    """
    Create Embeddings for the extracted topics using XLM-RoBERTa
    """
    if args.generate_embeddings:
        print("Generating Embeddings")
        # Load the XLM-R model and tokenizer with automatic device placement
        model_name = "xlm-roberta-large"
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaModel.from_pretrained(model_name, device_map="auto")

        # Set the model to evaluation mode
        model.eval()

        # Get the extracted topics
        with open(path_to_extracted_topics, "r") as f:
            extracted_topics = json.load(f)

        # Check if args.culture_one and args.culture_two are in the extracted_topics
        if args.culture_one in extracted_topics:
            if args.culture_two in extracted_topics[args.culture_one]:
                if symbol in extracted_topics[args.culture_one][args.culture_two]:
                    topics = extracted_topics[args.culture_one][args.culture_two][symbol]
        elif args.culture_two in extracted_topics:
            if args.culture_one in extracted_topics[args.culture_two]:
                if symbol in extracted_topics[args.culture_two][args.culture_one]:
                    topics = extracted_topics[args.culture_two][args.culture_one][symbol]
        else:
            print("Combination of cultures not found in the extracted_topics")
            exit()

        # See if embedding pickle file already exists
        try:
            with open(path_to_embeddings, "rb") as f:
                final_embeddings_dict = pickle.load(f)
        except FileNotFoundError:
            final_embeddings_dict = {}

        # Check if the culture_one and culture_two are in the final_embeddings_dict
        if args.culture_one in final_embeddings_dict:
            if args.culture_two in final_embeddings_dict[args.culture_one]:
                if symbol in final_embeddings_dict[args.culture_one][args.culture_two]:
                    print("Embeddings already generated")
                    exit()
            else:
                final_embeddings_dict[args.culture_one][args.culture_two] = {}
        elif args.culture_two in final_embeddings_dict:
            if args.culture_one in final_embeddings_dict[args.culture_two]:
                if symbol in final_embeddings_dict[args.culture_two][args.culture_one]:
                    print("Embeddings already generated")
                    exit()
            else:
                final_embeddings_dict[args.culture_two][args.culture_one] = {}
        else:
            final_embeddings_dict[args.culture_one] = {}
            final_embeddings_dict[args.culture_one][args.culture_two] = {}

        # Generate embeddings using batching
        embeddings = get_batched_embeddings(topics, model, tokenizer, batch_size=16)

        # Add the embeddings to the final_embeddings_dict
        final_embeddings_dict[args.culture_one][args.culture_two][symbol] = embeddings

        # Save the embeddings
        with open(path_to_embeddings, "wb") as f:
            pickle.dump(final_embeddings_dict, f)

    """
    Extract Keywords from the modelled topics
    """
    if args.extract_keywords:
        extract_keywords(args)