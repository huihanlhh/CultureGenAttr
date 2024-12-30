import argparse
import math
from collections import Counter
import csv
import string
import re
import numpy as np
import json
import math
import os
from tqdm import tqdm
import plotly.graph_objects as go
from multiprocessing import Pool

def load_json(path):
    with open(path, "r") as file:
        return json.load(file)

def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

def get_sorted_data(data, category):
    return sorted(data.items(), key=lambda x: x[1][category], reverse=True)


"""
Function for getting an overview of marking distributions
"""
def display_top_bottom(data, category, num=5):
    sorted_data = get_sorted_data(data, category)
    top_data = sorted_data[:num]
    bottom_data = sorted_data[-num:]
    print(f"\nTop {num} nationalities by {category}")
    for i, (nationality, stats) in enumerate(top_data):
        print(f"{i+1}. Nationality: {nationality}, {category}: {stats[category]}")
    print(f"\nBottom {num} nationalities by {category}")
    for i, (nationality, stats) in enumerate(bottom_data):
        print(f"{i+1}. Nationality: {nationality}, {category}: {stats[category]}")

def calculate_averages(data):
    total = len(data)
    categories = ["M", "TG", "OGS", "OGC", "U"]
    averages = {cat: sum([data[nat][cat] for nat in data]) / total for cat in categories}
    return averages

def overview_markings(data_path):
    data = load_json(data_path)
    categories = ["M", "TG", "OGS", "OGC", "U"]
    for category in categories:
        display_top_bottom(data, category)
    averages = calculate_averages(data)
    print("\nAverages:")
    for category, average in averages.items():
        print(f"Average {category}: {average:.5f}")

"""
Function to plot a pie chart for the distribution of markings in a culture's responses
"""
def plot_pie_chart(data, culture):
    # Labels and colors as given in the requirement
    labels = {
        "M": "Memo Association",
        "TG": "Weak Association Gen",
        "OGS": "Diffuse Association",
        "OGC": "Cross-Culture Gen",
        "U": "Untraceable Gen"
    }
    
    category_colors = {
        "M": '#636EFA',
        "TG": '#EF553B',
        "OGS": '#00CC96',
        "OGC": '#AB63FA',
        "U": '#FFA15A'
    }
    
    # Filter and prepare data
    filtered_data = {k: data[k] for k in data if data[k] > 0}
    sizes = [filtered_data[k] for k in filtered_data]
    label_values = [labels[k] for k in filtered_data]
    colors = [category_colors[k] for k in filtered_data]
    
    # Create the pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=label_values,
        values=sizes,
        textinfo='label+percent',
        hoverinfo='label+value',
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )])
    
    # Update layout to include legend on the side
    fig.update_layout(
        title_text=f'{culture} Cultural Response Analysis',
        annotations=[dict(text='', x=0.5, y=0.5, font_size=20, showarrow=False)],
        legend=dict(title="Categories", orientation="v", x=1, y=0.5)
    )
    
    # Save the plot
    fig.write_html(f"{args.home_dir}/probable_data/{culture}_{topic}_cultural_response_analysis.html")
    fig.show()

# Main Function call
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default="olmo-7b", help="Model name to use for classification")
    parser.add_argument("--topic", type=str, default="clothing", help="Topic of interest")
    parser.add_argument("--acc_symbols", action="store_true", help="Accumulate all symbols of a culture")
    parser.add_argument("--mark_symbols", action="store_true", help="Mark all symbols of a culture")
    parser.add_argument("--get_stats", action="store_true", help="Get stats of all symbols of a culture")
    parser.add_argument("--overview", action="store_true", help="Provide an overview of marking distributions")
    parser.add_argument("--plot_pie_chart", action="store_true", help="Plot a pie chart for cultural markings")
    parser.add_argument("--culture", type=str, help="Specify the culture to plot pie chart for")

    args = parser.parse_args()
    topic = args.topic

    # Path to non_diffuse_symbols 
    path_to_non_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_non_diffuse_symbols.json"

    # Path to diffuse symbols
    path_to_diffuse_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_diffuse_symbols.json"

    # Path to shortened responses
    path_to_shortened_responses = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_new_shortened.json"

    # Open the shortened responses
    with open(path_to_shortened_responses, "r") as f:
        shortened_responses = json.load(f)
    responses = shortened_responses[topic]["neighbor"]

    # Open the non general symbols
    with open(path_to_non_diffuse_symbols, "r") as f:
        non_diffuse_symbols = json.load(f)
    non_diffuse_symbols = non_diffuse_symbols[topic]

    # Open the general symbols
    with open(path_to_diffuse_symbols, "r") as f:
        diffuse_symbols = json.load(f)
    diffuse_symbols = diffuse_symbols[topic]

    # All symbols
    all_symbols = non_diffuse_symbols + diffuse_symbols
    
    # Loading nationalities from CSV
    with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    # Get nationalities
    nationalities = [nationality[1] for nationality in countries_nationalities_list]

    # path to save accumulated symbols
    path_to_acc_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_culture_response_wise_symbols_{topic}.json"
    
    """
    Generating Symbols of Each Culture
    """
    if args.acc_symbols:
        # Check if the file already exists
        if os.path.exists(path_to_acc_symbols):
            # with open(path_to_acc_symbols, "r") as f:
            #     acc_symbols_dict = json.load(f)
            acc_symbols_dict = {}
        else:
            acc_symbols_dict = {}

        # Iterate through the nationalities
        for nationality in nationalities:
            # Get the responses
            responses_ = responses[nationality]
            responses_new = responses_['male'] + responses_['female'] + responses_['']
            symbol_list = []
            for symbol in non_diffuse_symbols:
                # Iterate through each response and add if the symbol is present
                for response in responses_new:
                    if symbol in response:
                        symbol_list.append(symbol)

            for symbol in diffuse_symbols:
                # Iterate through each response and add if the symbol is present
                for response in responses_new:
                    if symbol in response:
                        # check if any non general symbol is present in the response and there is no ';'
                        if any([non_diffuse_symbol in response for non_diffuse_symbol in non_diffuse_symbols]) and ";" not in response:
                            continue
                        symbol_list.append(symbol)

            acc_symbols_dict[nationality] = symbol_list

        # Save the accumulated symbols
        with open(path_to_acc_symbols, "w") as f:
            json.dump(acc_symbols_dict, f, indent=4)  

    # path to save marked symbols
    path_to_marked_symbols = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_culture_response_wise_marked_symbols_{topic}.json"

    """
    Final Marking for the Dashboard
    """
    if args.mark_symbols:
        """
        Markings:
        M - Memorization
        TG - Weak Association Generalization traceable to memorization
        OGS - When symbol is diffuse association or traced to diffuse association symbol
        OGC - When memorized symbol of other culture is generated
        U - Unweak association generalization
        """

        """
        Memorization Stats and Symbols
        """
        # Path to memorized symbols
        path_to_memorized_symbols = f"{args.home_dir}/memoed_scripts/memorized_symbols_{args.topic}.json"

        # load memorized symbols
        with open(path_to_memorized_symbols, "r") as f:
            memorized_symbols = json.load(f)
        memorized_symbols = memorized_symbols["memorized_symbols"]

        # Path to memorization stats
        path_to_memorization_stats = f"{args.home_dir}/memoed_scripts/memorization_stats_olmo-7b_{topic}_Z=[2.6].json"

        # Load memorization stats
        with open(path_to_memorization_stats, "r") as f:
            memorization_stats = json.load(f)

        """
        Weak Association Generalizations
        """
        # Path to weak association generalization
        path_to_weak_association_generalization = f"{args.home_dir}/memoed_scripts/weak_association_generalizations_{args.model_name}_{topic}_Z=[2.6]_two.json"
        # Open weak association generalizations
        with open(path_to_weak_association_generalization, "r") as f:
            weak_association_generalizations = json.load(f)
        weak_association_generalizations_lst = []
        for key in weak_association_generalizations:
            weak_association_generalizations_lst.extend(weak_association_generalizations[key])
        weak_association_generalizations_lst = list(set(weak_association_generalizations_lst))

        """
        Traced to Diffuse Association
        """
        path_to_traced_to_diffuse = f"{args.home_dir}/memoed_scripts/traced_to_diffuse_{topic}.json"
        # Open traced to diffuse
        with open(path_to_traced_to_diffuse, "r") as f:
            traced_to_diffuse = json.load(f)
        traced_to_diffuse_lst = []
        for key in traced_to_diffuse:
            traced_to_diffuse_lst.extend(traced_to_diffuse[key])
        traced_to_diffuse_lst = list(set(traced_to_diffuse_lst))

        """
        Unweak association generalizations
        """
        # Path to unweak association generalization
        path_to_unweak_association_generalization = f"{args.home_dir}/memoed_scripts/unweak_association_generalizations_{topic}.json"
        # Open unweak association generalizations
        with open(path_to_unweak_association_generalization, "r") as f:
            unweak_association_generalizations = json.load(f)
        unweak_association_generalizations_lst = unweak_association_generalizations["unweak_association_generalizations"]

        """
        Load the accumulated symbols
        """
        with open(path_to_acc_symbols, "r") as f:
            acc_symbols_dict = json.load(f)

        # Final marking dict

        # Check if the file already exists
        if os.path.exists(path_to_marked_symbols):
            with open(path_to_marked_symbols, "r") as f:
                marked_symbols_dict = json.load(f)
        else:
            marked_symbols_dict = {}

        # Iterate through the nationalities
        for nationality in tqdm(acc_symbols_dict.keys()):
            # Lowercase nationality
            nationality_lower = nationality.lower()
            # Get the responses
            responses = acc_symbols_dict[nationality]

            # Get the list of memorized symbols of nationality
            memo_symbols_nationality = memorization_stats[nationality_lower]["symbols"]

            # Generate a marking list of same length as responses
            marking_lst = [""]*len(responses)

            # Iterate through each response
            for i, response in enumerate(responses):
                if response in memorized_symbols:
                    if response in memo_symbols_nationality:
                        """
                        Memorization
                        """
                        marking_lst[i] = "M"
                    else:
                        """
                        Cross-Culture Generalization
                        """
                        marking_lst[i] = "OGC"
                elif response in weak_association_generalizations_lst:
                    """
                    Weak Association from Memorized Association
                    """
                    marking_lst[i] = "TG"
                elif response in traced_to_diffuse_lst:
                    """
                    Weak Association from Diffuse Association
                    """
                    marking_lst[i] = "OGS-I"
                elif response in diffuse_symbols:
                    """
                    Diffuse Association
                    """
                    marking_lst[i] = "OGS"
                elif response in unweak_association_generalizations_lst:
                    """
                    Unweak association generalization
                    """
                    marking_lst[i] = "U"
                #print(f"Response: {response}, Marking: {marking_lst[i]}")

            marked_symbols_dict[nationality] = marking_lst

        # Save the marked symbols
        with open(path_to_marked_symbols, "w") as f:
            json.dump(marked_symbols_dict, f, indent=4)
                    
    """
    Store percentages for all nationalities
    """
    if args.get_stats:
        # Open the marked symbols
        with open(path_to_marked_symbols, "r") as f:
            marked_symbols_dict = json.load(f)

        # Path to store the percentages
        path_to_percentages = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_culture_response_wise_marked_symbols_percentages_{topic}_two.json"

        dictionary_percentages = {}

        # Initialize the dictionary
        for key in marked_symbols_dict.keys():
            dictionary_percentages[key] = {"M": 0, "TG": 0, "OGS-I": 0, "OGS": 0, "OGC": 0, "U": 0}

        # Iterate through the marked symbols
        for key in marked_symbols_dict.keys():
            marked_symbols = marked_symbols_dict[key]
            counter = Counter(marked_symbols)
            for key_ in counter.keys():
                percentage = counter[key_]/len(marked_symbols)
                # Round off to 5 decimal places
                percentage = round(percentage, 5)
                dictionary_percentages[key][key_] = percentage

        # Save the percentages
        with open(path_to_percentages, "w") as f:
            json.dump(dictionary_percentages, f, indent=4)

    """
    Provide an overview of marking distributions
    """
    if args.overview:
        markings_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_culture_response_wise_marked_symbols_percentages_{args.topic}.json"
        overview_markings(markings_path)

    """
    Plot a pie chart for the distribution of markings in a culture's responses
    """
    if args.plot_pie_chart and args.culture:
        # Path to marked symbols percentages
        data_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob=True_culture_response_wise_marked_symbols_percentages_{args.topic}.json"
        if os.path.exists(data_path):
            data = load_json(data_path)
            if args.culture in data:
                plot_pie_chart(data[args.culture], args.culture)
            else:
                print(f"No data available for culture: {args.culture}")
        else:
            print("Data file not found. Please ensure the correct path and data availability.")
