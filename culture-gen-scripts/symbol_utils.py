def process_generation_to_symbol_candidates(phrase):
    # remove redundant content indicator
    if "\"" in phrase:
        indices = [i for i in range(len(phrase)) if phrase.startswith("\"", i)]
        if len(indices) == 1:
            phrase = phrase[indices[0]+1:].strip()
        else:
            phrase = phrase[indices[0]+1: indices[1]].strip()
    if ":" in phrase:
        index = phrase.index(":")
        phrase = phrase[index+1:].strip()
    if "content is" in phrase:
        index = phrase.index("content is")
        phrase = phrase[index+1:].strip()
    # remove explanations
    if "(" in phrase:
        index = phrase.index("(")
        phrase = phrase[:index].strip()

    # remove full stops and punctuation marks in the phrase
    phrase = phrase.replace(".", "")
    phrase = phrase.replace(",", "")
    phrase = phrase.replace("?", "")
    phrase = phrase.replace("!", "")
    phrase = phrase.replace(";", "")
    phrase = phrase.replace(":", "")

    # If phrase has more than 1 word and first word is "a", "an" or "the", remove it
    tokens = phrase.split()
    if len(tokens) > 1 and tokens[0] in ["a", "an", "the"]:
        phrase = " ".join(tokens[1:])

    if phrase.strip() == "":
        return None
    # remove unfinished phrases
    tokens = phrase.split()
    if len(tokens[-1]) == 1:
        return None
    # remove topic words in answers
    removeable_phrases = ["a picture of ", "pictures of ", "statues of ", "a statue of "]
    for p in removeable_phrases:
        if p in phrase:
            phrase = phrase.replace(p, "")
    phrase = phrase.strip()
    return phrase