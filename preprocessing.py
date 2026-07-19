import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# helper to grab nltk data if missing
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        try:
            if res in ['punkt', 'punkt_tab']:
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


# setup nltk resources on import
download_nltk_resources()

# setup the basic nlp tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    """
    Cleans raw tweets by stripping URLs, mentions, and special chars, 
    then tokenizes and lemmatizes the words.
    """
    if not isinstance(text, str):
        return ""
    
    # lowercase everything
    text = text.lower()
    
    # drop links
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # remove user tags
    text = re.sub(r"@\w+", "", text)
    
    # keep the hashtag text but drop the '#' symbol itself
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # filter out weird characters and numbers, just keep letters
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # lemmatize and ditch stop words
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 1
    ]
    
    return " ".join(cleaned_tokens)
