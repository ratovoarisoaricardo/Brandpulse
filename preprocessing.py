import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Téléchargement automatique des ressources NLTK requises
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


# Initialisation des ressources
download_nltk_resources()

# Configuration du lemmatiseur et des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    """
    Fonction de nettoyage robuste pour les tweets.
    1. Mise en minuscules.
    2. Suppression des URLs (http/https).
    3. Suppression des mentions (@user).
    4. Suppression du symbole '#' tout en gardant le texte du hashtag.
    5. Suppression des caractères spéciaux et chiffres (conservation des lettres et espaces).
    6. Tokenisation.
    7. Suppression des mots vides (stopwords).
    8. Lemmatisation.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Mise en minuscules
    text = text.lower()
    
    # 2. Suppression des URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 3. Suppression des mentions (@user)
    text = re.sub(r"@\w+", "", text)
    
    # 4. Nettoyage des hashtags (supprimer '#' mais garder le mot, ex: #cool -> cool)
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # 5. Remplacer les sauts de ligne et conserver uniquement les lettres et espaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # 6. Tokenisation
    tokens = word_tokenize(text)
    
    # 7 & 8. Suppression des stopwords et Lemmatisation
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 1
    ]
    
    return " ".join(cleaned_tokens)
