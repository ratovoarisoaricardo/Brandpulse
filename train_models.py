import os
import urllib.request
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from preprocessing import clean_tweet

def download_data():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    data_path = 'data/Tweets.csv'
    urls = [
        'https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv',
        'https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/refs/heads/master/Tweets.csv',
        'https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv'
    ]
    
    if not os.path.exists(data_path):
        print("[-] Téléchargement du jeu de données Twitter Airline Sentiment...")
        success = False
        for url in urls:
            try:
                print(f"[-] Tentative de téléchargement depuis : {url}")
                urllib.request.urlretrieve(url, data_path)
                print("[+] Jeu de données téléchargé avec succès !")
                success = True
                break
            except Exception as e:
                print(f"[!] Échec du téléchargement depuis {url} : {e}")
        if not success:
            raise RuntimeError("Impossible de télécharger le jeu de données depuis les URLs sources.")
    else:
        print("[+] Le jeu de données existe déjà.")
    return data_path


def train_classical(df):
    print("\n=== ENTRAÎNEMENT DU MODÈLE CLASSIQUE (TF-IDF + REGRESSION LOGISTIQUE) ===")
    
    X = df['clean_text']
    y = df['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    
    # Sauvegarde du modèle
    joblib_path = 'models/classical_pipeline.pkl'
    import joblib
    joblib.dump(pipeline, joblib_path)
    print(f"[+] Modèle classique sauvegardé sous '{joblib_path}'.")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def train_deep_learning(df):
    print("\n=== ENTRAÎNEMENT DU MODÈLE DEEP LEARNING (LSTM) ===")
    
    # Mapping des labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['airline_sentiment'].map(label_map)
    
    X = df['clean_text'].values
    y = df['label'].values
    
    # Tokenisation
    max_features = 8000
    tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    
    # Sauvegarde du Tokenizer
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("[+] Tokenizer sauvegardé sous 'models/tokenizer.pkl'.")
    
    # Conversion en séquences et padding
    X_seq = tokenizer.texts_to_sequences(X)
    max_len = 40
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')
    y_cat = to_categorical(y, num_classes=3)
    
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    # Modèle LSTM
    embedding_dim = 64
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(32, dropout=0.2, recurrent_dropout=0.2) if tf.test.is_gpu_available() else LSTM(32, dropout=0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # Entraînement rapide
    print("Entraînement en cours (3 epochs max pour le script rapide)...")
    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Évaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    
    # Sauvegarde du modèle LSTM
    model_path = 'models/lstm_model.keras'
    model.save(model_path)
    print(f"[+] Modèle LSTM sauvegardé sous '{model_path}'.")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

if __name__ == '__main__':
    data_file = download_data()
    
    print("\n[-] Chargement des données...")
    df = pd.read_csv(data_file)
    
    print("[-] Nettoyage du texte...")
    df['clean_text'] = df['text'].apply(clean_tweet)
    df = df[df['clean_text'].str.strip() != '']
    
    # Exécution des entraînements
    classical_metrics = train_classical(df)
    dl_metrics = train_deep_learning(df)
    
    # Écriture d'un rapport de performance de base
    print("\n=== SYNTHÈSE DES PERFORMANCES ET COMPARAISON ===")
    print(f"{'Modèle':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    print(f"{'Régression Logistique':<25} | {classical_metrics['accuracy']:.4f}     | {classical_metrics['precision']:.4f}     | {classical_metrics['recall']:.4f}  | {classical_metrics['f1']:.4f}")
    print(f"{'LSTM (Deep Learning)':<25} | {dl_metrics['accuracy']:.4f}     | {dl_metrics['precision']:.4f}     | {dl_metrics['recall']:.4f}  | {dl_metrics['f1']:.4f}")
