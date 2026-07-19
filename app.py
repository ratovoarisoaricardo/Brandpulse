import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from preprocessing import clean_tweet

# Configuration de la page
st.set_page_config(
    page_title="BrandPulse AI - Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du style CSS personnalisé pour une esthétique premium
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .tweet-card {
        background-color: #1e293b;
        border-left: 5px solid #6366f1;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .sentiment-positive {
        color: #10b981;
        font-weight: bold;
        background-color: rgba(16, 185, 129, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
    .sentiment-neutral {
        color: #9ca3af;
        font-weight: bold;
        background-color: rgba(156, 163, 175, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
    .sentiment-negative {
        color: #ef4444;
        font-weight: bold;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_classical_model():
    model_path = 'models/classical_pipeline.pkl'
    if os.path.exists(model_path):
        import joblib
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_lstm_model():
    model_path = 'models/lstm_model.keras'
    tokenizer_path = 'models/tokenizer.pkl'
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    return None

classical_model = load_classical_model()
lstm_resources = load_lstm_model()

# --- MODÈLE HEURISTIQUE DE SECCOURS (Si aucun modèle n'est entraîné) ---
def heuristic_sentiment(text):
    text_lower = text.lower()
    # Mots clés positifs et négatifs simples
    pos_words = ['great', 'awesome', 'amazing', 'love', 'good', 'excellent', 'thanks', 'thank', 'best', 'cool', 'super', 'incroyable', 'excellent', 'parfait']
    neg_words = ['bad', 'worst', 'terrible', 'wait', 'delayed', 'delay', 'cancel', 'canceled', 'hate', 'rude', 'cold', 'slow', 'lost', 'hours', 'late', 'attendu', 'froid', 'nul']
    
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    
    if pos_count > neg_count:
        return 'positive', [0.1, 0.2, 0.7] # mock probs [neg, neu, pos]
    elif neg_count > pos_count:
        return 'negative', [0.7, 0.2, 0.1]
    else:
        return 'neutral', [0.2, 0.6, 0.2]

# --- FONCTION DE PRÉDICTION PRINCIPALE ---
def predict_sentiment(text, model_type):
    cleaned = clean_tweet(text)
    if not cleaned:
        return 'neutral', [0.33, 0.34, 0.33]
        
    if model_type == "Régression Logistique (TF-IDF)" and classical_model is not None:
        probs = classical_model.predict_proba([cleaned])[0]
        classes = classical_model.classes_
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        # Ordonner les probs sous forme [négatif, neutre, positif]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        ordered_probs = [
            probs[class_to_idx.get('negative', 0)],
            probs[class_to_idx.get('neutral', 1)],
            probs[class_to_idx.get('positive', 2)]
        ]
        return pred_label, ordered_probs
        
    elif model_type == "LSTM (Deep Learning)" and lstm_resources is not None:
        model, tokenizer = lstm_resources
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=40, padding='post', truncating='post')
        probs = model.predict(padded, verbose=0)[0] # [negative (0), neutral (1), positive (2)]
        classes = ['negative', 'neutral', 'positive']
        pred_idx = np.argmax(probs)
        return classes[pred_idx], list(probs)
        
    else:
        # Fallback si modèle non présent
        return heuristic_sentiment(text)

# --- POOL DE TWEETS POUR LA SIMULATION ---
SIMULATED_TWEETS = [
    "Just experienced the best customer service ever! Extremely happy with the quick response. #brandpulse",
    "Flight delayed by 5 hours. No explanation, rude staff, and cold food. Never flying with them again.",
    "Is anyone else experiencing issues with their website today? It seems to load very slowly.",
    "Thank you for resolving my booking issue so quickly, outstanding job!",
    "Had a decent flight, nothing special but it was on time.",
    "My luggage is lost. Again. This is the third time this year! Unacceptable service.",
    "Amazing experience, the staff went above and beyond to make us comfortable.",
    "Just a routine trip, everything went smoothly.",
    "Waiting in line for 2 hours just to be told the system is down. Wonderful.",
    "The new app update is fantastic! So much faster and cleaner.",
    "Very average experience. Not good, not bad.",
    "Worst support ever. I've been on hold for 40 minutes.",
    "Lovely weather today! Perfect day for a flight.",
    "Can someone help me with my flight cancellation refund? No one is replying to my emails.",
    "Smooth ride, friendly pilots, and comfortable seats. Five stars!",
    "The service was amazing!",
    "I waited 4 hours just to get a cold hamburger.",
    "Will they ever fix the boarding system? It's so disorganized."
]

# Initialisation de l'état de session pour le flux en direct
if 'tweet_history' not in st.session_state:
    # Créer un historique initial réaliste sur les 24 dernières heures
    history = []
    base_time = datetime.now() - timedelta(hours=24)
    sent_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for i in range(50):
        t = base_time + timedelta(minutes=random.randint(10, 25) * i)
        text = random.choice(SIMULATED_TWEETS)
        sentiment, probs = heuristic_sentiment(text)
        history.append({
            'timestamp': t,
            'text': text,
            'sentiment': sentiment,
            'probs': probs
        })
    st.session_state.tweet_history = history

if 'sim_active' not in st.session_state:
    st.session_state.sim_active = False

# --- HEADER DE L'APPLICATION ---
st.title("📊 BrandPulse AI — Analyse des Sentiments Twitter")
st.markdown("Plateforme marketing d'écoute sociale et d'analyse des émotions en temps réel.")

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3256/3256013.png", width=80)
st.sidebar.header("Configuration")

# Choix du modèle
model_option = st.sidebar.selectbox(
    "Modèle Actif",
    ("Régression Logistique (TF-IDF)", "LSTM (Deep Learning)")
)

# Indicateurs de statut des modèles
st.sidebar.markdown("### Statut des modèles")
if classical_model is not None:
    st.sidebar.success("✅ Classique (TF-IDF) chargé")
else:
    st.sidebar.warning("⚠️ Classique (TF-IDF) non détecté (Mode Secours)")

if lstm_resources is not None:
    st.sidebar.success("✅ LSTM Deep Learning chargé")
else:
    st.sidebar.warning("⚠️ LSTM non détecté (Mode Secours)")

st.sidebar.markdown("""
---
**À propos de BrandPulse AI**
Cette application classe les tweets en trois catégories (positif, neutre, négatif) à l'aide de pipelines NLP classiques ou d'apprentissage profond.
""")

# --- MISE EN PAGE PRINCIPALE (ONGLETS) ---
tab1, tab2 = st.tabs(["🔍 Analyse de Texte Unique", "📈 Flux en Direct (Simulation)"])

# --- ONGLET 1 : ANALYSE DE TEXTE UNIQUE ---
with tab1:
    st.header("Analyse de sentiment sur demande")
    st.write("Saisissez un tweet ou un message ci-dessous pour analyser instantanément les émotions qu'il contient.")
    
    user_input = st.text_area("Texte du Tweet", placeholder="Saisissez votre tweet ici...", height=100)
    
    # Exemples prédéfinis
    st.markdown("**Exemples rapides à tester :**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("Le service était incroyable !"):
            user_input = "Le service était incroyable !"
    with col_ex2:
        if st.button("J'ai attendu 4 heures juste pour avoir un hamburger froid."):
            user_input = "J'ai attendu 4 heures juste pour avoir un hamburger froid."
    with col_ex3:
        if st.button("Le vol a été retardé, mais l'équipage a été sympathique."):
            user_input = "Le vol a été retardé, mais l'équipage a été sympathique."
            
    if st.button("Analyser le Sentiment"):
        if user_input.strip() != "":
            with st.spinner("Analyse en cours..."):
                sentiment, probs = predict_sentiment(user_input, model_option)
                cleaned_text = clean_tweet(user_input)
                
                st.markdown("### Résultats de l'analyse")
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.write("**Texte nettoyé (NLP Pipeline) :**")
                    st.code(cleaned_text if cleaned_text else "[Texte vide après filtrage]")
                    
                    if sentiment == 'positive':
                        st.markdown(f"Sentiment Prédit : <span class='sentiment-positive'>POSITIF</span>", unsafe_allow_html=True)
                    elif sentiment == 'neutral':
                        st.markdown(f"Sentiment Prédit : <span class='sentiment-neutral'>NEUTRE</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"Sentiment Prédit : <span class='sentiment-negative'>NÉGATIF</span>", unsafe_allow_html=True)
                        
                with col_res2:
                    st.write("**Probabilités par classe :**")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Négatif', 'Neutre', 'Positif'],
                        'Confiance': [probs[0], probs[1], probs[2]]
                    })
                    fig_bar = px.bar(
                        prob_df, 
                        x='Confiance', 
                        y='Sentiment', 
                        orientation='h',
                        color='Sentiment',
                        color_discrete_map={'Négatif': '#ef4444', 'Neutre': '#9ca3af', 'Positif': '#10b981'},
                        text_auto='.1%'
                    )
                    fig_bar.update_layout(
                        showlegend=False, 
                        height=200, 
                        margin=dict(l=0, r=0, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Veuillez saisir du texte pour lancer l'analyse.")

# --- ONGLET 2 : FLUX EN DIRECT SIMULÉ ---
with tab2:
    st.header("Flux de tweets entrants en temps réel")
    st.write("Simulez un flux continu de tweets pour analyser l'évolution globale de la réputation de votre marque.")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        if st.session_state.sim_active:
            if st.button("⏸️ Pause Simulation"):
                st.session_state.sim_active = False
                st.rerun()
        else:
            if st.button("▶️ Démarrer la Simulation"):
                st.session_state.sim_active = True
                st.rerun()
                
    with col_ctrl2:
        st.write(f"Nombre total de tweets analysés : **{len(st.session_state.tweet_history)}**")
        
    # Exécution de la simulation
    if st.session_state.sim_active:
        # Simuler un nouveau tweet entrant
        new_text = random.choice(SIMULATED_TWEETS)
        # Légère altération pour ajouter de la diversité
        if random.random() > 0.6:
            new_text += f" #{random.choice(['airline', 'travel', 'fail', 'happy', 'brand'])}"
        
        sent, probs = predict_sentiment(new_text, model_option)
        st.session_state.tweet_history.append({
            'timestamp': datetime.now(),
            'text': new_text,
            'sentiment': sent,
            'probs': probs
        })
        
        # Limiter à 200 tweets dans l'historique pour éviter d'encombrer la mémoire
        if len(st.session_state.tweet_history) > 200:
            st.session_state.tweet_history.pop(0)
            
    # Calcul des statistiques sur l'historique
    df_hist = pd.DataFrame(st.session_state.tweet_history)
    counts = df_hist['sentiment'].value_counts()
    
    # KPIs en haut de page
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        pos_cnt = counts.get('positive', 0)
        st.markdown(f"<div class='metric-card'><h3>🟢 Positifs</h3><h2 style='color:#10b981'>{pos_cnt}</h2></div>", unsafe_allow_html=True)
    with col_kpi2:
        neu_cnt = counts.get('neutral', 0)
        st.markdown(f"<div class='metric-card'><h3>⚪ Neutres</h3><h2 style='color:#9ca3af'>{neu_cnt}</h2></div>", unsafe_allow_html=True)
    with col_kpi3:
        neg_cnt = counts.get('negative', 0)
        st.markdown(f"<div class='metric-card'><h3>🔴 Négatifs</h3><h2 style='color:#ef4444'>{neg_cnt}</h2></div>", unsafe_allow_html=True)
        
    st.markdown("### Visualisations analytiques")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Répartition globale des Sentiments**")
        pie_data = pd.DataFrame({
            'Sentiment': [s.capitalize() for s in counts.index],
            'Total': counts.values
        })
        fig_pie = px.pie(
            pie_data, 
            values='Total', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#9ca3af', 'Negative': '#ef4444'},
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            legend=dict(orientation="h", y=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.write("**Courbe de tendance (24 Dernières Heures)**")
        # Regrouper par tranches horaires
        df_hist['hour'] = pd.to_datetime(df_hist['timestamp']).dt.floor('H')
        trend_df = df_hist.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        
        fig_trend = px.line(
            trend_df, 
            x='hour', 
            y='count', 
            color='sentiment',
            labels={'hour': 'Temps', 'count': 'Nombre de Tweets', 'sentiment': 'Sentiment'},
            color_discrete_map={'positive': '#10b981', 'neutral': '#9ca3af', 'negative': '#ef4444'},
            markers=True
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#374151'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Affichage des derniers tweets reçus
    st.markdown("### Flux de tweets récents")
    recent_tweets = sorted(st.session_state.tweet_history, key=lambda x: x['timestamp'], reverse=True)[:5]
    
    for item in recent_tweets:
        t_str = pd.to_datetime(item['timestamp']).strftime('%H:%M:%S')
        sent_label = item['sentiment']
        if sent_label == 'positive':
            sent_tag = "<span class='sentiment-positive'>POSITIF</span>"
        elif sent_label == 'neutral':
            sent_tag = "<span class='sentiment-neutral'>NEUTRE</span>"
        else:
            sent_tag = "<span class='sentiment-negative'>NÉGATIF</span>"
            
        st.markdown(f"""
        <div class='tweet-card'>
            <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                <span style='color:#9ca3af; font-size:0.85rem;'>🕒 Reçu à {t_str}</span>
                {sent_tag}
            </div>
            <p style='margin:0; font-size:1rem; color:#f3f4f6;'>"{item['text']}"</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Si la simulation est active, forcer le rafraîchissement toutes les 2 secondes
    if st.session_state.sim_active:
        time.sleep(2)
        st.rerun()
