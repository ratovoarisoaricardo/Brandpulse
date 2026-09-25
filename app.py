import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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

# load custom css styling
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

# cache model loading so it doesn't reload on every UI refresh
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
    try:
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            return model, tokenizer
    except Exception as err:
        print("Note: TensorFlow/LSTM model load skipped due to environment:", err)
        return None
    return None

classical_model = load_classical_model()
lstm_resources = load_lstm_model()

# basic heuristic fallback in case the models aren't trained yet
def heuristic_sentiment(text):
    text_lower = text.lower()
    
    pos_words = ['great', 'awesome', 'amazing', 'love', 'good', 'excellent', 'thanks', 'thank', 'best', 'cool', 'super']
    neg_words = ['bad', 'worst', 'terrible', 'wait', 'delayed', 'delay', 'cancel', 'canceled', 'hate', 'rude', 'cold', 'slow', 'lost', 'hours', 'late']
    
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    
    if pos_count > neg_count:
        return 'positive', [0.1, 0.2, 0.7] # mock probs [neg, neu, pos]
    elif neg_count > pos_count:
        return 'negative', [0.7, 0.2, 0.1]
    else:
        return 'neutral', [0.2, 0.6, 0.2]

# main prediction routing
def predict_sentiment(text, model_type):
    cleaned = clean_tweet(text)
    if not cleaned:
        return 'neutral', [0.33, 0.34, 0.33]
        
    if model_type == "Logistic Regression (TF-IDF)" and classical_model is not None:
        probs = classical_model.predict_proba([cleaned])[0]
        classes = classical_model.classes_
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        
        # force order: [negative, neutral, positive]
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
        # fallback if model is missing
        return heuristic_sentiment(text)

# pool of tweets for the live simulation
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

# setup initial session state for the live feed
if 'tweet_history' not in st.session_state:
    # create some fake history over the last 24 hours
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

# app header
st.title("📊 BrandPulse AI — Twitter Sentiment Analysis")
st.markdown("Real-time marketing platform for social listening and emotion analysis.")

# sidebar config
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3256/3256013.png", width=80)
st.sidebar.header("Configuration")

model_option = st.sidebar.selectbox(
    "Active Model",
    ("Logistic Regression (TF-IDF)", "LSTM (Deep Learning)")
)

# show model statuses
st.sidebar.markdown("### Model Status")
if classical_model is not None:
    st.sidebar.success("✅ Classical (TF-IDF) loaded")
else:
    st.sidebar.warning("⚠️ Classical (TF-IDF) not found (Fallback Mode)")

if lstm_resources is not None:
    st.sidebar.success("✅ LSTM Deep Learning loaded")
else:
    st.sidebar.warning("⚠️ LSTM not found (Fallback Mode)")

st.sidebar.markdown("""
---
**About BrandPulse AI**
This app classifies tweets into three categories (positive, neutral, negative) using classical NLP pipelines or deep learning.
""")

# main layout tabs
tab1, tab2 = st.tabs(["🔍 Single Text Analysis", "📈 Live Feed (Simulation)"])

with tab1:
    st.header("On-Demand Sentiment Analysis")
    st.write("Enter a tweet or message below to instantly analyze its emotion.")
    
    user_input = st.text_area("Tweet Text", placeholder="Type your tweet here...", height=100)
    
    st.markdown("**Quick examples to test:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("The service was amazing!"):
            user_input = "The service was amazing!"
    with col_ex2:
        if st.button("I waited 4 hours just to get a cold burger."):
            user_input = "I waited 4 hours just to get a cold burger."
    with col_ex3:
        if st.button("The flight was delayed, but the crew was friendly."):
            user_input = "The flight was delayed, but the crew was friendly."
            
    if st.button("Analyze Sentiment"):
        if user_input.strip() != "":
            with st.spinner("Analyzing..."):
                sentiment, probs = predict_sentiment(user_input, model_option)
                cleaned_text = clean_tweet(user_input)
                
                st.markdown("### Analysis Results")
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.write("**Cleaned Text (NLP Pipeline):**")
                    st.code(cleaned_text if cleaned_text else "[Empty text after filtering]")
                    
                    if sentiment == 'positive':
                        st.markdown(f"Predicted Sentiment: <span class='sentiment-positive'>POSITIVE</span>", unsafe_allow_html=True)
                    elif sentiment == 'neutral':
                        st.markdown(f"Predicted Sentiment: <span class='sentiment-neutral'>NEUTRAL</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"Predicted Sentiment: <span class='sentiment-negative'>NEGATIVE</span>", unsafe_allow_html=True)
                        
                with col_res2:
                    st.write("**Class Probabilities:**")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Confidence': [probs[0], probs[1], probs[2]]
                    })
                    fig_bar = px.bar(
                        prob_df, 
                        x='Confidence', 
                        y='Sentiment', 
                        orientation='h',
                        color='Sentiment',
                        color_discrete_map={'Negative': '#ef4444', 'Neutral': '#9ca3af', 'Positive': '#10b981'},
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
                    st.plotly_chart(fig_bar, width="stretch")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Real-time Tweet Stream")
    st.write("Simulate a continuous feed of incoming tweets to monitor overall brand reputation.")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        if st.session_state.sim_active:
            if st.button("⏸️ Pause Simulation"):
                st.session_state.sim_active = False
                st.rerun()
        else:
            if st.button("▶️ Start Simulation"):
                st.session_state.sim_active = True
                st.rerun()
                
    with col_ctrl2:
        st.write(f"Total tweets analyzed: **{len(st.session_state.tweet_history)}**")
        
    # run the live simulation
    if st.session_state.sim_active:
        # simulate incoming tweet
        new_text = random.choice(SIMULATED_TWEETS)
        # add a random hashtag sometimes to vary it up
        if random.random() > 0.6:
            new_text += f" #{random.choice(['airline', 'travel', 'fail', 'happy', 'brand'])}"
        
        sent, probs = predict_sentiment(new_text, model_option)
        st.session_state.tweet_history.append({
            'timestamp': datetime.now(),
            'text': new_text,
            'sentiment': sent,
            'probs': probs
        })
        
        # keep memory footprint low by capping history at 200
        if len(st.session_state.tweet_history) > 200:
            st.session_state.tweet_history.pop(0)
            
    # calculate stats
    df_hist = pd.DataFrame(st.session_state.tweet_history)
    counts = df_hist['sentiment'].value_counts()
    
    # top KPIs
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        pos_cnt = counts.get('positive', 0)
        st.markdown(f"<div class='metric-card'><h3>🟢 Positive</h3><h2 style='color:#10b981'>{pos_cnt}</h2></div>", unsafe_allow_html=True)
    with col_kpi2:
        neu_cnt = counts.get('neutral', 0)
        st.markdown(f"<div class='metric-card'><h3>⚪ Neutral</h3><h2 style='color:#9ca3af'>{neu_cnt}</h2></div>", unsafe_allow_html=True)
    with col_kpi3:
        neg_cnt = counts.get('negative', 0)
        st.markdown(f"<div class='metric-card'><h3>🔴 Negative</h3><h2 style='color:#ef4444'>{neg_cnt}</h2></div>", unsafe_allow_html=True)
        
    st.markdown("### Analytics Visualizations")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Global Sentiment Distribution**")
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
        st.plotly_chart(fig_pie, width="stretch")
        
    with col_chart2:
        st.write("**Trend over Time (Last 24 Hours)**")
        # group by hours
        df_hist['hour'] = pd.to_datetime(df_hist['timestamp']).dt.floor('h')
        trend_df = df_hist.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        
        fig_trend = px.line(
            trend_df, 
            x='hour', 
            y='count', 
            color='sentiment',
            labels={'hour': 'Time', 'count': 'Tweet Count', 'sentiment': 'Sentiment'},
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
        st.plotly_chart(fig_trend, width="stretch")

    # display recent tweets
    st.markdown("### Recent Tweets Feed")
    recent_tweets = sorted(st.session_state.tweet_history, key=lambda x: x['timestamp'], reverse=True)[:5]
    
    for item in recent_tweets:
        t_str = pd.to_datetime(item['timestamp']).strftime('%H:%M:%S')
        sent_label = item['sentiment']
        if sent_label == 'positive':
            sent_tag = "<span class='sentiment-positive'>POSITIVE</span>"
        elif sent_label == 'neutral':
            sent_tag = "<span class='sentiment-neutral'>NEUTRAL</span>"
        else:
            sent_tag = "<span class='sentiment-negative'>NEGATIVE</span>"
            
        st.markdown(f"""
        <div class='tweet-card'>
            <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                <span style='color:#9ca3af; font-size:0.85rem;'>🕒 Received at {t_str}</span>
                {sent_tag}
            </div>
            <p style='margin:0; font-size:1rem; color:#f3f4f6;'>"{item['text']}"</p>
        </div>
        """, unsafe_allow_html=True)
        
    # force refresh every 2 seconds if simulation is running
    if st.session_state.sim_active:
        time.sleep(2)
        st.rerun()
