# BrandPulse AI 🚀

An intelligent Sentiment Analysis platform designed to automatically classify Twitter data into Positive, Neutral, or Negative sentiments. This project compares Classical NLP techniques (TF-IDF + Logistic Regression) with Modern Deep Learning architectures (Bi-Directional LSTM).

## 📁 Project Structure (Deliverables)

1. **The NLP Notebooks** (located in `notebooks/`)
   - `01_Preprocessing_and_Classical.ipynb`: Data cleaning, TF-IDF vectorization, and Baseline Models (Logistic Regression & Naive Bayes).
   - `02_Deep_Learning_LSTM.ipynb`: Advanced Sequence processing, Tokenization, and Bi-Directional LSTM training using TensorFlow/Keras.

2. **The Dashboard Code** (`app.py`)
   - A real-time Streamlit web application that visualizes sentiment distribution, trends over time, and allows custom text inference using the trained models.

3. **Performance Report** (`rapport_performances.md`)
   - A detailed comparison table of Accuracy and F1-Scores between the Classical and Deep Learning models, including Confusion Matrices analysis.

## 🛠️ Setup & Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run the Dashboard

```bash
streamlit run app.py
```
This will open the BrandPulse AI dashboard in your default web browser (typically at `http://localhost:8501`).

## 📊 Quick Performance Summary

| Model               | Accuracy | F1-Score | Strength |
|---------------------|----------|----------|----------|
| Logistic Regression | 77.02%   | 68.09%   | Fast, highly interpretable, excellent baseline. |
| LSTM (Deep Learning)| 73.29%   | 65.56%   | Context-aware, captures complex phrasing (requires larger datasets to beat baseline). |

*For more details, please see `rapport_performances.md`.*
