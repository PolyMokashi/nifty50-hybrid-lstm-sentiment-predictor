# Nifty 50 Hybrid LSTM & Sentiment Predictor

This repository contains the code for a comparative machine learning study evaluating the effectiveness of integrating real-time market sentiment with historical price data to predict Nifty 50 stock movements. 

The project compares a baseline **Random Forest Regressor**, a standard **LSTM**, and a **Dual-Input Hybrid LSTM**. The hybrid model processes numerical technical indicators alongside continuous sentiment scores scraped from news and social media, ultimately achieving the lowest RMSE and Mean Absolute Error (MAE).

## 🚀 Key Features

* **Automated Data Pipeline:** Fetches 7 years of Nifty 50 (`^NSEI`) market data using `yfinance` and engineers technical indicators (SMA_5, SMA_20, RSI, Returns).
* **Multi-Source Sentiment Scraping:** Integrates Google News, Reddit (via PRAW), and Twitter/X (via Tweepy) to aggregate real-time market discussions.
* **NLP Sentiment Scoring:** Utilizes **VADER** and Hugging Face's **BERT** pipeline to compute composite emotional polarity scores from scraped text.
* **Dual-Input Neural Network:** Employs the Keras Functional API to merge a 32-unit LSTM branch (processing historical prices) with a 16-unit LSTM branch (processing temporal sentiment).
* **Performance Evaluation:** Computes and visualizes RMSE, MAE, R² Scores, and Residual Errors using Scikit-Learn and Matplotlib.

## 🛠️ Tech Stack

* **Core:** Python 3, Pandas, NumPy
* **Deep Learning & ML:** TensorFlow, Keras, Scikit-Learn
* **Natural Language Processing:** Hugging Face `transformers`, `vaderSentiment`
* **Data Ingestion:** `yfinance`, `GoogleNews`, `praw` (Reddit API), `tweepy` (Twitter API)
* **Visualization:** Matplotlib

## 🧠 Model Architecture (Hybrid LSTM)

Unlike traditional time-series models that rely solely on historical price windows (Lookback = 10 days), the Hybrid model contextualizes market movements. 
1. **Price Branch:** Processes scaled arrays of `[Close, SMA_5, SMA_20, RSI, Return]`.
2. **Sentiment Branch:** Processes arrays of continuous sentiment polarity scores.
3. **Concatenation:** The outputs of both LSTM branches are concatenated and passed through dense layers to predict the $t+1$ unscaled closing price.

## 📊 Results Summary

The models were evaluated on the testing split (20% of the dataset) and compared via RMSE and MAE (in INR). 
* **Random Forest:** Exhibited the highest error due to limitations in learning sequential time-series patterns.
* **Baseline LSTM:** Captured temporal price dependencies, offering moderate accuracy.
* **Hybrid LSTM + Sentiment:** Delivered the best overall performance, demonstrating that integrating external sentiment signals yields tighter prediction margins and better anticipates sudden market reactions.

## ⚙️ Setup and Installation

**1. Clone the repository**
```bash
git clone [https://github.com/swarpya/nifty50-hybrid-lstm-sentiment-predictor.git](https://github.com/swarpya/nifty50-hybrid-lstm-sentiment-predictor.git)
cd nifty50-hybrid-lstm-sentiment-predictor
