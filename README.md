# 📈 Market Predictor – Forecasting European Stock Indices with Machine Learning

## 🎯 Project Objective

This project investigates whether it is possible to predict — better than random chance — the future movement of major **European stock indices** using **machine learning models**.

We focus on the following three benchmark indices:
- 🇫🇷 **CAC 40** (France)
- 🇪🇺 **STOXX Europe 600** (Pan-European, includes UK)
- 🇪🇺 **EURO STOXX 50** (Eurozone only)

---

## 🧠 Research Background & Literature

This project builds on recent research in financial machine learning, which highlights the role of **technical indicators**, **macroeconomic variables**, and **investor sentiment** in forecasting market behavior.

### 🔬 Key References

- **Kumbure et al. (2022)** – A comprehensive review of 138 ML-based stock forecasting studies (2000–2019). Most models use technical indicators (RSI, SMA, MACD) and techniques such as **SVM**, **neural networks**, and **LSTM**.
- **Liu & Long (2020)** – A hybrid architecture combining **Empirical Wavelet Transform**, **deep LSTM**, and **Extreme Learning Machine** for predicting daily closing prices of major indices. Outperforms standard LSTM and random forests.
- **Lin et al. (2021)** – Classify short-term market direction (up/down) using candlestick patterns and 21 technical indicators with various ML algorithms (logistic regression, k-NN, GBDT, LSTM).
- **Ko & Chang (2021)** – Show how integrating **investor sentiment** from news and forums via **BERT + LSTM-CNN** significantly boosts directional prediction performance.
- **Latif et al. (2023)** – Demonstrate that **macroeconomic indicators** (VIX, EPU, FSI, shadow rates) can outperform technical indicators in forecasting S&P 500 returns using deep learning models.

---

## 🧪 Research Hypotheses

We test the hypothesis that **machine learning models**, especially those using both technical and macroeconomic data, can outperform random guessing in predicting **daily returns** or **market direction** of European stock indices.

We will compare:
- **Regression models**: to forecast the exact daily return
- **Classification models**: to predict the direction (up/down)

---

## 🗃️ Data Sources

The project uses publicly available and easily accessible datasets:

- 📈 **Historical market data** via `yfinance` – OHLCV prices for CAC 40, EURO STOXX 50, and STOXX Europe 600.
- 📊 **Technical indicators** – Locally computed (SMA, EMA, RSI, MACD, etc.).
- 🌍 **Macroeconomic variables** – From **FRED**, **World Bank**, etc. (e.g., VIX, interest rates, oil prices, policy uncertainty).
- 💬 **(Optional)** Sentiment data from Kaggle datasets or news APIs.

