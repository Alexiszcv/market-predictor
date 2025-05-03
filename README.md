# 📈 Market Predictor – Prédiction des Indices Boursiers Européens par Machine Learning

## 🎯 Objectif du projet

Ce projet vise à explorer la **prédictibilité des grands indices boursiers européens** à l'aide de méthodes de **Machine Learning**. Nous cherchons à répondre à la question suivante :

> *Peut-on prédire — mieux que le hasard — l’évolution de grands indices boursiers européens en s'appuyant sur des modèles d’apprentissage automatique ?*

Nous nous concentrons ici sur trois indices emblématiques :
- 🇫🇷 **CAC 40** (France)
- 🇪🇺 **STOXX Europe 600** (Europe large, incluant la Grande-Bretagne)
- 🇪🇺 **EURO STOXX 50** (Zone euro uniquement)

---

## 🧠 Cadre de recherche et inspirations

La démarche du projet s’ancre dans l’état de l’art sur la prédiction boursière par ML, en combinant des sources **de prix**, **d’indicateurs techniques**, **macroéconomiques** et éventuellement **de sentiment**.

Nous nous appuyons notamment sur plusieurs travaux récents pour guider notre approche :

### 🔬 Recherches académiques clés

- **Kumbure et al. (2022)** – Revue de 138 articles sur le ML en finance. Montre la prédominance des indicateurs techniques (SMA, RSI, etc.) et des modèles non linéaires comme les **SVM**, **réseaux de neurones** et plus récemment **LSTM**. 
- **Liu & Long (2020)** – Proposent un modèle hybride combinant **ondelettes (EWT)**, **réseau LSTM optimisé**, et **Extreme Learning Machines** pour prédire les cours d'indices comme le S&P 500. Le modèle montre une nette amélioration par rapport aux approches classiques.
- **Lin et al. (2021)** – Se concentrent sur la **classification de la tendance** du marché (up/down), en combinant **indicateurs techniques** et **motifs de chandeliers japonais**, testés avec des SVM, forêts aléatoires et LSTM.
- **Ko & Chang (2021)** – Intègrent des **données de sentiment issues de forums et actualités** dans un modèle **LSTM-CNN**. Montre que le sentiment améliore significativement la prédiction.
- **Latif et al. (2023)** – Insistent sur la **puissance explicative des variables macroéconomiques** (VIX, EPU, FSI, taux SSR, etc.), parfois supérieures aux indicateurs techniques.

---

## 🧪 Hypothèse de recherche

Nous testons l’hypothèse que les modèles de ML, en particulier ceux intégrant des **variables macroéconomiques et techniques**, peuvent **battre le hasard** dans la prédiction des **rendements quotidiens ou directionnels** des indices européens.

Deux approches seront comparées :

- **Régression** : Prédire le rendement journalier de l’indice.
- **Classification** : Prédire la tendance haussière ou baissière du marché (up/down).

---

## 🗃️ Sources de données

Nous utilisons une combinaison de sources ouvertes, notamment :

- 📈 **Yahoo Finance API** (via `yfinance`) – Données historiques OHLCV des indices CAC 40, Euro Stoxx 50, Stoxx 600.
- 📊 **Indicateurs techniques** – Calculés localement (RSI, MACD, moyennes mobiles, etc.).
- 🌍 **Variables macroéconomiques** – Via **FRED**, **Banque Mondiale**, ou autres bases ouvertes (ex : VIX, taux d’intérêt, pétrole, EPU).
- 💬 **(Optionnel)** Données de sentiment via sources secondaires (Kaggle, news headlines, etc.).

---

## ⚙️ Organisation du projet

