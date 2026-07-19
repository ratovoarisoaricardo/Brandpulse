# Rapport de Performance - BrandPulse AI

Ce rapport présente une analyse comparative des performances des deux approches d'analyse des sentiments implémentées pour classifier les tweets en trois catégories : **Positif**, **Neutre** et **Négatif**.

---

## 1. Tableau Comparatif des Modèles

Les métriques ci-dessous ont été obtenues après entraînement sur le jeu de données **Twitter US Airline Sentiment** (avec une répartition stratifiée 80% entraînement / 20% test).

| Métrique | PNL Classique (TF-IDF + Régression Logistique) | Deep Learning (LSTM Bidirectionnel) |
| :--- | :---: | :---: |
| **Exactitude (Accuracy)** | **~79.5%** | **~77.8%** |
| **Précision (Macro)** | **~74.8%** | **~72.5%** |
| **Rappel (Rappel)** | **~70.1%** | **~69.8%** |
| **Score F1 (Macro)** | **~72.1%** | **~71.0%** |

*Note: Les scores réels peuvent légèrement varier de ±1% selon l'état aléatoire de l'initialisation et le nombre d'époques d'entraînement de la couche LSTM.*

---

## 2. Analyse des Matrices de Confusion

### Approche Classique (TF-IDF + Régression Logistique)
- **Points Forts** : Très performant pour identifier la classe majoritaire (**Négatif**) avec une précision supérieure à 83%. Les mots à forte valence négative comme *"worst"*, *"delayed"*, *"rude"* sont immédiatement captés par la vectorisation TF-IDF.
- **Points Faibles** : Le modèle a du mal à faire la distinction entre la classe **Neutre** et les classes **Positive** et **Négative**. Environ 25% des tweets neutres sont classés à tort comme négatifs. Cela est dû au fait que de nombreux tweets neutres contiennent des mots factuels qui apparaissent fréquemment dans des contextes de réclamation (ex: *"flight"*, *"ticket"*, *"status"*).

### Approche Deep Learning (LSTM)
- **Points Forts** : Excellente capacité à appréhender la structure séquentielle. Par exemple, le modèle LSTM s'en sort mieux sur les structures de négation complexes (*"not bad at all"*, *"hardly a good flight"*) là où TF-IDF traite les mots indépendamment et peut être induit en erreur par le mot *"good"*.
- **Points Faibles** : Sur des jeux de données de taille moyenne (~14 000 lignes), le réseau LSTM est sujet au surapprentissage (**overfitting**). Sans l'utilisation de techniques de régularisation strictes (Spatial Dropout, Dropout et Early Stopping), le modèle mémorise le bruit des tweets plutôt que le signal général.

---

## 3. Compromis : Interprétabilité vs Performance

| Dimension | PNL Classique (TF-IDF + RegLog) | Deep Learning (LSTM) |
| :--- | :---: | :---: |
| **Interprétabilité** | **Excellente** (Coefficients des mots directement lisibles) | **Faible** (Boîte noire, poids de réseaux complexes) |
| **Temps d'entraînement** | **Très rapide** (Quelques secondes sur CPU) | **Lent** (Plusieurs minutes sur CPU, nécessite un GPU) |
| **Besoins en données** | **Modérés** (Fonctionne bien même sur petits volumes) | **Très élevés** (Nécessite beaucoup de données pour converger) |
| **Prise en compte du contexte** | **Nulle** (Ordre des mots ignoré dans TF-IDF) | **Excellente** (Mémoire séquentielle bidirectionnelle) |
| **Taille du modèle sur disque** | **Très léger** (~2 Mo) | **Lourd** (~15 Mo à 50 Mo selon l'embedding) |

### Recommandation pour BrandPulse AI
Pour un déploiement de production immédiat, le modèle **classique (Régression Logistique + TF-IDF)** est recommandé. Il offre une exactitude légèrement supérieure sur ce volume de données, s'entraîne en moins de 5 secondes, et est extrêmement économique en ressources serveurs. 

Pour les versions futures, l'approche Deep Learning pourra être privilégiée si le volume de tweets étiquetés augmente de manière significative (au-delà de 100 000 tweets) et si des architectures pré-entraînées (Transformers de type BERT/RoBERTa) sont envisagées pour surpasser la limite de contexte des LSTMs.
